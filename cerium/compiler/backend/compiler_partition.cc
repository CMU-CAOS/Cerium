// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cerium/compiler/backend/compiler.h"

namespace Cerium {
namespace Backend {

template <typename ReceiveCipherFn>
Backend::Ciphertext CeriumCompiler::receive_ciphertext_common(
    const Backend::Ciphertext &input1,
    ReceiveCipherFn &&receive_cipher_operands) {
  assert(!input1.ct2.has_value());
  Backend::Ciphertext temp;
  temp.level = input1.level;
  temp.ct0 = make_treg_from(input1.ct0);
  temp.ct1 = make_treg_from(input1.ct1);
  receive_cipher_operands(temp.ct0, input1.ct0, temp.ct1, input1.ct1);
  return temp;
}

template <typename ReceiveFn>
void CeriumCompiler::receive_visit(const Frontend::Term::Ptr &term,
                                     const Frontend::Term::Ptr &args1,
                                     ReceiveFn &&receive_cipher) {
  std::visit(Overloaded{[&](Backend::Ciphertext &input) {
                          auto &output = initValue<Backend::Ciphertext>(term);
                          output.level = input.level;
                          output = receive_cipher(input);
                        },
                        [&](Backend::CiphertextVector &input) {
                          auto &output =
                              initValue<Backend::CiphertextVector>(term);
                          output.vec.resize(input.vec.size());
                          for (size_t i = 0; i < input.vec.size(); i++) {
                            output.vec[i] = receive_cipher(input.vec.at(i));
                          }
                        },
                        [&](auto &arg) {
                          throw std::runtime_error(
                              "Unsupported operation encountered");
                        }},
             Objects.at(args1));
}

void CeriumCompiler::partition(const Backend::PartitionInfo &partition) {
  current_partition_size = partition.partition_size;
  current_partition_id = partition.partition_id;
}

void CeriumCompiler::receive(const Frontend::Term::Ptr &term,
                                   const Frontend::Term::Ptr &args1) {
  auto src_partition_size = args1->getPartitionSize();
  auto src_partition_id = args1->getPartitionId();

  // TODO: Check for superset of partition size

  auto dest_partition_size = term->getPartitionSize();
  auto dest_partition_id = term->getPartitionId();
  Backend::PartitionInfo src_partition{src_partition_id, src_partition_size};
  Backend::PartitionInfo dest_partition{dest_partition_id, dest_partition_size};
  Backend::PartitionInfo kg_partition = dest_partition;

  if (src_partition.partition_size > dest_partition.partition_size) {
    kg_partition = src_partition;
  }

  auto kg_recv = create_kernel_group(kg_partition, KernelGroup::Rec);
  auto recv_instr =
      std::make_shared<ReceiveInstruction>(dest_partition, src_partition);
  kg_recv->push_instruction(recv_instr);

  auto receive_ct = [&](const Backend::Ciphertext &input1) {
    return receive_ciphertext_common(
        input1, [&](Backend::Polynomial &out0, const Backend::Polynomial &in0,
                    Backend::Polynomial &out1, const Backend::Polynomial &in1) {
          recv_instr->add_operands(out0, in0);
          recv_instr->add_operands(out1, in1);
        });
  };

  receive_visit(term, args1, receive_ct);
}

// Receive2 operates when the source partition is larger than the destination
// partition, and allows for multiple destination partitions to receive from the
// same source partition. This allows for more efficient communication patterns
// when the source partition is large and the destination partitions are small.
void CeriumCompiler::receive2(const Frontend::Term::Ptr &term,
                                const Frontend::Term::Ptr &args1) {

  if (!term->has<Frontend::ReceiveDestinationsAttribute>()) {
    throw std::runtime_error(
        "Receive2 op requires ReceiveDestinationsAttribute");
  }
  auto src_partition_size = args1->getPartitionSize();
  auto src_partition_id = args1->getPartitionId();

  // TODO: Check for superset of partition size

  Backend::PartitionInfo src_partition{src_partition_id, src_partition_size};
  Backend::PartitionInfo dest_partition{term->getPartitionId(),
                                        term->getPartitionSize()};
  if (dest_partition.partition_size > src_partition.partition_size) {
    throw std::runtime_error(
        "Src partition size must be larger than destination partition size");
  }

  std::vector<uint32_t> receive_dest_ids;
  auto &receive_dests_attribute =
      term->get<Frontend::ReceiveDestinationsAttribute>();
  if (receive_dests_attribute.empty()) {
    throw std::runtime_error("Receive destinations must not be empty");
  }
  for (auto &[size, id] : receive_dests_attribute) {
    receive_dest_ids.push_back(id);
    if (size != std::get<0>(receive_dests_attribute[0])) {
      throw std::runtime_error(
          "Receive destinations must all have the same partition size");
    }
  }

  auto kg_recv1 = create_kernel_group(src_partition, KernelGroup::Rec2);
  auto recv_instr1 = std::make_shared<ReceiveInstruction2>(
      dest_partition, receive_dest_ids, src_partition);
  kg_recv1->push_instruction(recv_instr1);

  auto kg_recv0 = create_kernel_group(src_partition, KernelGroup::Rec2);
  auto recv_instr0 = std::make_shared<ReceiveInstruction2>(
      dest_partition, receive_dest_ids, src_partition);
  kg_recv0->push_instruction(recv_instr0);

  auto receive_ct = [&](const Backend::Ciphertext &input1) {
    return receive_ciphertext_common(
        input1, [&](Backend::Polynomial &out0, const Backend::Polynomial &in0,
                    Backend::Polynomial &out1, const Backend::Polynomial &in1) {
          recv_instr0->add_operands(out0, in0);
          recv_instr1->add_operands(out1, in1);
        });
  };

  receive_visit(term, args1, receive_ct);
}

void CeriumCompiler::reduce(const Frontend::Term::Ptr &term,
                              const std::vector<Frontend::Term::Ptr> &args) {
  if (!term->has<Frontend::ReduceSourcesAttribute>()) {
    throw std::runtime_error("Reduce op requires ReduceSourcesAttribute");
  }
  auto reduction_source = term->get<Frontend::ReduceSourcesAttribute>();
  if (reduction_source.size() != args.size()) {
    throw std::runtime_error("Reduce source metadata must match operand count");
  }
  if (args.empty()) {
    throw std::runtime_error("Reduce op expects at least one operand");
  }
  std::vector<uint32_t> reduce_source_ids;
  for (auto i = 0; i < args.size(); i++) {
    if (reduction_source[i].first != args[i]->getPartitionSize()) {
      throw std::runtime_error(
          "Reduce source partition size does not match operand metadata");
    }
    if (reduction_source[i].first != args[0]->getPartitionSize()) {
      throw std::runtime_error(
          "Reduce operands must share a source partition size");
    }
    if (reduction_source[i].second != args[i]->getPartitionId()) {
      throw std::runtime_error(
          "Reduce source partition id does not match operand metadata");
    }
    reduce_source_ids.push_back(reduction_source[i].second);
  }
  auto src_partition_size = args[0]->getPartitionSize();
  auto src_partition_id = args[0]->getPartitionId();

  // TODO: Check for superset of partition size

  auto dest_partition_size = term->getPartitionSize();
  auto dest_partition_id = term->getPartitionId();

  if (src_partition_size == 0) {
    throw std::runtime_error("Source partition size must be greater than zero");
  }
  if (dest_partition_size <= src_partition_size) {
    throw std::runtime_error(
        "Dest partition size must be larger than source partition size");
  }
  if (dest_partition_size % src_partition_size != 0) {
    throw std::runtime_error(
        "Dest partition size must be a multiple of the source partition size");
  }
  auto kg_partition =
      Backend::PartitionInfo{dest_partition_id, dest_partition_size};
  auto sync_size = kg_partition.partition_size;

  auto kg_ags1 = create_kernel_group(kg_partition, KernelGroup::Agg3);
  auto ags_instruction1 =
      std::make_shared<AggregateScatterInstruction3>(sync_size,
                                                     src_partition_size);
  kg_ags1->push_instruction(ags_instruction1);

  auto kg_ags2 = create_kernel_group(kg_partition, KernelGroup::Agg3);
  auto ags_instruction2 =
      std::make_shared<AggregateScatterInstruction3>(sync_size,
                                                     src_partition_size);
  kg_ags2->push_instruction(ags_instruction2);

  auto ags_ct = [&](const std::vector<Frontend::Term::Ptr> &inputs) {
    assert(isCipher(inputs[0]));
    auto input1 = getCiphertext(inputs[0]);
    assert(!input1.ct2.has_value());
    Backend::Ciphertext temp;
    temp.level = input1.level;
    temp.ct0 = make_treg_from(input1.ct0);
    temp.ct1 = make_treg_from(input1.ct1);

    auto num_operands = dest_partition_size / src_partition_size;
    std::vector<Backend::Polynomial> ct0s(num_operands, make_zero_treg()),
        ct1s(num_operands, make_zero_treg());

    for (int i = 0; i < reduction_source.size(); i++) {
      assert(isCipher(inputs[i]));
      auto input_ct = getCiphertext(inputs[i]);
      assert(input_ct.level == temp.level);
      assert(!input_ct.ct2.has_value());
      ct0s[reduction_source[i].second] = input_ct.ct0;
      ct1s[reduction_source[i].second] = input_ct.ct1;
    }
    ags_instruction2->add_operands(temp.ct1, ct1s);
    ags_instruction1->add_operands(temp.ct0, ct0s);
    return temp;
  };

  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            output.level = input.level;
            output = ags_ct(args);
          },
          // [&](Backend::CiphertextVector & input) {
          //   auto & output = initValue<Backend::CiphertextVector>(term);
          //   output.vec.resize(input.vec.size());
          //   for(size_t i = 0; i < input.vec.size(); i++) {
          //     output.vec[i] = receive_ct(input.vec.at(i));
          //   }
          // },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args[0]));
}

void CeriumCompiler::set_instruction_limbs() {

  for (size_t j = 0; j < mod_partitions; j++) {
    auto &k_groups = kernel_groups_split[j];
    for (auto i = 0; i < k_groups.size(); i++) {
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &kg = k_groups[i];
      if (!kg) {
        continue;
      }
      for (auto &instr : kg->instructions()) {
        instr->set_limbs(j % kg->partition_size(), kg->partition_size());
      }
    }
  }
}

void CeriumCompiler::set_temporary_values() {

  std::map<Backend::TermIndexType, KernelGroup::IndexType> created_at_kg;
  for (auto i = 0; i < mod_partitions; i++) {
    for (auto j = 0; j < kernel_groups_split[i].size(); j++) {
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &kg = kernel_groups_split[i][j];
      if (!kg) {
        continue;
      }
      auto idx = j;
      auto &instrs = kg->instructions();
      for (auto &instr : instrs) {
        auto &dests = instr->dests();
        for (auto &dst : dests) {
          auto dst_idx = dst.term_idx();
          created_at_kg[dst_idx] = idx;
          if (dst.is_output()) {
            dst.set_temp(false);
          }
        }
        auto &srcs = instr->srcs();
        for (auto &src : srcs) {
          if (src.is_input()) {
            src.set_temp(false);
            continue;
          }
          auto src_idx = src.term_idx();
          if (idx != created_at_kg[src_idx]) {
            src.set_temp(false);
          }
        }
      }
    }
  }
}

void CeriumCompiler::split_kernels_limbwise() {

  using OpCode = KernelInstruction::OpCode;

  auto split_instruction_shares_sud =
      [&](std::shared_ptr<KernelInstruction> &instr, uint16_t partition_id,
          uint16_t partition_size) {
        using LimbType = Backend::Term::LimbType;
        assert(instr->opcode() == OpCode::SuD ||
               instr->opcode() == OpCode::SuD2);
        size_t val_count = 2;
        if (std::dynamic_pointer_cast<SudInstruction>(instr) ||
            std::dynamic_pointer_cast<SudInstruction2>(instr)) {
          val_count = 3;
        }
        auto &srcs = instr->srcs();
        auto &dests = instr->dests();
        for (size_t i = 0; i < srcs.size(); i++) {
          auto &src = srcs[i];
          std::set<uint16_t> shares;
          if (i % val_count == val_count - 1 &&
              src.limbtype() == LimbType::Spl) {
            shares = src.shares();
          } else {
            for (auto &share : src.shares()) {
              if (share % partition_size == partition_id) {
                shares.insert(share);
              }
            }
          }
          Backend::Polynomial src_;
          src_ = make_new_term_share_from(&src);
          src_.set_shares(shares);
          src = src_;
        }
        for (auto &dst : dests) {
          auto dst_ = make_new_term_share_from(&dst);
          std::set<uint16_t> shares;
          for (auto &share : dst.shares()) {
            if (share % partition_size == partition_id) {
              shares.insert(share);
            }
          }
          dst_.set_shares(shares);
          dst = dst_;
        }
      };

  auto split_instruction_shares_mod =
      [&](std::shared_ptr<KernelInstruction> &instr, uint16_t partition_id,
          uint16_t partition_size) {
        using LimbType = Backend::Term::LimbType;
        assert(instr->opcode() == OpCode::Mod);
        auto &srcs = instr->srcs();
        auto &dests = instr->dests();
        for (size_t i = 0; i < srcs.size(); i++) {
          auto &src = srcs[i];
          Backend::Polynomial src_;
          src_ = make_new_term_share_from(&src);
          src_.set_shares(src.shares());
          src = src_;
        }
        for (auto &dst : dests) {
          auto dst_ = make_new_term_share_from(&dst);
          std::set<uint16_t> shares;
          for (auto &share : dst.shares()) {
            if (share % partition_size == partition_id) {
              shares.insert(share);
            }
          }
          dst_.set_shares(shares);
          dst = dst_;
        }
      };

  auto split_instruction_shares_msd =
      [&](std::shared_ptr<KernelInstruction> &instr, uint16_t partition_id,
          uint16_t partition_size) {
        using LimbType = Backend::Term::LimbType;
        assert(instr->opcode() == OpCode::Msd);
        size_t val_count = 3;
        auto &srcs = instr->srcs();
        auto &dests = instr->dests();
        for (size_t i = 0; i < srcs.size(); i++) {
          auto &src = srcs[i];
          std::set<uint16_t> shares;
          if (i % val_count != 0 && src.limbtype() == LimbType::Spl) {
            shares = src.shares();
          } else {
            for (auto &share : src.shares()) {
              if (share % partition_size == partition_id) {
                shares.insert(share);
              }
            }
          }
          Backend::Polynomial src_;
          src_ = make_new_term_share_from(&src);
          src_.set_shares(shares);
          src = src_;
        }
        for (auto &dst : dests) {
          auto dst_ = make_new_term_share_from(&dst);
          std::set<uint16_t> shares;
          for (auto &share : dst.shares()) {
            if (share % partition_size == partition_id) {
              shares.insert(share);
            }
          }
          dst_.set_shares(shares);
          dst = dst_;
        }
      };

  auto split_instruction_shares = [&](std::shared_ptr<KernelInstruction> &instr,
                                      uint16_t partition_id,
                                      uint16_t partition_size) {
    if (instr->opcode() == OpCode::SuD) {
      return split_instruction_shares_sud(instr, partition_id, partition_size);
    } else if (instr->opcode() == OpCode::SuD2) {
      return split_instruction_shares_sud(instr, partition_id, partition_size);
    } else if (instr->opcode() == OpCode::Mod) {
      return split_instruction_shares_mod(instr, partition_id, partition_size);
    } else if (instr->opcode() == OpCode::Msd) {
      return split_instruction_shares_msd(instr, partition_id, partition_size);
    }
    auto &srcs = instr->srcs();
    auto &dests = instr->dests();
    for (auto &src : srcs) {
      std::set<uint16_t> shares;
      for (auto &share : src.shares()) {
        if (share % partition_size == partition_id) {
          shares.insert(share);
        }
      }
      Backend::Polynomial src_;
      if (src.is_eval_key()) {
        auto jt = evalkey_digits_to_term.find({src.term_idx(), shares});
        if (jt != evalkey_digits_to_term.end()) {
          src_ = jt->second;
        } else {
          src_ = make_copy_treg_from(src);
          src_.set_shares(shares);
          src_.term_share()->update_evalkey_term_symbol();
          evalkey_digits_to_term[{src.term_idx(), shares}] = src_;
        }
      } else {
        src_ = make_new_term_share_from(&src);
        src_.set_shares(shares);
      }
      src = src_;
    }
    for (auto &dst : dests) {
      auto dst_ = make_new_term_share_from(&dst);
      std::set<uint16_t> shares;
      for (auto &share : dst.shares()) {
        if (share % partition_size == partition_id) {
          shares.insert(share);
        }
      }
      dst_.set_shares(shares);
      dst = dst_;
    }
  };

  auto split_dist = [&](KernelGroupPtr &kg, uint8_t partition_start,
                        uint8_t partition_end) {
    assert(kg->type() == KernelGroup::Type::Dist);
    auto &instrs = kg->instructions();
    std::vector<KernelGroupPtr> split(mod_partitions);
    for (auto i = 0; i < mod_partitions; i++) {
      if (i < partition_start || i >= partition_end) {
        continue;
      }
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &new_kg = split[i] = std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      for (auto &instr : instrs) {
        auto dist_inst = std::dynamic_pointer_cast<DistRecvInstruction>(instr);
        assert(dist_inst);
        auto new_inst = std::make_shared<DistRecvInstruction2>(
            kg->partition_size(), i % kg->partition_size());
        auto dests = dist_inst->dests();
        auto srcs = dist_inst->srcs();
        assert(dests.size() == srcs.size());
        for (size_t n = 0; n < srcs.size(); n++) {
          auto &src = srcs[n];
          auto shares = src.shares();
          std::vector<Backend::Polynomial> new_srcs;
          for (auto ii = 0; ii < kg->partition_size(); ii++) {
            std::set<uint16_t> sh;
            for (auto &share : shares) {
              if (share % kg->partition_size() == ii % kg->partition_size()) {
                sh.insert(share);
              }
            }
            auto new_term = make_new_term_share_from(&src);
            new_term.set_shares(sh);
            new_srcs.push_back(new_term);
          }
          new_inst->add_operands(dests[n], new_srcs);
        }
        new_kg->push_instruction(std::move(new_inst));
      }
    }
    return split;
  };

  auto split_agg = [&](KernelGroupPtr &kg, uint8_t partition_start,
                       uint8_t partition_end) {
    assert(kg->type() == KernelGroup::Type::Agg);
    auto &instrs = kg->instructions();
    std::vector<KernelGroupPtr> split(mod_partitions);
    for (auto i = 0; i < mod_partitions; i++) {
      if (i < partition_start || i >= partition_end) {
        continue;
      }
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &new_kg = split[i] = std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      for (auto &instr : instrs) {
        auto agg_inst =
            std::dynamic_pointer_cast<AggregateScatterInstruction>(instr);
        assert(agg_inst);
        auto new_inst = std::make_shared<AggregateScatterInstruction2>(
            kg->partition_size(), i % kg->partition_size());
        auto dests = agg_inst->dests();
        auto srcs = agg_inst->srcs();
        assert(dests.size() == srcs.size());
        for (size_t n = 0; n < srcs.size(); n++) {
          auto &dst = dests[n];
          auto shares = dst.shares();
          std::vector<Backend::Polynomial> new_dests;
          for (auto ii = partition_start; ii < partition_end; ii++) {
            std::set<uint16_t> sh;
            for (auto &share : shares) {
              if (share % kg->partition_size() == ii % kg->partition_size()) {
                sh.insert(share);
              }
            }
            auto new_term = make_new_term_share_from(&dst);
            new_term.set_shares(sh);
            new_dests.push_back(new_term);
          }
          new_inst->add_operands(new_dests, srcs[n]);
        }
        new_kg->push_instruction(std::move(new_inst));
      }
    }
    return split;
  };

  auto split_agg3 = [&](KernelGroupPtr &kg, uint8_t partition_start,
                        uint8_t partition_end) {
    assert(kg->type() == KernelGroup::Type::Agg3);
    auto &instrs = kg->instructions();
    std::vector<KernelGroupPtr> split(mod_partitions);
    for (auto i = 0; i < mod_partitions; i++) {
      if (i < partition_start || i >= partition_end) {
        continue;
      }
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &new_kg = split[i] = std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      for (auto &instr : instrs) {
        auto agg_inst =
            std::dynamic_pointer_cast<AggregateScatterInstruction3>(instr);
        assert(agg_inst);
        auto new_inst = std::make_shared<AggregateScatterInstruction2>(
            kg->partition_size(), i % kg->partition_size());
        auto dests = agg_inst->dests();
        auto srcs = agg_inst->srcs();
        auto sync_size = agg_inst->sync_size();
        auto src_partition_size = agg_inst->src_partition_size();
        auto num_srcs_per_dest = sync_size / src_partition_size;
        assert(sync_size == kg->partition_size());
        assert(dests.size() * sync_size == srcs.size() * src_partition_size);
        assert(sync_size == partition_end - partition_start);
        for (size_t n = 0; n < dests.size(); n++) {
          auto &dst = dests[n];
          auto shares = dst.shares();
          std::vector<Backend::Polynomial> new_dests;
          Backend::Polynomial new_src;
          for (auto ii = partition_start; ii < partition_end; ii++) {
            std::set<uint16_t> sh;
            for (auto &share : shares) {
              if (share % kg->partition_size() == ii % kg->partition_size()) {
                sh.insert(share);
              }
            }
            auto new_term = make_new_term_share_from(&dst);
            new_term.set_shares(sh);
            new_dests.push_back(new_term);
          }
          auto src_index =
              n * num_srcs_per_dest + ((i % sync_size) / src_partition_size);
          new_src = srcs[src_index];
          std::set<uint16_t> sh;
          for (auto &share : new_src.shares()) {
            if (share % src_partition_size == i % src_partition_size) {
              sh.insert(share);
            }
          }
          new_src = make_new_term_share_from(&new_src);
          new_src.set_shares(sh);
          new_inst->add_operands(new_dests, new_src);
        }
        new_kg->push_instruction(std::move(new_inst));
      }
    }
    return split;
  };

  auto split_receive = [&](KernelGroupPtr &kg, uint8_t partition_start,
                           uint8_t partition_end) {
    assert(kg->type() == KernelGroup::Type::Rec);
    auto &instrs = kg->instructions();
    std::vector<KernelGroupPtr> split(mod_partitions);
    for (auto i = 0; i < mod_partitions; i++) {
      if (i < partition_start || i >= partition_end) {
        continue;
      }
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &new_kg = split[i] = std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      for (auto &instr : instrs) {
        auto recv_inst = std::dynamic_pointer_cast<ReceiveInstruction>(instr);
        assert(recv_inst);
        auto new_inst = std::make_shared<DistRecvInstruction2>(
            kg->partition_size(), i % kg->partition_size());
        auto dests = recv_inst->dests();
        auto srcs = recv_inst->srcs();
        const auto &src_partition = recv_inst->src_partition();
        const auto &dest_partition = recv_inst->dest_partition();
        assert(dests.size() == srcs.size());
        for (size_t n = 0; n < srcs.size(); n++) {
          auto &src = srcs[n];
          auto shares = src.shares();
          std::vector<Backend::Polynomial> new_srcs;
          for (auto ii = partition_start; ii < partition_end; ii++) {
            std::set<uint16_t> sh;
            if (ii >=
                    src_partition.partition_size * src_partition.partition_id &&
                ii < src_partition.partition_size *
                         (src_partition.partition_id + 1)) {
              for (auto &share : shares) {
                if (share % src_partition.partition_size ==
                    ii % src_partition.partition_size) {
                  sh.insert(share);
                }
              }
            }
            auto new_term = make_new_term_share_from(&src);
            new_term.set_shares(sh);
            new_srcs.push_back(new_term);
          }
          auto dest = make_new_term_share_from(&dests[n]);
          if (i < dest_partition.partition_size * dest_partition.partition_id ||
              i >= dest_partition.partition_size *
                       (dest_partition.partition_id + 1)) {
            dest.set_shares({});
          } else {
            dest.set_shares(dests[n].shares());
          }
          new_inst->add_operands(dest, new_srcs);
        }
        new_kg->push_instruction(std::move(new_inst));
      }
    }
    return split;
  };

  auto split_receive2 = [&](KernelGroupPtr &kg, uint8_t partition_start,
                            uint8_t partition_end) {
    assert(kg->type() == KernelGroup::Type::Rec2);
    auto &instrs = kg->instructions();
    std::vector<KernelGroupPtr> split(mod_partitions);
    for (auto i = 0; i < mod_partitions; i++) {
      if (i < partition_start || i >= partition_end) {
        continue;
      }
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &new_kg = split[i] = std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      for (auto &instr : instrs) {
        auto recv_inst = std::dynamic_pointer_cast<ReceiveInstruction2>(instr);
        assert(recv_inst);
        auto new_inst = std::make_shared<DistRecvInstruction2>(
            kg->partition_size(), i % kg->partition_size());
        auto dests = recv_inst->dests();
        auto srcs = recv_inst->srcs();
        const auto &src_partition = recv_inst->src_partition();
        const auto &dest_partition = recv_inst->dest_partition();
        const auto &dest_partition_ids = recv_inst->dest_partition_ids();
        assert(dests.size() == srcs.size());
        for (size_t n = 0; n < srcs.size(); n++) {
          auto &src = srcs[n];
          auto shares = src.shares();
          std::vector<Backend::Polynomial> new_srcs;
          for (auto ii = partition_start; ii < partition_end; ii++) {
            std::set<uint16_t> sh;
            if (ii >=
                    src_partition.partition_size * src_partition.partition_id &&
                ii < src_partition.partition_size *
                         (src_partition.partition_id + 1)) {
              for (auto &share : shares) {
                if (share % src_partition.partition_size ==
                    ii % src_partition.partition_size) {
                  sh.insert(share);
                }
              }
            }
            auto new_term = make_new_term_share_from(&src);
            new_term.set_shares(sh);
            new_srcs.push_back(new_term);
          }
          auto dest = make_new_term_share_from(&dests[n]);
          bool i_in_destination = false;
          for (auto &dest_partition_id : dest_partition_ids) {
            if (i >= (dest_partition.partition_size * dest_partition_id) &&
                i < (dest_partition.partition_size * (dest_partition_id + 1))) {
              i_in_destination = true;
              break;
            }
          }
          if (i_in_destination) {
            dest.set_shares(dests[n].shares());
          } else {
            dest.set_shares({});
          }
          new_inst->add_operands(dest, new_srcs);
        }
        new_kg->push_instruction(std::move(new_inst));
      }
    }
    return split;
  };

  auto split_all_reduce = [&](KernelGroupPtr &kg, uint8_t partition_start,
                              uint8_t partition_end) {
    assert(kg->type() == KernelGroup::Type::Ard);
    auto &instrs = kg->instructions();
    std::vector<KernelGroupPtr> split(mod_partitions);
    for (auto i = 0; i < mod_partitions; i++) {
      if (i < partition_start || i >= partition_end) {
        continue;
      }
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &new_kg = split[i] = std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      for (auto &instr : instrs) {
        auto ard_inst = std::dynamic_pointer_cast<AllReduceInstruction>(instr);
        assert(ard_inst);
        auto new_inst = std::make_shared<AllReduceInstruction>(
            kg->partition_size(), i % kg->partition_size());
        auto dests = ard_inst->dests();
        auto srcs = ard_inst->srcs();
        assert(dests.size() == srcs.size());
        for (size_t n = 0; n < srcs.size(); n++) {
          auto &dst = dests[n];
          auto &src = srcs[n];
          auto sh = dst.shares();
          auto new_dest = make_new_term_share_from(&dst);
          new_dest.set_shares(sh);
          auto new_src = make_new_term_share_from(&src);
          new_src.set_shares(sh);
          new_inst->add_operands(new_dest, new_src);
        }
        new_kg->push_instruction(std::move(new_inst));
      }
    }
    return split;
  };

  auto split_all_reduce3 = [&](KernelGroupPtr &kg, uint8_t partition_start,
                               uint8_t partition_end) {
    assert(kg->type() == KernelGroup::Type::Ard3);
    auto &instrs = kg->instructions();
    std::vector<KernelGroupPtr> split(mod_partitions);
    for (auto i = 0; i < mod_partitions; i++) {
      if (i < partition_start || i >= partition_end) {
        continue;
      }
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &new_kg = split[i] = std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      for (auto &instr : instrs) {
        auto ard_inst = std::dynamic_pointer_cast<AllReduceInstruction3>(instr);
        assert(ard_inst);
        auto new_inst = std::make_shared<AllReduceInstruction>(
            kg->partition_size(), i % kg->partition_size());
        auto dests = ard_inst->dests();
        auto srcs = ard_inst->srcs();
        auto sync_size = ard_inst->sync_size();
        auto src_partition_size = ard_inst->src_partition_size();
        auto num_srcs_per_sync = ard_inst->num_srcs_per_sync();
        assert(sync_size == kg->partition_size());
        assert(dests.size() * num_srcs_per_sync == srcs.size());
        assert(src_partition_size * num_srcs_per_sync == sync_size);
        assert(sync_size == partition_end - partition_start);
        for (size_t n = 0; n < dests.size(); n++) {
          auto &dst = dests[n];
          auto shares = dst.shares();
          Backend::Polynomial new_src;
          auto src_index =
              n * num_srcs_per_sync + ((i % sync_size) / src_partition_size);
          new_src = srcs[src_index];
          std::set<uint16_t> sh;
          for (auto &share : new_src.shares()) {
            if (share % src_partition_size == i % src_partition_size) {
              sh.insert(share);
            }
          }
          new_src = make_new_term_share_from(&new_src);
          new_src.set_shares(sh);
          auto new_dest = make_new_term_share_from(&dst);
          new_inst->add_operands(new_dest, new_src);
        }
        new_kg->push_instruction(std::move(new_inst));
      }
    }
    return split;
  };

  auto split_call = [&](KernelGroupPtr &kg, uint8_t partition_start,
                        uint8_t partition_end) {
    // TODO: Change this function to handle each arg separately like
    // We don't want to have to keep aligning data at function call boundaries

    assert(kg->type() == KernelGroup::Type::Call);
    auto &instrs = kg->instructions();
    std::vector<KernelGroupPtr> split(mod_partitions);
    for (auto i = 0; i < mod_partitions; i++) {
      if (i < partition_start || i >= partition_end) {
        continue;
      }
      using OpCode = Backend::LimbInstruction::OpCode;
      auto &new_kg = split[i] = std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      auto partition_size = kg->partition_size();
      auto partition_id = i % partition_size;
      for (auto &instr : instrs) {
        auto call_inst = std::dynamic_pointer_cast<CallInstruction>(instr);
        assert(call_inst);
        auto new_inst =
            std::dynamic_pointer_cast<CallInstruction>(call_inst->clone());
        assert(new_inst);
        auto &dests = new_inst->dests_map();
        auto &srcs = new_inst->srcs_map();

        const auto &args_partition_info = call_inst->args_partition_info();

        for (auto &[k, src] : srcs) {
          std::set<uint16_t> shares;
          auto &part_info = args_partition_info.at(k);
          for (auto &share : src.shares()) {
            if ((part_info.partition_id + (share % part_info.partition_size)) %
                    partition_size ==
                partition_id) {
              shares.insert(share);
            }
          }
          Backend::Polynomial src_;
          src_ = make_new_term_share_from(&src);
          src_.set_shares(shares);
          src = src_;
        }
        for (auto &[k, dst] : dests) {
          auto dst_ = make_new_term_share_from(&dst);
          std::set<uint16_t> shares;
          auto &part_info = args_partition_info.at(k);
          for (auto &share : dst.shares()) {
            if ((part_info.partition_id + (share % part_info.partition_size)) %
                    partition_size ==
                partition_id) {
              shares.insert(share);
            }
          }
          dst_.set_shares(shares);
          dst = dst_;
        }
        new_kg->push_instruction(std::move(new_inst));
      }
    }
    return split;
  };

  kernel_groups_split.resize(mod_partitions);
  for (auto i = 0; i < mod_partitions; i++) {
    kernel_groups_split[i].resize(kernel_groups.size());
  }
  for (auto j = 0; j < kernel_groups.size(); j++) {
    auto &kg = kernel_groups[j];
    auto kg_partition_start = kg->partition_id() * kg->partition_size();
    auto kg_partition_end = kg_partition_start + kg->partition_size();
    auto &instrs = kg->instructions();
    if (kg->type() == KernelGroup::Type::Dist) {
      auto split = split_dist(kg, kg_partition_start, kg_partition_end);
      for (auto i = 0; i < mod_partitions; i++) {
        kernel_groups_split[i][j] = split[i];
      }
      continue;
    } else if (kg->type() == KernelGroup::Type::Agg) {
      auto split = split_agg(kg, kg_partition_start, kg_partition_end);
      for (auto i = 0; i < mod_partitions; i++) {
        kernel_groups_split[i][j] = split[i];
      }
      continue;
    } else if (kg->type() == KernelGroup::Type::Agg3) {
      auto split = split_agg3(kg, kg_partition_start, kg_partition_end);
      for (auto i = 0; i < mod_partitions; i++) {
        kernel_groups_split[i][j] = split[i];
      }
      continue;
    } else if (kg->type() == KernelGroup::Type::Rec) {
      auto split = split_receive(kg, kg_partition_start, kg_partition_end);
      for (auto i = 0; i < mod_partitions; i++) {
        kernel_groups_split[i][j] = split[i];
      }
      continue;
    } else if (kg->type() == KernelGroup::Type::Rec2) {
      auto split = split_receive2(kg, kg_partition_start, kg_partition_end);
      for (auto i = 0; i < mod_partitions; i++) {
        kernel_groups_split[i][j] = split[i];
      }
      continue;
    } else if (kg->type() == KernelGroup::Type::Ard) {
      auto split = split_all_reduce(kg, kg_partition_start, kg_partition_end);
      for (auto i = 0; i < mod_partitions; i++) {
        kernel_groups_split[i][j] = split[i];
      }
      continue;
    } else if (kg->type() == KernelGroup::Type::Ard3) {
      auto split = split_all_reduce3(kg, kg_partition_start, kg_partition_end);
      for (auto i = 0; i < mod_partitions; i++) {
        kernel_groups_split[i][j] = split[i];
      }
      continue;
    } else if (kg->type() == KernelGroup::Type::Call) {
      auto split = split_call(kg, kg_partition_start, kg_partition_end);
      for (auto i = 0; i < mod_partitions; i++) {
        kernel_groups_split[i][j] = split[i];
      }
      continue;
    }
    for (auto i = 0; i < mod_partitions; i++) {
      using OpCode = Backend::LimbInstruction::OpCode;
      if (i < kg_partition_start || i >= kg_partition_end) {
        continue;
      }
      auto &new_kg = kernel_groups_split[i][j] =
          std::make_shared<KernelGroup>(kg);
      auto idx = new_kg->index();
      for (auto &instr : instrs) {
        auto instr_clone = instr->clone();
        if (!kg->dont_split_modular()) {
          split_instruction_shares(instr_clone, i % kg->partition_size(),
                                   kg->partition_size());
        }
        new_kg->push_instruction(std::move(instr_clone));
      }
    }
  }
  kernel_groups = kernel_groups_split[0];
}

#ifdef CERIUM_USE_SPLIT_KERNELS_LIMBWISE_V2
void CeriumCompiler::split_kernels_limbwise_v2() {
  using OpCode = KernelInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;

  auto shares_for_partition = [](const std::set<uint16_t> &shares,
                                 uint16_t partition_id,
                                 uint16_t partition_size) {
    std::set<uint16_t> result;
    for (auto share : shares) {
      if (share % partition_size == partition_id) {
        result.insert(share);
      }
    }
    return result;
  };

  auto clone_with_shares = [&](const Backend::Polynomial &poly,
                               const std::set<uint16_t> &shares) {
    auto clone = make_new_term_share_from(&poly);
    clone.set_shares(shares);
    return clone;
  };

  auto clone_eval_key_with_shares = [&](const Backend::Polynomial &poly,
                                        const std::set<uint16_t> &shares) {
    auto cached = evalkey_digits_to_term.find({poly.term_idx(), shares});
    if (cached != evalkey_digits_to_term.end()) {
      return cached->second;
    }

    auto clone = make_copy_treg_from(poly);
    clone.set_shares(shares);
    clone.term_share()->update_evalkey_term_symbol();
    evalkey_digits_to_term[{poly.term_idx(), shares}] = clone;
    return clone;
  };

  auto split_regular_src = [&](const Backend::Polynomial &src,
                               uint16_t partition_id, uint16_t partition_size) {
    auto shares =
        shares_for_partition(src.shares(), partition_id, partition_size);
    if (src.is_eval_key()) {
      return clone_eval_key_with_shares(src, shares);
    }
    return clone_with_shares(src, shares);
  };

  auto split_regular_dest = [&](const Backend::Polynomial &dst,
                                uint16_t partition_id,
                                uint16_t partition_size) {
    return clone_with_shares(
        dst, shares_for_partition(dst.shares(), partition_id, partition_size));
  };

  auto split_instruction_shares_sud =
      [&](std::shared_ptr<KernelInstruction> &instr, uint16_t partition_id,
          uint16_t partition_size) {
        assert(instr->opcode() == OpCode::SuD ||
               instr->opcode() == OpCode::SuD2);
        size_t val_count = 2;
        if (std::dynamic_pointer_cast<SudInstruction>(instr) ||
            std::dynamic_pointer_cast<SudInstruction2>(instr)) {
          val_count = 3;
        }

        auto &srcs = instr->srcs();
        for (size_t i = 0; i < srcs.size(); i++) {
          auto shares =
              (i % val_count == val_count - 1 &&
               srcs[i].limbtype() == LimbType::Spl)
                  ? srcs[i].shares()
                  : shares_for_partition(srcs[i].shares(), partition_id,
                                         partition_size);
          srcs[i] = clone_with_shares(srcs[i], shares);
        }

        for (auto &dst : instr->dests()) {
          dst = split_regular_dest(dst, partition_id, partition_size);
        }
      };

  auto split_instruction_shares_mod =
      [&](std::shared_ptr<KernelInstruction> &instr, uint16_t partition_id,
          uint16_t partition_size) {
        assert(instr->opcode() == OpCode::Mod);
        for (auto &src : instr->srcs()) {
          src = clone_with_shares(src, src.shares());
        }
        for (auto &dst : instr->dests()) {
          dst = split_regular_dest(dst, partition_id, partition_size);
        }
      };

  auto split_instruction_shares_msd =
      [&](std::shared_ptr<KernelInstruction> &instr, uint16_t partition_id,
          uint16_t partition_size) {
        throw std::runtime_error("Deprectated");
        assert(instr->opcode() == OpCode::Msd);
        size_t val_count = 3;
        auto &srcs = instr->srcs();
        for (size_t i = 0; i < srcs.size(); i++) {
          auto shares =
              (i % val_count != 0 && srcs[i].limbtype() == LimbType::Spl)
                  ? srcs[i].shares()
                  : shares_for_partition(srcs[i].shares(), partition_id,
                                         partition_size);
          srcs[i] = clone_with_shares(srcs[i], shares);
        }

        for (auto &dst : instr->dests()) {
          dst = split_regular_dest(dst, partition_id, partition_size);
        }
      };

  auto split_instruction_shares = [&](std::shared_ptr<KernelInstruction> &instr,
                                      uint16_t partition_id,
                                      uint16_t partition_size) {
    if (instr->opcode() == OpCode::SuD || instr->opcode() == OpCode::SuD2) {
      return split_instruction_shares_sud(instr, partition_id, partition_size);
    }
    if (instr->opcode() == OpCode::Mod) {
      return split_instruction_shares_mod(instr, partition_id, partition_size);
    }
    if (instr->opcode() == OpCode::Msd) {
      return split_instruction_shares_msd(instr, partition_id, partition_size);
    }

    for (auto &src : instr->srcs()) {
      src = split_regular_src(src, partition_id, partition_size);
    }
    for (auto &dst : instr->dests()) {
      dst = split_regular_dest(dst, partition_id, partition_size);
    }
  };

  auto split_kernel_group = [&](KernelGroupPtr &kg, auto rewrite_instruction) {
    auto partition_start = kg->partition_id() * kg->partition_size();
    auto partition_end = partition_start + kg->partition_size();
    std::vector<KernelGroupPtr> split(mod_partitions);

    for (uint16_t partition = 0; partition < mod_partitions; partition++) {
      if (partition < partition_start || partition >= partition_end) {
        continue;
      }
      auto &new_kg = split[partition] = std::make_shared<KernelGroup>(kg);
      for (auto &instr : kg->instructions()) {
        rewrite_instruction(new_kg, instr, partition, partition_start,
                            partition_end);
      }
    }
    return split;
  };

  auto push_split = [&](size_t kg_index, std::vector<KernelGroupPtr> &&split) {
    for (size_t partition = 0; partition < kernel_groups_split.size();
         partition++) {
      kernel_groups_split[partition][kg_index] = split[partition];
    }
  };

  auto split_dest_across_group = [&](const Backend::Polynomial &dst,
                                     uint16_t partition_start,
                                     uint16_t partition_end,
                                     uint16_t partition_size) {
    std::vector<Backend::Polynomial> split_dests;
    for (uint16_t partition = partition_start; partition < partition_end;
         partition++) {
      split_dests.push_back(clone_with_shares(
          dst, shares_for_partition(dst.shares(), partition % partition_size,
                                    partition_size)));
    }
    return split_dests;
  };

  auto split_dist_srcs = [&](const Backend::Polynomial &src,
                             uint16_t partition_size) {
    std::vector<Backend::Polynomial> split_srcs;
    for (uint16_t partition_id = 0; partition_id < partition_size;
         partition_id++) {
      split_srcs.push_back(clone_with_shares(
          src,
          shares_for_partition(src.shares(), partition_id, partition_size)));
    }
    return split_srcs;
  };

  auto split_receive_srcs = [&](const Backend::Polynomial &src,
                                const Backend::PartitionInfo &src_partition,
                                uint16_t partition_start,
                                uint16_t partition_end) {
    std::vector<Backend::Polynomial> split_srcs;
    for (uint16_t partition = partition_start; partition < partition_end;
         partition++) {
      std::set<uint16_t> shares;
      if (partition >=
              src_partition.partition_size * src_partition.partition_id &&
          partition <
              src_partition.partition_size * (src_partition.partition_id + 1)) {
        shares = shares_for_partition(src.shares(),
                                      partition % src_partition.partition_size,
                                      src_partition.partition_size);
      }
      split_srcs.push_back(clone_with_shares(src, shares));
    }
    return split_srcs;
  };

  auto split_destination_for_receive =
      [&](const Backend::Polynomial &dst, uint16_t partition,
          const Backend::PartitionInfo &dest_partition) {
        auto dest = make_new_term_share_from(&dst);
        if (partition <
                dest_partition.partition_size * dest_partition.partition_id ||
            partition >= dest_partition.partition_size *
                             (dest_partition.partition_id + 1)) {
          dest.set_shares({});
        } else {
          dest.set_shares(dst.shares());
        }
        return dest;
      };

  auto split_destination_for_receive2 =
      [&](const Backend::Polynomial &dst, uint16_t partition,
          const Backend::PartitionInfo &dest_partition,
          const std::vector<uint32_t> &dest_partition_ids) {
        bool in_destination = false;
        for (auto dest_partition_id : dest_partition_ids) {
          if (partition >= dest_partition.partition_size * dest_partition_id &&
              partition <
                  dest_partition.partition_size * (dest_partition_id + 1)) {
            in_destination = true;
            break;
          }
        }

        auto dest = make_new_term_share_from(&dst);
        dest.set_shares(in_destination ? dst.shares() : std::set<uint16_t>{});
        return dest;
      };

  auto split_dist = [&](KernelGroupPtr &kg) {
    assert(kg->type() == KernelGroup::Type::Dist);
    return split_kernel_group(
        kg, [&](KernelGroupPtr &new_kg,
                const std::shared_ptr<KernelInstruction> &instr,
                uint16_t partition, uint16_t, uint16_t) {
          auto dist_inst =
              std::dynamic_pointer_cast<DistRecvInstruction>(instr);
          assert(dist_inst);
          auto new_inst = std::make_shared<DistRecvInstruction2>(
              kg->partition_size(), partition % kg->partition_size());
          auto dests = dist_inst->dests();
          auto srcs = dist_inst->srcs();
          assert(dests.size() == srcs.size());
          for (size_t n = 0; n < srcs.size(); n++) {
            new_inst->add_operands(
                dests[n], split_dist_srcs(srcs[n], kg->partition_size()));
          }
          new_kg->push_instruction(std::move(new_inst));
        });
  };

  auto split_agg = [&](KernelGroupPtr &kg) {
    assert(kg->type() == KernelGroup::Type::Agg);
    return split_kernel_group(
        kg,
        [&](KernelGroupPtr &new_kg,
            const std::shared_ptr<KernelInstruction> &instr, uint16_t partition,
            uint16_t partition_start, uint16_t partition_end) {
          auto agg_inst =
              std::dynamic_pointer_cast<AggregateScatterInstruction>(instr);
          assert(agg_inst);
          auto new_inst = std::make_shared<AggregateScatterInstruction2>(
              kg->partition_size(), partition % kg->partition_size());
          auto dests = agg_inst->dests();
          auto srcs = agg_inst->srcs();
          assert(dests.size() == srcs.size());
          for (size_t n = 0; n < srcs.size(); n++) {
            new_inst->add_operands(
                split_dest_across_group(dests[n], partition_start,
                                        partition_end, kg->partition_size()),
                srcs[n]);
          }
          new_kg->push_instruction(std::move(new_inst));
        });
  };

  auto split_agg3 = [&](KernelGroupPtr &kg) {
    assert(kg->type() == KernelGroup::Type::Agg3);
    return split_kernel_group(
        kg,
        [&](KernelGroupPtr &new_kg,
            const std::shared_ptr<KernelInstruction> &instr, uint16_t partition,
            uint16_t partition_start, uint16_t partition_end) {
          auto agg_inst =
              std::dynamic_pointer_cast<AggregateScatterInstruction3>(instr);
          assert(agg_inst);
          auto new_inst = std::make_shared<AggregateScatterInstruction2>(
              kg->partition_size(), partition % kg->partition_size());
          auto dests = agg_inst->dests();
          auto srcs = agg_inst->srcs();
          auto sync_size = agg_inst->sync_size();
          auto src_partition_size = agg_inst->src_partition_size();
          auto num_srcs_per_dest = sync_size / src_partition_size;
          assert(sync_size == kg->partition_size());
          assert(dests.size() * sync_size == srcs.size() * src_partition_size);
          assert(sync_size == partition_end - partition_start);
          for (size_t n = 0; n < dests.size(); n++) {
            auto new_src =
                srcs[n * num_srcs_per_dest +
                     ((partition % sync_size) / src_partition_size)];
            std::set<uint16_t> sh;
            for (auto &share : new_src.shares()) {
              if (share % src_partition_size ==
                  partition % src_partition_size) {
                sh.insert(share);
              }
            }
            new_src = make_new_term_share_from(&new_src);
            new_src.set_shares(sh);
            new_inst->add_operands(
                split_dest_across_group(dests[n], partition_start,
                                        partition_end, kg->partition_size()),
                new_src);
          }
          new_kg->push_instruction(std::move(new_inst));
        });
  };

  auto split_receive = [&](KernelGroupPtr &kg) {
    assert(kg->type() == KernelGroup::Type::Rec);
    return split_kernel_group(
        kg,
        [&](KernelGroupPtr &new_kg,
            const std::shared_ptr<KernelInstruction> &instr, uint16_t partition,
            uint16_t partition_start, uint16_t partition_end) {
          auto recv_inst = std::dynamic_pointer_cast<ReceiveInstruction>(instr);
          assert(recv_inst);
          auto new_inst = std::make_shared<DistRecvInstruction2>(
              kg->partition_size(), partition % kg->partition_size());
          auto dests = recv_inst->dests();
          auto srcs = recv_inst->srcs();
          assert(dests.size() == srcs.size());
          for (size_t n = 0; n < srcs.size(); n++) {
            auto srcs_split =
                split_receive_srcs(srcs[n], recv_inst->src_partition(),
                                   partition_start, partition_end);
            auto dest = split_destination_for_receive(
                dests[n], partition, recv_inst->dest_partition());
            new_inst->add_operands(dest, srcs_split);
          }
          new_kg->push_instruction(std::move(new_inst));
        });
  };

  auto split_receive2 = [&](KernelGroupPtr &kg) {
    assert(kg->type() == KernelGroup::Type::Rec2);
    return split_kernel_group(
        kg,
        [&](KernelGroupPtr &new_kg,
            const std::shared_ptr<KernelInstruction> &instr, uint16_t partition,
            uint16_t partition_start, uint16_t partition_end) {
          auto recv_inst =
              std::dynamic_pointer_cast<ReceiveInstruction2>(instr);
          assert(recv_inst);
          auto new_inst = std::make_shared<DistRecvInstruction2>(
              kg->partition_size(), partition % kg->partition_size());
          auto dests = recv_inst->dests();
          auto srcs = recv_inst->srcs();
          assert(dests.size() == srcs.size());
          for (size_t n = 0; n < srcs.size(); n++) {
            auto srcs_split =
                split_receive_srcs(srcs[n], recv_inst->src_partition(),
                                   partition_start, partition_end);
            auto dest = split_destination_for_receive2(
                dests[n], partition, recv_inst->dest_partition(),
                recv_inst->dest_partition_ids());
            new_inst->add_operands(dest, srcs_split);
          }
          new_kg->push_instruction(std::move(new_inst));
        });
  };

  auto split_all_reduce = [&](KernelGroupPtr &kg) {
    assert(kg->type() == KernelGroup::Type::Ard);
    return split_kernel_group(
        kg, [&](KernelGroupPtr &new_kg,
                const std::shared_ptr<KernelInstruction> &instr,
                uint16_t partition, uint16_t, uint16_t) {
          auto ard_inst =
              std::dynamic_pointer_cast<AllReduceInstruction>(instr);
          assert(ard_inst);
          auto new_inst = std::make_shared<AllReduceInstruction>(
              kg->partition_size(), partition % kg->partition_size());
          auto dests = ard_inst->dests();
          auto srcs = ard_inst->srcs();
          assert(dests.size() == srcs.size());
          for (size_t n = 0; n < srcs.size(); n++) {
            auto shares = dests[n].shares();
            auto new_dest = clone_with_shares(dests[n], shares);
            auto new_src = clone_with_shares(srcs[n], shares);
            new_inst->add_operands(new_dest, new_src);
          }
          new_kg->push_instruction(std::move(new_inst));
        });
  };

  auto split_all_reduce3 = [&](KernelGroupPtr &kg) {
    assert(kg->type() == KernelGroup::Type::Ard3);
    return split_kernel_group(
        kg,
        [&](KernelGroupPtr &new_kg,
            const std::shared_ptr<KernelInstruction> &instr, uint16_t partition,
            uint16_t partition_start, uint16_t partition_end) {
          auto ard_inst =
              std::dynamic_pointer_cast<AllReduceInstruction3>(instr);
          assert(ard_inst);
          auto new_inst = std::make_shared<AllReduceInstruction>(
              kg->partition_size(), partition % kg->partition_size());
          auto dests = ard_inst->dests();
          auto srcs = ard_inst->srcs();
          auto sync_size = ard_inst->sync_size();
          auto src_partition_size = ard_inst->src_partition_size();
          auto num_srcs_per_sync = ard_inst->num_srcs_per_sync();
          assert(sync_size == kg->partition_size());
          assert(dests.size() * num_srcs_per_sync == srcs.size());
          assert(src_partition_size * num_srcs_per_sync == sync_size);
          assert(sync_size == partition_end - partition_start);
          for (size_t n = 0; n < dests.size(); n++) {
            auto new_dest = make_new_term_share_from(&dests[n]);
            auto src_index =
                n * num_srcs_per_sync +
                ((partition % sync_size) / src_partition_size);
            auto new_src = srcs[src_index];
            std::set<uint16_t> sh;
            for (auto &share : new_src.shares()) {
              if (share % src_partition_size ==
                  partition % src_partition_size) {
                sh.insert(share);
              }
            }
            new_src = make_new_term_share_from(&new_src);
            new_src.set_shares(sh);
            new_inst->add_operands(new_dest, new_src);
          }
          new_kg->push_instruction(std::move(new_inst));
        });
  };

  auto split_call_arg = [&](Backend::Polynomial &arg, const std::string &name,
                            const Backend::PartitionInfo &part_info,
                            uint16_t partition, uint16_t partition_size) {
    auto partition_id = partition % partition_size;
    std::set<uint16_t> shares;
    for (auto share : arg.shares()) {
      if ((part_info.partition_id + (share % part_info.partition_size)) %
              partition_size ==
          partition_id) {
        shares.insert(share);
      }
    }
    arg = clone_with_shares(arg, shares);
  };

  auto split_call = [&](KernelGroupPtr &kg) {
    assert(kg->type() == KernelGroup::Type::Call);
    return split_kernel_group(
        kg, [&](KernelGroupPtr &new_kg,
                const std::shared_ptr<KernelInstruction> &instr,
                uint16_t partition, uint16_t, uint16_t) {
          auto call_inst = std::dynamic_pointer_cast<CallInstruction>(instr);
          assert(call_inst);
          auto new_inst =
              std::dynamic_pointer_cast<CallInstruction>(call_inst->clone());
          assert(new_inst);

          const auto &args_partition_info = call_inst->args_partition_info();
          for (auto &[name, src] : new_inst->srcs_map()) {
            split_call_arg(src, name, args_partition_info.at(name), partition,
                           kg->partition_size());
          }
          for (auto &[name, dst] : new_inst->dests_map()) {
            split_call_arg(dst, name, args_partition_info.at(name), partition,
                           kg->partition_size());
          }
          new_kg->push_instruction(std::move(new_inst));
        });
  };

  auto split_generic = [&](KernelGroupPtr &kg) {
    return split_kernel_group(
        kg, [&](KernelGroupPtr &new_kg,
                const std::shared_ptr<KernelInstruction> &instr,
                uint16_t partition, uint16_t, uint16_t) {
          auto instr_clone = instr->clone();
          if (!kg->dont_split_modular()) {
            split_instruction_shares(instr_clone,
                                     partition % kg->partition_size(),
                                     kg->partition_size());
          }
          new_kg->push_instruction(std::move(instr_clone));
        });
  };

  kernel_groups_split.resize(mod_partitions);
  for (auto partition = 0; partition < mod_partitions; partition++) {
    kernel_groups_split[partition].resize(kernel_groups.size());
  }
  for (size_t kg_index = 0; kg_index < kernel_groups.size(); kg_index++) {
    auto &kg = kernel_groups[kg_index];
    switch (kg->type()) {
    case KernelGroup::Type::Dist:
      push_split(kg_index, split_dist(kg));
      break;
    case KernelGroup::Type::Agg:
      push_split(kg_index, split_agg(kg));
      break;
    case KernelGroup::Type::Agg3:
      push_split(kg_index, split_agg3(kg));
      break;
    case KernelGroup::Type::Rec:
      push_split(kg_index, split_receive(kg));
      break;
    case KernelGroup::Type::Rec2:
      push_split(kg_index, split_receive2(kg));
      break;
    case KernelGroup::Type::Ard:
      push_split(kg_index, split_all_reduce(kg));
      break;
    case KernelGroup::Type::Ard3:
      push_split(kg_index, split_all_reduce3(kg));
      break;
    case KernelGroup::Type::Call:
      push_split(kg_index, split_call(kg));
      break;
    default:
      push_split(kg_index, split_generic(kg));
      break;
    }
  }

  kernel_groups = kernel_groups_split[0];
}
#endif

} // namespace Backend
} // namespace Cerium
