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

void CeriumCompiler::validate_partition_state(
    const PartitionInfo &partition) const {
  if (mod_partitions == 0) {
    throw std::runtime_error("mod_partitions must be greater than zero");
  }
  if (partition.partition_size == 0) {
    throw std::runtime_error("partition_size must be greater than zero");
  }
  if (partition.partition_size > mod_partitions) {
    throw std::runtime_error(
        "partition_size " + std::to_string(partition.partition_size) +
        " exceeds mod_partitions " + std::to_string(mod_partitions));
  }
}

Backend::Polynomial CeriumCompiler::make_term(
    uint16_t level,
    const std::function<void(Backend::Term &)> &configure_term) {
  auto index = nextTermIndex++;
  auto term = std::make_shared<Backend::Term>(index);
  terms[index] = term;
  configure_term(*term);

  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;
  return Backend::Polynomial(term_share, level);
}

Backend::Polynomial CeriumCompiler::make_input(uint16_t level) {
  return make_term(level, [](Backend::Term &term) { term.set_as_input(); });
}

Backend::Polynomial CeriumCompiler::make_function_input(uint16_t level) {
  return make_term(level, [](Backend::Term &term) {
    term.set_as_input();
    term.set_as_function_arg(true);
  });
}

Backend::Polynomial CeriumCompiler::make_plaintext(uint16_t level) {
  return make_term(level, [](Backend::Term &term) { term.set_as_plaintext(); });
}

Backend::Polynomial CeriumCompiler::make_scalar(uint16_t level) {
  return make_term(level, [](Backend::Term &term) {
    term.set_as_plaintext();
    term.set_as_scalar();
  });
}

std::shared_ptr<KernelGroup>
CeriumCompiler::create_kernel_group(const PartitionInfo &partition,
                                      KernelGroup::Type type) {
  validate_partition_state(partition);
  auto kg = std::make_shared<KernelGroup>(kernel_groups.size(),
                                          partition.partition_size,
                                          partition.partition_id, type);
  kernel_groups.push_back(kg);
  return kg;
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size});
}

std::shared_ptr<KernelGroup>
CeriumCompiler::create_kernel_group(std::shared_ptr<KernelGroup> kg_other) {
  validate_partition_state(
      PartitionInfo{kg_other->partition_id(), kg_other->partition_size()});
  auto kg = std::make_shared<KernelGroup>(kernel_groups.size(), kg_other);
  kernel_groups.push_back(kg);
  assert(kg->dont_split_modular() == kg_other->dont_split_modular());
  return kg;
}

std::shared_ptr<KernelGroup>
CeriumCompiler::create_kernel_group_intt(const PartitionInfo &partition) {
  validate_partition_state(partition);
  auto kg = std::make_shared<KernelGroup>(
      kernel_groups.size(), partition.partition_size, partition.partition_id,
      KernelGroup::Type::Int);
  kernel_groups.push_back(kg);
  return kg;
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_intt() {
  return create_kernel_group_intt(
      PartitionInfo{current_partition_id, current_partition_size});
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_bconv() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size},
      KernelGroup::Type::Bco);
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_ntt() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size},
      KernelGroup::Type::Ntt);
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_sud2() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size},
      KernelGroup::Type::SuD2);
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_pmul() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size},
      KernelGroup::Type::Pmu);
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_dist() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size},
      KernelGroup::Type::Dist);
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_agg() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size},
      KernelGroup::Type::Agg);
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_ard() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size},
      KernelGroup::Type::Ard);
}

std::shared_ptr<KernelGroup> CeriumCompiler::create_kernel_group_call() {
  return create_kernel_group(
      PartitionInfo{current_partition_id, current_partition_size},
      KernelGroup::Type::Call);
}

Backend::Polynomial CeriumCompiler::make_treg(uint16_t level) {
  auto index = nextTermIndex++;
  auto term = std::make_shared<Backend::Term>(index);
  terms[index] = term;
  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;
  return Backend::Polynomial(term_share, level);
}

Backend::Polynomial CeriumCompiler::make_zero_treg() {
  auto treg = make_treg(0);
  treg.set_as_zero(true);
  return treg;
}

Backend::Polynomial
CeriumCompiler::make_treg_from(const Backend::Polynomial &other) {
  auto index = nextTermIndex++;
  auto term = std::make_shared<Backend::Term>(index);
  terms[index] = term;
  auto level = other.level();

  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;
  return Backend::Polynomial(term_share, level);
}

Backend::Polynomial
CeriumCompiler::make_copy_treg_from(const Backend::Polynomial &other) {
  auto index = nextTermIndex++;
  auto term = std::make_shared<Backend::Term>(index, other.term());
  terms[index] = term;
  auto level = other.level();

  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;
  auto poly = Backend::Polynomial(term_share, level);
  poly.set_bignode_idx(other.bignode_index());
  poly.set_limbtype(other.limbtype());
  poly.set_limbs(other.limbs());
  poly.set_shares(other.shares());
  poly.set_adjustment_size(other.adjustment_size());
  poly.set_ext_adjustment_size(other.ext_adjustment_size());
  poly.set_upper_ext_adjustment_size(other.upper_ext_adjustment_size());
  return poly;
}

Backend::Polynomial
CeriumCompiler::make_rescale_treg_from(const Backend::Polynomial &other,
                                         const uint32_t rescale_levels) {
  auto index = nextTermIndex++;
  auto term = std::make_shared<Backend::Term>(index);
  terms[index] = term;
  assert(other.level() > rescale_levels);
  auto level = other.level() - rescale_levels;

  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;
  return Backend::Polynomial(term_share, level);
}

Backend::Polynomial
CeriumCompiler::make_bcor_from(const Backend::Polynomial &other) {
  auto index = nextTermIndex++;
  auto term = std::make_shared<Backend::Term>(index);
  terms[index] = term;
  term->set_as_bcor();
  auto level = other.level();

  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;
  return Backend::Polynomial(term_share, level);
}

void CeriumCompiler::set_output(Frontend::Term::Ptr &args1,
                                  const std::string &name) {
  Backend::Ciphertext &input1 =
      std::get<Backend::Ciphertext>(Objects.at(args1));
  assert(!input1.ct2.has_value());
  input1.ct0.set_output(true);
  input1.ct1.set_output(true);

  input1.ct0.set_symbol(name + ":c0");
  input1.ct1.set_symbol(name + ":c1");
}

Backend::Polynomial
CeriumCompiler::make_new_term_share_from(const Backend::Polynomial *other) {
  auto level = other->level();
  auto term = other->term();
  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;
  auto poly = Backend::Polynomial(term_share, level);
  poly.set_bignode_idx(other->bignode_index());
  poly.set_limbtype(other->limbtype());
  poly.set_adjustment_size(other->adjustment_size());
  poly.set_ext_adjustment_size(other->ext_adjustment_size());
  poly.set_upper_ext_adjustment_size(other->upper_ext_adjustment_size());
  return poly;
}

Backend::Polynomial
CeriumCompiler::make_new_term_share_from(const Backend::Polynomial &other) {
  return make_new_term_share_from(&other);
}

Backend::Polynomial
CeriumCompiler::make_modswitch_treg_from(Backend::Polynomial &other) {
  auto term = other.term();
  auto level = other.level() - 1;
  assert(level > 0);

  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;

  auto poly = Backend::Polynomial(term_share, level);
  poly.set_bignode_idx(other.bignode_index());
  return poly;
}

Backend::Polynomial CeriumCompiler::make_double_rescale_treg_from(
    const Backend::Polynomial &other) {
  auto index = nextTermIndex++;
  auto term = std::make_shared<Backend::Term>(index);
  terms[index] = term;
  auto level = other.level() - 2;
  assert(level > 0);

  auto index_ts = nextTermShareIndex++;
  auto term_share = std::make_shared<Backend::TermShare>(index_ts, term, level);
  term_shares[index_ts] = term_share;
  return Backend::Polynomial(term_share, level);
}

CeriumCompiler::RuntimeValuePair
CeriumCompiler::get_runtime_value_pair(const Frontend::Term::Ptr &args1,
                                         const Frontend::Term::Ptr &args2) {
  auto o1 = Objects.at(args1);
  auto o2 = Objects.at(args2);

  return std::visit(Overloaded{[&](auto &lhs) -> RuntimeValuePair {
                      return std::visit(
                          Overloaded{[&](auto &rhs) -> RuntimeValuePair {
                            return RuntimeValuePair{std::make_pair(lhs, rhs)};
                          }},
                          o2);
                    }},
                    o1);
}

void CeriumCompiler::add(const Frontend::Term::Ptr &term,
                               const Frontend::Term::Ptr &args1,
                               const Frontend::Term::Ptr &args2) {

  if (isPlain2(args1)) {
    assert(!isPlain2(args2));
    add(term, args2, args1);
    return;
  }

  auto pair = get_runtime_value_pair(args1, args2);

  std::shared_ptr<KernelGroup> kg = create_kernel_group();
  std::shared_ptr<BinOpInstruction> add_instr =
      std::make_shared<BinOpInstruction>(Backend::LimbInstruction::OpCode::Add);
  kg->push_instruction(add_instr);

  auto add_ct_ct = [&](Backend::Ciphertext &output,
                       const Backend::Ciphertext &input1,
                       const Backend::Ciphertext &input2) {
    output.level = input1.level;
    if (output.level != input2.level) {
      throw std::runtime_error("Ciphertext levels do not match");
    }
    output.ct0 = make_treg_from(input1.ct0);
    output.ct1 = make_treg_from(input1.ct1);
    add_instr->add_operands(output.ct0, input1.ct0, input2.ct0);
    add_instr->add_operands(output.ct1, input1.ct1, input2.ct1);
    if (input1.ct2.has_value() && input2.ct2.has_value()) {
      auto t2 = make_treg_from(input1.ct2.value());
      add_instr->add_operands(t2, input1.ct2.value(), input2.ct2.value());
      output.ct2 = t2;
    } else if (input1.ct2.has_value()) {
      output.ct2 = input1.ct2;
    } else {
      output.ct2 = input2.ct2;
    }
  };

  auto add_ct_pt = [&](Backend::Ciphertext &output,
                       const Backend::Ciphertext &input1,
                       const Backend::Plaintext &input2) {
    output.level = input1.level;
    if (output.level != input2.level) {
      throw std::runtime_error("Levels do not match");
    }
    output.ct1 = input1.ct1;
    output.ct2 = input1.ct2;
    output.ct0 = make_treg_from(input1.ct0);
    add_instr->add_operands(output.ct0, input1.ct0, input2.pt);
  };

  std::visit(
      Overloaded{
          [&](std::pair<Backend::Ciphertext, Backend::Ciphertext> &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            add_ct_ct(output, input.first, input.second);
          },
          [&](std::pair<Backend::Ciphertext, Backend::Plaintext> &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            add_ct_pt(output, input.first, input.second);
          },
          [&](std::pair<Backend::CiphertextVector, Backend::Ciphertext>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              add_ct_ct(output.vec.at(i), input.first.vec.at(i), input.second);
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::Plaintext> &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              add_ct_pt(output.vec.at(i), input.first.vec.at(i), input.second);
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::CiphertextVector>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            if (input.first.vec.size() != input.second.vec.size()) {
              throw std::runtime_error("Vector sizes do not match");
            }
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              add_ct_ct(output.vec.at(i), input.first.vec.at(i),
                        input.second.vec.at(i));
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::PlaintextVector>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            if (input.first.vec.size() != input.second.vec.size()) {
              throw std::runtime_error("Vector sizes do not match");
            }
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              add_ct_pt(output.vec.at(i), input.first.vec.at(i),
                        input.second.vec.at(i));
            }
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      pair);
}

void CeriumCompiler::mul(const Frontend::Term::Ptr &term,
                               const Frontend::Term::Ptr &args1,
                               const Frontend::Term::Ptr &args2) {

  if (isPlain2(args1)) {
    assert(!isPlain2(args2));
    mul(term, args2, args1);
    return;
  }

  auto pair = get_runtime_value_pair(args1, args2);

  using OpCode = Backend::LimbInstruction::OpCode;
  std::shared_ptr<KernelGroup> kg = create_kernel_group();
  auto mul_instr = std::make_shared<BinOpInstruction>(OpCode::Mul);
  auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
  kg->push_instruction(mul_instr);

  auto mul_ct_ct = [&](Backend::Ciphertext &output,
                       const Backend::Ciphertext &input1,
                       const Backend::Ciphertext &input2) {
    output.level = input1.level;
    if (output.level != input2.level) {
      throw std::runtime_error("Ciphertext levels do not match");
    }
    if (input1.ct2.has_value() || input2.ct2.has_value()) {
      throw std::runtime_error("ERROR: Values not relinearized");
    }

    output.ct0 = make_treg_from(input1.ct0);
    mul_instr->add_operands(output.ct0, input1.ct0, input2.ct0);

    auto t2 = make_treg_from(input1.ct1);
    mul_instr->add_operands(t2, input1.ct1, input2.ct1);
    output.ct2 = t2;

    auto t01 = make_treg_from(input1.ct0);
    auto t10 = make_treg_from(input1.ct1);
    output.ct1 = make_treg_from(input1.ct1);

    mul_instr->add_operands(t01, input1.ct0, input2.ct1);
    mul_instr->add_operands(t10, input1.ct1, input2.ct0);
    add_instr->add_operands(output.ct1, t01, t10);
  };

  auto mul_ct_pt = [&](Backend::Ciphertext &output,
                       const Backend::Ciphertext &input1,
                       const Backend::Plaintext &input2) {
    output.level = input1.level;
    output.ct0 = make_treg_from(input1.ct0);
    mul_instr->add_operands(output.ct0, input1.ct0, input2.pt);

    output.ct1 = make_treg_from(input1.ct1);
    mul_instr->add_operands(output.ct1, input1.ct1, input2.pt);

    if (input1.ct2.has_value()) {
      auto t2 = make_treg_from(input1.ct2.value());
      mul_instr->add_operands(t2, input1.ct2.value(), input2.pt);
      output.ct2 = t2;
    }
  };

  std::visit(
      Overloaded{
          [&](std::pair<Backend::Ciphertext, Backend::Ciphertext> &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            mul_ct_ct(output, input.first, input.second);
          },
          [&](std::pair<Backend::Ciphertext, Backend::Plaintext> &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            mul_ct_pt(output, input.first, input.second);
          },
          [&](std::pair<Backend::CiphertextVector, Backend::Ciphertext>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              mul_ct_ct(output.vec.at(i), input.first.vec.at(i), input.second);
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::Plaintext> &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              mul_ct_pt(output.vec.at(i), input.first.vec.at(i), input.second);
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::CiphertextVector>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            if (input.first.vec.size() != input.second.vec.size()) {
              throw std::runtime_error("Vector sizes do not match");
            }
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              mul_ct_ct(output.vec.at(i), input.first.vec.at(i),
                        input.second.vec.at(i));
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::PlaintextVector>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            if (input.first.vec.size() != input.second.vec.size()) {
              throw std::runtime_error("Vector sizes do not match");
            }
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              mul_ct_pt(output.vec.at(i), input.first.vec.at(i),
                        input.second.vec.at(i));
            }
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      pair);

  if (add_instr->num_destinations() > 0) {
    kg->push_instruction(add_instr);
  }
}

void CeriumCompiler::sub(const Frontend::Term::Ptr &term,
                               const Frontend::Term::Ptr &args1,
                               const Frontend::Term::Ptr &args2) {

  auto pair = get_runtime_value_pair(args1, args2);

  using OpCode = Backend::LimbInstruction::OpCode;
  std::shared_ptr<KernelGroup> kg = create_kernel_group();
  auto sub_instr = std::make_shared<BinOpInstruction>(OpCode::Sub);
  auto neg_instr = std::make_shared<UnOpInstruction>(OpCode::Neg);
  kg->push_instruction(sub_instr);

  auto sub_ct_ct = [&](Backend::Ciphertext &output,
                       const Backend::Ciphertext &input1,
                       const Backend::Ciphertext &input2) {
    output.level = input1.level;
    if (output.level != input2.level) {
      throw std::runtime_error("Ciphertext levels do not match");
    }
    output.ct0 = make_treg_from(input1.ct0);
    output.ct1 = make_treg_from(input1.ct1);
    sub_instr->add_operands(output.ct0, input1.ct0, input2.ct0);
    sub_instr->add_operands(output.ct1, input1.ct1, input2.ct1);
    if (input1.ct2.has_value() && input2.ct2.has_value()) {
      auto t2 = make_treg_from(input1.ct2.value());
      sub_instr->add_operands(t2, input1.ct2.value(), input2.ct2.value());
      output.ct2 = t2;
    } else if (input1.ct2.has_value()) {
      output.ct2 = input1.ct2;
    } else if (input2.ct2.has_value()) {
      auto t2 = make_treg_from(input2.ct2.value());
      neg_instr->add_operands(t2, input2.ct2.value());
      output.ct2 = t2;
    }
  };

  auto sub_ct_pt = [&](Backend::Ciphertext &output,
                       const Backend::Ciphertext &input1,
                       const Backend::Plaintext &input2) {
    output.level = input1.level;
    if (output.level != input2.level) {
      throw std::runtime_error("Levels do not match");
    }
    output.ct1 = input1.ct1;
    output.ct2 = input1.ct2;
    output.ct0 = make_treg_from(input1.ct0);
    sub_instr->add_operands(output.ct0, input1.ct0, input2.pt);
  };

  auto sub_pt_ct = [&](Backend::Ciphertext &output,
                       const Backend::Plaintext &input1,
                       const Backend::Ciphertext &input2) {
    output.level = input1.level;
    if (output.level != input2.level) {
      throw std::runtime_error("Levels do not match");
    }
    output.ct0 = make_treg_from(input2.ct0);
    sub_instr->add_operands(output.ct0, input1.pt, input2.ct0);
    output.ct1 = make_treg_from(input2.ct1);
    neg_instr->add_operands(output.ct1, input2.ct1);
    if (input2.ct2.has_value()) {
      auto t2 = make_treg_from(input2.ct2.value());
      neg_instr->add_operands(t2, input2.ct2.value());
      output.ct2 = t2;
    }
  };

  std::visit(
      Overloaded{
          [&](std::pair<Backend::Ciphertext, Backend::Ciphertext> &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            sub_ct_ct(output, input.first, input.second);
          },
          [&](std::pair<Backend::Ciphertext, Backend::Plaintext> &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            sub_ct_pt(output, input.first, input.second);
          },
          [&](std::pair<Backend::Plaintext, Backend::Ciphertext> &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            sub_pt_ct(output, input.first, input.second);
          },
          [&](std::pair<Backend::CiphertextVector, Backend::Ciphertext>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              sub_ct_ct(output.vec.at(i), input.first.vec.at(i), input.second);
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::Plaintext> &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              sub_ct_pt(output.vec.at(i), input.first.vec.at(i), input.second);
            }
          },
          [&](std::pair<Backend::Plaintext, Backend::CiphertextVector> &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.second.vec.size());
            for (size_t i = 0; i < input.second.vec.size(); i++) {
              sub_pt_ct(output.vec.at(i), input.first, input.second.vec.at(i));
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::CiphertextVector>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            if (input.first.vec.size() != input.second.vec.size()) {
              throw std::runtime_error("Vector sizes do not match");
            }
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              sub_ct_ct(output.vec.at(i), input.first.vec.at(i),
                        input.second.vec.at(i));
            }
          },
          [&](std::pair<Backend::CiphertextVector, Backend::PlaintextVector>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            if (input.first.vec.size() != input.second.vec.size()) {
              throw std::runtime_error("Vector sizes do not match");
            }
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              sub_ct_pt(output.vec.at(i), input.first.vec.at(i),
                        input.second.vec.at(i));
            }
          },
          [&](std::pair<Backend::PlaintextVector, Backend::CiphertextVector>
                  &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output.vec.resize(input.first.vec.size());
            if (input.first.vec.size() != input.second.vec.size()) {
              throw std::runtime_error("Vector sizes do not match");
            }
            for (size_t i = 0; i < input.first.vec.size(); i++) {
              sub_pt_ct(output.vec.at(i), input.first.vec.at(i),
                        input.second.vec.at(i));
            }
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      pair);

  if (neg_instr->num_destinations() > 0) {
    kg->push_instruction(neg_instr);
  }
}

void CeriumCompiler::rotate(Backend::Ciphertext &output,
                              const Frontend::Term::Ptr &args1, int32_t rot_idx,
                              const bool use_keyswitch2) {
  Backend::Ciphertext &input1 =
      std::get<Backend::Ciphertext>(Objects.at(args1));
  rotate_internal(output, input1, rot_idx, use_keyswitch2);
}
void CeriumCompiler::rotate_internal(Backend::Ciphertext &output,
                                       Backend::Ciphertext &input1,
                                       int32_t rot_idx, bool use_keyswitch2) {
  output.level = input1.level;
  assert(!input1.ct2.has_value());
  using OpCode = Backend::LimbInstruction::OpCode;
  if (rot_idx == 0) {
    output.ct0 = input1.ct0;
    output.ct1 = input1.ct1;
    return;
  }

  Backend::Ciphertext temp;
  auto kg = create_kernel_group();

  temp.ct0 = make_treg_from(input1.ct0);
  temp.ct1 = make_treg_from(input1.ct0);
  auto rot_instr = std::make_shared<RotationInstruction>();
  rot_instr->add_operands(rot_idx, temp.ct0, input1.ct0);
  rot_instr->add_operands(rot_idx, temp.ct1, input1.ct1);
  kg->push_instruction(rot_instr);
  if (use_keyswitch2) {
    output = keyswitch2(temp, Backend::EvalKeyType::Rotation, rot_idx);
  } else {
    output = keyswitch(temp, Backend::EvalKeyType::Rotation, rot_idx);
  }
}

#if 0
  void CeriumCompiler::rotate3_internal(Backend::Ciphertext &output,
                                         Backend::Ciphertext &input1,
                                         int32_t rot_idx, bool use_keyswitch2) {
    output.level = input1.level;
    assert(!input1.ct2.has_value());
    // using SSAInstruction = Backend::PolynomialInstruction;
    using OpCode = Backend::LimbInstruction::OpCode;
    if (rot_idx == 0) {
      output.ct0 = input1.ct0;
      output.ct1 = input1.ct1;
      return;
    }

    Backend::Ciphertext temp;
    auto kg = create_kernel_group();

    temp.ct0 = make_treg_from(input1.ct0);
    temp.ct1 = make_treg_from(input1.ct0);
    auto rot_instr =
        std::make_shared<RotationInstruction>();
    rot_instr->add_operands(rot_idx, temp.ct0, input1.ct0);
    rot_instr->add_operands(rot_idx, temp.ct1, input1.ct1);
    kg->push_instruction(rot_instr);
    // output = keyswitch(temp, Backend::EvalKeyType::Rotation, rot_idx);
    output = keyswitch3(temp, Backend::EvalKeyType::Rotation, rot_idx);
  }
#endif

void CeriumCompiler::rotate(const Frontend::Term::Ptr &term,
                                  const Frontend::Term::Ptr &args1,
                                  int32_t rot_idx, const bool use_keyswitch2) {

  if (rot_idx == 0) {
    std::visit(Overloaded{[&](Backend::Ciphertext &input) {
                            auto &output = initValue<Backend::Ciphertext>(term);
                            output = input;
                          },
                          [&](Backend::CiphertextVector &input) {
                            auto &output =
                                initValue<Backend::CiphertextVector>(term);
                            output.vec.resize(input.vec.size());
                            for (size_t i = 0; i < input.vec.size(); i++) {
                              output.vec[i] = input.vec[i];
                            }
                          },
                          [&](auto &arg) {
                            throw std::runtime_error(
                                "Unsupported operation encountered");
                          }},
               Objects.at(args1));
    return;
  }

  std::shared_ptr<KernelGroup> kg = create_kernel_group();
  using OpCode = Backend::LimbInstruction::OpCode;
  auto rot_instr = std::make_shared<RotationInstruction>();
  kg->push_instruction(rot_instr);

  auto rotate_ct = [&](const Backend::Ciphertext &input1) {
    assert(!input1.ct2.has_value());
    Backend::Ciphertext temp;
    temp.level = input1.level;
    temp.ct0 = make_treg_from(input1.ct0);
    temp.ct1 = make_treg_from(input1.ct0);
    rot_instr->add_operands(rot_idx, temp.ct0, input1.ct0);
    rot_instr->add_operands(rot_idx, temp.ct1, input1.ct1);
    return temp;
  };

  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            output.level = input.level;
            auto temp = rotate_ct(input);
            output = keyswitch(temp, Backend::EvalKeyType::Rotation, rot_idx);
          },
          [&](Backend::CiphertextVector &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            Backend::CiphertextVector temp_vec;
            temp_vec.vec.resize(input.vec.size());
            output.vec.resize(input.vec.size());
            for (size_t i = 0; i < input.vec.size(); i++) {
              temp_vec.vec[i] = rotate_ct(input.vec.at(i));
            }
            output = keyswitch_vec(temp_vec, Backend::EvalKeyType::Rotation,
                                   rot_idx);
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args1));
}

void CeriumCompiler::conjugate(const Frontend::Term::Ptr &term,
                                     const Frontend::Term::Ptr &args1,
                                     const bool use_keyswitch2) {

  std::shared_ptr<KernelGroup> kg = create_kernel_group();
  using OpCode = Backend::LimbInstruction::OpCode;
  auto con_instr = std::make_shared<UnOpInstruction>(OpCode::Con);
  kg->push_instruction(con_instr);

  auto conjugate_ct = [&](const Backend::Ciphertext &input1) {
    assert(!input1.ct2.has_value());
    Backend::Ciphertext temp;
    temp.level = input1.level;
    temp.ct0 = make_treg_from(input1.ct0);
    temp.ct1 = make_treg_from(input1.ct0);
    con_instr->add_operands(temp.ct0, input1.ct0);
    con_instr->add_operands(temp.ct1, input1.ct1);
    return temp;
  };

  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            output.level = input.level;
            auto temp = conjugate_ct(input);
            if (use_keyswitch2) {
              output = keyswitch2(temp, Backend::EvalKeyType::Conjugation);
            } else {
              output = keyswitch(temp, Backend::EvalKeyType::Conjugation);
            }
          },
          [&](Backend::CiphertextVector &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            Backend::CiphertextVector temp_vec;
            temp_vec.vec.resize(input.vec.size());
            output.vec.resize(input.vec.size());
            for (size_t i = 0; i < input.vec.size(); i++) {
              temp_vec.vec[i] = conjugate_ct(input.vec.at(i));
            }
            if (use_keyswitch2) {
              throw std::runtime_error(
                  "keyswitch2 not supported for conjugate_vec");
            }
            output = keyswitch_vec(temp_vec, Backend::EvalKeyType::Conjugation);
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args1));
}

void CeriumCompiler::relinearize(const Frontend::Term::Ptr &term,
                                       const Frontend::Term::Ptr &args1) {

  uint32_t rescale_levels = 0;
  if (term->has<Frontend::RescaleLevelsAttribute>()) {
    rescale_levels = term->get<Frontend::RescaleLevelsAttribute>();
  }

  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            output = keyswitch(input, Backend::EvalKeyType::Relinearization, 0,
                               rescale_levels);
          },
          [&](Backend::CiphertextVector &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            output = keyswitch_vec(input, Backend::EvalKeyType::Relinearization,
                                   0, rescale_levels);
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args1));
}

void CeriumCompiler::relinearize2(const Frontend::Term::Ptr &term,
                                        const Frontend::Term::Ptr &args1) {

  uint32_t rescale_levels = 0;
  if (term->has<Frontend::RescaleLevelsAttribute>()) {
    rescale_levels = term->get<Frontend::RescaleLevelsAttribute>();
  }
  assert(rescale_levels == 0);

  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) {
            auto &output = initValue<Backend::Ciphertext>(term);
            output = keyswitch2(input, Backend::EvalKeyType::Relinearization, 0,
                                rescale_levels);
          },
          [&](Backend::CiphertextVector &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            throw std::runtime_error("Unimplemented keyswitch2_vec");
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args1));
}

#if 0
  void CeriumCompiler::relinearize3(const Frontend::Term::Ptr & term, const Frontend::Term::Ptr &args1) {

    uint32_t rescale_levels = 0;
    if(term->has<Frontend::RescaleLevelsAttribute>()) {
      rescale_levels = term->get<Frontend::RescaleLevelsAttribute>();
    }

    std::visit(
        Overloaded{
            [&](Backend::Ciphertext &input) {
              auto & output = initValue<Backend::Ciphertext>(term);
              output = keyswitch3(input, Backend::EvalKeyType::Relinearization,0,rescale_levels);
            },
            [&](Backend::CiphertextVector & input) {
              throw std::runtime_error(
                  "Unimplemented keyswitch2_vec");
              // auto & output = initValue<Backend::CiphertextVector>(term);
              // output = keyswitch_vec(input, Backend::EvalKeyType::Relinearization,0,rescale_levels);
            },
            [&](auto & arg) {
              throw std::runtime_error(
                  "Unsupported operation encountered");
            }
        },
        Objects.at(args1));
  }
#endif

std::vector<Backend::CiphertextVector>
CeriumCompiler::hoisted_input_broadcast_keyswitching_internal_vec(
    const Backend::CiphertextVector &input,
    const std::vector<int32_t> rotation_indices) {

  // TODO: Fix the evalkey numbers here. They should be equal to sqrt(2 * c *
  // BS)

  using LimbType = Backend::Term::LimbType;
  using OpCode = Backend::LimbInstruction::OpCode;

  std::vector<Backend::CiphertextVector> output_vec;

  auto level = input.vec[0].ct1.level();
  auto key_switch_type = KeySwitch::KeySwitchType::Broadcast;
  auto [extension_size, digits] =
      compute_keyswitch_split(key_switch_type, level);

  Backend::Polynomial b0_in;
  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();

  KernelGroupPtr kg_dist;
  KernelGroupPtr kg_mov;
  auto dist_instr = std::make_shared<DistRecvInstruction>();
  auto mov_instr = std::make_shared<UnOpInstruction>(OpCode::Mov);

  if (current_partition_size > 1) {
    kg_dist = create_kernel_group_dist();
    kg_mov = create_kernel_group();
    kg_dist->push_instruction(dist_instr);
    kg_mov->push_instruction(mov_instr);
  }

  kg_intt0->push_instruction(intt_instr0);
  auto kg_bconv0 = create_kernel_group_bconv();
  auto bconv_instr0 = std::make_shared<BconvInstruction>();
  kg_bconv0->push_instruction(bconv_instr0);

  auto kg_ntt0 = create_kernel_group_ntt();
  auto ntt_inst0 = std::make_shared<NttInstruction>();
  kg_ntt0->push_instruction(ntt_inst0);

  auto make_parext_pair_from = [&](Backend::Polynomial &inp,
                                   const uint64_t extension_size) {
    auto t0 = make_treg_from(inp);
    t0.set_limbtype(LimbType::Par);
    auto t1 = make_treg_from(inp);
    t1.set_limbtype(LimbType::Ext);
    t1.set_extension_size(extension_size);
    return std::pair{t0, t1};
  };

  auto make_parext_pair_from__ =
      [&](std::pair<Backend::Polynomial, Backend::Polynomial> &inp,
          const uint64_t extension_size) {
        auto t0 = make_treg_from(std::get<0>(inp));
        t0.set_limbtype(LimbType::Par);
        auto t1 = make_treg_from(std::get<1>(inp));
        t1.set_limbtype(LimbType::Ext);
        t1.set_extension_size(extension_size);
        return std::pair{t0, t1};
      };

  std::vector<PolyPair> t1_vec;

  for (auto &i : input.vec) {
    auto t0 = i.ct1;
    if (current_partition_size > 1) {
      auto t0_intt = make_treg_from(t0);
      intt_instr0->add_operands(t0_intt, t0);
      auto t0_dist = make_treg_from(t0_intt);
      dist_instr->add_operands(t0_dist, t0_intt);
      b0_in = make_bcor_from(t0_dist);
      mov_instr->add_operands(b0_in, t0_dist);
    } else {
      b0_in = make_bcor_from(t0);
      intt_instr0->add_operands(b0_in, t0);
    }
    auto b0 = make_bcor_from(b0_in);
    b0.set_limbtype(LimbType::PaE);
    b0.set_extension_size(extension_size);
    bconv_instr0->add_operands(b0, b0_in);

    auto t1 = make_parext_pair_from(t0, extension_size);
    ntt_inst0->add_operands(t1.first, t1.second, b0, t0, b0_in);
    t1_vec.push_back(t1);
  }

  DigitMap<Backend::Polynomial> digit_map;
  auto kg_in = kg_intt0;
  if (kg_mov) {
    kg_in = kg_mov;
  }
  split_digit_wise_internal(digits, kg_in, digit_map);
  split_digit_wise_internal(digits, kg_bconv0, digit_map);
  split_digit_wise_internal(digits, kg_ntt0, digit_map);
  kg_in->set_dont_split_modular(true);
  kg_bconv0->set_dont_split_modular(true);
  kg_ntt0->set_dont_split_modular(true);

  std::shared_ptr<KernelGroup> kg_evkmul = create_kernel_group();
  std::shared_ptr<KernelGroup> kg_evkmul_ext = create_kernel_group();
  std::shared_ptr<KernelGroup> kg_intt = create_kernel_group_intt();
  std::shared_ptr<KernelGroup> kg_bconv = create_kernel_group_bconv();
  std::shared_ptr<KernelGroup> kg_sud = create_kernel_group_ntt();
  std::shared_ptr<KernelGroup> kg_add = create_kernel_group();
  std::shared_ptr<KernelGroup> kg_rot = create_kernel_group();

  auto intt_instr = std::make_shared<InttInstruction>();
  auto bconv_instr = std::make_shared<BconvInstruction>();
  auto sud_instr = std::make_shared<SudInstruction>();

  auto mad_inst = std::make_shared<MadInstruction>();
  auto mad_inst_ext = std::make_shared<MadInstruction>();
  kg_evkmul->push_instruction(mad_inst);
  kg_evkmul_ext->push_instruction(mad_inst_ext);

  for (int i = 0; i < rotation_indices.size(); i++) {
    int32_t rot_idx = rotation_indices[i];
    if (rot_idx == 0) {
      output_vec.push_back(input);
      continue;
    }

    Backend::CiphertextVector rotate_vec;

    for (auto n = 0; n < input.vec.size(); n++) {
      auto evalkeys =
          get_evalkey(input.vec[n].ct1, Backend::EvalKeyType::RotationInv,
                      -rot_idx, key_switch_type, extension_size);
      auto evk0 = evalkeys.first;
      auto evk1 = evalkeys.second;

      auto t1 = t1_vec[n];

      auto t2 = make_parext_pair_from__(t1, extension_size);
      auto t3 = make_parext_pair_from__(t1, extension_size);
      mad_inst->add_operands(t2.first, t1.first, evk0.first);
      mad_inst_ext->add_operands(t2.second, t1.second, evk0.second);
      mad_inst->add_operands(t3.first, t1.first, evk1.first);
      mad_inst_ext->add_operands(t3.second, t1.second, evk1.second);

      auto b2_in = make_bcor_from(t2.second);
      auto b3_in = make_bcor_from(t3.second);
      b2_in.set_limbtype(LimbType::Ext);
      b2_in.set_extension_size(extension_size);
      b3_in.set_limbtype(LimbType::Ext);
      b3_in.set_extension_size(extension_size);
      intt_instr->add_operands(b2_in, t2.second);
      intt_instr->add_operands(b3_in, t3.second);

      auto b2 = make_bcor_from(t2.first);
      auto b3 = make_bcor_from(t3.first);
      b2.set_limbtype(LimbType::Par);
      b3.set_limbtype(LimbType::Par);
      bconv_instr->add_operands(b2, b2_in);
      bconv_instr->add_operands(b3, b3_in);

      auto t4 = make_treg_from(t2.first);
      auto t5 = make_treg_from(t3.first);
      auto t6 = make_treg_from(t4);
      auto t7 = make_treg_from(t5);

      sud_instr->add_operands(t4, t2.first, b2, b2_in);
      sud_instr->add_operands(t5, t3.first, b3, b3_in);

      auto rot_instr = std::make_shared<RotationInstruction>();
      auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
      auto in_rot = make_treg_from(input.vec[n].ct0);
      add_instr->add_operands(in_rot, t4, input.vec[n].ct0);
      rot_instr->add_operands(rot_idx, t6, in_rot);
      rot_instr->add_operands(rot_idx, t7, t5);
      kg_add->push_instruction(add_instr);
      kg_rot->push_instruction(rot_instr);

      Backend::Ciphertext rotate;
      rotate.ct0 = t6;
      rotate.ct1 = t7;
      rotate.level = t6.level();
      rotate_vec.vec.push_back(rotate);
    }
    output_vec.push_back(rotate_vec);
  }

  split_digit_wise_internal_mad(digits, kg_evkmul, digit_map);
  split_digit_wise_internal_mad(digits, kg_evkmul_ext, digit_map);
  kg_evkmul->set_dont_split_modular(true);
  kg_evkmul_ext->set_dont_split_modular(true);

  kg_intt->push_instruction(intt_instr);
  kg_bconv->push_instruction(bconv_instr);
  kg_sud->push_instruction(sud_instr);

  return output_vec;
}
Backend::CiphertextVector
CeriumCompiler::bsgs_giantstep_accumulate_keyswitch_iterations_vec(
    const std::vector<Backend::CiphertextVector> &babysteps,
    const std::vector<Backend::Plaintext> &plaintexts,
    const std::vector<int32_t> giantstep_rotation_indices,
    const KernelGroup::SplitType &split, const uint32_t extension_size____,
    const uint32_t dnum_glo, const uint8_t rescale_levels) {

  using LimbType = Backend::Term::LimbType;
  using OpCode = Backend::LimbInstruction::OpCode;

  std::shared_ptr<Backend::BigNode> bn;
  std::optional<Backend::Polynomial>
      zero_rot_idx_sum; /* Sum of terms that are not rotated*/

  auto num_ct = babysteps[0].vec.size();
  Backend::CiphertextVector giantstep;
  giantstep.vec.resize(num_ct);

  std::vector<Backend::Polynomial> giantstep_ct0_sum(num_ct);
  std::vector<std::optional<Backend::Polynomial>>
      giantstep_zero_rot_idx_ct1_sum(num_ct);
  std::vector<std::optional<PolyPair>> giantstep_k0_sum(num_ct);
  std::vector<std::optional<PolyPair>> giantstep_k1_sum(num_ct);

  size_t num_babysteps = babysteps.size();
  auto keyswitch_type = KeySwitch::KeySwitchType::Aggregation;

  auto [extension_size, digits] =
      compute_keyswitch_split(keyswitch_type, babysteps[0].vec[0].ct1.level());

  auto kg_mul = create_kernel_group();

  std::vector<std::shared_ptr<BinOpInstruction>> mul_instr0(num_babysteps);
  std::vector<std::shared_ptr<BinOpInstruction>> mul_instr1(num_babysteps);
  std::vector<std::shared_ptr<BinOpInstruction>> add_instr0(num_babysteps);
  std::vector<std::shared_ptr<BinOpInstruction>> add_instr1(num_babysteps);

  for (int j = 0; j < num_babysteps; j++) {
    mul_instr0[j] = std::make_shared<BinOpInstruction>(OpCode::MuP);
    mul_instr1[j] = std::make_shared<BinOpInstruction>(OpCode::MuP);
    add_instr0[j] = std::make_shared<BinOpInstruction>(OpCode::Add);
    add_instr1[j] = std::make_shared<BinOpInstruction>(OpCode::Add);
  }

  std::vector<Backend::CiphertextVector> babystep_accumulated(
      giantstep_rotation_indices.size());
  for (int i = 0; i < giantstep_rotation_indices.size(); i++) {
    babystep_accumulated[i].vec.resize(num_ct);
    auto rot_idx = giantstep_rotation_indices[i];
    for (int j = 0; j < num_babysteps; j++) {
      auto &plaintext = plaintexts[j + i * num_babysteps];
      for (auto n = 0; n < num_ct; n++) {
        auto t0 = make_treg_from(babysteps[j].vec[n].ct0);
        auto t1 = make_treg_from(babysteps[j].vec[n].ct1);
        mul_instr0[j]->add_operands(t0, plaintext.pt, babysteps[j].vec[n].ct0);
        mul_instr1[j]->add_operands(t1, plaintext.pt, babysteps[j].vec[n].ct1);

        Backend::Ciphertext t;
        if (j == 0) {
          t.ct0 = t0;
          t.ct1 = t1;
          t.level = t0.level();
        } else {
          t.ct0 = make_treg_from(t0);
          t.ct1 = make_treg_from(t1);
          t.level = t0.level();
          add_instr0[j]->add_operands(t.ct0, t0,
                                      babystep_accumulated[i].vec[n].ct0);
          add_instr1[j]->add_operands(t.ct1, t1,
                                      babystep_accumulated[i].vec[n].ct1);
        }
        babystep_accumulated[i].vec[n] = t;
      }
    }
  }

  for (int j = 0; j < num_babysteps; j++) {
    kg_mul->push_instruction(std::move(mul_instr0[j]));
    kg_mul->push_instruction(std::move(mul_instr1[j]));
    kg_mul->push_instruction(std::move(add_instr0[j]));
    kg_mul->push_instruction(std::move(add_instr1[j]));
  }

  auto make_parext_pair_from = [&](Backend::Polynomial &inp,
                                   const uint64_t extension_size) {
    auto t0 = make_treg_from(inp);
    t0.set_limbtype(LimbType::Usp);
    t0.set_adjustment_size(-rescale_levels);
    auto t1 = make_treg_from(inp);
    t1.set_limbtype(LimbType::Ext);
    t1.set_extension_size(extension_size);
    t1.set_adjustment_size(rescale_levels);
    return std::pair{t0, t1};
  };

  auto make_parext_pair_from__ =
      [&](std::pair<Backend::Polynomial, Backend::Polynomial> &inp,
          const uint64_t extension_size) {
        auto t0 = make_treg_from(std::get<0>(inp));
        t0.set_limbtype(LimbType::Usp);
        t0.set_adjustment_size(-rescale_levels);
        auto t1 = make_treg_from(std::get<1>(inp));
        t1.set_limbtype(LimbType::Ext);
        t1.set_extension_size(extension_size);
        t1.set_adjustment_size(rescale_levels);
        return std::pair{t0, t1};
      };

  auto kg_rot = create_kernel_group();
  auto kg_add = create_kernel_group();
  std::vector<std::shared_ptr<RotationInstruction>> rot_instrs;
  for (int i = 0; i < giantstep_rotation_indices.size(); i++) {
    auto rot_idx = giantstep_rotation_indices[i];
    auto rot_instr = std::make_shared<RotationInstruction>();
    rot_instrs.push_back(rot_instr);
    kg_rot->push_instruction(rot_instr);
  }
  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();

  auto kg_bconv0 = create_kernel_group_bconv();
  auto bconv_instr0 = std::make_shared<BconvInstruction>();

  auto kg_ntt0 = create_kernel_group_ntt();
  auto ntt_inst0 = std::make_shared<NttInstruction>();

  auto kg_evkmul = create_kernel_group();
  auto kg_evkmul_ext = create_kernel_group();
  auto mad_inst = std::make_shared<MadInstruction>();
  kg_evkmul->push_instruction(mad_inst);
  auto mad_inst_ext = std::make_shared<MadInstruction>();
  kg_evkmul_ext->push_instruction(mad_inst_ext);

  auto kg_acc = create_kernel_group();
  auto kg_acc_ext = create_kernel_group();

  auto acc_inst = std::make_shared<BinOpInstruction>(OpCode::Add);
  auto acc_inst_ext = std::make_shared<BinOpInstruction>(OpCode::Add);
  kg_acc->push_instruction(acc_inst);
  kg_acc_ext->push_instruction(acc_inst_ext);
  for (int n = num_ct - 1; n >= 0; n--) {

    for (int i = 0; i < giantstep_rotation_indices.size(); i++) {
      auto rot_idx = giantstep_rotation_indices[i];
      Backend::Polynomial rot0;
      Backend::Polynomial rot1;
      if (rot_idx == 0) {
        rot0 = babystep_accumulated[i].vec[n].ct0;
        rot1 = babystep_accumulated[i].vec[n].ct1;
      } else {
        rot0 = make_treg_from(babystep_accumulated[i].vec[n].ct0);
        rot1 = make_treg_from(babystep_accumulated[i].vec[n].ct1);

        auto rot_instr = rot_instrs[i];
        rot_instr->add_operands(rot_idx, rot0,
                                babystep_accumulated[i].vec[n].ct0);
        rot_instr->add_operands(rot_idx, rot1,
                                babystep_accumulated[i].vec[n].ct1);
      }
      if (i == 0) {
        giantstep_ct0_sum[n] = rot0;
      } else {
        auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
        auto tsum = make_treg_from(rot0);
        add_instr->add_operands(tsum, rot0, giantstep_ct0_sum[n]);
        giantstep_ct0_sum[n] = tsum;
        kg_add->push_instruction(add_instr);
      }
      if (rot_idx == 0) {
        if (!giantstep_zero_rot_idx_ct1_sum[n].has_value()) {
          giantstep_zero_rot_idx_ct1_sum[n] = rot1;
        } else {

          auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
          auto tsum = make_treg_from(rot0);
          add_instr->add_operands(tsum, rot1,
                                  giantstep_zero_rot_idx_ct1_sum[n].value());
          giantstep_zero_rot_idx_ct1_sum[n] = tsum;
          kg_add->push_instruction(add_instr);
        }
        continue;
      }

      auto evalkeys = get_evalkey(rot1, Backend::EvalKeyType::Rotation, rot_idx,
                                  keyswitch_type, extension_size);
      auto evk0 = evalkeys.first;
      auto evk1 = evalkeys.second;
      evk0.first.set_adjustment_size(-rescale_levels);
      evk0.second.set_adjustment_size(rescale_levels);
      evk1.first.set_adjustment_size(-rescale_levels);
      evk1.second.set_adjustment_size(rescale_levels);

      auto t0 = rot1;

      Backend::Polynomial b0_in;
      b0_in = make_bcor_from(t0);
      intt_instr0->add_operands(b0_in, t0);

      auto b0 = make_bcor_from(b0_in);
      b0.set_limbtype(LimbType::ExC);
      b0.set_extension_size(extension_size);
      bconv_instr0->add_operands(b0, b0_in);

      auto t1 = make_parext_pair_from(t0, extension_size);
      ntt_inst0->add_operands(t1.first, t1.second, b0, t0, b0_in);

      auto t2 = make_parext_pair_from__(t1, extension_size);
      auto t3 = make_parext_pair_from__(t1, extension_size);
      mad_inst->add_operands(t2.first, t1.first, evk0.first);
      mad_inst->add_operands(t3.first, t1.first, evk1.first);

      mad_inst_ext->add_operands(t2.second, t1.second, evk0.second);
      mad_inst_ext->add_operands(t3.second, t1.second, evk1.second);

      if (!giantstep_k0_sum[n].has_value()) {
        giantstep_k0_sum[n] = t2;
      } else {
        auto &k0_sum_val = giantstep_k0_sum[n].value();
        auto tsum = make_parext_pair_from__(t2, extension_size);
        acc_inst->add_operands(tsum.first, t2.first, k0_sum_val.first);
        acc_inst_ext->add_operands(tsum.second, t2.second, k0_sum_val.second);
        giantstep_k0_sum[n] = tsum;
      }

      if (!giantstep_k1_sum[n].has_value()) {
        giantstep_k1_sum[n] = t3;
      } else {
        auto &k1_sum_val = giantstep_k1_sum[n].value();
        auto tsum = make_parext_pair_from__(t2, extension_size);
        acc_inst->add_operands(tsum.first, t3.first, k1_sum_val.first);
        acc_inst_ext->add_operands(tsum.second, t3.second, k1_sum_val.second);
        giantstep_k1_sum[n] = tsum;
      }
    }
  }

  kg_intt0->push_instruction(intt_instr0);
  kg_bconv0->push_instruction(bconv_instr0);
  kg_ntt0->push_instruction(ntt_inst0);
  split_digit_wise(digits, kg_intt0, kg_bconv0, kg_ntt0, kg_evkmul,
                   kg_evkmul_ext, false /*don't split modular*/);

  kg_evkmul->merge_kg(kg_acc);
  kg_evkmul_ext->merge_kg(kg_acc_ext);

  auto kg_pmul = create_kernel_group_pmul();
  auto pmul_inst = std::make_shared<PmuInstruction>();
  kg_pmul->push_instruction(pmul_inst);

  auto kg_intt = create_kernel_group_intt();
  auto intt_instr = std::make_shared<InttInstruction>();
  kg_intt->push_instruction(intt_instr);

  auto kg_bconv = create_kernel_group_bconv();
  auto bconv_instr = std::make_shared<BconvInstruction>();
  kg_bconv->push_instruction(bconv_instr);

  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();
  kg_sud->push_instruction(sud_instr);

  for (auto n = 0; n < num_ct; n++) {

    assert(giantstep_k0_sum[n].has_value());
    assert(giantstep_k1_sum[n].has_value());

    auto &k0 = giantstep_k0_sum[n].value();
    auto &k1 = giantstep_k1_sum[n].value();

    auto k0_temp = make_parext_pair_from__(k0, extension_size);
    auto k0_base = make_treg_from(k0.second);
    k0_base.set_limbtype(LimbType::Ext);
    k0_base.set_extension_size(extension_size);
    k0_base.set_adjustment_size(0);
    pmul_inst->add_operands(k0_temp.first, k0.first, giantstep_ct0_sum[n],
                            k0_base);
    pmul_inst->add_operands(k0_temp.second, k0.second, giantstep_ct0_sum[n],
                            k0_base);
    k0 = k0_temp;

    if (giantstep_zero_rot_idx_ct1_sum[n].has_value()) {

      auto k1_temp = make_parext_pair_from__(k1, extension_size);
      auto k1_base = make_treg_from(k1.second);
      k1_base.set_limbtype(LimbType::Ext);
      k1_base.set_extension_size(extension_size);
      k1_base.set_adjustment_size(0);
      pmul_inst->add_operands(k1_temp.first, k1.first,
                              giantstep_zero_rot_idx_ct1_sum[n].value(),
                              k1_base);
      pmul_inst->add_operands(k1_temp.second, k1.second,
                              giantstep_zero_rot_idx_ct1_sum[n].value(),
                              k1_base);
      k1 = k1_temp;
    }

    auto b0_in = make_bcor_from(k0.second);
    auto b1_in = make_bcor_from(k1.second);
    b0_in.set_limbtype(LimbType::Ext);
    b0_in.set_extension_size(extension_size);
    b0_in.set_adjustment_size(rescale_levels);
    b1_in.set_limbtype(LimbType::Ext);
    b1_in.set_extension_size(extension_size);
    b1_in.set_adjustment_size(rescale_levels);
    intt_instr->add_operands(b0_in, k0.second);
    intt_instr->add_operands(b1_in, k1.second);

// 1 : Sud then aggregate scatter
// 0 : All Reduce and aggregate scatter then sud
#if 1
    auto b0 = make_bcor_from(k0.first);
    auto b1 = make_bcor_from(k1.first);
    b0.set_limbtype(LimbType::Usp);
    b1.set_limbtype(LimbType::Usp);
    b0.set_adjustment_size(-rescale_levels);
    b1.set_adjustment_size(-rescale_levels);
    bconv_instr->add_operands(b0, b0_in);
    bconv_instr->add_operands(b1, b1_in);

    Backend::Polynomial t4, t5;
    if (rescale_levels == 0) {
      t4 = make_treg_from(k0.first);
      t5 = make_treg_from(k1.first);
    } else {
      t4 = make_rescale_treg_from(k0.first, rescale_levels);
      t5 = make_rescale_treg_from(k1.first, rescale_levels);
    }
    sud_instr->add_operands(t4, k0.first, b0, b0_in);
    sud_instr->add_operands(t5, k1.first, b1, b1_in);

    if (current_partition_size > 1) {
      t4.set_limbtype(LimbType::Usp);
      t5.set_limbtype(LimbType::Usp);
      auto kg_ags1 = create_kernel_group_agg();
      auto kg_ags2 = create_kernel_group_agg();
      auto ags_instr1 = std::make_shared<AggregateScatterInstruction>();
      auto ags_instr2 = std::make_shared<AggregateScatterInstruction>();
      kg_ags1->push_instruction(ags_instr1);
      kg_ags2->push_instruction(ags_instr2);
      auto t4_ag = make_treg_from(t4);
      auto t5_ag = make_treg_from(t5);
      ags_instr1->add_operands(t5_ag, t5);
      ags_instr2->add_operands(t4_ag, t4);
      t4 = t4_ag;
      t5 = t5_ag;
    }
#else
#endif
    Backend::Ciphertext ret_ct;
    ret_ct.ct0 = t4;
    ret_ct.ct1 = t5;
    ret_ct.level = t4.level();
    giantstep.vec[n] = ret_ct;
  }

  return giantstep;
}

Backend::CiphertextVector
CeriumCompiler::rotate_multiply_accumulate_internal_vec(
    const Backend::CiphertextVector &input,
    const std::vector<Backend::Plaintext> &plaintexts,
    const std::vector<int32_t> rotation_indices, const uint8_t rescale_levels) {


  using LimbType = Backend::Term::LimbType;
  using OpCode = Backend::LimbInstruction::OpCode;

  std::shared_ptr<Backend::BigNode> bn;

  auto num_ct = input.vec.size();
  Backend::CiphertextVector output;
  output.vec.resize(num_ct);

  std::vector<std::optional<PolyPair>> rotate_k0_sum(num_ct);
  std::vector<std::optional<PolyPair>> rotate_k1_sum(num_ct);

  size_t num_rotations = rotation_indices.size();
  auto keyswitch_type = KeySwitch::KeySwitchType::Aggregation;

  auto [extension_size, digits] =
      compute_keyswitch_split(keyswitch_type, input.vec[0].ct1.level());

  auto kg_rot_mul = create_kernel_group();

  std::vector<Backend::Polynomial> rotate_multiplies_ct0_sum(num_ct);
  std::vector<std::optional<Backend::Polynomial>>
      rotate_multiplies_ct1_zero_rot_idx_sum(num_ct);
  std::vector<std::vector<Backend::Polynomial>> rotate_multiplies(
      num_rotations);

  for (int j = 0; j < num_rotations; j++) {
    auto rotation_idx = rotation_indices[j];
    Backend::CiphertextVector t;
    if (rotation_idx == 0) {
      auto mul_instr = std::make_shared<BinOpInstruction>(OpCode::MuP);
      auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
      kg_rot_mul->push_instruction(mul_instr);
      kg_rot_mul->push_instruction(add_instr);
      for (auto i = 0; i < num_ct; i++) {
        auto &ct = input.vec[i];
        auto t2 = make_treg_from(ct.ct0);
        auto t3 = make_treg_from(ct.ct1);
        mul_instr->add_operands(t2, plaintexts[j].pt, ct.ct0);
        mul_instr->add_operands(t3, plaintexts[j].pt, ct.ct1);
        if (j == 0) {
          rotate_multiplies_ct0_sum[i] = t2;
        } else {
          auto &t4 = rotate_multiplies_ct0_sum[i];
          auto tsum = make_treg_from(t2);
          add_instr->add_operands(tsum, t2, t4);
          rotate_multiplies_ct0_sum[i] = tsum;
        }
        if (!rotate_multiplies_ct1_zero_rot_idx_sum[i].has_value()) {
          rotate_multiplies_ct1_zero_rot_idx_sum[i] = t3;
        } else {
          auto &t5 = rotate_multiplies_ct1_zero_rot_idx_sum[i].value();
          auto tsum = make_treg_from(t5);
          add_instr->add_operands(tsum, t3, t5);
          rotate_multiplies_ct1_zero_rot_idx_sum[i] = tsum;
        }
      }
      continue;
    }
    auto rot_instr = std::make_shared<RotationInstruction>();
    auto mul_instr = std::make_shared<BinOpInstruction>(OpCode::MuP);
    auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
    kg_rot_mul->push_instruction(rot_instr);
    kg_rot_mul->push_instruction(mul_instr);
    kg_rot_mul->push_instruction(add_instr);
    rotate_multiplies[j].resize(num_ct);
    for (auto i = 0; i < num_ct; i++) {
      auto &ct = input.vec[i];
      auto t0 = make_treg_from(ct.ct0);
      auto t1 = make_treg_from(ct.ct1);
      rot_instr->add_operands(rotation_idx, t0, ct.ct0);
      rot_instr->add_operands(rotation_idx, t1, ct.ct1);
      auto t2 = make_treg_from(ct.ct0);
      auto &t3 = rotate_multiplies[j][i] = make_treg_from(ct.ct1);
      mul_instr->add_operands(t2, plaintexts[j].pt, t0);
      mul_instr->add_operands(t3, plaintexts[j].pt, t1);
      if (j == 0) {
        rotate_multiplies_ct0_sum[i] = t2;
      } else {
        auto &t4 = rotate_multiplies_ct0_sum[i];
        auto tsum = make_treg_from(t2);
        add_instr->add_operands(tsum, t2, t4);
        rotate_multiplies_ct0_sum[i] = tsum;
      }
    }
  }

  auto make_parext_pair_from = [&](Backend::Polynomial &inp,
                                   const uint64_t extension_size) {
    auto t0 = make_treg_from(inp);
    t0.set_limbtype(LimbType::Usp);
    t0.set_adjustment_size(-rescale_levels);
    auto t1 = make_treg_from(inp);
    t1.set_limbtype(LimbType::Ext);
    t1.set_extension_size(extension_size);
    t1.set_adjustment_size(rescale_levels);
    return std::pair{t0, t1};
  };

  auto make_parext_pair_from__ =
      [&](std::pair<Backend::Polynomial, Backend::Polynomial> &inp,
          const uint64_t extension_size) {
        auto t0 = make_treg_from(std::get<0>(inp));
        t0.set_limbtype(LimbType::Usp);
        t0.set_adjustment_size(-rescale_levels);
        auto t1 = make_treg_from(std::get<1>(inp));
        t1.set_limbtype(LimbType::Ext);
        t1.set_extension_size(extension_size);
        t1.set_adjustment_size(rescale_levels);
        return std::pair{t0, t1};
      };

  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();

  auto kg_bconv0 = create_kernel_group_bconv();
  auto bconv_instr0 = std::make_shared<BconvInstruction>();

  auto kg_ntt0 = create_kernel_group_ntt();
  auto ntt_inst0 = std::make_shared<NttInstruction>();

  auto kg_evkmul = create_kernel_group();
  auto kg_evkmul_ext = create_kernel_group();
  auto mad_inst = std::make_shared<MadInstruction>();
  kg_evkmul->push_instruction(mad_inst);
  auto mad_inst_ext = std::make_shared<MadInstruction>();
  kg_evkmul_ext->push_instruction(mad_inst_ext);

  auto kg_acc = create_kernel_group();
  auto kg_acc_ext = create_kernel_group();

  auto acc_inst = std::make_shared<BinOpInstruction>(OpCode::Add);
  auto acc_inst_ext = std::make_shared<BinOpInstruction>(OpCode::Add);
  kg_acc->push_instruction(acc_inst);
  kg_acc_ext->push_instruction(acc_inst_ext);
  for (auto n = 0; n < num_ct; n++) {

    for (int i = 0; i < rotation_indices.size(); i++) {
      auto rot_idx = rotation_indices[i];
      if (rot_idx == 0) {
        continue;
      }
      auto rot1 = rotate_multiplies[i][n];

      auto evalkeys = get_evalkey(rot1, Backend::EvalKeyType::Rotation, rot_idx,
                                  keyswitch_type, extension_size);
      auto evk0 = evalkeys.first;
      auto evk1 = evalkeys.second;
      evk0.first.set_adjustment_size(-rescale_levels);
      evk0.second.set_adjustment_size(rescale_levels);
      evk1.first.set_adjustment_size(-rescale_levels);
      evk1.second.set_adjustment_size(rescale_levels);

      auto t0 = rot1;

      Backend::Polynomial b0_in;
      b0_in = make_bcor_from(t0);
      intt_instr0->add_operands(b0_in, t0);

      auto b0 = make_bcor_from(b0_in);
      b0.set_limbtype(LimbType::ExC);
      b0.set_extension_size(extension_size);
      bconv_instr0->add_operands(b0, b0_in);

      auto t1 = make_parext_pair_from(t0, extension_size);
      ntt_inst0->add_operands(t1.first, t1.second, b0, t0, b0_in);

      auto t2 = make_parext_pair_from__(t1, extension_size);
      auto t3 = make_parext_pair_from__(t1, extension_size);
      mad_inst->add_operands(t2.first, t1.first, evk0.first);
      mad_inst->add_operands(t3.first, t1.first, evk1.first);

      mad_inst_ext->add_operands(t2.second, t1.second, evk0.second);
      mad_inst_ext->add_operands(t3.second, t1.second, evk1.second);

      if (!rotate_k0_sum[n].has_value()) {
        rotate_k0_sum[n] = t2;
      } else {
        auto &k0_sum_val = rotate_k0_sum[n].value();
        auto tsum = make_parext_pair_from__(t2, extension_size);
        acc_inst->add_operands(tsum.first, t2.first, k0_sum_val.first);
        acc_inst_ext->add_operands(tsum.second, t2.second, k0_sum_val.second);
        rotate_k0_sum[n] = tsum;
      }

      if (!rotate_k1_sum[n].has_value()) {
        rotate_k1_sum[n] = t3;
      } else {
        auto &k1_sum_val = rotate_k1_sum[n].value();
        auto tsum = make_parext_pair_from__(t2, extension_size);
        acc_inst->add_operands(tsum.first, t3.first, k1_sum_val.first);
        acc_inst_ext->add_operands(tsum.second, t3.second, k1_sum_val.second);
        rotate_k1_sum[n] = tsum;
      }
    }
  }

  kg_intt0->push_instruction(intt_instr0);
  kg_bconv0->push_instruction(bconv_instr0);
  kg_ntt0->push_instruction(ntt_inst0);
  split_digit_wise(digits, kg_intt0, kg_bconv0, kg_ntt0, kg_evkmul,
                   kg_evkmul_ext, false /*don't split modular*/);

  kg_evkmul->merge_kg(kg_acc);
  kg_evkmul_ext->merge_kg(kg_acc_ext);

  auto kg_pmul = create_kernel_group_pmul();
  auto pmul_inst = std::make_shared<PmuInstruction>();
  kg_pmul->push_instruction(pmul_inst);

  auto kg_intt = create_kernel_group_intt();
  auto intt_instr = std::make_shared<InttInstruction>();
  kg_intt->push_instruction(intt_instr);

  auto kg_bconv = create_kernel_group_bconv();
  auto bconv_instr = std::make_shared<BconvInstruction>();
  kg_bconv->push_instruction(bconv_instr);

  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();
  kg_sud->push_instruction(sud_instr);

  for (auto n = 0; n < num_ct; n++) {

    assert(rotate_k0_sum[n].has_value());
    assert(rotate_k1_sum[n].has_value());

    auto &k0 = rotate_k0_sum[n].value();
    auto &k1 = rotate_k1_sum[n].value();

    auto k0_temp = make_parext_pair_from__(k0, extension_size);
    auto k0_base = make_treg_from(k0.second);
    k0_base.set_limbtype(LimbType::Ext);
    k0_base.set_extension_size(extension_size);
    k0_base.set_adjustment_size(0);
    pmul_inst->add_operands(k0_temp.first, k0.first,
                            rotate_multiplies_ct0_sum[n], k0_base);
    pmul_inst->add_operands(k0_temp.second, k0.second,
                            rotate_multiplies_ct0_sum[n], k0_base);
    k0 = k0_temp;

    if (rotate_multiplies_ct1_zero_rot_idx_sum[n].has_value()) {

      auto k1_temp = make_parext_pair_from__(k1, extension_size);
      auto k1_base = make_treg_from(k1.second);
      k1_base.set_limbtype(LimbType::Ext);
      k1_base.set_extension_size(extension_size);
      k1_base.set_adjustment_size(0);
      pmul_inst->add_operands(k1_temp.first, k1.first,
                              rotate_multiplies_ct1_zero_rot_idx_sum[n].value(),
                              k1_base);
      pmul_inst->add_operands(k1_temp.second, k1.second,
                              rotate_multiplies_ct1_zero_rot_idx_sum[n].value(),
                              k1_base);
      k1 = k1_temp;
    }

    auto b0_in = make_bcor_from(k0.second);
    auto b1_in = make_bcor_from(k1.second);
    b0_in.set_limbtype(LimbType::Ext);
    b0_in.set_extension_size(extension_size);
    b0_in.set_adjustment_size(rescale_levels);
    b1_in.set_limbtype(LimbType::Ext);
    b1_in.set_extension_size(extension_size);
    b1_in.set_adjustment_size(rescale_levels);
    intt_instr->add_operands(b0_in, k0.second);
    intt_instr->add_operands(b1_in, k1.second);

// 1 : Sud then aggregate scatter
// 0 : All Reduce and aggregate scatter then sud
#if 1
    auto b0 = make_bcor_from(k0.first);
    auto b1 = make_bcor_from(k1.first);
    b0.set_limbtype(LimbType::Usp);
    b1.set_limbtype(LimbType::Usp);
    b0.set_adjustment_size(-rescale_levels);
    b1.set_adjustment_size(-rescale_levels);
    bconv_instr->add_operands(b0, b0_in);
    bconv_instr->add_operands(b1, b1_in);

    Backend::Polynomial t4, t5;
    if (rescale_levels == 0) {
      t4 = make_treg_from(k0.first);
      t5 = make_treg_from(k1.first);
    } else {
      t4 = make_rescale_treg_from(k0.first, rescale_levels);
      t5 = make_rescale_treg_from(k1.first, rescale_levels);
    }
    sud_instr->add_operands(t4, k0.first, b0, b0_in);
    sud_instr->add_operands(t5, k1.first, b1, b1_in);

    if (current_partition_size > 1) {
      t4.set_limbtype(LimbType::Usp);
      t5.set_limbtype(LimbType::Usp);
      auto kg_ags = create_kernel_group_agg();
      auto ags_instr = std::make_shared<AggregateScatterInstruction>();
      kg_ags->push_instruction(ags_instr);
      auto t4_ag = make_treg_from(t4);
      auto t5_ag = make_treg_from(t5);
      ags_instr->add_operands(t5_ag, t5);
      ags_instr->add_operands(t4_ag, t4);
      t4 = t4_ag;
      t5 = t5_ag;
    }
#else
#endif
    Backend::Ciphertext ret_ct;
    ret_ct.ct0 = t4;
    ret_ct.ct1 = t5;
    ret_ct.level = t4.level();
    output.vec[n] = ret_ct;
  }

  return output;
}

Backend::CiphertextVector
CeriumCompiler::multiply_rotate_accumulate_internal_vec(
    const Backend::CiphertextVector &input,
    const std::vector<Backend::Plaintext> &plaintexts,
    const std::vector<int32_t> rotation_indices, const uint8_t rescale_levels) {

  using LimbType = Backend::Term::LimbType;
  using OpCode = Backend::LimbInstruction::OpCode;

  std::shared_ptr<Backend::BigNode> bn;

  auto num_ct = input.vec.size();
  Backend::CiphertextVector output;
  output.vec.resize(num_ct);

  std::vector<std::optional<PolyPair>> rotate_k0_sum(num_ct);
  std::vector<std::optional<PolyPair>> rotate_k1_sum(num_ct);

  size_t num_rotations = rotation_indices.size();
  auto keyswitch_type = KeySwitch::KeySwitchType::Aggregation;

  auto [extension_size, digits] =
      compute_keyswitch_split(keyswitch_type, input.vec[0].ct1.level());

  auto kg_mul = create_kernel_group();
  auto kg_rot = create_kernel_group();

  std::vector<Backend::Polynomial> rotate_multiplies_ct0_sum(num_ct);
  std::vector<std::optional<Backend::Polynomial>>
      rotate_multiplies_ct1_zero_rot_idx_sum(num_ct);
  std::vector<std::vector<Backend::Polynomial>> rotate_multiplies(
      num_rotations);

  for (int j = 0; j < num_rotations; j++) {
    auto rotation_idx = rotation_indices[j];
    Backend::CiphertextVector t;
    if (rotation_idx == 0) {
      auto mul_instr = std::make_shared<BinOpInstruction>(OpCode::MuP);
      auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
      kg_mul->push_instruction(mul_instr);
      kg_rot->push_instruction(add_instr);
      for (auto i = 0; i < num_ct; i++) {
        auto &ct = input.vec[i];
        auto t2 = make_treg_from(ct.ct0);
        auto t3 = make_treg_from(ct.ct1);
        mul_instr->add_operands(t2, plaintexts[j].pt, ct.ct0);
        mul_instr->add_operands(t3, plaintexts[j].pt, ct.ct1);
        if (j == 0) {
          rotate_multiplies_ct0_sum[i] = t2;
        } else {
          auto &t4 = rotate_multiplies_ct0_sum[i];
          auto tsum = make_treg_from(t2);
          add_instr->add_operands(tsum, t2, t4);
          rotate_multiplies_ct0_sum[i] = tsum;
        }
        if (!rotate_multiplies_ct1_zero_rot_idx_sum[i].has_value()) {
          rotate_multiplies_ct1_zero_rot_idx_sum[i] = t3;
        } else {
          auto &t5 = rotate_multiplies_ct1_zero_rot_idx_sum[i].value();
          auto tsum = make_treg_from(t5);
          add_instr->add_operands(tsum, t3, t5);
          rotate_multiplies_ct1_zero_rot_idx_sum[i] = tsum;
        }
      }
      continue;
    }
    auto mul_instr = std::make_shared<BinOpInstruction>(OpCode::MuP);
    auto rot_instr = std::make_shared<RotationInstruction>();
    auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
    kg_mul->push_instruction(mul_instr);
    kg_rot->push_instruction(rot_instr);
    kg_rot->push_instruction(add_instr);
    rotate_multiplies[j].resize(num_ct);
    for (auto i = 0; i < num_ct; i++) {
      auto &ct = input.vec[i];
      auto t0 = make_treg_from(ct.ct0);
      auto t1 = make_treg_from(ct.ct1);
      mul_instr->add_operands(t0, ct.ct0, plaintexts[j].pt);
      mul_instr->add_operands(t1, ct.ct1, plaintexts[j].pt);
      auto t2 = make_treg_from(t0);
      auto &t3 = rotate_multiplies[j][i] = make_treg_from(t1);
      rot_instr->add_operands(rotation_idx, t2, t0);
      rot_instr->add_operands(rotation_idx, t3, t1);
      if (j == 0) {
        rotate_multiplies_ct0_sum[i] = t2;
      } else {
        auto &t4 = rotate_multiplies_ct0_sum[i];
        auto tsum = make_treg_from(t2);
        add_instr->add_operands(tsum, t2, t4);
        rotate_multiplies_ct0_sum[i] = tsum;
      }
    }
  }

  auto make_parext_pair_from = [&](Backend::Polynomial &inp,
                                   const uint64_t extension_size) {
    auto t0 = make_treg_from(inp);
    t0.set_limbtype(LimbType::Usp);
    t0.set_adjustment_size(-rescale_levels);
    auto t1 = make_treg_from(inp);
    t1.set_limbtype(LimbType::Ext);
    t1.set_extension_size(extension_size);
    t1.set_adjustment_size(rescale_levels);
    return std::pair{t0, t1};
  };

  auto make_parext_pair_from__ =
      [&](std::pair<Backend::Polynomial, Backend::Polynomial> &inp,
          const uint64_t extension_size) {
        auto t0 = make_treg_from(std::get<0>(inp));
        t0.set_limbtype(LimbType::Usp);
        t0.set_adjustment_size(-rescale_levels);
        auto t1 = make_treg_from(std::get<1>(inp));
        t1.set_limbtype(LimbType::Ext);
        t1.set_extension_size(extension_size);
        t1.set_adjustment_size(rescale_levels);
        return std::pair{t0, t1};
      };

  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();

  auto kg_bconv0 = create_kernel_group_bconv();
  auto bconv_instr0 = std::make_shared<BconvInstruction>();

  auto kg_ntt0 = create_kernel_group_ntt();
  auto ntt_inst0 = std::make_shared<NttInstruction>();

  auto kg_evkmul = create_kernel_group();
  auto kg_evkmul_ext = create_kernel_group();
  auto mad_inst = std::make_shared<MadInstruction>();
  kg_evkmul->push_instruction(mad_inst);
  auto mad_inst_ext = std::make_shared<MadInstruction>();
  kg_evkmul_ext->push_instruction(mad_inst_ext);

  auto kg_acc = create_kernel_group();
  auto kg_acc_ext = create_kernel_group();

  auto acc_inst = std::make_shared<BinOpInstruction>(OpCode::Add);
  auto acc_inst_ext = std::make_shared<BinOpInstruction>(OpCode::Add);
  kg_acc->push_instruction(acc_inst);
  kg_acc_ext->push_instruction(acc_inst_ext);
  for (auto n = 0; n < num_ct; n++) {

    for (int i = 0; i < rotation_indices.size(); i++) {
      auto rot_idx = rotation_indices[i];
      if (rot_idx == 0) {
        continue;
      }
      auto rot1 = rotate_multiplies[i][n];

      auto evalkeys = get_evalkey(rot1, Backend::EvalKeyType::Rotation, rot_idx,
                                  keyswitch_type, extension_size);
      auto evk0 = evalkeys.first;
      auto evk1 = evalkeys.second;
      evk0.first.set_adjustment_size(-rescale_levels);
      evk0.second.set_adjustment_size(rescale_levels);
      evk1.first.set_adjustment_size(-rescale_levels);
      evk1.second.set_adjustment_size(rescale_levels);

      auto t0 = rot1;

      Backend::Polynomial b0_in;
      b0_in = make_bcor_from(t0);
      intt_instr0->add_operands(b0_in, t0);

      auto b0 = make_bcor_from(b0_in);
      b0.set_limbtype(LimbType::ExC);
      b0.set_extension_size(extension_size);
      bconv_instr0->add_operands(b0, b0_in);

      auto t1 = make_parext_pair_from(t0, extension_size);
      ntt_inst0->add_operands(t1.first, t1.second, b0, t0, b0_in);

      auto t2 = make_parext_pair_from__(t1, extension_size);
      auto t3 = make_parext_pair_from__(t1, extension_size);
      mad_inst->add_operands(t2.first, t1.first, evk0.first);
      mad_inst->add_operands(t3.first, t1.first, evk1.first);

      mad_inst_ext->add_operands(t2.second, t1.second, evk0.second);
      mad_inst_ext->add_operands(t3.second, t1.second, evk1.second);

      if (!rotate_k0_sum[n].has_value()) {
        rotate_k0_sum[n] = t2;
      } else {
        auto &k0_sum_val = rotate_k0_sum[n].value();
        auto tsum = make_parext_pair_from__(t2, extension_size);
        acc_inst->add_operands(tsum.first, t2.first, k0_sum_val.first);
        acc_inst_ext->add_operands(tsum.second, t2.second, k0_sum_val.second);
        rotate_k0_sum[n] = tsum;
      }

      if (!rotate_k1_sum[n].has_value()) {
        rotate_k1_sum[n] = t3;
      } else {
        auto &k1_sum_val = rotate_k1_sum[n].value();
        auto tsum = make_parext_pair_from__(t2, extension_size);
        acc_inst->add_operands(tsum.first, t3.first, k1_sum_val.first);
        acc_inst_ext->add_operands(tsum.second, t3.second, k1_sum_val.second);
        rotate_k1_sum[n] = tsum;
      }
    }
  }

  kg_intt0->push_instruction(intt_instr0);
  kg_bconv0->push_instruction(bconv_instr0);
  kg_ntt0->push_instruction(ntt_inst0);
  split_digit_wise(digits, kg_intt0, kg_bconv0, kg_ntt0, kg_evkmul,
                   kg_evkmul_ext, false /*don't split modular*/);

  kg_evkmul->merge_kg(kg_acc);
  kg_evkmul_ext->merge_kg(kg_acc_ext);

  auto kg_pmul = create_kernel_group_pmul();
  auto pmul_inst = std::make_shared<PmuInstruction>();
  kg_pmul->push_instruction(pmul_inst);

  auto kg_intt = create_kernel_group_intt();
  auto intt_instr = std::make_shared<InttInstruction>();
  kg_intt->push_instruction(intt_instr);

  auto kg_bconv = create_kernel_group_bconv();
  auto bconv_instr = std::make_shared<BconvInstruction>();
  kg_bconv->push_instruction(bconv_instr);

  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();
  kg_sud->push_instruction(sud_instr);

  for (auto n = 0; n < num_ct; n++) {

    assert(rotate_k0_sum[n].has_value());
    assert(rotate_k1_sum[n].has_value());

    auto &k0 = rotate_k0_sum[n].value();
    auto &k1 = rotate_k1_sum[n].value();

    auto k0_temp = make_parext_pair_from__(k0, extension_size);
    auto k0_base = make_treg_from(k0.second);
    k0_base.set_limbtype(LimbType::Ext);
    k0_base.set_extension_size(extension_size);
    k0_base.set_adjustment_size(0);
    pmul_inst->add_operands(k0_temp.first, k0.first,
                            rotate_multiplies_ct0_sum[n], k0_base);
    pmul_inst->add_operands(k0_temp.second, k0.second,
                            rotate_multiplies_ct0_sum[n], k0_base);
    k0 = k0_temp;

    if (rotate_multiplies_ct1_zero_rot_idx_sum[n].has_value()) {

      auto k1_temp = make_parext_pair_from__(k1, extension_size);
      auto k1_base = make_treg_from(k1.second);
      k1_base.set_limbtype(LimbType::Ext);
      k1_base.set_extension_size(extension_size);
      k1_base.set_adjustment_size(0);
      pmul_inst->add_operands(k1_temp.first, k1.first,
                              rotate_multiplies_ct1_zero_rot_idx_sum[n].value(),
                              k1_base);
      pmul_inst->add_operands(k1_temp.second, k1.second,
                              rotate_multiplies_ct1_zero_rot_idx_sum[n].value(),
                              k1_base);
      k1 = k1_temp;
    }

    auto b0_in = make_bcor_from(k0.second);
    auto b1_in = make_bcor_from(k1.second);
    b0_in.set_limbtype(LimbType::Ext);
    b0_in.set_extension_size(extension_size);
    b0_in.set_adjustment_size(rescale_levels);
    b1_in.set_limbtype(LimbType::Ext);
    b1_in.set_extension_size(extension_size);
    b1_in.set_adjustment_size(rescale_levels);
    intt_instr->add_operands(b0_in, k0.second);
    intt_instr->add_operands(b1_in, k1.second);

// 1 : Sud then aggregate scatter
// 0 : All Reduce and aggregate scatter then sud
#if 1
    auto b0 = make_bcor_from(k0.first);
    auto b1 = make_bcor_from(k1.first);
    b0.set_limbtype(LimbType::Usp);
    b1.set_limbtype(LimbType::Usp);
    b0.set_adjustment_size(-rescale_levels);
    b1.set_adjustment_size(-rescale_levels);
    bconv_instr->add_operands(b0, b0_in);
    bconv_instr->add_operands(b1, b1_in);

    Backend::Polynomial t4, t5;
    if (rescale_levels == 0) {
      t4 = make_treg_from(k0.first);
      t5 = make_treg_from(k1.first);
    } else {
      t4 = make_rescale_treg_from(k0.first, rescale_levels);
      t5 = make_rescale_treg_from(k1.first, rescale_levels);
    }
    sud_instr->add_operands(t4, k0.first, b0, b0_in);
    sud_instr->add_operands(t5, k1.first, b1, b1_in);

    if (current_partition_size > 1) {
      t4.set_limbtype(LimbType::Usp);
      t5.set_limbtype(LimbType::Usp);
      auto kg_ags = create_kernel_group_agg();
      auto ags_instr = std::make_shared<AggregateScatterInstruction>();
      kg_ags->push_instruction(ags_instr);
      auto t4_ag = make_treg_from(t4);
      auto t5_ag = make_treg_from(t5);
      ags_instr->add_operands(t5_ag, t5);
      ags_instr->add_operands(t4_ag, t4);
      t4 = t4_ag;
      t5 = t5_ag;
    }
#else
#endif
    Backend::Ciphertext ret_ct;
    ret_ct.ct0 = t4;
    ret_ct.ct1 = t5;
    ret_ct.level = t4.level();
    output.vec[n] = ret_ct;
  }

  return output;
}

Backend::CiphertextVector CeriumCompiler::rotate_accumulate_internal(
    const std::vector<Backend::CiphertextVector> &inputs,
    const std::vector<int32_t> rotation_indices, const uint32_t dnum_glo,
    const uint8_t rescale_levels) {

  using LimbType = Backend::Term::LimbType;
  using OpCode = Backend::LimbInstruction::OpCode;

  std::shared_ptr<Backend::BigNode> bn;
  std::optional<Backend::Polynomial>
      zero_rot_idx_sum; /* Sum of terms that are not rotated*/


  auto num_ct = inputs[0].vec.size();
  Backend::CiphertextVector outputs;
  outputs.vec.resize(num_ct);

  std::vector<Backend::Polynomial> giantstep_ct0_sum(num_ct);
  std::vector<std::optional<Backend::Polynomial>>
      giantstep_zero_rot_idx_ct1_sum(num_ct);
  std::vector<std::optional<PolyPair>> giantstep_k0_sum(num_ct);
  std::vector<std::optional<PolyPair>> giantstep_k1_sum(num_ct);

  auto keyswitch_type = KeySwitch::KeySwitchType::Aggregation;

  auto [extension_size, digits] =
      compute_keyswitch_split(keyswitch_type, inputs[0].vec[0].ct1.level());

  auto kg_mul = create_kernel_group();

  auto make_parext_pair_from = [&](Backend::Polynomial &inp,
                                   const uint64_t extension_size) {
    auto t0 = make_treg_from(inp);
    t0.set_limbtype(LimbType::Usp);
    t0.set_adjustment_size(-rescale_levels);
    auto t1 = make_treg_from(inp);
    t1.set_limbtype(LimbType::Ext);
    t1.set_extension_size(extension_size);
    t1.set_adjustment_size(rescale_levels);
    return std::pair{t0, t1};
  };

  auto make_parext_pair_from__ =
      [&](std::pair<Backend::Polynomial, Backend::Polynomial> &inp,
          const uint64_t extension_size) {
        auto t0 = make_treg_from(std::get<0>(inp));
        t0.set_limbtype(LimbType::Usp);
        t0.set_adjustment_size(-rescale_levels);
        auto t1 = make_treg_from(std::get<1>(inp));
        t1.set_limbtype(LimbType::Ext);
        t1.set_extension_size(extension_size);
        t1.set_adjustment_size(rescale_levels);
        return std::pair{t0, t1};
      };

  auto kg_rot = create_kernel_group();
  std::vector<std::shared_ptr<RotationInstruction>> rot_instrs;
  for (int i = 0; i < rotation_indices.size(); i++) {
    auto rot_idx = rotation_indices[i];
    auto rot_instr = std::make_shared<RotationInstruction>();
    rot_instrs.push_back(rot_instr);
    kg_rot->push_instruction(rot_instr);
  }
  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();

  auto kg_bconv0 = create_kernel_group_bconv();
  auto bconv_instr0 = std::make_shared<BconvInstruction>();

  auto kg_ntt0 = create_kernel_group_ntt();
  auto ntt_inst0 = std::make_shared<NttInstruction>();

  auto kg_evkmul = create_kernel_group();
  auto kg_evkmul_ext = create_kernel_group();
  auto mad_inst = std::make_shared<MadInstruction>();
  kg_evkmul->push_instruction(mad_inst);
  auto mad_inst_ext = std::make_shared<MadInstruction>();
  kg_evkmul_ext->push_instruction(mad_inst_ext);

  auto kg_acc = create_kernel_group();
  auto kg_acc_ext = create_kernel_group();

  auto acc_inst = std::make_shared<BinOpInstruction>(OpCode::Add);
  auto acc_inst_ext = std::make_shared<BinOpInstruction>(OpCode::Add);
  kg_acc->push_instruction(acc_inst);
  kg_acc_ext->push_instruction(acc_inst_ext);
  for (int n = num_ct - 1; n >= 0; n--) {

    for (int i = 0; i < rotation_indices.size(); i++) {
      auto rot_idx = rotation_indices[i];
      Backend::Polynomial rot0;
      Backend::Polynomial rot1;
      if (rot_idx == 0) {
        rot0 = inputs[i].vec[n].ct0;
        rot1 = inputs[i].vec[n].ct1;
      } else {
        rot0 = make_treg_from(inputs[i].vec[n].ct0);
        rot1 = make_treg_from(inputs[i].vec[n].ct1);

        auto rot_instr = rot_instrs[i];
        rot_instr->add_operands(rot_idx, rot0, inputs[i].vec[n].ct0);
        rot_instr->add_operands(rot_idx, rot1, inputs[i].vec[n].ct1);
      }
      if (i == 0) {
        giantstep_ct0_sum[n] = rot0;
      } else {
        auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
        auto tsum = make_treg_from(rot0);
        add_instr->add_operands(tsum, rot0, giantstep_ct0_sum[n]);
        giantstep_ct0_sum[n] = tsum;
        kg_rot->push_instruction(add_instr);
      }
      if (rot_idx == 0) {
        if (!giantstep_zero_rot_idx_ct1_sum[n].has_value()) {
          giantstep_zero_rot_idx_ct1_sum[n] = rot1;
        } else {

          auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
          auto tsum = make_treg_from(rot0);
          add_instr->add_operands(tsum, rot1,
                                  giantstep_zero_rot_idx_ct1_sum[n].value());
          giantstep_zero_rot_idx_ct1_sum[n] = tsum;
          kg_rot->push_instruction(add_instr);
        }
        continue;
      }

      auto evalkeys = get_evalkey(rot1, Backend::EvalKeyType::Rotation, rot_idx,
                                  keyswitch_type, extension_size);
      auto evk0 = evalkeys.first;
      auto evk1 = evalkeys.second;
      evk0.first.set_adjustment_size(-rescale_levels);
      evk0.second.set_adjustment_size(rescale_levels);
      evk1.first.set_adjustment_size(-rescale_levels);
      evk1.second.set_adjustment_size(rescale_levels);

      auto t0 = rot1;

      Backend::Polynomial b0_in;
      b0_in = make_bcor_from(t0);
      intt_instr0->add_operands(b0_in, t0);

      auto b0 = make_bcor_from(b0_in);
      b0.set_limbtype(LimbType::ExC);
      b0.set_extension_size(extension_size);
      bconv_instr0->add_operands(b0, b0_in);

      auto t1 = make_parext_pair_from(t0, extension_size);
      ntt_inst0->add_operands(t1.first, t1.second, b0, t0, b0_in);

      auto t2 = make_parext_pair_from__(t1, extension_size);
      auto t3 = make_parext_pair_from__(t1, extension_size);
      mad_inst->add_operands(t2.first, t1.first, evk0.first);
      mad_inst->add_operands(t3.first, t1.first, evk1.first);

      mad_inst_ext->add_operands(t2.second, t1.second, evk0.second);
      mad_inst_ext->add_operands(t3.second, t1.second, evk1.second);

      if (!giantstep_k0_sum[n].has_value()) {
        giantstep_k0_sum[n] = t2;
      } else {
        auto &k0_sum_val = giantstep_k0_sum[n].value();
        auto tsum = make_parext_pair_from__(t2, extension_size);
        acc_inst->add_operands(tsum.first, t2.first, k0_sum_val.first);
        acc_inst_ext->add_operands(tsum.second, t2.second, k0_sum_val.second);
        giantstep_k0_sum[n] = tsum;
      }

      if (!giantstep_k1_sum[n].has_value()) {
        giantstep_k1_sum[n] = t3;
      } else {
        auto &k1_sum_val = giantstep_k1_sum[n].value();
        auto tsum = make_parext_pair_from__(t2, extension_size);
        acc_inst->add_operands(tsum.first, t3.first, k1_sum_val.first);
        acc_inst_ext->add_operands(tsum.second, t3.second, k1_sum_val.second);
        giantstep_k1_sum[n] = tsum;
      }
    }
  }

  kg_intt0->push_instruction(intt_instr0);
  kg_bconv0->push_instruction(bconv_instr0);
  kg_ntt0->push_instruction(ntt_inst0);
  split_digit_wise(digits, kg_intt0, kg_bconv0, kg_ntt0, kg_evkmul,
                   kg_evkmul_ext, false /*don't split modular*/);

  kg_evkmul->merge_kg(kg_acc);
  kg_evkmul_ext->merge_kg(kg_acc_ext);

  auto kg_pmul = create_kernel_group_pmul();
  auto pmul_inst = std::make_shared<PmuInstruction>();
  kg_pmul->push_instruction(pmul_inst);

  auto kg_intt = create_kernel_group_intt();
  auto intt_instr = std::make_shared<InttInstruction>();
  kg_intt->push_instruction(intt_instr);

  auto kg_bconv = create_kernel_group_bconv();
  auto bconv_instr = std::make_shared<BconvInstruction>();
  kg_bconv->push_instruction(bconv_instr);

  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();
  kg_sud->push_instruction(sud_instr);

  for (auto n = 0; n < num_ct; n++) {

    assert(giantstep_k0_sum[n].has_value());
    assert(giantstep_k1_sum[n].has_value());

    auto &k0 = giantstep_k0_sum[n].value();
    auto &k1 = giantstep_k1_sum[n].value();

    auto k0_temp = make_parext_pair_from__(k0, extension_size);
    auto k0_base = make_treg_from(k0.second);
    k0_base.set_limbtype(LimbType::Ext);
    k0_base.set_extension_size(extension_size);
    k0_base.set_adjustment_size(0);
    pmul_inst->add_operands(k0_temp.first, k0.first, giantstep_ct0_sum[n],
                            k0_base);
    pmul_inst->add_operands(k0_temp.second, k0.second, giantstep_ct0_sum[n],
                            k0_base);
    k0 = k0_temp;

    if (giantstep_zero_rot_idx_ct1_sum[n].has_value()) {

      auto k1_temp = make_parext_pair_from__(k1, extension_size);
      auto k1_base = make_treg_from(k1.second);
      k1_base.set_limbtype(LimbType::Ext);
      k1_base.set_extension_size(extension_size);
      k1_base.set_adjustment_size(0);
      pmul_inst->add_operands(k1_temp.first, k1.first,
                              giantstep_zero_rot_idx_ct1_sum[n].value(),
                              k1_base);
      pmul_inst->add_operands(k1_temp.second, k1.second,
                              giantstep_zero_rot_idx_ct1_sum[n].value(),
                              k1_base);
      k1 = k1_temp;
    }

    auto b0_in = make_bcor_from(k0.second);
    auto b1_in = make_bcor_from(k1.second);
    b0_in.set_limbtype(LimbType::Ext);
    b0_in.set_extension_size(extension_size);
    b0_in.set_adjustment_size(rescale_levels);
    b1_in.set_limbtype(LimbType::Ext);
    b1_in.set_extension_size(extension_size);
    b1_in.set_adjustment_size(rescale_levels);
    intt_instr->add_operands(b0_in, k0.second);
    intt_instr->add_operands(b1_in, k1.second);

// 1 : Sud then aggregate scatter
// 0 : All Reduce and aggregate scatter then sud
#if 1
    auto b0 = make_bcor_from(k0.first);
    auto b1 = make_bcor_from(k1.first);
    b0.set_limbtype(LimbType::Usp);
    b1.set_limbtype(LimbType::Usp);
    b0.set_adjustment_size(-rescale_levels);
    b1.set_adjustment_size(-rescale_levels);
    bconv_instr->add_operands(b0, b0_in);
    bconv_instr->add_operands(b1, b1_in);

    Backend::Polynomial t4, t5;
    if (rescale_levels == 0) {
      t4 = make_treg_from(k0.first);
      t5 = make_treg_from(k1.first);
    } else {
      t4 = make_rescale_treg_from(k0.first, rescale_levels);
      t5 = make_rescale_treg_from(k1.first, rescale_levels);
    }
    sud_instr->add_operands(t4, k0.first, b0, b0_in);
    sud_instr->add_operands(t5, k1.first, b1, b1_in);

    if (current_partition_size > 1) {
      t4.set_limbtype(LimbType::Usp);
      t5.set_limbtype(LimbType::Usp);
      auto kg_ags = create_kernel_group_agg();
      auto ags_instr = std::make_shared<AggregateScatterInstruction>();
      kg_ags->push_instruction(ags_instr);
      auto t4_ag = make_treg_from(t4);
      auto t5_ag = make_treg_from(t5);
      ags_instr->add_operands(t5_ag, t5);
      ags_instr->add_operands(t4_ag, t4);
      t4 = t4_ag;
      t5 = t5_ag;
    }
#else
#endif
    Backend::Ciphertext ret_ct;
    ret_ct.ct0 = t4;
    ret_ct.ct1 = t5;
    ret_ct.level = t4.level();
    outputs.vec[n] = ret_ct;
  }

  return outputs;
}

CeriumCompiler::PolyPair
CeriumCompiler::bsgs_giantstep_accumulate_keyswitch_iterations(
    const std::tuple<std::vector<Backend::Polynomial>,
                     std::vector<Backend::Polynomial>> &babysteps,
    const std::vector<Backend::Plaintext> &plaintexts,
    const std::vector<int32_t> giantstep_rotation_indices,
    const KernelGroup::SplitType &split, const uint32_t extension_size____,
    const uint32_t dnum_glo, const uint8_t rescale_levels) {
  using LimbType = Backend::Term::LimbType;
  using OpCode = Backend::LimbInstruction::OpCode;

  std::shared_ptr<Backend::BigNode> bn;
  std::optional<Backend::Polynomial>
      zero_rot_idx_sum; /* Sum of terms that are not rotated*/

  std::vector<Backend::Polynomial> babystep_rotations_ct0 =
      std::get<0>(babysteps);
  std::vector<Backend::Polynomial> babystep_rotations_ct1 =
      std::get<1>(babysteps);

  Backend::Polynomial giantstep_ct0_sum;

  std::optional<Backend::Polynomial> giantstep_zero_rot_idx_ct1_sum;
  std::optional<PolyPair> giantstep_k0_sum;
  std::optional<PolyPair> giantstep_k1_sum;

  size_t num_babysteps = babystep_rotations_ct0.size();
  auto keyswitch_type = KeySwitch::KeySwitchType::Aggregation;

  auto [extension_size, digits] = compute_keyswitch_split(
      keyswitch_type, babystep_rotations_ct1[0].level());

  auto kg_mul = create_kernel_group();
  auto mul_instr = std::make_shared<BinOpInstruction>(OpCode::MuP);
  auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);

  std::vector<Backend::Polynomial> babystep_accumulated_ct0(
      giantstep_rotation_indices.size());
  std::vector<Backend::Polynomial> babystep_accumulated_ct1(
      giantstep_rotation_indices.size());
  for (int i = 0; i < giantstep_rotation_indices.size(); i++) {

    auto rot_idx = giantstep_rotation_indices[i];
    for (int j = 0; j < num_babysteps; j++) {
      auto &plaintext = plaintexts[j + i * num_babysteps];
      auto t0 = make_treg_from(babystep_rotations_ct0[j]);
      auto t1 = make_treg_from(babystep_rotations_ct1[j]);
      mul_instr->add_operands(t0, plaintext.pt, babystep_rotations_ct0[j]);
      mul_instr->add_operands(t1, plaintext.pt, babystep_rotations_ct1[j]);

      if (j == 0) {
        babystep_accumulated_ct0[i] = t0;
        babystep_accumulated_ct1[i] = t1;
      } else {
        auto t2 = make_treg_from(t0);
        auto t3 = make_treg_from(t1);
        add_instr->add_operands(t2, t0, babystep_accumulated_ct0[i]);
        add_instr->add_operands(t3, t1, babystep_accumulated_ct1[i]);
        babystep_accumulated_ct0[i] = t2;
        babystep_accumulated_ct1[i] = t3;
      }
    }
  }
  kg_mul->push_instruction(mul_instr);
  kg_mul->push_instruction(add_instr);

  auto kg_rot = create_kernel_group();
  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();

  auto kg_bconv0 = create_kernel_group_bconv();
  auto bconv_instr0 = std::make_shared<BconvInstruction>();

  auto kg_ntt0 = create_kernel_group_ntt();
  auto ntt_inst0 = std::make_shared<NttInstruction>();

  auto kg_evkmul = create_kernel_group();
  auto kg_evkmul_ext = create_kernel_group();
  auto kg_acc = create_kernel_group();
  auto kg_acc_ext = create_kernel_group();

  auto make_parext_pair_from = [&](Backend::Polynomial &inp,
                                   const uint64_t extension_size) {
    auto t0 = make_treg_from(inp);
    t0.set_limbtype(LimbType::Usp);
    t0.set_adjustment_size(-rescale_levels);
    auto t1 = make_treg_from(inp);
    t1.set_limbtype(LimbType::Ext);
    t1.set_extension_size(extension_size);
    t1.set_adjustment_size(rescale_levels);
    return std::pair{t0, t1};
  };

  auto make_parext_pair_from__ =
      [&](std::pair<Backend::Polynomial, Backend::Polynomial> &inp,
          const uint64_t extension_size) {
        auto t0 = make_treg_from(std::get<0>(inp));
        t0.set_limbtype(LimbType::Usp);
        t0.set_adjustment_size(-rescale_levels);
        auto t1 = make_treg_from(std::get<1>(inp));
        t1.set_limbtype(LimbType::Ext);
        t1.set_extension_size(extension_size);
        t1.set_adjustment_size(rescale_levels);
        return std::pair{t0, t1};
      };
  for (int i = 0; i < giantstep_rotation_indices.size(); i++) {

    auto rot_idx = giantstep_rotation_indices[i];
    Backend::Polynomial rot0;
    Backend::Polynomial rot1;
    if (rot_idx == 0) {
      rot0 = babystep_accumulated_ct0[i];
      rot1 = babystep_accumulated_ct1[i];
    } else {
      rot0 = make_treg_from(babystep_accumulated_ct0[i]);
      rot1 = make_treg_from(babystep_accumulated_ct1[i]);

      auto rot_instr = std::make_shared<RotationInstruction>();
      rot_instr->add_operands(rot_idx, rot0, babystep_accumulated_ct0[i]);
      rot_instr->add_operands(rot_idx, rot1, babystep_accumulated_ct1[i]);
      kg_rot->push_instruction(rot_instr);
    }

    if (i == 0) {
      giantstep_ct0_sum = rot0;
    } else {

      auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
      auto tsum = make_treg_from(rot0);
      add_instr->add_operands(tsum, rot0, giantstep_ct0_sum);
      giantstep_ct0_sum = tsum;
      kg_rot->push_instruction(add_instr);
    }

    if (rot_idx == 0) {
      if (!giantstep_zero_rot_idx_ct1_sum.has_value()) {
        giantstep_zero_rot_idx_ct1_sum = rot1;
      } else {

        auto add_instr = std::make_shared<BinOpInstruction>(OpCode::Add);
        auto tsum = make_treg_from(rot0);
        add_instr->add_operands(tsum, rot1,
                                giantstep_zero_rot_idx_ct1_sum.value());
        giantstep_zero_rot_idx_ct1_sum = tsum;
        kg_rot->push_instruction(add_instr);
      }
      continue;
    }

    auto evalkeys = get_evalkey(rot1, Backend::EvalKeyType::Rotation, rot_idx,
                                keyswitch_type, extension_size);
    auto evk0 = evalkeys.first;
    auto evk1 = evalkeys.second;
    evk0.first.set_adjustment_size(-rescale_levels);
    evk0.second.set_adjustment_size(rescale_levels);
    evk1.first.set_adjustment_size(-rescale_levels);
    evk1.second.set_adjustment_size(rescale_levels);

    auto t0 = rot1;

    Backend::Polynomial b0_in;
    b0_in = make_bcor_from(t0);
    intt_instr0->add_operands(b0_in, t0);

    auto b0 = make_bcor_from(b0_in);
    b0.set_limbtype(LimbType::ExC);
    b0.set_extension_size(extension_size);
    bconv_instr0->add_operands(b0, b0_in);

    auto t1 = make_parext_pair_from(t0, extension_size);
    ntt_inst0->add_operands(t1.first, t1.second, b0, t0, b0_in);

    auto t2 = make_parext_pair_from__(t1, extension_size);
    auto t3 = make_parext_pair_from__(t1, extension_size);
    auto mad_inst = std::make_shared<MadInstruction>();
    mad_inst->add_operands(t2.first, t1.first, evk0.first);
    mad_inst->add_operands(t3.first, t1.first, evk1.first);
    kg_evkmul->push_instruction(mad_inst);

    auto mad_inst_ext = std::make_shared<MadInstruction>();
    mad_inst_ext->add_operands(t2.second, t1.second, evk0.second);
    mad_inst_ext->add_operands(t3.second, t1.second, evk1.second);
    kg_evkmul_ext->push_instruction(mad_inst_ext);

    auto acc_inst = std::make_shared<BinOpInstruction>(OpCode::Add);
    auto acc_inst_ext = std::make_shared<BinOpInstruction>(OpCode::Add);

    if (!giantstep_k0_sum.has_value()) {
      giantstep_k0_sum = t2;
    } else {
      auto &k0_sum_val = giantstep_k0_sum.value();
      auto tsum = make_parext_pair_from__(t2, extension_size);
      acc_inst->add_operands(tsum.first, t2.first, k0_sum_val.first);
      acc_inst_ext->add_operands(tsum.second, t2.second, k0_sum_val.second);
      giantstep_k0_sum = tsum;
    }

    if (!giantstep_k1_sum.has_value()) {
      giantstep_k1_sum = t3;
    } else {
      auto &k1_sum_val = giantstep_k1_sum.value();
      auto tsum = make_parext_pair_from__(t2, extension_size);
      acc_inst->add_operands(tsum.first, t3.first, k1_sum_val.first);
      acc_inst_ext->add_operands(tsum.second, t3.second, k1_sum_val.second);
      giantstep_k1_sum = tsum;
    }
    kg_acc->push_instruction(acc_inst);
    kg_acc_ext->push_instruction(acc_inst_ext);
  }

  kg_intt0->push_instruction(intt_instr0);
  kg_bconv0->push_instruction(bconv_instr0);
  kg_ntt0->push_instruction(ntt_inst0);
  split_digit_wise(digits, kg_intt0, kg_bconv0, kg_ntt0, kg_evkmul,
                   kg_evkmul_ext, false /*don't split modular*/);

  kg_evkmul->merge_kg(kg_acc);
  kg_evkmul_ext->merge_kg(kg_acc_ext);

  assert(giantstep_k0_sum.has_value());
  assert(giantstep_k1_sum.has_value());

  auto &k0 = giantstep_k0_sum.value();
  auto &k1 = giantstep_k1_sum.value();

  auto kg_pmul = create_kernel_group_pmul();
  auto k0_temp = make_parext_pair_from__(k0, extension_size);
  auto k0_base = make_treg_from(k0.second);
  k0_base.set_limbtype(LimbType::Ext);
  k0_base.set_extension_size(extension_size);
  k0_base.set_adjustment_size(0);
  auto pmul_inst = std::make_shared<PmuInstruction>();
  pmul_inst->add_operands(k0_temp.first, k0.first, giantstep_ct0_sum, k0_base);
  pmul_inst->add_operands(k0_temp.second, k0.second, giantstep_ct0_sum,
                          k0_base);
  k0 = k0_temp;

  if (giantstep_zero_rot_idx_ct1_sum.has_value()) {

    auto k1_temp = make_parext_pair_from__(k1, extension_size);
    auto k1_base = make_treg_from(k1.second);
    k1_base.set_limbtype(LimbType::Ext);
    k1_base.set_extension_size(extension_size);
    k1_base.set_adjustment_size(0);
    pmul_inst->add_operands(k1_temp.first, k1.first,
                            giantstep_zero_rot_idx_ct1_sum.value(), k1_base);
    pmul_inst->add_operands(k1_temp.second, k1.second,
                            giantstep_zero_rot_idx_ct1_sum.value(), k1_base);
    k1 = k1_temp;
  }

  kg_pmul->push_instruction(pmul_inst);

  auto kg_intt = create_kernel_group_intt();
  auto b0_in = make_bcor_from(k0.second);
  auto b1_in = make_bcor_from(k1.second);
  b0_in.set_limbtype(LimbType::Ext);
  b0_in.set_extension_size(extension_size);
  b0_in.set_adjustment_size(rescale_levels);
  b1_in.set_limbtype(LimbType::Ext);
  b1_in.set_extension_size(extension_size);
  b1_in.set_adjustment_size(rescale_levels);
  auto intt_instr = std::make_shared<InttInstruction>();
  intt_instr->add_operands(b0_in, k0.second);
  intt_instr->add_operands(b1_in, k1.second);
  kg_intt->push_instruction(intt_instr);

// 1 : Sud then aggregate scatter
// 0 : All Reduce and aggregate scatter then sud
#if 1
  auto kg_bconv = create_kernel_group_bconv();
  auto b0 = make_bcor_from(k0.first);
  auto b1 = make_bcor_from(k1.first);
  b0.set_limbtype(LimbType::Usp);
  b1.set_limbtype(LimbType::Usp);
  b0.set_adjustment_size(-rescale_levels);
  b1.set_adjustment_size(-rescale_levels);
  auto bconv_instr = std::make_shared<BconvInstruction>();
  bconv_instr->add_operands(b0, b0_in);
  bconv_instr->add_operands(b1, b1_in);
  kg_bconv->push_instruction(bconv_instr);

  Backend::Polynomial t4, t5;
  if (rescale_levels == 0) {
    t4 = make_treg_from(k0.first);
    t5 = make_treg_from(k1.first);
  } else {
    t4 = make_rescale_treg_from(k0.first, rescale_levels);
    t5 = make_rescale_treg_from(k1.first, rescale_levels);
  }
  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();
  sud_instr->add_operands(t4, k0.first, b0, b0_in);
  sud_instr->add_operands(t5, k1.first, b1, b1_in);
  kg_sud->push_instruction(sud_instr);

  if (current_partition_size > 1) {
    t4.set_limbtype(LimbType::Usp);
    t5.set_limbtype(LimbType::Usp);
    auto kg_ags = create_kernel_group_agg();
    auto ags_instr = std::make_shared<AggregateScatterInstruction>();
    auto t4_ag = make_treg_from(t4);
    auto t5_ag = make_treg_from(t5);
    ags_instr->add_operands(t5_ag, t5);
    ags_instr->add_operands(t4_ag, t4);
    kg_ags->push_instruction(ags_instr);
    t4 = t4_ag;
    t5 = t5_ag;
  }
#else
  if (current_partition_size > 1) {
    auto kg_ard = create_kernel_group_ard();
    auto kg_ags = create_kernel_group_agg();
    auto ard_instr = std::make_shared<AllReduceInstruction>();
    auto ags_instr = std::make_shared<AggregateScatterInstruction>();
    auto k0_ag = make_treg_from(k0.first);
    auto k1_ag = make_treg_from(k1.first);
    k0_ag.set_adjustment_size(-rescale_levels);
    k1_ag.set_adjustment_size(-rescale_levels);
    auto b0_in_red = make_new_term_share_from(&b0_in);
    auto b1_in_red = make_new_term_share_from(&b1_in);
    b0_in_red.set_extension_size(extension_size);
    b1_in_red.set_extension_size(extension_size);
    ard_instr->add_operands(b0_in_red, b0_in);
    ard_instr->add_operands(b1_in_red, b1_in);
    ags_instr->add_operands(k0_ag, k0.first);
    ags_instr->add_operands(k1_ag, k1.first);
    kg_ard->push_instruction(ard_instr);
    kg_ags->push_instruction(ags_instr);
    b0_in = b0_in_red;
    b1_in = b1_in_red;
    k0.first = k0_ag;
    k1.first = k1_ag;
  }

  auto kg_bconv = create_kernel_group_bconv();
  auto b0 = make_bcor_from(k0.first);
  auto b1 = make_bcor_from(k1.first);
  // b0.set_limbtype(LimbType::Par);
  // b1.set_limbtype(LimbType::Par);
  b0.set_adjustment_size(-rescale_levels);
  b1.set_adjustment_size(-rescale_levels);
  auto bconv_instr = std::make_shared<BconvInstruction>();
  bconv_instr->add_operands(b0, b0_in);
  bconv_instr->add_operands(b1, b1_in);
  kg_bconv->push_instruction(bconv_instr);

  Backend::Polynomial t4, t5;
  if (rescale_levels == 0) {
    t4 = make_treg_from(k0.first);
    t5 = make_treg_from(k1.first);
  } else {
    t4 = make_rescale_treg_from(k0.first, rescale_levels);
    t5 = make_rescale_treg_from(k1.first, rescale_levels);
  }
  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();
  sud_instr->add_operands(t4, k0.first, b0, b0_in);
  sud_instr->add_operands(t5, k1.first, b1, b1_in);
  kg_sud->push_instruction(sud_instr);
#endif

  return PolyPair(t4, t5);
}

void CeriumCompiler::hoisted_input_broadcast(
    const Frontend::Term::Ptr &term, std::vector<int32_t> rotation_indices,
    const uint32_t rescale_levels) {
  auto &args = term->getOperands();
  assert(isCipher(args[0]));

  using OpCode = Backend::LimbInstruction::OpCode;
  Backend::CiphertextVector input1;
  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) {
            input1.vec.push_back(input);
          },
          [&](Backend::CiphertextVector &input) {
            throw std::runtime_error(
                "Unsupported Hoisted Input Broadcast for CiphertextVector");
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args[0]));
  auto level = input1.vec[0].level;
  assert(rescale_levels == 0);

  auto key_switch_type = KeySwitch::KeySwitchType::Broadcast;
  auto [extension_size, digits] =
      compute_keyswitch_split(key_switch_type, level);

  std::vector<Backend::CiphertextVector> keyswitch;
  keyswitch = hoisted_input_broadcast_keyswitching_internal_vec(
      input1, rotation_indices);

  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) {
            auto &output = initValue<Backend::CiphertextVector>(term);
            for (auto &k : keyswitch) {
              output.vec.push_back(k.vec[0]);
            }
          },
          [&](Backend::CiphertextVector &input) {
            throw std::runtime_error(
                "Unsupported Hoisted Input Broadcast for CiphertextVector");
          },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args[0]));
}

void CeriumCompiler::rotate_accumulate(const Frontend::Term::Ptr &term,
                                         std::vector<int32_t> rotation_indices,
                                         const uint32_t rescale_levels) {
  auto &args = term->getOperands();
  for (auto &arg : args) {
    assert(isCipher(arg));
  }

  assert(rotation_indices.size() == args.size());

  using OpCode = Backend::LimbInstruction::OpCode;
  std::vector<Backend::CiphertextVector> input1;
  for (auto &arg : args) {
    Backend::Ciphertext &arg_value =
        std::get<Backend::Ciphertext>(Objects.at(arg));
    Backend::CiphertextVector i;
    i.vec.push_back(arg_value);
    input1.push_back(i);
  }

  auto level = input1[0].vec[0].level;
  assert(rescale_levels == 0);

  Backend::CiphertextVector keyswitch_giantstep;
  keyswitch_giantstep =
      rotate_accumulate_internal(input1, rotation_indices, 1, rescale_levels);

  std::visit(Overloaded{[&](Backend::Ciphertext &input) {
                          auto &output = initValue<Backend::Ciphertext>(term);
                          output = keyswitch_giantstep.vec[0];
                        },
                        [&](Backend::CiphertextVector &input) {
                          throw std::runtime_error(
                              "Rot Acc unsupported for Ciphertext Vector");
                        },
                        [&](auto &arg) {
                          throw std::runtime_error(
                              "Unsupported operation encountered");
                        }},
             Objects.at(args[0]));
}

void CeriumCompiler::rotate_multiply_accumulate_common(
    const Frontend::Term::Ptr &term,
    const std::vector<Frontend::Term::Ptr> &args,
    std::vector<int32_t> rotation_indices, const uint32_t rescale_levels,
    const std::function<Backend::CiphertextVector(
        const Backend::CiphertextVector &input1,
        const std::vector<Backend::Plaintext> &plaintexts,
        const std::vector<int32_t> &rotation_indices,
        const uint32_t rescale_levels)> &build_output) {
  assert(args.size() == rotation_indices.size() + 1);
  assert(isCipher(args[0]) || isCiphertextVector(args[0]));

  std::vector<Backend::Plaintext> plaintext_inputs;
  plaintext_inputs.reserve(args.size() - 1);
  for (size_t i = 1; i < args.size(); ++i) {
    assert(isPlain(args[i]));
    plaintext_inputs.push_back(
        std::get<Backend::Plaintext>(Objects.at(args[i])));
  }

  Backend::CiphertextVector input1;
  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) { input1.vec.push_back(input); },
          [&](Backend::CiphertextVector &input) {
            input1 = input;
            const auto level = input1.vec[0].level;
            for (size_t i = 1; i < input1.vec.size(); ++i) {
              assert(level == input1.vec[i].level);
            }
          },
          [&](auto &) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args[0]));

  const auto level = input1.vec[0].level;
  assert(rescale_levels < level);

  const auto output_vec =
      build_output(input1, plaintext_inputs, rotation_indices, rescale_levels);

  std::visit(Overloaded{[&](Backend::Ciphertext &) {
                          auto &output = initValue<Backend::Ciphertext>(term);
                          output = output_vec.vec[0];
                        },
                        [&](Backend::CiphertextVector &) {
                          auto &output =
                              initValue<Backend::CiphertextVector>(term);
                          output = output_vec;
                        },
                        [&](auto &) {
                          throw std::runtime_error(
                              "Unsupported operation encountered");
                        }},
             Objects.at(args[0]));
}

void CeriumCompiler::rotate_multiply_accumulate(
    const Frontend::Term::Ptr &term,
    const std::vector<Frontend::Term::Ptr> &args,
    std::vector<int32_t> rotation_indices, const uint32_t rescale_levels) {
  rotate_multiply_accumulate_common(
      term, args, std::move(rotation_indices), rescale_levels,
      [this](const Backend::CiphertextVector &input1,
             const std::vector<Backend::Plaintext> &plaintext_inputs,
             const std::vector<int32_t> &rotation_indices,
             const uint32_t rescale_levels) {
        return rotate_multiply_accumulate_internal_vec(
            input1, plaintext_inputs, rotation_indices, rescale_levels);
      });
}

void CeriumCompiler::multiply_rotate_accumulate(
    const Frontend::Term::Ptr &term,
    const std::vector<Frontend::Term::Ptr> &args,
    std::vector<int32_t> rotation_indices, const uint32_t rescale_levels) {
  rotate_multiply_accumulate_common(
      term, args, std::move(rotation_indices), rescale_levels,
      [this](const Backend::CiphertextVector &input1,
             const std::vector<Backend::Plaintext> &plaintext_inputs,
             const std::vector<int32_t> &rotation_indices,
             const uint32_t rescale_levels) {
        return multiply_rotate_accumulate_internal_vec(
            input1, plaintext_inputs, rotation_indices, rescale_levels);
      });
}
void CeriumCompiler::bsgs_multiply_accumulate(
    const Frontend::Term::Ptr &term,
    const std::vector<Frontend::Term::Ptr> &args,
    std::vector<int32_t> babystep_rotation_indices,
    std::vector<int32_t> giantstep_rotation_indices,
    const uint32_t rescale_levels) {
  assert(args.size() ==
         babystep_rotation_indices.size() * giantstep_rotation_indices.size() +
             1);
  assert(isCipher(args[0]) || isCiphertextVector(args[0]));
  std::vector<Backend::Plaintext> plaintext_inputs;
  for (int i = 1; i < args.size(); i++) {
    assert(isPlain(args[i]));
    Backend::Plaintext &plaintext_input =
        std::get<Backend::Plaintext>(Objects.at(args[i]));
    plaintext_inputs.push_back(plaintext_input);
  }

  using OpCode = Backend::LimbInstruction::OpCode;
  Backend::CiphertextVector input1;
  std::visit(Overloaded{[&](Backend::Ciphertext &input) {
                          input1.vec.push_back(input);
                        },
                        [&](Backend::CiphertextVector &input) {
                          input1 = input;
                          auto &level = input1.vec[0].level;
                          for (size_t i = 1; i < input1.vec.size(); i++) {
                            assert(level == input1.vec[i].level);
                          }
                        },
                        [&](auto &arg) {
                          throw std::runtime_error(
                              "Unsupported operation encountered");
                        }},
             Objects.at(args[0]));
  auto level = input1.vec[0].level;
  assert(rescale_levels < level);

  auto key_switch_type = KeySwitch::KeySwitchType::Broadcast;
  auto [extension_size, digits] =
      compute_keyswitch_split(key_switch_type, level);

  std::vector<Backend::CiphertextVector> keyswitch_babystep;
  keyswitch_babystep = hoisted_input_broadcast_keyswitching_internal_vec(
      input1, babystep_rotation_indices);

  Backend::CiphertextVector keyswitch_giantstep;
  keyswitch_giantstep = bsgs_giantstep_accumulate_keyswitch_iterations_vec(
      keyswitch_babystep, plaintext_inputs, giantstep_rotation_indices, digits,
      extension_size, 1, rescale_levels);

  std::visit(Overloaded{[&](Backend::Ciphertext &input) {
                          auto &output = initValue<Backend::Ciphertext>(term);
                          output = keyswitch_giantstep.vec[0];
                        },
                        [&](Backend::CiphertextVector &input) {
                          auto &output =
                              initValue<Backend::CiphertextVector>(term);
                          output = keyswitch_giantstep;
                        },
                        [&](auto &arg) {
                          throw std::runtime_error(
                              "Unsupported operation encountered");
                        }},
             Objects.at(args[0]));
}

} // namespace Backend
} // namespace Cerium
