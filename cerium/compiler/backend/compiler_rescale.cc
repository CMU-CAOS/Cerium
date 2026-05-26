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

void CeriumCompiler::to_ephemeral(Backend::Ciphertext &output,
                                    const Frontend::Term::Ptr &args1) {

  Backend::Ciphertext &input1 =
      std::get<Backend::Ciphertext>(Objects.at(args1));
  output.level = input1.level;
  assert(!input1.ct2.has_value());
  using OpCode = Backend::LimbInstruction::OpCode;
  output = keyswitch2(input1, Backend::EvalKeyType::Ephemeral);
}

void CeriumCompiler::bootstrap_mod_raise(Backend::Ciphertext &output,
                                           const Frontend::Term::Ptr &args1,
                                           const uint16_t raise_to_level,
                                           bool use_ephemeral_key) {

  Backend::Ciphertext &input1 =
      std::get<Backend::Ciphertext>(Objects.at(args1));
  assert(!input1.ct2.has_value());
  output.level = raise_to_level;
  using OpCode = Backend::LimbInstruction::OpCode;

  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;

  // TODO: Set the limbtypes....
  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();
  auto t0 = make_treg_from(input1.ct0);
  auto t1 = make_treg_from(input1.ct1);
  intt_instr0->add_operands(t0, input1.ct0);
  intt_instr0->add_operands(t1, input1.ct1);
  kg_intt0->push_instruction(intt_instr0);

  KernelGroupPtr kg_dist;
  auto dist_instr = std::make_shared<DistRecvInstruction>();

  if (current_partition_size > 1) {
    kg_dist = create_kernel_group_dist();
    auto t0_dist = make_treg_from(t0);
    auto t1_dist = make_treg_from(t1);
    dist_instr->add_operands(t0_dist, t0);
    dist_instr->add_operands(t1_dist, t1);
    kg_dist->push_instruction(dist_instr);
    t0 = t0_dist;
    t1 = t1_dist;
  }

  auto t2 = make_treg_from(input1.ct0);
  auto t3 = make_treg_from(input1.ct1);
  auto kg_resolve = create_kernel_group();
  auto resolve_instr = std::make_shared<ResolveInstruction>();
  resolve_instr->add_operands(t2, t0);
  resolve_instr->add_operands(t3, t1);
  kg_resolve->push_instruction(resolve_instr);
  kg_resolve->set_dont_split_modular(true);

  auto t4 = make_treg(raise_to_level);
  auto kg_mod0 = create_kernel_group();
  auto mod_instr0 = std::make_shared<ModInstruction>();
  mod_instr0->add_operands(t4, t2);
  kg_mod0->push_instruction(mod_instr0);

  auto kg_ntt0 = create_kernel_group_ntt();
  auto t6 = make_treg_from(t4);
  auto ntt_instr0 = std::make_shared<NttInstructionSingle>();
  ntt_instr0->add_operands(t6, t4);
  kg_ntt0->push_instruction(ntt_instr0);

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

  auto raised_level = raise_to_level + 1;
  auto extension_size = input1.level; 
  PolyPair t5;
  t5.first = make_treg(raised_level);
  t5.first.set_limbtype(LimbType::Par);
  t5.second = make_treg(raised_level);
  t5.second.set_limbtype(LimbType::Ext);
  t5.second.set_extension_size(extension_size);

  auto kg_mod1 = create_kernel_group();
  auto mod_instr1 = std::make_shared<ModInstruction>();
  mod_instr1->add_operands(t5.first, t3);
  mod_instr1->add_operands(t5.second, t3);
  kg_mod1->push_instruction(mod_instr1);

  auto t7 = make_parext_pair_from__(t5, extension_size);
  auto kg_ntt = create_kernel_group_ntt();
  auto ntt_instr = std::make_shared<NttInstructionSingle>();
  ntt_instr->add_operands(t7.first, t5.first);
  ntt_instr->add_operands(t7.second, t5.second);
  kg_ntt->push_instruction(ntt_instr);

  auto level = input1.level;
  auto key_switch_type = KeySwitch::KeySwitchType::Broadcast;
  auto [extension_size_, digits] =
      compute_keyswitch_split(key_switch_type, level);
  assert(extension_size == extension_size_);
  std::pair<PolyPair, PolyPair> evalkeys;
  if (use_ephemeral_key) {
    evalkeys = get_evalkey(input1.ct1, Backend::EvalKeyType::Bootstrap2,
                           raise_to_level, key_switch_type, extension_size);
  } else {
    evalkeys = get_evalkey(input1.ct1, Backend::EvalKeyType::Bootstrap,
                           raise_to_level, key_switch_type, extension_size);
  }
  auto evk0 = evalkeys.first;
  auto evk1 = evalkeys.second;

  auto kg_evkmul = create_kernel_group();
  auto t8 = make_parext_pair_from__(t7, extension_size);
  auto t9 = make_parext_pair_from__(t7, extension_size);
  auto mad_inst = std::make_shared<MadInstruction>();
  mad_inst->add_operands(t8.first, t7.first, evk0.first);
  mad_inst->add_operands(t9.first, t7.first, evk1.first);
  kg_evkmul->push_instruction(mad_inst);

  auto kg_evkmul_ext = create_kernel_group();
  auto mad_inst_ext = std::make_shared<MadInstruction>();
  mad_inst_ext->add_operands(t8.second, t7.second, evk0.second);
  mad_inst_ext->add_operands(t9.second, t7.second, evk1.second);
  kg_evkmul_ext->push_instruction(mad_inst_ext);

  assert(digits.size() == 1);
  DigitMap<Backend::Polynomial> digit_map;
  split_digit_wise_internal(digits, kg_mod1, digit_map);
  split_digit_wise_internal(digits, kg_ntt, digit_map);
  split_digit_wise_internal_mad(digits, kg_evkmul, digit_map);
  split_digit_wise_internal_mad(digits, kg_evkmul_ext, digit_map);
  kg_mod1->set_dont_split_modular(true);
  kg_ntt->set_dont_split_modular(true);
  kg_evkmul->set_dont_split_modular(true);
  kg_evkmul_ext->set_dont_split_modular(true);

  auto kg_intt = create_kernel_group_intt();
  auto b8_in = make_bcor_from(t8.second);
  auto b9_in = make_bcor_from(t9.second);
  b8_in.set_limbtype(LimbType::Ext);
  b8_in.set_extension_size(extension_size);
  b9_in.set_limbtype(LimbType::Ext);
  b9_in.set_extension_size(extension_size);
  auto intt_instr = std::make_shared<InttInstruction>();
  intt_instr->add_operands(b8_in, t8.second);
  intt_instr->add_operands(b9_in, t9.second);
  kg_intt->push_instruction(intt_instr);

  auto kg_bconv = create_kernel_group_bconv();
  auto b8 = make_bcor_from(t8.first);
  auto b9 = make_bcor_from(t9.first);
  b8.set_limbtype(LimbType::Par);
  b9.set_limbtype(LimbType::Par);
  auto bconv_instr = std::make_shared<BconvInstruction>();
  bconv_instr->add_operands(b8, b8_in);
  bconv_instr->add_operands(b9, b9_in);
  kg_bconv->push_instruction(bconv_instr);

  Backend::Ciphertext temp;
  temp.ct0 = make_treg_from(t8.first);
  temp.ct1 = make_treg_from(t9.first);
  temp.level = t8.first.level();
  Backend::Ciphertext temp_rescale;
  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();
  sud_instr->add_operands(temp.ct0, t8.first, b8, b8_in);
  sud_instr->add_operands(temp.ct1, t9.first, b9, b9_in);
  kg_sud->push_instruction(sud_instr);

  rescale_internal(temp_rescale, temp);

  auto kg_add = create_kernel_group();
  output.ct0 = make_treg_from(temp_rescale.ct0);
  auto add_instr0 = std::make_shared<BinOpInstruction>(OpCode::Add);
  add_instr0->add_operands(output.ct0, temp_rescale.ct0, t6);
  kg_add->push_instruction(add_instr0);

  output.ct1 = temp_rescale.ct1;

}

void CeriumCompiler::rescale(const Frontend::Term::Ptr &term,
                                   const Frontend::Term::Ptr &args1) {

  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;

  auto kg_intt = create_kernel_group_intt();
  auto intt_instruction = std::make_shared<InttInstruction>();
  kg_intt->push_instruction(intt_instruction);

  KernelGroupPtr kg_drm0, kg_drm1;
  auto drm_instruction0 = std::make_shared<DistRecvInstruction>();
  auto drm_instruction1 = std::make_shared<DistRecvInstruction>();

  if (current_partition_size > 1) {
    kg_drm1 = create_kernel_group_dist();
    kg_drm1->push_instruction(drm_instruction1);
    kg_drm0 = create_kernel_group_dist();
    kg_drm0->push_instruction(drm_instruction0);
  }

  auto kg_sud = create_kernel_group_ntt();
  auto sud_instruction = std::make_shared<BinOpInstruction>(OpCode::SuD);
  kg_sud->push_instruction(sud_instruction);

  auto rescale_ct = [&](Backend::Ciphertext &input) {
    assert(!input.ct2.has_value());
    Backend::Ciphertext output;
    output.level = input.level - 1;
    Backend::LimbIndexType last_limbshare = input.level - 1;

    auto t1_lastlimb = make_new_term_share_from(&input.ct1);
    auto t0_lastlimb = make_new_term_share_from(&input.ct0);
    t1_lastlimb.set_shares(std::set<Backend::LimbIndexType>{last_limbshare});
    t0_lastlimb.set_shares(std::set<Backend::LimbIndexType>{last_limbshare});

    auto t1_remaining_limbs = make_new_term_share_from(&input.ct1);
    auto t0_remaining_limbs = make_new_term_share_from(&input.ct0);
    auto remaining_limbs = t1_remaining_limbs.shares();
    remaining_limbs.erase(last_limbshare);
    t1_remaining_limbs.set_shares(remaining_limbs);
    t0_remaining_limbs.set_shares(remaining_limbs);

    auto t1_lastlimb_intt = make_treg_from(input.ct1);
    auto t0_lastlimb_intt = make_treg_from(input.ct0);
    t1_lastlimb_intt.set_shares(
        std::set<Backend::LimbIndexType>{last_limbshare});
    t0_lastlimb_intt.set_shares(
        std::set<Backend::LimbIndexType>{last_limbshare});

    intt_instruction->add_operands(t0_lastlimb_intt, t0_lastlimb);
    intt_instruction->add_operands(t1_lastlimb_intt, t1_lastlimb);

    if (current_partition_size > 1) {
      auto t0_ll_intt_dist = make_treg_from(t0_lastlimb_intt);
      auto t1_ll_intt_dist = make_treg_from(t1_lastlimb_intt);
      t0_ll_intt_dist.set_shares(
          std::set<Backend::LimbIndexType>{last_limbshare});
      t1_ll_intt_dist.set_shares(
          std::set<Backend::LimbIndexType>{last_limbshare});
      drm_instruction1->add_operands(t1_ll_intt_dist, t1_lastlimb_intt);
      drm_instruction0->add_operands(t0_ll_intt_dist, t0_lastlimb_intt);
      t0_lastlimb_intt = t0_ll_intt_dist;
      t1_lastlimb_intt = t1_ll_intt_dist;
    }

    output.ct0 = make_rescale_treg_from(input.ct0);
    output.ct1 = make_rescale_treg_from(input.ct1);

    sud_instruction->add_operands(output.ct0, t0_remaining_limbs,
                                  t0_lastlimb_intt);
    sud_instruction->add_operands(output.ct1, t1_remaining_limbs,
                                  t1_lastlimb_intt);

    return output;
  };

  std::visit(Overloaded{[&](Backend::Ciphertext &input) {
                          auto &output = initValue<Backend::Ciphertext>(term);
                          output = rescale_ct(input);
                        },
                        [&](Backend::CiphertextVector &input) {
                          auto &output =
                              initValue<Backend::CiphertextVector>(term);
                          output.vec.resize(input.vec.size());
                          for (size_t i = 0; i < input.vec.size(); i++) {
                            output.vec[i] = rescale_ct(input.vec.at(i));
                          }
                        },
                        [&](auto &arg) {
                          throw std::runtime_error(
                              "Unsupported operation encountered");
                        }},
             Objects.at(args1));
}

void CeriumCompiler::rescale_internal(Backend::Ciphertext &output,
                                        const Backend::Ciphertext &input1) {

  output.level = input1.level - 1;
  using OpCode = Backend::LimbInstruction::OpCode;

  Backend::LimbIndexType last_limbshare = input1.level - 1;

  auto t1_lastlimb = make_new_term_share_from(&input1.ct1);
  auto t0_lastlimb = make_new_term_share_from(&input1.ct0);
  t1_lastlimb.set_shares(std::set<Backend::LimbIndexType>{last_limbshare});
  t0_lastlimb.set_shares(std::set<Backend::LimbIndexType>{last_limbshare});

  auto t1_remaining_limbs = make_new_term_share_from(&input1.ct1);
  auto t0_remaining_limbs = make_new_term_share_from(&input1.ct0);
  auto remaining_limbs = t1_remaining_limbs.shares();
  remaining_limbs.erase(last_limbshare);
  t1_remaining_limbs.set_shares(remaining_limbs);
  t0_remaining_limbs.set_shares(remaining_limbs);

  auto t1_lastlimb_intt = make_treg_from(input1.ct1);
  auto t0_lastlimb_intt = make_treg_from(input1.ct0);
  t1_lastlimb_intt.set_shares(std::set<Backend::LimbIndexType>{last_limbshare});
  t0_lastlimb_intt.set_shares(std::set<Backend::LimbIndexType>{last_limbshare});

  auto kg_intt = create_kernel_group_intt();
  auto intt_instruction = std::make_shared<InttInstruction>();
  intt_instruction->add_operands(t0_lastlimb_intt, t0_lastlimb);
  intt_instruction->add_operands(t1_lastlimb_intt, t1_lastlimb);
  kg_intt->push_instruction(intt_instruction);

  if (current_partition_size > 1) {
    auto kg_drm0 = create_kernel_group_dist();
    auto kg_drm1 = create_kernel_group_dist();
    auto drm_instruction0 = std::make_shared<DistRecvInstruction>();
    auto drm_instruction1 = std::make_shared<DistRecvInstruction>();
    auto t0_ll_intt_dist = make_treg_from(t0_lastlimb_intt);
    auto t1_ll_intt_dist = make_treg_from(t1_lastlimb_intt);
    t0_ll_intt_dist.set_shares(
        std::set<Backend::LimbIndexType>{last_limbshare});
    t1_ll_intt_dist.set_shares(
        std::set<Backend::LimbIndexType>{last_limbshare});
    drm_instruction0->add_operands(t0_ll_intt_dist, t0_lastlimb_intt);
    drm_instruction1->add_operands(t1_ll_intt_dist, t1_lastlimb_intt);
    kg_drm0->push_instruction(drm_instruction0);
    kg_drm1->push_instruction(drm_instruction1);
    t0_lastlimb_intt = t0_ll_intt_dist;
    t1_lastlimb_intt = t1_ll_intt_dist;
  }

  output.ct0 = make_rescale_treg_from(input1.ct0);
  output.ct1 = make_rescale_treg_from(input1.ct1);

  auto kg_sud = create_kernel_group_ntt();
  auto sud_instruction = std::make_shared<BinOpInstruction>(OpCode::SuD);
  sud_instruction->add_operands(output.ct0, t0_remaining_limbs,
                                t0_lastlimb_intt);
  sud_instruction->add_operands(output.ct1, t1_remaining_limbs,
                                t1_lastlimb_intt);
  kg_sud->push_instruction(sud_instruction);
}

void CeriumCompiler::double_rescale(const Frontend::Term::Ptr &term,
                                          const Frontend::Term::Ptr &args1) {
  Backend::Ciphertext *input_ptr = nullptr;
  Backend::CiphertextVector *input_vec_ptr = nullptr;

  std::visit(
      Overloaded{
          [&](Backend::Ciphertext &input) { input_ptr = &input; },
          [&](Backend::CiphertextVector &input) { input_vec_ptr = &input; },
          [&](auto &arg) {
            throw std::runtime_error("Unsupported operation encountered");
          }},
      Objects.at(args1));

  KernelGroupPtr kg_intt;
  std::shared_ptr<InttInstruction> intt_instr;
  KernelGroupPtr kg_drm;
  std::shared_ptr<DistRecvInstruction> drm_instruction;
  KernelGroupPtr kg_resolve;
  std::shared_ptr<ResolveInstruction> resolve_instr;
  KernelGroupPtr kg_mod;
  KernelGroupPtr kg_sud;
  std::shared_ptr<SudInstruction> sud_instruction;

  auto double_rescale_ct = [&](Backend::Ciphertext &input1,
                               Backend::Ciphertext &output,
                               std::shared_ptr<ModInstruction> &mod_instr) {
    assert(!input1.ct2.has_value());
    output.level = input1.level - 2;
    auto level = input1.level;

    Backend::LimbIndexType last_limbshare = level - 1;
    Backend::LimbIndexType second_last_limbshare = level - 2;

    auto limbshare_to_drop = std::set<Backend::LimbIndexType>(
        {last_limbshare, second_last_limbshare});

    auto t0_lastlimb = make_new_term_share_from(&input1.ct0);
    auto t1_lastlimb = make_new_term_share_from(&input1.ct1);
    t0_lastlimb.set_shares(limbshare_to_drop);
    t1_lastlimb.set_shares(limbshare_to_drop);

    auto t0_remaining_limbs = make_new_term_share_from(&input1.ct0);
    auto t1_remaining_limbs = make_new_term_share_from(&input1.ct1);
    auto remaining_limbs = t0_remaining_limbs.shares();
    remaining_limbs.erase(last_limbshare);
    remaining_limbs.erase(second_last_limbshare);
    t0_remaining_limbs.set_shares(remaining_limbs);
    t1_remaining_limbs.set_shares(remaining_limbs);

    auto t0_lastlimb_intt = make_treg_from(input1.ct0);
    auto t1_lastlimb_intt = make_treg_from(input1.ct1);
    t0_lastlimb_intt.set_shares(limbshare_to_drop);
    t1_lastlimb_intt.set_shares(limbshare_to_drop);

    intt_instr->add_operands(t0_lastlimb_intt, t0_lastlimb);
    intt_instr->add_operands(t1_lastlimb_intt, t1_lastlimb);

    if (current_partition_size > 1) {
      auto t0_ll_intt_dist = make_treg_from(t0_lastlimb_intt);
      auto t1_ll_intt_dist = make_treg_from(t1_lastlimb_intt);
      t0_ll_intt_dist.set_shares(limbshare_to_drop);
      t1_ll_intt_dist.set_shares(limbshare_to_drop);
      drm_instruction->add_operands(t0_ll_intt_dist, t0_lastlimb_intt);
      drm_instruction->add_operands(t1_ll_intt_dist, t1_lastlimb_intt);
      t0_lastlimb_intt = t0_ll_intt_dist;
      t1_lastlimb_intt = t1_ll_intt_dist;
    }

    auto t2 = make_treg_from(t0_lastlimb_intt);
    auto t3 = make_treg_from(t1_lastlimb_intt);
    t2.set_shares(limbshare_to_drop);
    t3.set_shares(limbshare_to_drop);
    resolve_instr->add_operands(t2, t0_lastlimb_intt);
    resolve_instr->add_operands(t3, t1_lastlimb_intt);

    auto t4 = make_double_rescale_treg_from(input1.ct0);
    auto t5 = make_double_rescale_treg_from(input1.ct1);
    mod_instr->add_operands(t4, t2);
    mod_instr->add_operands(t5, t3);

    output.ct0 = make_double_rescale_treg_from(input1.ct0);
    output.ct1 = make_double_rescale_treg_from(input1.ct1);
    sud_instruction->add_operands(output.ct0, t0_remaining_limbs, t4,
                                  t0_lastlimb_intt);
    sud_instruction->add_operands(output.ct1, t1_remaining_limbs, t5,
                                  t1_lastlimb_intt);
  };

  if (input_ptr != nullptr) {
    kg_intt = create_kernel_group_intt();
    intt_instr = std::make_shared<InttInstruction>();

    if (current_partition_size > 1) {
      kg_drm = create_kernel_group_dist();
      drm_instruction = std::make_shared<DistRecvInstruction>();
    }

    kg_resolve = create_kernel_group();
    resolve_instr = std::make_shared<ResolveInstruction>();
    kg_mod = create_kernel_group();
    auto mod_instr = std::make_shared<ModInstruction>();
    kg_sud = create_kernel_group_ntt();
    sud_instruction = std::make_shared<SudInstruction>();

    auto &output = initValue<Backend::Ciphertext>(term);
    double_rescale_ct(*input_ptr, output, mod_instr);

    kg_intt->push_instruction(intt_instr);
    if (current_partition_size > 1) {
      kg_drm->push_instruction(drm_instruction);
    }
    kg_resolve->push_instruction(resolve_instr);
    kg_resolve->set_dont_split_modular(true);
    kg_mod->push_instruction(mod_instr);
    kg_sud->push_instruction(sud_instruction);
    return;
  }

  if (input_vec_ptr != nullptr) {
    kg_intt = create_kernel_group_intt();
    intt_instr = std::make_shared<InttInstruction>();

    if (current_partition_size > 1) {
      kg_drm = create_kernel_group_dist();
      drm_instruction = std::make_shared<DistRecvInstruction>();
    }

    kg_resolve = create_kernel_group();
    resolve_instr = std::make_shared<ResolveInstruction>();
    kg_mod = create_kernel_group();
    kg_sud = create_kernel_group_ntt();
    sud_instruction = std::make_shared<SudInstruction>();

    auto &output_vec = initValue<Backend::CiphertextVector>(term);
    output_vec.vec.resize(input_vec_ptr->vec.size());
    for (size_t i = 0; i < input_vec_ptr->vec.size(); i++) {
      auto mod_instr = std::make_shared<ModInstruction>();
      double_rescale_ct(input_vec_ptr->vec.at(i), output_vec.vec.at(i),
                        mod_instr);
      kg_mod->push_instruction(mod_instr);
    }

    kg_intt->push_instruction(intt_instr);
    if (current_partition_size > 1) {
      kg_drm->push_instruction(drm_instruction);
    }
    kg_resolve->push_instruction(resolve_instr);
    kg_resolve->set_dont_split_modular(true);
    kg_sud->push_instruction(sud_instruction);
    return;
  }

  throw std::runtime_error("Unsupported operation encountered");
}

void CeriumCompiler::mod_switch(const Frontend::Term::Ptr &term,
                                      const Frontend::Term::Ptr &args1) {

  using OpCode = Backend::LimbInstruction::OpCode;

  auto mod_switch_ct = [&](Backend::Ciphertext &input1) {
    Backend::Ciphertext temp;
    temp.ct1 = make_modswitch_treg_from(input1.ct1);
    temp.ct0 = make_modswitch_treg_from(input1.ct0);
    if (input1.ct2.has_value()) {
      temp.ct2 = make_modswitch_treg_from(input1.ct2.value());
    }

    temp.level = input1.level - 1;
    return temp;
  };

  auto mod_switch_pt = [&](Backend::Plaintext &input1) {
    Backend::Plaintext temp;
    temp.pt = make_modswitch_treg_from(input1.pt);
    temp.level = input1.level - 1;
    return temp;
  };

  std::visit(Overloaded{[&](Backend::Ciphertext &input) {
                          auto &output = initValue<Backend::Ciphertext>(term);
                          output = mod_switch_ct(input);
                        },
                        [&](Backend::CiphertextVector &input) {
                          auto &output =
                              initValue<Backend::CiphertextVector>(term);
                          output.vec.resize(input.vec.size());
                          for (size_t i = 0; i < input.vec.size(); i++) {
                            output.vec[i] = mod_switch_ct(input.vec.at(i));
                          }
                        },
                        [&](Backend::Plaintext &input) {
                          auto &output = initValue<Backend::Plaintext>(term);
                          output = mod_switch_pt(input);
                        },
                        [&](Backend::PlaintextVector &input) {
                          auto &output =
                              initValue<Backend::PlaintextVector>(term);
                          output.vec.resize(input.vec.size());
                          for (size_t i = 0; i < input.vec.size(); i++) {
                            output.vec[i] = mod_switch_pt(input.vec.at(i));
                          }
                        },
                        [&](auto &arg) {
                          throw std::runtime_error(
                              "Unsupported operation encountered");
                        }},
             Objects.at(args1));
}

} // namespace Backend
} // namespace Cerium
