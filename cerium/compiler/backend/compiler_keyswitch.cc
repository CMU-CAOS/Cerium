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

namespace {

using namespace Cerium::Backend;

template <typename MakeTregFn>
std::pair<Polynomial, Polynomial>
make_parext_pair_from(MakeTregFn &&make_treg_from, Polynomial &inp,
                      const uint64_t extension_size,
                      const int32_t rescale_levels) {
  auto t0 = make_treg_from(inp);
  t0.set_limbtype(Term::LimbType::Par);
  if (rescale_levels != 0) {
    t0.set_adjustment_size(-rescale_levels);
  }
  auto t1 = make_treg_from(inp);
  t1.set_limbtype(Term::LimbType::Ext);
  t1.set_extension_size(extension_size);
  if (rescale_levels != 0) {
    t1.set_adjustment_size(rescale_levels);
  }
  return {t0, t1};
}

template <typename MakeTregFn>
std::pair<Polynomial, Polynomial> make_parext_pair_from_pair(
    MakeTregFn &&make_treg_from, std::pair<Polynomial, Polynomial> &inp,
    const uint64_t extension_size, const int32_t rescale_levels) {
  auto t0 = make_treg_from(std::get<0>(inp));
  t0.set_limbtype(Term::LimbType::Par);
  if (rescale_levels != 0) {
    t0.set_adjustment_size(-rescale_levels);
  }
  auto t1 = make_treg_from(std::get<1>(inp));
  t1.set_limbtype(Term::LimbType::Ext);
  t1.set_extension_size(extension_size);
  if (rescale_levels != 0) {
    t1.set_adjustment_size(rescale_levels);
  }
  return {t0, t1};
}

template <typename EvalKeyPair>
void set_evalkey_adjustment_sizes(EvalKeyPair &evk0, EvalKeyPair &evk1,
                                  uint32_t rescale_levels) {
  evk0.first.set_adjustment_size(-rescale_levels);
  evk0.second.set_adjustment_size(rescale_levels);
  evk1.first.set_adjustment_size(-rescale_levels);
  evk1.second.set_adjustment_size(rescale_levels);
}

std::tuple<uint64_t, KernelGroup::SplitType>
compute_keyswitch_split_by_stride(uint64_t level, uint64_t dnum_glo,
                                  uint64_t dnum_loc) {
  using SplitType = KernelGroup::SplitType;
  assert(dnum_loc > 0);
  assert(dnum_glo > 0);

  auto split_size = level / dnum_loc;
  auto extension_size = (level % dnum_loc) ? split_size + 1 : split_size;
  extension_size = (extension_size % dnum_glo) == 0
                       ? extension_size / dnum_glo
                       : (extension_size / dnum_glo) + 1;

  SplitType split;
  for (auto ii = 0; ii < dnum_loc; ++ii) {
    auto count = 0;
    std::set<uint16_t> s;
    for (auto l = ii; l < level; l += dnum_loc) {
      s.insert(l);
      ++count;
      if (count == extension_size) {
        split.push_back(s);
        s.clear();
        count = 0;
      }
    }
    if (!s.empty()) {
      split.push_back(s);
    }
  }
  return {extension_size, split};
}

} // namespace

namespace Cerium {
namespace Backend {

std::pair<CeriumCompiler::PolyPair, CeriumCompiler::PolyPair>
CeriumCompiler::get_evalkey(const Backend::Polynomial &input,
                              Backend::EvalKeyType evk_type, int32_t rot_idx,
                              KeySwitch::KeySwitchType key_switch_type,
                              uint16_t extension_size) {
  uint16_t level = input.level();

  using LimbType = Backend::Term::LimbType;
  LimbType limbtype = LimbType::Usp; // Type of the first part of the evalkey
  LimbType limbtype_ =
      LimbType::Ext; // Type of the extension part of the evalkey

  bool evk_dont_split_shares = false;

  if (evk_type == Backend::EvalKeyType::Bootstrap ||
      evk_type == Backend::EvalKeyType::Bootstrap2) {
    level = rot_idx + 1; // overloading rot_idx to get the level of the output
    extension_size = input.level(); // + 1;
    limbtype = LimbType::Par; // In a bootstrap operation, the first half of the
                              // evalkey can be split across chiplets
    evk_dont_split_shares = true;
  } else if (key_switch_type == KeySwitch::KeySwitchType::Broadcast) {
    limbtype = LimbType::Par;
    evk_dont_split_shares = true;
  } else if (key_switch_type == KeySwitch::KeySwitchType::Naive) {
    limbtype = LimbType::Par;
    limbtype_ = LimbType::Ext_Split;
    evk_dont_split_shares = true;
  }

  auto evalkey_map_idx = CeriumCompiler::EvalKeyIndexType(
      key_switch_type, evk_type, level, extension_size, current_partition_size,
      current_partition_id, rot_idx);
  auto it = evalkey_input_indices.find(evalkey_map_idx);
  uint64_t evalkey_index;

  if (it == evalkey_input_indices.end()) {
    std::stringstream s;
    s << "K:" << evk_type << ":" << level << ":" << extension_size << ":"
      << rot_idx << ":";
    std::string evalkey_name = std::move(s.str());
    auto eval_key0 = make_input(level);
    eval_key0.set_limbtype(limbtype);
    eval_key0.set_symbol(evalkey_name + "c0");
    auto eval_key1 = make_input(level);
    eval_key1.set_limbtype(limbtype);
    eval_key1.set_symbol(evalkey_name + "c1");
    auto eval_key0_ = make_input(level);
    eval_key0_.set_limbtype(limbtype_);
    eval_key0_.set_extension_size(extension_size);
    eval_key0_.set_symbol(evalkey_name + "c0");
    auto eval_key1_ = make_input(level);
    eval_key1_.set_limbtype(limbtype_);
    eval_key1_.set_extension_size(extension_size);
    eval_key1_.set_symbol(evalkey_name + "c1");

    evalkey_index = eval_key0.term_share_idx();
    evalkey_input_indices[evalkey_map_idx] = evalkey_index;
  } else {
    evalkey_index = it->second;
  }

  PolyPair evk0;
  PolyPair evk1;
  {
    Backend::Polynomial poly0;
    Backend::Polynomial poly0_;
    Backend::Polynomial poly1;
    Backend::Polynomial poly1_;

    auto &term0 = term_shares[evalkey_index];
    auto &term1 = term_shares[evalkey_index + 1];
    auto &term0_ = term_shares[evalkey_index + 2];
    auto &term1_ = term_shares[evalkey_index + 3];
    term0->term()->set_as_eval_key(evk_type, 0, evk_dont_split_shares);
    term0_->term()->set_as_eval_key(evk_type, 0, evk_dont_split_shares);
    term1->term()->set_as_eval_key(evk_type, 1, evk_dont_split_shares);
    term1_->term()->set_as_eval_key(evk_type, 1, evk_dont_split_shares);

    poly0 = Backend::Polynomial(term0, level);
    poly1 = Backend::Polynomial(term1, level);
    poly0_ = Backend::Polynomial(term0_, level);
    poly1_ = Backend::Polynomial(term1_, level);

    evk0 = PolyPair(poly0, poly0_);
    evk1 = PolyPair(poly1, poly1_);
  }
  return std::move(
      std::pair<CeriumCompiler::PolyPair, CeriumCompiler::PolyPair>(evk0,
                                                                        evk1));
}

std::tuple<uint64_t, KernelGroup::SplitType>
CeriumCompiler::compute_keyswitch_split_aggregation(
    KeySwitch::KeySwitchType key_switch_type, uint64_t level) {
  using SplitType = KernelGroup::SplitType;
  auto dnums = KeySwitch::get_keyswitch_dnum(
      key_switch_type, level, mod_partitions, current_partition_size);
  auto dnum_glo = dnums.first;
  auto dnum_loc = dnums.second;
  assert(dnum_loc > 0);
  assert(dnum_glo > 0);

  auto split_size = level / dnum_loc;
  auto extension_size = (level % dnum_loc) ? split_size + 1 : split_size;
  extension_size = (extension_size % dnum_glo) == 0
                       ? extension_size / dnum_glo
                       : (extension_size / dnum_glo) + 1;

  SplitType split;

  auto count = 0;
  for (auto ii = 0; ii < dnum_loc; ii++) {
    auto bound = ii < (level % dnum_loc) ? split_size + 1 : split_size;
    std::set<uint16_t> s;
    for (auto i = 0; i < bound; i++) {
      s.insert(s.end(), count);
      count++;
    }
    split.push_back(s);
  }
  return {extension_size, split};
}

std::tuple<uint64_t, KernelGroup::SplitType>
CeriumCompiler::compute_keyswitch_split(
    KeySwitch::KeySwitchType key_switch_type, uint64_t level) {
  if (key_switch_type == KeySwitch::KeySwitchType::Aggregation) {
    return compute_keyswitch_split_aggregation(key_switch_type, level);
  }
  auto dnums = KeySwitch::get_keyswitch_dnum(
      key_switch_type, level, mod_partitions, current_partition_size);
  auto dnum_glo = dnums.first;
  auto dnum_loc = dnums.second;
  assert(dnum_glo == 1);
  return compute_keyswitch_split_by_stride(level, dnum_glo, dnum_loc);
}

std::tuple<uint64_t, KernelGroup::SplitType>
CeriumCompiler::compute_keyswitch2_split(
    KeySwitch::KeySwitchType key_switch_type, uint64_t level) {
  if (key_switch_type == KeySwitch::KeySwitchType::Aggregation) {
    throw std::runtime_error("Keyswitch2 does not support aggregation split");
  }
  return compute_keyswitch_split_by_stride(level, 1, level);
}

Backend::Ciphertext CeriumCompiler::keyswitch_ciphertext_impl(
    Backend::Ciphertext &input, Backend::EvalKeyType evk_type, int32_t rot_idx,
    const uint32_t rescale_levels, bool use_keyswitch2_split) {
  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;

  Backend::Ciphertext output;

  auto key_switch_type = KeySwitch::KeySwitchType::Broadcast;
  Backend::Polynomial inp;
  if (evk_type == Backend::EvalKeyType::Relinearization) {
    inp = input.ct2.value();
  } else {
    inp = input.ct1;
  }

  auto [extension_size, digits] =
      use_keyswitch2_split
          ? compute_keyswitch2_split(key_switch_type, inp.level())
          : compute_keyswitch_split(key_switch_type, inp.level());
  auto evalkeys =
      get_evalkey(inp, evk_type, rot_idx, key_switch_type, extension_size);
  auto evk0 = evalkeys.first;
  auto evk1 = evalkeys.second;
  set_evalkey_adjustment_sizes(evk0, evk1, rescale_levels);

  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();

  KernelGroupPtr kg_dist;
  KernelGroupPtr kg_mov;
  auto dist_instr = std::make_shared<DistRecvInstruction>();
  auto mov_instr = std::make_shared<UnOpInstruction>(OpCode::Mov);

  if (current_partition_size > 1) {
    // Need to implement the case when rescale_levels are > 0
    // TODO: Broadcast adjusted limbs of the input
    if(rescale_levels != 0) {
      throw std::runtime_error("Rescaling not supported for keyswitch with partitioned input");
    }
    kg_dist = create_kernel_group_dist();
    kg_mov = create_kernel_group();
  }

  auto kg_bconv0 = create_kernel_group_bconv();
  auto bconv_instr0 = std::make_shared<BconvInstruction>();

  auto kg_ntt0 = create_kernel_group_ntt();
  auto ntt_inst0 = std::make_shared<NttInstruction>();

  auto kg_evkmul = create_kernel_group();
  auto kg_evkmul_ext = create_kernel_group();

  auto kg_pmul = create_kernel_group_pmul();
  auto pmul_instr = std::make_shared<PmuInstruction>();

  auto kg_intt = create_kernel_group_intt();
  auto intt_instr = std::make_shared<InttInstruction>();

  auto kg_bconv = create_kernel_group_bconv();
  auto bconv_instr = std::make_shared<BconvInstruction>();
  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();

  auto make_treg = [this](const Backend::Polynomial &value) {
    return this->make_treg_from(value);
  };

  auto t0 = inp;
  Backend::Polynomial b0_in;
  if (current_partition_size > 1) {
    auto t0_intt = make_treg_from(t0);
    intt_instr0->add_operands(t0_intt, t0);
    auto t0_dist = make_treg_from(t0_intt);
    dist_instr->add_operands(t0_dist, t0_intt);
    b0_in = make_bcor_from(t0_dist);
    mov_instr->add_operands(b0_in, t0_dist);
    kg_dist->push_instruction(dist_instr);
    kg_mov->push_instruction(mov_instr);
  } else {
    b0_in = make_bcor_from(t0);
    intt_instr0->add_operands(b0_in, t0);
  }

  auto b0 = make_bcor_from(b0_in);
  b0.set_limbtype(LimbType::PaE);
  b0.set_extension_size(extension_size);
  bconv_instr0->add_operands(b0, b0_in);

  auto t1 =
      make_parext_pair_from(make_treg, t0, extension_size, rescale_levels);
  ntt_inst0->add_operands(t1.first, t1.second, b0, inp, inp);

  auto t2 =
      make_parext_pair_from_pair(make_treg, t1, extension_size, rescale_levels);
  auto t3 =
      make_parext_pair_from_pair(make_treg, t1, extension_size, rescale_levels);
  auto mad_inst = std::make_shared<MadInstruction>();
  mad_inst->add_operands(t2.first, t1.first, evk0.first);
  mad_inst->add_operands(t3.first, t1.first, evk1.first);
  kg_evkmul->push_instruction(mad_inst);

  auto mad_inst_ext = std::make_shared<MadInstruction>();
  mad_inst_ext->add_operands(t2.second, t1.second, evk0.second);
  mad_inst_ext->add_operands(t3.second, t1.second, evk1.second);
  kg_evkmul_ext->push_instruction(mad_inst_ext);

  auto t2_temp =
      make_parext_pair_from_pair(make_treg, t2, extension_size, rescale_levels);
  auto t2_base = make_treg_from(t2.second);
  t2_base.set_limbtype(LimbType::Ext);
  t2_base.set_extension_size(extension_size);
  t2_base.set_adjustment_size(0);
  pmul_instr->add_operands(t2_temp.first, t2.first, input.ct0, t2_base);
  pmul_instr->add_operands(t2_temp.second, t2.second, input.ct0, t2_base);
  t2 = t2_temp;

  if (evk_type == Backend::EvalKeyType::Relinearization) {
    auto t3_temp = make_parext_pair_from_pair(make_treg, t3, extension_size,
                                              rescale_levels);
    auto t3_base = make_treg_from(t3.second);
    t3_base.set_limbtype(LimbType::Ext);
    t3_base.set_extension_size(extension_size);
    t3_base.set_adjustment_size(0);
    pmul_instr->add_operands(t3_temp.first, t3.first, input.ct1, t3_base);
    pmul_instr->add_operands(t3_temp.second, t3.second, input.ct1, t3_base);
    t3 = t3_temp;
  }

  auto b2_in = make_bcor_from(t2.second);
  auto b3_in = make_bcor_from(t3.second);
  b2_in.set_limbtype(LimbType::Ext);
  b2_in.set_extension_size(extension_size);
  b2_in.set_adjustment_size(rescale_levels);
  b3_in.set_limbtype(LimbType::Ext);
  b3_in.set_extension_size(extension_size);
  b3_in.set_adjustment_size(rescale_levels);
  intt_instr->add_operands(b2_in, t2.second);
  intt_instr->add_operands(b3_in, t3.second);

  auto b2 = make_bcor_from(t2.first);
  auto b3 = make_bcor_from(t3.first);
  b2.set_limbtype(LimbType::Par);
  b3.set_limbtype(LimbType::Par);
  b2.set_adjustment_size(-rescale_levels);
  b3.set_adjustment_size(-rescale_levels);
  bconv_instr->add_operands(b2, b2_in);
  bconv_instr->add_operands(b3, b3_in);

  Backend::Polynomial t4, t5;
  if (rescale_levels == 0) {
    t4 = make_treg_from(t2.first);
    t5 = make_treg_from(t3.first);
  } else {
    t4 = make_rescale_treg_from(t2.first, rescale_levels);
    t5 = make_rescale_treg_from(t3.first, rescale_levels);
  }

  sud_instr->add_operands(t4, t2.first, b2, b2_in);
  sud_instr->add_operands(t5, t3.first, b3, b3_in);

  output.level = inp.level() - rescale_levels;
  output.ct0 = t4;
  output.ct1 = t5;

  kg_intt0->push_instruction(intt_instr0);
  kg_bconv0->push_instruction(bconv_instr0);
  kg_ntt0->push_instruction(ntt_inst0);
  auto kg_in = kg_intt0;
  if (kg_mov) {
    kg_in = kg_mov;
  }
  split_digit_wise(digits, kg_in, kg_bconv0, kg_ntt0, kg_evkmul, kg_evkmul_ext);

  kg_pmul->push_instruction(pmul_instr);
  kg_intt->push_instruction(intt_instr);
  kg_bconv->push_instruction(bconv_instr);
  kg_sud->push_instruction(sud_instr);
  return output;
}

Backend::Ciphertext
CeriumCompiler::keyswitch2(Backend::Ciphertext &input,
                             Backend::EvalKeyType evk_type, int32_t rot_idx,
                             const uint32_t rescale_levels) {
  return keyswitch_ciphertext_impl(input, evk_type, rot_idx, rescale_levels,
                                   true);
}

Backend::Ciphertext CeriumCompiler::keyswitch(Backend::Ciphertext &input,
                                                Backend::EvalKeyType evk_type,
                                                int32_t rot_idx,
                                                const uint32_t rescale_levels) {
  return keyswitch_ciphertext_impl(input, evk_type, rot_idx, rescale_levels,
                                   false);
}

Backend::CiphertextVector
CeriumCompiler::keyswitch_vec(Backend::CiphertextVector &input_vec,
                                Backend::EvalKeyType evk_type, int32_t rot_idx,
                                const uint32_t rescale_levels) {
  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;
  using SplitType = KernelGroup::SplitType;

  Backend::CiphertextVector output_vec;

  assert(!input_vec.vec.empty());
  auto key_switch_type = KeySwitch::KeySwitchType::Broadcast;
  for (size_t i = 0; i < input_vec.vec.size(); i++) {
    assert(input_vec.vec.at(i).level == input_vec.vec.at(0).level);
  }
  Backend::Polynomial inp;
  if (evk_type == Backend::EvalKeyType::Relinearization) {
    inp = input_vec.vec.at(0).ct2.value();
  } else {
    inp = input_vec.vec.at(0).ct1;
  }

  auto [extension_size, digits] =
      compute_keyswitch_split(key_switch_type, inp.level());
  auto evalkeys =
      get_evalkey(inp, evk_type, rot_idx, key_switch_type, extension_size);
  auto evk0 = evalkeys.first;
  auto evk1 = evalkeys.second;
  set_evalkey_adjustment_sizes(evk0, evk1, rescale_levels);

  auto kg_intt0 = create_kernel_group_intt();
  auto intt_instr0 = std::make_shared<InttInstruction>();

  KernelGroupPtr kg_drm;
  KernelGroupPtr kg_mov;
  auto dist_instr = std::make_shared<DistRecvInstruction>();
  auto mov_instr = std::make_shared<UnOpInstruction>(OpCode::Mov);
  if (current_partition_size > 1) {
    kg_drm = create_kernel_group_dist();
    kg_mov = create_kernel_group();
  }

  auto kg_bconv0 = create_kernel_group_bconv();
  auto bconv_instr0 = std::make_shared<BconvInstruction>();

  auto kg_ntt0 = create_kernel_group_ntt();
  auto ntt_inst0 = std::make_shared<NttInstruction>();

  auto kg_evkmul = create_kernel_group();
  auto kg_evkmul_ext = create_kernel_group();

  auto kg_pmul = create_kernel_group_pmul();
  auto pmul_instr = std::make_shared<PmuInstruction>();

  auto kg_intt = create_kernel_group_intt();
  auto intt_instr = std::make_shared<InttInstruction>();

  auto kg_bconv = create_kernel_group_bconv();
  auto bconv_instr = std::make_shared<BconvInstruction>();
  auto kg_sud = create_kernel_group_ntt();
  auto sud_instr = std::make_shared<SudInstruction>();

  auto make_treg = [this](const Backend::Polynomial &value) {
    return this->make_treg_from(value);
  };

  for (size_t i = 0; i < input_vec.vec.size(); i++) {
    if (evk_type == Backend::EvalKeyType::Relinearization) {
      inp = input_vec.vec.at(i).ct2.value();
    } else {
      inp = input_vec.vec.at(i).ct1;
    }
    auto t0 = inp;

    Backend::Polynomial b0_in;

    if (current_partition_size > 1) {
      // Need to implement the case when rescale_levels are > 0
      // TODO: Broadcast adjusted limbs of the input
      assert(rescale_levels == 0);
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

    auto t1 =
        make_parext_pair_from(make_treg, t0, extension_size, rescale_levels);
    ntt_inst0->add_operands(t1.first, t1.second, b0, inp, b0_in);

    auto t2 = make_parext_pair_from_pair(make_treg, t1, extension_size,
                                         rescale_levels);
    auto t3 = make_parext_pair_from_pair(make_treg, t1, extension_size,
                                         rescale_levels);
    auto mad_inst = std::make_shared<MadInstruction>();
    mad_inst->add_operands(t2.first, t1.first, evk0.first);
    mad_inst->add_operands(t3.first, t1.first, evk1.first);
    kg_evkmul->push_instruction(mad_inst);

    auto mad_inst_ext = std::make_shared<MadInstruction>();
    mad_inst_ext->add_operands(t2.second, t1.second, evk0.second);
    mad_inst_ext->add_operands(t3.second, t1.second, evk1.second);
    kg_evkmul_ext->push_instruction(mad_inst_ext);

    auto t2_temp = make_parext_pair_from_pair(make_treg, t2, extension_size,
                                              rescale_levels);
    auto t2_base = make_treg_from(t2.second);
    t2_base.set_limbtype(LimbType::Ext);
    t2_base.set_extension_size(extension_size);
    t2_base.set_adjustment_size(0);
    pmul_instr->add_operands(t2_temp.first, t2.first, input_vec.vec.at(i).ct0,
                             t2_base);
    pmul_instr->add_operands(t2_temp.second, t2.second, input_vec.vec.at(i).ct0,
                             t2_base);
    t2 = t2_temp;

    if (evk_type == Backend::EvalKeyType::Relinearization) {
      auto t3_temp = make_parext_pair_from_pair(make_treg, t3, extension_size,
                                                rescale_levels);
      auto t3_base = make_treg_from(t3.second);
      t3_base.set_limbtype(LimbType::Ext);
      t3_base.set_extension_size(extension_size);
      t3_base.set_adjustment_size(0);
      pmul_instr->add_operands(t3_temp.first, t3.first, input_vec.vec.at(i).ct1,
                               t3_base);
      pmul_instr->add_operands(t3_temp.second, t3.second,
                               input_vec.vec.at(i).ct1, t3_base);
      t3 = t3_temp;
    }

    auto b2_in = make_bcor_from(t2.second);
    auto b3_in = make_bcor_from(t3.second);
    b2_in.set_limbtype(LimbType::Ext);
    b2_in.set_extension_size(extension_size);
    b2_in.set_adjustment_size(rescale_levels);
    b3_in.set_limbtype(LimbType::Ext);
    b3_in.set_extension_size(extension_size);
    b3_in.set_adjustment_size(rescale_levels);
    intt_instr->add_operands(b2_in, t2.second);
    intt_instr->add_operands(b3_in, t3.second);

    auto b2 = make_bcor_from(t2.first);
    auto b3 = make_bcor_from(t3.first);
    b2.set_limbtype(LimbType::Par);
    b3.set_limbtype(LimbType::Par);
    b2.set_adjustment_size(-rescale_levels);
    b3.set_adjustment_size(-rescale_levels);
    bconv_instr->add_operands(b2, b2_in);
    bconv_instr->add_operands(b3, b3_in);

    Backend::Polynomial t4, t5;
    if (rescale_levels == 0) {
      t4 = make_treg_from(t2.first);
      t5 = make_treg_from(t3.first);
    } else {
      t4 = make_rescale_treg_from(t2.first, rescale_levels);
      t5 = make_rescale_treg_from(t3.first, rescale_levels);
    }

    sud_instr->add_operands(t4, t2.first, b2, b2_in);
    sud_instr->add_operands(t5, t3.first, b3, b3_in);

    Backend::Ciphertext output;
    output.level = inp.level() - rescale_levels;
    output.ct0 = t4;
    output.ct1 = t5;

    output_vec.vec.push_back(output);
  }

  kg_intt0->push_instruction(intt_instr0);
  auto kg_in = kg_intt0;
  if (current_partition_size > 1) {
    kg_drm->push_instruction(dist_instr);
    kg_mov->push_instruction(mov_instr);
    kg_in = kg_mov;
  }
  kg_bconv0->push_instruction(bconv_instr0);
  kg_ntt0->push_instruction(ntt_inst0);
  split_digit_wise(digits, kg_in, kg_bconv0, kg_ntt0, kg_evkmul, kg_evkmul_ext);

  kg_pmul->push_instruction(pmul_instr);
  kg_intt->push_instruction(intt_instr);
  kg_bconv->push_instruction(bconv_instr);
  kg_sud->push_instruction(sud_instr);
  return output_vec;
}

void CeriumCompiler::split_digit_wise_internal(
    const KernelGroup::SplitType &split, std::shared_ptr<KernelGroup> &kg,
    DigitMap<Backend::Polynomial> &digit_map) {

  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;
  auto &instrs = kg->instructions();
  for (auto &instr : instrs) {
    auto &srcs = instr->srcs();
    auto &dests = instr->dests();
    auto srcs_copy = srcs;
    auto dests_copy = dests;
    srcs.clear();
    dests.clear();
    for (auto &digit : split) {
      for (auto &src_ : srcs_copy) {
        auto &shares = src_.shares();
        std::set<uint16_t> intersect;
        std::set_intersection(shares.begin(), shares.end(), digit.begin(),
                              digit.end(),
                              std::inserter(intersect, intersect.begin()));
        auto it = digit_map.find({src_.term_idx(), intersect});
        if (it != digit_map.end()) {
          srcs.push_back(it->second);
        } else {
          assert(src_.limbtype() == LimbType::Spl ||
                 src_.limbtype() == LimbType::SplPar);
          auto src = make_new_term_share_from(&src_);
          src.set_shares(intersect);
          srcs.push_back(src);
        }
      }
      for (auto &dst_ : dests_copy) {
        auto dst = make_copy_treg_from(dst_);
        auto &shares = dst.shares();
        std::set<uint16_t> intersect;
        std::set_intersection(shares.begin(), shares.end(), digit.begin(),
                              digit.end(),
                              std::inserter(intersect, intersect.begin()));
        dst.set_shares(intersect);
        dests.push_back(dst);
        digit_map.insert({{dst_.term_idx(), intersect}, dst});
      }
    }
  }
}

void CeriumCompiler::split_digit_wise_internal_mad(
    const KernelGroup::SplitType &split, std::shared_ptr<KernelGroup> &kg,
    DigitMap<Backend::Polynomial> &digit_map, bool dont_split_modular) {

  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;
  auto &instructions = kg->instructions();
  auto instrs_copy = instructions;
  instructions.clear();
  for (auto &instr_ : instrs_copy) {
    assert(instr_->opcode() == OpCode::Mad);
    size_t count = 0;
    std::vector<Backend::Polynomial> prev_dests;
    std::set<uint16_t> accumulated;
    for (auto &digit : split) {
      decltype(accumulated) uni;
      set_union(digit.begin(), digit.end(), accumulated.begin(),
                accumulated.end(), std::inserter(uni, uni.begin()));
      accumulated = std::move(uni);

      std::shared_ptr<KernelInstruction> instr;
      if (count == 0) {
        instr = std::make_shared<BinOpInstruction>(OpCode::Mul);
      } else {
        instr = std::make_shared<MadInstruction>(true);
      }
      auto &srcs = instr->srcs();
      auto &dests = instr->dests();
      size_t num_srcs = 0;
      for (auto &src_ : instr_->srcs()) {
        auto &shares = src_.shares();
        std::set<uint16_t> intersect;
        std::set_intersection(shares.begin(), shares.end(), digit.begin(),
                              digit.end(),
                              std::inserter(intersect, intersect.begin()));
        auto it = digit_map.find({src_.term_idx(), intersect});
        if (it != digit_map.end()) {
          srcs.push_back(it->second);
        } else if (src_.is_eval_key()) {
          auto jt = evalkey_digits_to_term.find({src_.term_idx(), intersect});
          if (jt != evalkey_digits_to_term.end()) {
            srcs.push_back(jt->second);
          } else {
            auto src = make_copy_treg_from(src_);
            src.set_shares(intersect);
            srcs.push_back(src);
            if (dont_split_modular) {
              src.term_share()->update_evalkey_term_symbol();
            }
            evalkey_digits_to_term[{src_.term_idx(), intersect}] = src;
          }
        } else {
          assert(src_.limbtype() == LimbType::Spl);
          auto src = make_new_term_share_from(&src_);
          src.set_shares(intersect);
          srcs.push_back(src);
        }

        if (count != 0 && num_srcs % 2 == 1) {
          srcs.push_back(prev_dests[num_srcs / 2]);
        }
        num_srcs++;
      }
      for (auto &dst_ : instr_->dests()) {
        if (count == split.size() - 1) {
          dests.push_back(dst_);
          continue;
        }
        auto dst = make_copy_treg_from(dst_);
        dst.set_shares(accumulated);
        dests.push_back(dst);
      }
      prev_dests = dests;
      instructions.push_back(instr);
      count++;
    }
  }
}

void CeriumCompiler::split_digit_wise(
    const KernelGroup::SplitType &split, std::shared_ptr<KernelGroup> &intt,
    std::shared_ptr<KernelGroup> &drm, std::shared_ptr<KernelGroup> &bconv,
    std::shared_ptr<KernelGroup> &ntt, std::shared_ptr<KernelGroup> &evkmul,
    std::shared_ptr<KernelGroup> &evkmul_ext) {

  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;

  DigitMap<Backend::Polynomial> digit_map;
  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;

  if (drm) {
    split_digit_wise_internal(split, drm, digit_map);
    drm->set_dont_split_modular(true);
  } else {
    split_digit_wise_internal(split, intt, digit_map);
  }
  split_digit_wise_internal(split, bconv, digit_map);
  split_digit_wise_internal(split, ntt, digit_map);
  split_digit_wise_internal_mad(split, evkmul, digit_map);
  split_digit_wise_internal_mad(split, evkmul_ext, digit_map);
  bconv->set_dont_split_modular(true);
  ntt->set_dont_split_modular(true);
  evkmul->set_dont_split_modular(true);
  evkmul_ext->set_dont_split_modular(true);
}

void CeriumCompiler::split_digit_wise(
    const KernelGroup::SplitType &split, std::shared_ptr<KernelGroup> &in,
    std::shared_ptr<KernelGroup> &bconv, std::shared_ptr<KernelGroup> &ntt,
    std::shared_ptr<KernelGroup> &evkmul,
    std::shared_ptr<KernelGroup> &evkmul_ext, bool set_dont_split_modular) {

  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;

  DigitMap<Backend::Polynomial> digit_map;
  using OpCode = Backend::LimbInstruction::OpCode;
  using LimbType = Backend::Term::LimbType;

  split_digit_wise_internal(split, in, digit_map);
  split_digit_wise_internal(split, bconv, digit_map);
  split_digit_wise_internal(split, ntt, digit_map);
  split_digit_wise_internal_mad(split, evkmul, digit_map,
                                set_dont_split_modular);
  split_digit_wise_internal_mad(split, evkmul_ext, digit_map,
                                set_dont_split_modular);
  if (set_dont_split_modular) {
    in->set_dont_split_modular(true);
    bconv->set_dont_split_modular(true);
    ntt->set_dont_split_modular(true);
    evkmul->set_dont_split_modular(true);
    evkmul_ext->set_dont_split_modular(true);
  }
}

} // namespace Backend
} // namespace Cerium
