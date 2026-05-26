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

#include "cuda_kernel.h"

#include "cerium/compiler/backend/limb.h"
#include "cerium/compiler/util/overloaded.h"
#include "logger.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace Cerium {
namespace Backend {

std::string indent(int indent_depth) {
  std::string tab = "";
  for (auto i = 0; i < indent_depth; i++) {
    tab += "\t";
  }
  return tab;
};

std::string FusedKernelLimb::limb_to_variable(const Backend::Limb *limb) {
  std::string name = "r_";
  if (!limb->term()) {
    return "0";
  }
  return name + limb->term()->name();
}

const std::set<LimbIndexType> &FusedKernelLimb::pmu_base() const {
  return pmu_base_;
}

const std::set<LimbIndexType> &FusedKernelLimb::sud_base() const {
  return sud_base_;
}

bool FusedKernelLimb::reduction_required(const Limb *limb) {
  return reduction_reqd_.find(*limb) != reduction_reqd_.end();
}

std::string FusedKernelLimb::handle_add(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  std::stringstream s;
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();

  s << "auto " << limb_to_variable(dests[0]) << " = ";

  switch (opCode) {
  case OpCode::Add:
    if (reduction_required(srcs[0]) || reduction_required(srcs[1])) {
      s << "add2(" << limb_to_variable(srcs[0]) << ", "
        << limb_to_variable(srcs[1]) << ", modulus_, modulus_square_);";
      reduction_reqd_.insert(*dests[0]);
    } else {
      s << "add(" << limb_to_variable(srcs[0]) << ", "
        << limb_to_variable(srcs[1]) << ", modulus_);";
    }
    break;
  case OpCode::Sub:
    if (reduction_required(srcs[0]) || reduction_required(srcs[1])) {
      s << "subtract2(" << limb_to_variable(srcs[0]) << ", "
        << limb_to_variable(srcs[1]) << ", modulus_, modulus_square_);";
      reduction_reqd_.insert(*dests[0]);
    } else {
      s << "subtract(" << limb_to_variable(srcs[0]) << ", "
        << limb_to_variable(srcs[1]) << ", modulus_);";
    }

    break;
  case OpCode::Neg:
    if (reduction_required(srcs[0])) {
      s << "subtract2(0" << ", " << limb_to_variable(srcs[0])
        << ", modulus_, modulus_square_);";
      reduction_reqd_.insert(*dests[0]);
    } else {
      s << "negate(" << limb_to_variable(srcs[0]) << ", modulus_);";
    }
    break;
  default:
    throw std::invalid_argument("Invalid Argument");
  }
  return s.str();
}

std::string FusedKernelLimb::handle_mad(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  std::stringstream s;
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();

  s << "auto " << limb_to_variable(dests[0]) << " = ";

  switch (opCode) {
  case OpCode::Mad:
    if (reduction_required(srcs[0]) || reduction_required(srcs[1])) {
      s << "multiply_add2(";
      if (reduction_required(srcs[0])) {
        s << "reduce2(" << limb_to_variable(srcs[0])
          << ", modulus_, barrett_ratio_,barrett_k_)";
      } else {
        s << limb_to_variable(srcs[0]);
      }
      s << ", ";
      if (reduction_required(srcs[1])) {
        s << "reduce2(" << limb_to_variable(srcs[1])
          << ", modulus_, barrett_ratio_,barrett_k_)";
      } else {
        s << limb_to_variable(srcs[1]);
      }
      s << ", " << limb_to_variable(srcs[2]) << ",modulus_,modulus_square_);";
      reduction_reqd_.insert(*dests[0]);
    } else {
      s << "multiply_add2(" << limb_to_variable(srcs[0]) << ", "
        << limb_to_variable(srcs[1]) << ", " << limb_to_variable(srcs[2])
        << ",modulus_,modulus_square_);";
      reduction_reqd_.insert(*dests[0]);
    }
    break;
  default:
    throw std::invalid_argument("Invalid Argument");
  }
  return s.str();
}

std::string FusedKernelLimb::handle_mul(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  std::stringstream s;
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();

  s << "auto " << limb_to_variable(dests[0]) << " = ";

  switch (opCode) {
  case OpCode::Mul:
  case OpCode::MuP:
    if (reduction_required(srcs[0]) || reduction_required(srcs[1])) {
      s << "multiply2(";
      if (reduction_required(srcs[0])) {
        s << "reduce2(" << limb_to_variable(srcs[0])
          << ", modulus_, barrett_ratio_,barrett_k_)";
      } else {
        s << limb_to_variable(srcs[0]);
      }
      s << ", ";
      if (reduction_required(srcs[1])) {
        s << "reduce2(" << limb_to_variable(srcs[1])
          << ", modulus_, barrett_ratio_,barrett_k_)";
      } else {
        s << limb_to_variable(srcs[1]);
      }
      s << ",modulus_,barrett_ratio_,barrett_k_);";
      reduction_reqd_.insert(*dests[0]);
    } else {
      s << "multiply2(" << limb_to_variable(srcs[0]) << ", "
        << limb_to_variable(srcs[1]) << ",modulus_,barrett_ratio_,barrett_k_);";
      reduction_reqd_.insert(*dests[0]);
    }
    break;
  default:
    throw std::invalid_argument("Invalid Argument");
  }
  return s.str();
}

std::string FusedKernelLimb::handle_pmu(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  std::stringstream s;
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();

  s << "auto " << limb_to_variable(dests[0]) << " = ";
  s << "add2(" << limb_to_variable(srcs[0]) << ",";

  switch (opCode) {
  case OpCode::Pmu:
    if (reduction_required(srcs[1])) {
      s << "multiply2(";
      s << "reduce2(" << limb_to_variable(srcs[1])
        << ", modulus_, barrett_ratio_,barrett_k_)";
      s << ", ";
      s << "pmu_,modulus_,barrett_ratio_,barrett_k_)";
      reduction_reqd_.insert(*dests[0]);
    } else {
      s << "multiply2(" << limb_to_variable(srcs[1]) << ", "
        << "pmu_,modulus_,barrett_ratio_,barrett_k_)";
      reduction_reqd_.insert(*dests[0]);
    }
    break;
  default:
    throw std::invalid_argument("Invalid Argument");
  }
  s << ", modulus_, modulus_square_);";
  return s.str();
}

std::string FusedKernelLimb::handle_rot(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {

  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  std::stringstream s;
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();
  // TODO: Optimize by moving the permutation to thread local storage

#ifndef ROT_WRITE
  s << "auto " << limb_to_variable(dests[0]) << " = ";
#endif

  switch (opCode) {
  case OpCode::Rot: {
    auto rot_instr = std::dynamic_pointer_cast<UnOpLimbInstruction>(limb_instr);
    assert(rot_instr);
#ifndef ROT_WRITE
    s << "ld_global_u32";
    s << "(&v_" << srcs[0]->term()->name();
    s << "[ld_nc_global_u32(&rot_";
    if (rot_instr->rotate_idx() >= 0) {
      s << rot_instr->rotate_idx();
    } else {
      s << "m" << -rot_instr->rotate_idx();
    }
    s << "[i])]);";
#else
    s << "";
    s << "v_" << dests[0]->term()->name();
    s << "[ld_global_u32_evict_last(&rot_";
    if (rot_instr->rotate_idx() >= 0) {
      s << rot_instr->rotate_idx();
    } else {
      s << "m" << -rot_instr->rotate_idx();
    }
    s << "[i])] = ";
    if (reduction_required(srcs[0])) {
      s << "reduce2(" << limb_to_variable(srcs[0])
        << ", modulus_, barrett_ratio_,barrett_k_)";
    } else {
      s << limb_to_variable(srcs[0]);
    }
    s << ";";
#endif
    break;
  }
  case OpCode::Con: {
#ifndef ROT_WRITE
    s << "ld_global_u32";
    s << "(&v_" << srcs[0]->term()->name();
    s << "[ld_nc_global_u32(&rot_con[i])]);";
#else
    s << "";
    s << "v_" << dests[0]->term()->name();
    s << "[ld_nc_global_u32(&rot_con";
    s << "[i])] = ";
    if (reduction_required(srcs[0])) {
      s << "reduce2(" << limb_to_variable(srcs[0])
        << ", modulus_, barrett_ratio_,barrett_k_)";
    } else {
      s << limb_to_variable(srcs[0]);
    }
    s << ";";
#endif
    break;
  }
  default:
    throw std::invalid_argument("Invalid Argument");
  }
  return s.str();
}

std::string FusedKernelLimb::handle_rsv(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  assert(opCode == OpCode::Rsv);
  auto rsv_instr =
      std::dynamic_pointer_cast<ResolveLimbInstruction>(limb_instr);

  std::stringstream s;
  const auto dest = rsv_instr->polynomial_dests()[0];
  const auto src = rsv_instr->srcs()[0];

  auto num_dest_limbs = dest->num_limbs();
  const auto &dest_limbs = dest->limbs();
  if (num_dest_limbs != 2) {
    throw std::runtime_error("Unimplemented RSV For more than 2 limbs");
  }

  s << "rsv_write_" << num_dest_limbs << "(";
  s << "dest, ";
  s << limb_to_variable(src) << "_" << src->limb_idx()
    << ", factors_, modulus, barrett_ratio, barrett_k); ";
  s << "factors_ += " << num_dest_limbs << ";";
  return s.str();
}

std::string FusedKernelLimb::handle_mod(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  assert(opCode == OpCode::Mod);
  auto mod_instr = std::dynamic_pointer_cast<ModLimbInstruction>(limb_instr);

  std::stringstream s;
  const auto src = mod_instr->polynomial_srcs()[0];
  const auto dest = mod_instr->dests()[0];

  auto num_src_limbs = src->num_limbs();
  const auto &src_limbs = src->limbs();
  if (num_src_limbs != 2) {
    throw std::runtime_error("Unimplemented MOD For more than 2 limbs");
  }

  auto poly_name = src->term()->name();

  s << "auto " << limb_to_variable(dest);
  s << " = mod_read_" << num_src_limbs << "(";
  for (auto &l : src_limbs) {
    s << "r_" << poly_name << "_" << l << ", ";
  }
  s << "modulus_, barrett_ratio_, barrett_k_); ";
  return s.str();
}

std::string FusedKernelLimb::handle_rsm(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  assert(opCode == OpCode::RsM);
  auto mod_instr =
      std::dynamic_pointer_cast<ResolveModLimbInstruction>(limb_instr);
  assert(mod_instr);

  std::stringstream s;
  const auto src = mod_instr->polynomial_srcs()[0];
  const auto dest = mod_instr->dests()[0];

  auto num_src_limbs = src->num_limbs();
  const auto &src_limbs = src->limbs();
  if (num_src_limbs != 2) {
    throw std::runtime_error("Unimplemented MOD For more than 2 limbs");
  }

  auto poly_name = src->term()->name();

  s << "auto " << limb_to_variable(dest);
  s << " = mod_read_" << num_src_limbs << "(";
  for (auto &l : src_limbs) {
    s << "r_" << poly_name << "_" << l << ", ";
  }
  s << "modulus_, barrett_ratio_, barrett_k_); ";
  return s.str();
}

std::string FusedKernelLimb::handle_mov(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  std::stringstream s;
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();
  if (opCode != OpCode::Mov) {
    throw std::invalid_argument("Must be called with mov instruction");
  }
  s << "auto " << limb_to_variable(dests[0]) << " = "
    << limb_to_variable(srcs[0]) << ";";
  return s.str();
}

std::ostream &FusedKernelLimb::handle_mov(
    std::ostream &s,
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
    uint8_t kernel_count, int indent_depth) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();
  if (opCode != OpCode::Mov) {
    throw std::invalid_argument("Must be called with mov instruction");
  }

  if (kernel_count == 1) {
    return s;
  }

  const auto limb_idx = dests[0]->limb_idx();
  std::string src_name = "";
  auto src = srcs[0];
  src_name = src->term()->name() + "_" + std::to_string(src->limb_idx());
  std::string dest_name =
      dests[0]->term()->name() + "_" + std::to_string(dests[0]->limb_idx());
  s << indent(indent_depth) << "const auto " << "v_" << src_name << " = "
    << "args_inputs[pos_" << src_name << "];\n";
  s << indent(indent_depth) << "auto " << "v_" << dest_name << " = "
    << "args_outputs[pos_" << dest_name << "];\n";
  s << indent(indent_depth) << "mov(v_" << dest_name << ", v_" << src_name
    << ");\n";

  return s;
}

std::ostream &FusedKernelLimb::handle_bco(
    std::ostream &s,
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
    int indent_depth) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  auto bco_instr =
      std::dynamic_pointer_cast<BaseConvLimbInstruction>(limb_instr);
  assert(bco_instr);
  const auto &dests = bco_instr->polynomial_dests();
  const auto &srcs = bco_instr->polynomial_srcs();

  std::string src_name = srcs[0]->term()->name();
  std::string dest_name = dests[0]->term()->name();
  return s;
}

void FusedKernelLimb::move_instructions_closer() {
  return;
  if (instructions_.size() == 1) {
    return;
  }
  std::vector<std::shared_ptr<LimbInstruction>> new_order;
  LimbMap<uint32_t> created_at;
  int count = 0;
  std::vector<std::set<uint32_t>> parents(instructions_.size());
  std::vector<std::set<uint32_t>> children(instructions_.size());
  std::vector<uint32_t> zero_indegree;
  for (auto &instr : instructions_) {
    auto dests = instr->dests();
    auto srcs = instr->srcs();
    for (auto &dest : dests) {
      created_at[*dest] = count;
    }
    for (auto &src : srcs) {
      auto it = created_at.find(*src);
      if (it == created_at.end() && !src->is_temp()) {
        continue;
      }
      parents[count].insert(created_at.at(*src));
      children[created_at.at(*src)].insert(count);
    }
    count++;
  }

  for (int i = 0; i < instructions_.size(); i++) {
    if (parents[i].empty()) {
      zero_indegree.push_back(i);
    }
  }

  while (!zero_indegree.empty()) {
    auto i = zero_indegree.back();
    zero_indegree.pop_back();
    new_order.push_back(instructions_[i]);
    for (auto &c : children[i]) {
      parents[c].erase(i);
      if (parents[c].empty()) {
        zero_indegree.push_back(c);
      }
    }
  }

  assert(new_order.size() == instructions_.size());
  instructions_ = std::move(new_order);
}

std::string FusedKernelLimb::instr_to_cuda(
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  std::stringstream s;
  switch (opCode) {
  case OpCode::Add:
  case OpCode::Sub:
  case OpCode::Neg:
    s << handle_add(limb_instr);
    break;
  case OpCode::Mul:
  case OpCode::MuP:
    s << handle_mul(limb_instr);
    break;
  case OpCode::Pmu:
    s << handle_pmu(limb_instr);
    break;
  case OpCode::Mad:
    s << handle_mad(limb_instr);
    break;
  case OpCode::Rot:
  case OpCode::Con:
    s << handle_rot(limb_instr);
    break;
  case OpCode::Rsv:
    s << handle_rsv(limb_instr);
    break;
  case OpCode::Mod:
    s << handle_mod(limb_instr);
    break;
  case OpCode::RsM:
    s << handle_rsm(limb_instr);
    break;
  case OpCode::Bco:
    handle_bco(s, limb_instr, 0);
    break;
  case OpCode::Mov:
    s << handle_mov(limb_instr);
    break;
  }
  return s.str();
}

std::ostream &FusedKernelLimb::ntt_instr_to_cuda(
    std::ostream &s,
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
    uint8_t kernel_count, int indent_depth) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  switch (opCode) {
  case OpCode::Ntt:
  case OpCode::SuD:
  case OpCode::SuD2:
  case OpCode::Msd:
    handle_ntt(s, limb_instr, kernel_count, indent_depth);
    break;
  case OpCode::Int:
    handle_intt(s, limb_instr, kernel_count, indent_depth);
    break;
  case OpCode::Mov: {
    handle_mov(s, limb_instr, kernel_count, indent_depth);
    break;
  }
  default:
    throw std::invalid_argument("Invalid Argument");
  }
  return s;
}

std::ostream &FusedKernelLimb::bco_instr_to_cuda(
    std::ostream &s,
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
    int indent_depth) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  switch (opCode) {
  case OpCode::Bco:
    handle_bco(s, limb_instr, indent_depth);
    break;
  default:
    throw std::invalid_argument("Invalid Argument");
  }
  return s;
}

void FusedKernelLimb::push_limb_instruction(
    std::shared_ptr<LimbInstruction> &&limb_instruction) {
  using OpCode = Backend::LimbInstruction::OpCode;
  if (!limb_instruction) {
    return;
  }
  const auto &srcs = limb_instruction->srcs();
  const auto &dests = limb_instruction->dests();

  for (const auto &dst : dests) {
    for (const auto &s : srcs) {
      limb_dependencies_map_[*dst].insert(*s);
      if (!s->is_temp()) {
        continue;
      }
      limb_dependencies_map_[*dst].insert(limb_dependencies_map_[*s].begin(),
                                          limb_dependencies_map_[*s].end());
    }
  }

  for (const auto &src : srcs) {
    if (src->is_temp()) {
      continue;
    }
    if (src->is_scalar()) {
      kernel_inputs_scalar_.insert(*src);
    } else if (src->is_plaintext_remapable()) {
      kernel_inputs_remapable_.insert(*src);
    } else {
      if (kernel_outputs_.find(*src) == kernel_outputs_.end()) {
        kernel_inputs_.insert(*src);
      }
    }
  }
  for (const auto &dst : dests) {
    if (dst->is_temp()) {
      continue;
    }
    kernel_outputs_.insert(*dst);
  }
  if (limb_instruction->opcode() == OpCode::SuD) {
    auto sud_instr =
        std::dynamic_pointer_cast<SuDLimbInstruction>(limb_instruction);
    if (sud_instr) {
      sud_base_ = sud_instr->divide_bases();
    } else {
      sud_base_.insert(limb_instruction->srcs()[1]->limb_idx());
    }
    // HACK: Add this to rot_inputs to prevent overwriting this register
    rot_inputs_.insert(*limb_instruction->srcs()[1]);
  } else if (limb_instruction->opcode() == OpCode::SuD2) {
    auto sud_instr =
        std::dynamic_pointer_cast<SuDLimbInstruction2>(limb_instruction);
    if (sud_instr) {
      sud_base_ = sud_instr->divide_bases();
    }
    // HACK: Add this to rot_inputs to prevent overwriting this register
    rot_inputs_.insert(*limb_instruction->srcs()[1]);
    rot_inputs_.insert(*limb_instruction->srcs()[2]);
    rot_inputs_.insert(*limb_instruction->srcs()[3]);
  } else if (limb_instruction->opcode() == OpCode::Rot) {
    auto rot_instr =
        std::dynamic_pointer_cast<UnOpLimbInstruction>(limb_instruction);
    assert(rot_instr);
    rot_indices_.insert(rot_instr->rotate_idx());
    auto src = limb_instruction->srcs()[0];
    auto dest = limb_instruction->dests()[0];
    assert(!src->is_temp() || !dest->is_temp());
    rot_inputs_.insert(*src);
    rot_outputs_.insert(*dest);
  } else if (limb_instruction->opcode() == OpCode::Con) {
    rot_indices_.insert(0);
    auto src = limb_instruction->srcs()[0];
    auto dest = limb_instruction->dests()[0];
    assert(!src->is_temp() || !dest->is_temp());
    rot_inputs_.insert(*src);
    rot_outputs_.insert(*dest);
  } else if (limb_instruction->opcode() == OpCode::Rsv) {
    auto rsv_instr =
        std::dynamic_pointer_cast<ResolveLimbInstruction>(limb_instruction);
    assert(rsv_instr);
    auto dst = rsv_instr->polynomial_dests()[0];
    auto &limbs = dst->limbs();
    for (auto &l : limbs) {
      kernel_outputs_.insert(Limb(dst, l));
    }
    rsv_base_ = dst->limbs();
  } else if (limb_instruction->opcode() == OpCode::Mod) {
    auto mod_instr =
        std::dynamic_pointer_cast<ModLimbInstruction>(limb_instruction);
    assert(mod_instr);
    auto src = mod_instr->polynomial_srcs()[0];
    auto &limbs = src->limbs();
    for (auto &l : limbs) {
      kernel_inputs_.insert(Limb(src, l));
      rot_inputs_.insert(Limb(src, l));
    }
  } else if (limb_instruction->opcode() == OpCode::Msd) {
    auto msd_instr =
        std::dynamic_pointer_cast<MsdLimbInstruction>(limb_instruction);
    assert(msd_instr);
    auto src = msd_instr->polynomial_srcs()[0];
    auto &limbs = src->limbs();
    for (auto &l : limbs) {
      kernel_inputs_.insert(Limb(src, l));
      rot_inputs_.insert(Limb(src, l));
    }
    sud_base_ = msd_instr->divide_bases();
  } else if (limb_instruction->opcode() == OpCode::RsM) {
    auto mod_instr =
        std::dynamic_pointer_cast<ResolveModLimbInstruction>(limb_instruction);
    assert(mod_instr);
    auto src = mod_instr->polynomial_srcs()[0];
    auto &limbs = src->limbs();
    for (auto &l : limbs) {
      kernel_inputs_.insert(Limb(src, l));
      rot_inputs_.insert(Limb(src, l));
    }
  } else if (limb_instruction->opcode() == OpCode::Pmu) {
    auto pmu_instr =
        std::dynamic_pointer_cast<PmuLimbInstruction>(limb_instruction);
    assert(pmu_instr);
    pmu_base_ = pmu_instr->divide_bases();
  } else if (limb_instruction->opcode() == OpCode::Drm) {
    auto drm_instr =
        std::dynamic_pointer_cast<DistRecvLimbInstruction>(limb_instruction);
    assert(drm_instr);
    auto src = drm_instr->polynomial_srcs()[0];
    std::unordered_map<LimbIndexType, std::tuple<uint32_t, Limb>> dest_map;
    std::unordered_map<LimbIndexType, std::tuple<uint32_t, Limb>> src_map;
    uint32_t src_index = 0;
    uint32_t max_num_limbs = 0;
    auto dest = drm_instr->polynomial_dests()[0];
    auto all_srcs = drm_instr->all_srcs();
    for (auto &src : all_srcs) {
      max_num_limbs = std::max(max_num_limbs, src->num_limbs());
    }
    for (auto &l : src->limbs()) {
      kernel_inputs_.insert(Limb(src, l));
      src_map[l] = {src_index++, Limb(src, l)};
    }
    for (size_t i = 0; i < all_srcs.size(); i++) {
      size_t dest_index = 0;
      for (auto &l : all_srcs[i]->limbs()) {
        dest_map[l] = {i * max_num_limbs + dest_index++, Limb()};
      }
    }
    for (auto &l : dest->limbs()) {
      kernel_outputs_.insert(Limb(dest, l));
      auto &[idx, limb] = dest_map.at(l);
      dest_map[l] = {idx, Limb(dest, l)};
    }
    CL_LOG3("Max Num Limbs: {}", max_num_limbs);
    comm_srcs.resize(max_num_limbs);
    comm_dests.resize(all_srcs.size(), std::vector<Limb>(max_num_limbs));
    for (auto &[l, src_l] : src_map) {
      auto [index, src] = src_l;
      CL_LOG3("Src Map {} {} {}", l, index, src);
      comm_srcs[index] = src;
    }
    for (auto &[l, dest_l] : dest_map) {
      auto [index, dest] = dest_l;
      CL_LOG3("Dest Map {} {} {}", l, index, dest);
      comm_dests[index / max_num_limbs][index % max_num_limbs] = dest;
    }
    comm_num_limbs_ = max_num_limbs;
    comm_size_ = drm_instr->sync_size();
  } else if (limb_instruction->opcode() == OpCode::Ags) {
    auto ags_instr = std::dynamic_pointer_cast<AggregateScatterLimbInstruction>(
        limb_instruction);
    assert(ags_instr);
    auto src = ags_instr->polynomial_srcs()[0];
    auto dest = ags_instr->polynomial_dests()[0];
    std::map<LimbIndexType, std::tuple<uint32_t, Limb>> dest_map;
    std::map<LimbIndexType, std::tuple<uint32_t, Limb>> src_map;
    uint32_t dest_index = 0;
    uint32_t max_num_limbs = 0;
    for (auto &l : src->limbs()) {
      kernel_inputs_.insert(Limb(src, l));
    }
    for (auto &l : dest->limbs()) {
      kernel_outputs_.insert(Limb(dest, l));
      dest_map[l] = {dest_index++, Limb(dest, l)};
    }
    auto all_dests = ags_instr->all_dests();
    for (auto &dst : all_dests) {
      max_num_limbs = std::max(max_num_limbs, dst->num_limbs());
    }
    for (size_t i = 0; i < all_dests.size(); i++) {
      size_t src_index = 0;
      for (auto &l : all_dests[i]->limbs()) {
        if (src->limbs().find(l) != src->limbs().end()) {
          src_map[l] = {i * max_num_limbs + src_index++, Limb(src, l)};
        } else {
          src_map[l] = {i * max_num_limbs + src_index++, Limb()};
        }
      }
    }
    for (auto &l : dest->limbs()) {
      assert(src_map.find(l) != src_map.end());
    }
    CL_LOG3("Max Num Limbs: {}", max_num_limbs);
    comm_srcs.resize(max_num_limbs);
    comm_dests.resize(all_dests.size(), std::vector<Limb>(max_num_limbs));
    for (auto &[l, dst_l] : dest_map) {
      auto [index, dst] = dst_l;
      CL_LOG3("Scatter Map {} {}", index, dst);
      comm_srcs[index] = dst;
    }
    for (auto &[l, src_l] : src_map) {
      auto [index, src] = src_l;
      CL_LOG3("Gather Map {} {}", index, src);
      comm_dests[index / max_num_limbs][index % max_num_limbs] = src;
    }
    comm_num_limbs_ = max_num_limbs;
    comm_size_ = ags_instr->sync_size();
  } else if (limb_instruction->opcode() == OpCode::Ard) {
    auto ard_instr =
        std::dynamic_pointer_cast<AllReduceLimbInstruction>(limb_instruction);
    assert(ard_instr);
    auto src = ard_instr->polynomial_srcs()[0];
    auto dest = ard_instr->polynomial_dests()[0];
    std::map<LimbIndexType, std::tuple<uint32_t, Limb>> dest_map;
    std::map<LimbIndexType, std::tuple<uint32_t, Limb>> src_map;
    uint32_t dest_index = 0;
    uint32_t max_num_limbs = 0;
    for (auto &l : src->limbs()) {
      kernel_inputs_.insert(Limb(src, l));
    }
    for (auto &l : dest->limbs()) {
      kernel_outputs_.insert(Limb(dest, l));
      dest_map[l] = {dest_index++, Limb(dest, l)};
    }
    max_num_limbs = dest->limbs().size();
    size_t src_index = 0;
    for (auto &l : dest->limbs()) {
      if(src->limbs().find(l) != src->limbs().end()) {
        src_map[l] = {src_index++, Limb(src, l)};
      } else {
        src_map[l] = {src_index++, Limb()};
      }
    }
    CL_LOG3("Max Num Limbs: {}", max_num_limbs);
    comm_srcs.resize(max_num_limbs);
    comm_dests.resize(1, std::vector<Limb>(max_num_limbs));
    for (auto &[l, dst_l] : dest_map) {
      auto [index, dst] = dst_l;
      CL_LOG3("Scatter Map {} {}", index, dst);
      comm_srcs[index] = dst;
    }
    for (auto &[l, src_l] : src_map) {
      auto [index, src] = src_l;
      CL_LOG3("Gather Map {} {}", index, src);
      comm_dests[index / max_num_limbs][index % max_num_limbs] = src;
    }
    comm_num_limbs_ = max_num_limbs;
    comm_size_ = ard_instr->sync_size();
  } else if (limb_instruction->opcode() == OpCode::Call) {
    auto call_instr =
        std::dynamic_pointer_cast<CallLimbInstruction>(limb_instruction);
    assert(call_instr);
    auto srcs = call_instr->srcs_map();
    auto dests = call_instr->dests_map();
    for (auto &[k, src] : srcs) {
      for (auto &l : src.limbs()) {
        call_function_args[k + "(" + std::to_string(l) + ")"] = Limb(src, l);
        call_function_args_map[Limb(src, l)].push_back(k + "(" +
                                                       std::to_string(l) + ")");
        kernel_inputs_.insert(Limb(src, l));
      }
    }
    for (auto &[k, dest] : dests) {
      for (auto &l : dest.limbs()) {
        call_function_args[k + "(" + std::to_string(l) + ")"] = Limb(dest, l);
        call_function_args_map[Limb(dest, l)].push_back(
            k + "(" + std::to_string(l) + ")");
        kernel_outputs_.insert(Limb(dest, l));
      }
    }
    call_function_name = call_instr->function_name();
  }

  instructions_.push_back(limb_instruction);
}

std::ostream &FusedKernelLimb::write_ntt_kernel(std::ostream &s,
                                                uint ntt_kernel_count,
                                                KernelType2 &kernel_type,
                                                int indent_depth) {

  using OpCode = Backend::LimbInstruction::OpCode;
  s << indent(indent_depth) << "const auto modulus_ = modulii[" << limb_idx_
    << "];\n";
  if (kernel_type == KernelType2::Ntt || kernel_type == KernelType2::SuD) {
    s << indent(indent_depth)
      << "const auto power_of_roots_ = power_of_roots + (" +
             std::to_string(limb_idx_) + "*NTT_TABLE_SIZE);\n";
  } else if (kernel_type == KernelType2::Int) {
    s << indent(indent_depth)
      << "const auto inverse_power_of_roots_div_two_ = "
         "inverse_power_of_roots_div_two + (" +
             std::to_string(limb_idx_) + "*NTT_TABLE_SIZE);\n";
  }

  if (ntt_kernel_count == 1 && kernel_type == KernelType2::SuD) {
    s << indent(indent_depth) << "const auto sud_factor_ = sud_factors["
      << limb_idx_ << "];\n";
  }
  for (auto &limb_instr : instructions_) {
    s << indent(indent_depth) << "//\t" << limb_instr->ppOp() << "\n";
    ntt_instr_to_cuda(s, limb_instr, ntt_kernel_count, indent_depth);
  }

  return s;
}

std::ostream &FusedKernelLimb::write(std::ostream &s, int indent_depth) {
  return s;
}

std::ostream &operator<<(std::ostream &s, const FusedKernelLimb &fkl) {
  for (const auto &instruction : fkl.instructions_) {
    s << instruction->ppOp() << "\n";
  }
  return s;
}

std::string FusedKernelLimb::prettyPrint() {
  std::stringstream ss;
  ss << *this << "\n";
  return ss.str();
}

void FusedKernel::write_load_register(std::ostream &s, const Limb &v,
                                      int indent_depth) {
  s << indent(indent_depth) << "auto r_" << v.term()->name() << " = ";
  if (v.is_input() && !v.is_function_arg()) {
    s << "ld_nc_global_u32";
  } else {
    s << "ld_global_u32";
  }
  s << "(&v_" << v.term()->name() << "[";
  if (v.is_plaintext() && v.plaintext_repeat_size() != 0 &&
      v.plaintext_repeat_size() != 32768) {
    s << "i >> " << 15 - std::log2(v.plaintext_repeat_size());
  } else {
    s << "i";
  }
  s << "]);\n";
}

void FusedKernel::foreach_map_limb_set_valuetype(
    std::vector<std::vector<ValueType>> &map_limb_set,
    std::function<void(size_t pos, ValueType &)> func) {
  for (auto i = 0; i < map_limb_set.size(); i++) {
    for (auto &l : map_limb_set[i]) {
      func(i, l);
    }
  }
}

void FusedKernel::foreach_map_limb_set(
    std::vector<std::vector<ValueType>> &map_limb_set,
    std::function<void(size_t pos, Limb &)> func) {
  for (auto i = 0; i < map_limb_set.size(); i++) {
    for (auto &l : map_limb_set[i]) {
      func(i, std::get<0>(l));
    }
  }
}

void FusedKernel::push_limb_kernel(std::shared_ptr<FusedKernelLimb> &&fkl) {
  if (fkl->num_instructions() == 0) {
    return;
  }
  limb_blocks_.push_back(fkl);
  fkl->set_ntt_kernel(is_ntt_kernel_);

  rot_indices_.insert(fkl->rot_indices_.begin(), fkl->rot_indices_.end());
  rot_inputs_.insert(fkl->rot_inputs_.begin(), fkl->rot_inputs_.end());
  rot_outputs_.insert(fkl->rot_outputs_.begin(), fkl->rot_outputs_.end());
}

std::ostream &FusedKernel::write_kernel_int(std::ostream &s,
                                            int ntt_kernel_count,
                                            int indent_depth) {

  auto indent = [&]() {
    std::string tab = "";
    for (auto i = 0; i < indent_depth; i++) {
      tab += "\t";
    }
    return tab;
  };

  std::string kernel_header =
      kernel_signature + " _kernel_int_" + std::to_string(kernel_idx_) + "_" +
      std::to_string(ntt_kernel_count) +
      "(LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const "
      "uint32_t * bases, const LimbDataType * modulii,  const LimbDataType2 * "
      "inverse_power_of_roots_div_two ) {";
  constexpr std::string_view kernel_end = "}";

  constexpr std::string_view if_header_0 = "if(j == ";
  constexpr std::string_view if_header_1 = ") {";
  constexpr std::string_view if_end = "}";

  s << kernel_header << "\n";
  indent_depth++;
  std::size_t count = 0;
  if (!limb_blocks_.empty()) {
    s << indent() << "__shared__ LimbDataType shmem[intShMemSize"
      << ntt_kernel_count << "];\n";
    s << indent() << "auto j = blockIdx.y;\n";
    s << indent() << "auto limb_idx = ldu_global_u32(&bases[j]);\n";
    s << indent() << "auto modulus_ = ldu_global_u32(&modulii[limb_idx]);\n";
    s << indent()
      << "const auto inverse_power_of_roots_div_two_ = "
         "inverse_power_of_roots_div_two + (limb_idx"
         "*NTT_TABLE_SIZE);\n";

    auto &l = limb_blocks_[0];
    s << indent()
      << "constexpr size_t num_outputs = " << l->kernel_outputs_.size()
      << ";\n";
    s << indent()
      << "constexpr size_t num_inputs = " << l->kernel_inputs_.size() << ";\n";

    size_t count = 0;
    for (auto &inp : l->kernel_inputs_) {
      s << indent() << "auto v_" << inp.term()->name() << " = "
        << "ldu_global_u32_ptr(&args_inputs[num_inputs*j + " << count
        << "]);\n";
      count++;
    }
    count = 0;
    for (auto &out : l->kernel_outputs_) {
      s << indent() << "auto v_" << out.term()->name() << " = "
        << "ldu_global_u32_ptr(&args_outputs[num_outputs*j + " << count
        << "]);\n";
      count++;
    }

    for (auto &limb_instr : l->instructions_) {
      s << indent() << "//\t" << limb_instr->ppOp() << "\n";
      l->ntt_instr_to_cuda(s, limb_instr, ntt_kernel_count, indent_depth);
    }
  }

  s << "\n";
  indent_depth--;
  s << indent() << kernel_end << "\n";
  return s;
}

std::ostream &FusedKernel::write_kernel_ntt(std::ostream &s,
                                            int ntt_kernel_count,
                                            int indent_depth) {

  std::string kernel_header =
      kernel_signature + "_kernel_ntt_" + std::to_string(kernel_idx_) + "_" +
      std::to_string(ntt_kernel_count) +
      "(LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const "
      "uint32_t * bases, const LimbDataType * modulii, const LimbDataType2 * "
      "power_of_roots ";
  constexpr std::string_view kernel_end = "}";
  if (kernel_type_ == KernelType::SuD && ntt_kernel_count == 1) {
    kernel_header += ", const LimbDataType * sud_factors, const LimbDataType * "
                     "barrett_ratio, const uint32_t * barrett_k";
  } else if (kernel_type_ == KernelType::Msd && ntt_kernel_count == 0) {
    kernel_header +=
        ", const LimbDataType * barrett_ratio, const uint32_t * barrett_k";
  } else if (kernel_type_ == KernelType::Msd && ntt_kernel_count == 1) {
    kernel_header += ", const LimbDataType * sud_factors, const LimbDataType * "
                     "barrett_ratio, const uint32_t * barrett_k";
  }
  kernel_header += ") {";

  s << kernel_header << "\n";
  indent_depth++;
  if (!limb_blocks_.empty()) {
    s << indent(indent_depth) << "__shared__ LimbDataType shmem[nttShMemSize"
      << ntt_kernel_count << "];\n";
    s << indent(indent_depth) << "auto j = blockIdx.y;\n";
    s << indent(indent_depth)
      << "const auto limb_idx = ldu_global_u32(&bases[j]);\n";
    s << indent(indent_depth)
      << "const auto modulus_ = ldu_global_u32(&modulii[limb_idx]);\n";
    s << indent(indent_depth)
      << "const auto power_of_roots_ = "
         "power_of_roots + (limb_idx"
         "*NTT_TABLE_SIZE);\n";

    auto &l = limb_blocks_[0];
    s << indent(indent_depth)
      << "constexpr size_t num_outputs = " << l->kernel_outputs_.size()
      << ";\n";
    s << indent(indent_depth)
      << "constexpr size_t num_inputs = " << l->kernel_inputs_.size() << ";\n";

    size_t count = 0;
    for (auto &inp : l->kernel_inputs_) {
      s << indent(indent_depth) << "auto v_" << inp.term()->name();
      if (kernel_type_ == KernelType::Msd) {
        s << "_" << inp.limb_idx();
      }
      s << " = " << "ldu_global_u32_ptr(&args_inputs[num_inputs*j + " << count
        << "]);\n";
      count++;
    }
    count = 0;
    for (auto &out : l->kernel_outputs_) {
      s << indent(indent_depth) << "auto v_" << out.term()->name() << " = "
        << "ldu_global_u32_ptr(&args_outputs[num_outputs*j + " << count
        << "]);\n";
      count++;
    }

    for (auto &limb_instr : l->instructions_) {
      s << indent(indent_depth) << "//\t" << limb_instr->ppOp() << "\n";
      l->ntt_instr_to_cuda(s, limb_instr, ntt_kernel_count, indent_depth);
    }
  }

  s << "\n";
  indent_depth--;
  s << indent(indent_depth) << kernel_end << "\n";
  return s;
}

std::ostream &FusedKernel::write_kernel_empty(std::ostream &s,
                                              int indent_depth) {

  auto indent = [&]() {
    std::string tab = "";
    for (auto i = 0; i < indent_depth; i++) {
      tab += "\t";
    }
    return tab;
  };

  std::string kernel_header =
      kernel_signature + "_kernel_" + std::to_string(kernel_idx_) +
      "(LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const "
      "LimbDataType * args_scalars, const uint32_t * bases, const LimbDataType "
      "* modulii, const LimbDataType * barrett_ratios, const uint32_t * "
      "barrett_k, const uint32_t ** rotate_map) {";
  constexpr std::string_view kernel_end = "}";

  s << kernel_header << "\n";
  indent_depth++;

  s << indent() << kernel_end << "\n";
  return s;
}

std::ostream &FusedKernel::write_kernel_generic(std::ostream &s,
                                                int indent_depth) {

  auto indent = [&]() {
    std::string tab = "";
    for (auto i = 0; i < indent_depth; i++) {
      tab += "\t";
    }
    return tab;
  };

  std::string kernel_header =
      kernel_signature + "__launch_bounds__(BLOCK_DIM_X) " + "_kernel_" +
      std::to_string(kernel_idx_) +
      "(LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const "
      "LimbDataType * args_scalars, const LimbDataType ** remapables_base, "
      "const size_t * args_remapables, const uint32_t * bases, const "
      "LimbDataType * modulii, const LimbDataType * barrett_ratios, const "
      "uint32_t * barrett_k, const uint32_t ** rotate_map) {";
  constexpr std::string_view kernel_end = "}";

  s << kernel_header << "\n";
  indent_depth++;

  std::size_t count = 0;
  s << indent() << "//Rots\n";
  count = 0;
  for (auto &r : rot_indices_) {
    s << indent() << "auto rot_";
    if (r < 0) {
      s << "m" << -r;
    } else if (r > 0) {
      s << r;
    } else {
      s << "con";
    }
    s << " = ldu_global_u32_ptr(&rotate_map[" << count << "]);\n";
    count++;
  }

  s << indent() << "//-------------------\n";

  s << indent() << "int j = blockIdx.y;\n";
  if (!limb_blocks_.empty()) {

    s << indent() << "auto limb_idx = ldu_global_u32(&bases[j]);\n";
    s << indent() << "auto modulus_ = ldu_global_u32(&modulii[limb_idx]);\n";
    s << indent()
      << "auto barrett_ratio_ = ldu_global_u32(&barrett_ratios[limb_idx]);\n";
    s << indent()
      << "auto barrett_k_ = ldu_global_u32(&barrett_k[limb_idx]);\n";
    s << indent()
      << "auto modulus_square_ = multiply_64(modulus_ , modulus_);\n";

    auto &l = limb_blocks_[0];
    s << indent()
      << "constexpr size_t num_outputs = " << l->kernel_outputs_.size()
      << ";\n";
    s << indent()
      << "constexpr size_t num_inputs = " << l->kernel_inputs_.size() << ";\n";
    s << indent()
      << "constexpr size_t num_scalars = " << l->kernel_inputs_scalar_.size()
      << ";\n";
    s << indent() << "constexpr size_t num_remapables = "
      << l->kernel_inputs_remapable_.size() << ";\n";

    size_t count = 0;
    for (auto &inp : l->kernel_inputs_) {
      s << indent() << "auto v_" << inp.term()->name() << " = "
        << "ldu_global_u32_ptr(&args_inputs[num_inputs*j + " << count
        << "]);\n";
      count++;
    }
    if (!l->kernel_inputs_remapable_.empty()) {
      s << indent() << "auto remapables_base_ = *remapables_base;\n";
    }
    count = 0;
    for (auto &inp : l->kernel_inputs_remapable_) {
      s << indent() << "auto v_" << inp.term()->name() << " = "
        << "remapables_base_ + args_remapables[num_remapables*j + " << count
        << "];\n";
      count++;
    }
    count = 0;
    for (auto &inp_sca : l->kernel_inputs_scalar_) {
      s << indent() << "auto r_" << inp_sca.term()->name() << " = "
        << "ldu_global_u32(&args_scalars[num_scalars*j + " << count << "]);\n";
      count++;
    }
    count = 0;
    for (auto &out : l->kernel_outputs_) {
      s << indent() << "auto v_" << out.term()->name() << " = "
        << "ldu_global_u32_ptr(&args_outputs[num_outputs*j + " << count
        << "]);\n";
      count++;
    }

    constexpr std::string_view loop_header =
        "{\n\tint i = threadIdx.x + blockDim.x * blockIdx.x;";
    constexpr std::string_view loop_end = "}";

    s << indent() << loop_header << "\n";
    indent_depth++;
    for (const auto &v : l->kernel_inputs_) {
      write_load_register(s, v, indent_depth);
    }
    for (const auto &v : l->kernel_inputs_remapable_) {
      write_load_register(s, v, indent_depth);
    }

    for (auto &limb_instr : l->instructions_) {
      s << indent() << "//\t" << limb_instr->ppOp() << "\n";
      s << indent() << l->instr_to_cuda(limb_instr) << "\n";
    }

    for (const auto &v : l->kernel_outputs_) {
      if (l->reduction_required(&v)) {
        s << indent() << "st_na_global_u32(&v_" << v.term()->name() << "[i], "
          << "reduce2(r_" << v.term()->name()
          << ",modulus_,barrett_ratio_,barrett_k_));\n";

#ifndef ROT_WRITE
      } else {
#else
      } else if (l->rot_outputs_.find(v) == l->rot_outputs_.end()) {
#endif
        s << indent() << "st_na_global_u32(&v_" << v.term()->name() << "[i], "
          << "r_" << v.term()->name() << ");\n";
      }
    }

    indent_depth--;
    s << indent() << loop_end << "\n";
  }

  indent_depth--;
  s << indent() << kernel_end << "\n";
  return s;
}

std::ostream &FusedKernel::write_kernel(std::ostream &s, int indent_depth) {
  if (limb_blocks_.empty()) {
    return s;
  }

  switch (kernel_type_) {
  case KernelType::Rsv:
    return write_kernel_rsv(s, indent_depth);
  case KernelType::Mod:
    return write_kernel_mod(s, indent_depth);
  case KernelType::Pmu:
    return write_kernel_pmu(s, indent_depth);
  case KernelType::Int:
  case KernelType::Ntt:
  case KernelType::SuD:
  case KernelType::SuD2:
  case KernelType::Msd:
  case KernelType::Bco:
  case KernelType::Drm:
  case KernelType::Ags:
  case KernelType::Ard:
  case KernelType::Call:
    return s;
  default:
    return write_kernel_generic(s, indent_depth);
  }
}

std::ostream &FusedKernel::write_function(std::ostream &s, int indent_depth) {

  if (limb_blocks_.empty()) {
    return s;
  }
  if (kernel_type_ == KernelType::Int || kernel_type_ == KernelType::Ntt ||
      kernel_type_ == KernelType::SuD || kernel_type_ == KernelType::SuD2) {
    return s;
  }
  if (kernel_type_ == KernelType::Bco) {
    return s;
  }
  if (kernel_type_ == KernelType::Ard || kernel_type_ == KernelType::Drm ||
      kernel_type_ == KernelType::Ags) {
    return s;
  }

  if (kernel_type_ == KernelType::Msd) {
    throw std::runtime_error(
        "Msd kernel type is not supported in write_function");
  } else if (kernel_type_ == KernelType::Rsv) {
    function_definition_ =
        "__host__ void _function_" + std::to_string(kernel_idx_) +
        "_(cudaStream_t & stream, LimbDataType ** args_outputs, const "
        "LimbDataType ** args_inputs, const LimbDataType * factors, const "
        "LimbDataType * modulus, const LimbDataType * barrett_ratio, const "
        "uint32_t barrett_k)";
    constexpr std::string_view function_end = "}";

    s << indent(indent_depth) << function_definition_ << "{ \n";
    indent_depth++;
    s << indent(indent_depth) << "dim3 gridSize(GRID_DIM_X,"
      << limb_blocks_.size() << ",1);\n";
    s << indent(indent_depth) << "dim3 blockSize(BLOCK_DIM_X,1,1);\n";
    s << indent(indent_depth) << "_kernel_rsv_" << kernel_idx_
      << "<<<gridSize,blockSize,0,stream>>>(args_outputs, args_inputs, "
         "factors, modulus, barrett_ratio, barrett_k);\n";
    indent_depth--;
    s << indent(indent_depth) << function_end << "\n";
  } else if (kernel_type_ == KernelType::Call) {
    // Nothing to Do
    function_definition_ =
        "__host__ void _function_" + std::to_string(kernel_idx_) +
        "_(cudaStream_t & stream, LimbDataType ** args_outputs, const size_t * "
        "num_output_bases, const LimbDataType ** args_inputs, const size_t * "
        "num_input_bases, const LimbDataType ** base_conversion_factors, const "
        "LimbDataType ** output_modulii, const LimbDataType ** barrett_ratios, "
        "const uint32_t ** barrett_k, const size_t shared_mem_size)";
    constexpr std::string_view function_end = "}";
    s << indent(indent_depth) << function_definition_ << "{ \n";
    indent_depth++;
    indent_depth--;
    s << indent(indent_depth) << function_end << "\n";
  } else {
    function_definition_ =
        "__host__ void _function_" + std::to_string(kernel_idx_) +
        "_(cudaStream_t & stream, LimbDataType ** args_outputs, const "
        "LimbDataType ** args_inputs, const LimbDataType * args_scalars, const "
        "LimbDataType ** remapables_base, const size_t * args_remapables, "
        "const uint32_t * bases, const LimbDataType * modulii, const "
        "LimbDataType * barrett_ratios, const uint32_t * barrett_k, const "
        "uint32_t ** rotation_map)";
    constexpr std::string_view function_end = "}";

    s << indent(indent_depth) << function_definition_ << "{ \n";
    indent_depth++;
    s << indent(indent_depth) << "dim3 gridSize(GRID_DIM_X,"
      << limb_blocks_.size() << ",1);\n";
    s << indent(indent_depth) << "dim3 blockSize(BLOCK_DIM_X,1,1);\n";
    s << indent(indent_depth) << "_kernel_" << kernel_idx_
      << "<<<gridSize,blockSize,0,stream>>>(args_outputs, args_inputs, "
         "args_scalars, remapables_base, args_remapables, bases, modulii, "
         "barrett_ratios, barrett_k, rotation_map);\n";

    indent_depth--;
    s << indent(indent_depth) << "CHECK_CUDA_ERROR();\n";
    s << indent(indent_depth) << function_end << "\n";
  }

  return s;
}

bool FusedKernel::no_code_to_write() const {
  if (limb_blocks_.empty()) {
    return true;
  }
  switch (kernel_type_) {
  case KernelType::Int:
  case KernelType::Ntt:
  case KernelType::SuD:
  case KernelType::SuD2:
  case KernelType::Msd:
  case KernelType::Bco:
  case KernelType::Ags:
  case KernelType::Drm:
  case KernelType::Ard:
    return true;
  default:
    return false;
  }
}

std::ostream &FusedKernel::write_code(std::ostream &s, int indent_depth) {
  write_kernel(s, indent_depth);
  write_function(s, indent_depth);
  write_function_cugraph(s, indent_depth);
  return s;
}

std::string FusedKernel::prettyPrint() {
  std::stringstream ss;
  write_code(ss);
  ss << "\n";
  return ss.str();
}

std::string FusedKernel::printInstruction() {
  std::stringstream ss;
  write_instruction(ss);
  return ss.str();
}

void FusedKernel::complete() {

  fused_kernel_inputs_set_.clear();
  fused_kernel_outputs_set_.clear();
  fused_kernel_inputs_scalar_set_.clear();
  fused_kernel_inputs_remapable_set_.clear();
  limb_dependencies_map_.clear();
  bases_.clear();
  comm_num_limbs_ = 0;
  comm_size_ = 0;
  comm_srcs.clear();
  comm_dests.clear();

  fused_kernel_inputs_set_.resize(limb_blocks_.size());
  fused_kernel_inputs_scalar_set_.resize(limb_blocks_.size());
  fused_kernel_inputs_remapable_set_.resize(limb_blocks_.size());
  fused_kernel_outputs_set_.resize(limb_blocks_.size());
  limb_dependencies_map_.resize(limb_blocks_.size());

  for (size_t l = 0; l < limb_blocks_.size(); l++) {
    auto &lb = limb_blocks_[l];
    auto &inputs = lb->kernel_inputs_;
    fused_kernel_inputs_set_[l].insert(fused_kernel_inputs_set_[l].end(),
                                       inputs.begin(), inputs.end());
    auto &inputs_scalar = lb->kernel_inputs_scalar_;
    fused_kernel_inputs_scalar_set_[l].insert(
        fused_kernel_inputs_scalar_set_[l].end(), inputs_scalar.begin(),
        inputs_scalar.end());
    auto &inputs_remapable = lb->kernel_inputs_remapable_;
    fused_kernel_inputs_remapable_set_[l].insert(
        fused_kernel_inputs_remapable_set_[l].end(), inputs_remapable.begin(),
        inputs_remapable.end());
    auto &outputs = lb->kernel_outputs_;
    fused_kernel_outputs_set_[l].insert(fused_kernel_outputs_set_[l].end(),
                                        outputs.begin(), outputs.end());
    bases_.push_back(lb->limb_idx_);
    for (auto &[k, v] : lb->limb_dependencies_map_) {
      if (k.is_temp()) {
        continue;
      }
      limb_dependencies_map_[l][k] = v;
    }
    lb->limb_dependencies_map_.clear();
  }
  for (size_t l = 0; l < limb_blocks_.size(); l++) {
    auto &lb = limb_blocks_[l];
    pmu_bases_ = lb->pmu_base_;
    sud_bases_ = lb->sud_base_;
    rsv_bases_ = lb->rsv_base_;
  }
  for (size_t l = 0; l < limb_blocks_.size(); l++) {
    auto &lb = limb_blocks_[l];
    comm_num_limbs_ += lb->comm_num_limbs_;
    if (l == 0) {
      comm_size_ = lb->comm_size_;
      comm_dests.resize(lb->comm_dests.size());
    } else {
      assert(comm_size_ == lb->comm_size_);
      assert(comm_dests.size() == lb->comm_dests.size());
    }
    for (auto &s : limb_blocks_[l]->comm_srcs) {
      comm_srcs.push_back(s);
    }
    for (size_t i = 0; i < lb->comm_dests.size(); i++) {
      CL_LOG3("comm_dests.size() = {} lb->comm_dests.size() = {}",
              comm_dests.size(), lb->comm_dests.size());
      for (auto &d : lb->comm_dests[i]) {
        comm_dests[i].push_back(d);
      }
    }
  }
  assert(comm_srcs.size() == comm_num_limbs_);
  if (kernel_type_ == KernelType::Ags || kernel_type_ == KernelType::Ard) {
    bases_.clear();
    for (auto &s : comm_srcs) {
      auto &l = std::get<0>(s);
      if (l.term()) {
        bases_.push_back(l.limb_idx());
      } else {
        bases_.push_back(0);
      }
    }
  }

  for (size_t l = 0; l < limb_blocks_.size(); l++) {
    auto &lb = limb_blocks_[l];
    call_function_name_ = lb->call_function_name;
    call_function_args_ = lb->call_function_args;
    call_function_args_map_ = lb->call_function_args_map;
  }

  if (kernel_type_ == KernelType::Generic) {
    for (size_t l = 0; l < limb_blocks_.size(); l++) {
      auto &lb = limb_blocks_[l];
      lb->move_instructions_closer();
    }
  }
}

std::ostream &FusedKernelLimb::handle_intt(
    std::ostream &s,
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
    const uint8_t kernel_count, int indent_depth = 0) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();
  // TODO: Optimize by moving the permutation to thread local storage

  if (kernel_count == 0) {
    if (opCode == OpCode::Int) {
      const auto limb_idx = dests[0]->limb_idx();
      std::string src_name = "";
      auto src = srcs[0];
      src_name = src->term()->name();
      std::string dest_name = dests[0]->term()->name();
      s << indent(indent_depth) << "intt0(v_" << dest_name << ", v_" << src_name
        << ", inverse_power_of_roots_div_two_, "
           "modulus_, shmem);\n";
    } else {
      throw std::invalid_argument("Invalid Argument");
    }
  } else if (kernel_count == 1) {
    if (opCode == OpCode::Int) {
      const auto limb_idx = dests[0]->limb_idx();
      std::string dest_name = dests[0]->term()->name();
      s << indent(indent_depth) << "intt1(v_" << dest_name << ", v_"
        << dest_name
        << ", inverse_power_of_roots_div_two_, "
           "modulus_, shmem);\n";
    } else {
      throw std::invalid_argument("Invalid Argument");
    }
  } else if (kernel_count == 2) {
    if (opCode == OpCode::Int) {
      const auto limb_idx = dests[0]->limb_idx();
      std::string src_name = "";
      auto src = srcs[0];
      src_name = src->term()->name();
      std::string dest_name = dests[0]->term()->name();
      s << indent(indent_depth) << "intt0_prelim(v_" << dest_name << ", v_"
        << src_name
        << ", inverse_power_of_roots_div_two_, "
           "modulus_, shmem);\n";
    } else {
      throw std::invalid_argument("Invalid Argument");
    }
  } else {
    throw std::invalid_argument("Invalid Argument");
  }

  return s;
}

std::ostream &FusedKernelLimb::handle_ntt(
    std::ostream &s,
    const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
    const uint8_t kernel_count, int indent_depth = 0) {
  using OpCode = Backend::LimbInstruction::OpCode;
  auto opCode = limb_instr->opcode();
  const auto &dests = limb_instr->dests();
  const auto &srcs = limb_instr->srcs();
  // TODO: Optimize by moving the permutation to thread local storage

  // TODO: Fix this by deleting dead instructions
  bool all_dests_dead = true;
  for (auto &dest : dests) {
    if (!dest->is_temp()) {
      all_dests_dead = false;
      break;
    }
  }

  if (all_dests_dead) {
    return s;
  }

  if (kernel_count == 0) {
    if (opCode == OpCode::Ntt) {
      std::string src_name = "";
      auto src = srcs[0];
      src_name = src->term()->name();
      std::string src_prefix = "v_";
      std::string dest_name = dests[0]->term()->name();
      s << indent(indent_depth) << "ntt0(v_" << dest_name << ", " << src_prefix
        << src_name << ", power_of_roots_, modulus_, shmem);\n";
    } else if (opCode == OpCode::SuD) {
      std::string src_name = "";
      auto src = srcs[1];
      src_name = src->term()->name();
      std::string src_prefix = "v_";
      std::string dest_name = dests[0]->term()->name();
      s << indent(indent_depth) << "ntt0_sud(v_" << dest_name << ", "
        << src_prefix << src_name << ", power_of_roots_, modulus_, shmem);\n";
    } else if (opCode == OpCode::SuD2) {
      std::string src_name = "";
      auto src = srcs[1];
      src_name = src->term()->name();
      std::string src_prefix = "v_";
      std::string dest_name = dests[0]->term()->name();
      s << indent(indent_depth) << "ntt0_sud2(v_" << dest_name << ", "
        << src_prefix << src_name << ", power_of_roots_, modulus_, shmem);\n";
    } else if (opCode == OpCode::Msd) {
      std::string src_name = "";
      auto src = limb_instr->polynomial_srcs()[0];
      auto num_src_limbs = src->num_limbs();
      const auto &src_limbs = src->limbs();
      if (num_src_limbs != 2) {
        throw std::runtime_error("Unimplemented MSD For more than 2 limbs");
      }
      src_name = src->term()->name();
      std::string src_prefix = "v_";
      std::string dest_name = dests[0]->term()->name();
      s << indent(indent_depth) << "ntt0_msd(v_" << dest_name << ", ";
      for (auto &l : src_limbs) {
        s << src_prefix << src_name << "_" << l << ", ";
      }
      s << "power_of_roots_, modulus_, "
           "ldu_global_u32(&barrett_ratio[limb_idx]), "
           "ldu_global_u32(&barrett_k[limb_idx]), shmem);\n";
    } else {
      throw std::invalid_argument("Invalid Argument");
    }
  } else if (kernel_count == 1) {
    if (opCode == OpCode::Ntt) {
      std::string dest_name = dests[0]->term()->name();
      s << indent(indent_depth) << "ntt1(v_" << dest_name << ", v_" << dest_name
        << ", power_of_roots_, modulus_, shmem);\n";
    } else if (opCode == OpCode::SuD || opCode == OpCode::SuD2) {
      std::string dest_name = dests[0]->term()->name();
      std::string src_name = srcs[0]->term()->name();
      s << indent(indent_depth) << "ntt1_sud(v_" << dest_name << ", v_"
        << dest_name << ", v_" << src_name
        << ", power_of_roots_, ldu_global_u32(&sud_factors[limb_idx]), "
           "modulus_, "
           "ldu_global_u32(&barrett_ratio[limb_idx]), "
           "ldu_global_u32(&barrett_k[limb_idx]), shmem);\n";
    } else if (opCode == OpCode::Msd) {
      std::string dest_name = dests[0]->term()->name();
      std::string src_name = srcs[0]->term()->name();
      s << indent(indent_depth) << "ntt1_sud(v_" << dest_name << ", v_"
        << dest_name << ", v_" << src_name << "_" << srcs[0]->limb_idx()
        << ", power_of_roots_, ldu_global_u32(&sud_factors[limb_idx]), "
           "modulus_, "
           "ldu_global_u32(&barrett_ratio[limb_idx]), "
           "ldu_global_u32(&barrett_k[limb_idx]), shmem);\n";
    } else {
      throw std::invalid_argument("Invalid Argument");
    }
  } else {
    throw std::invalid_argument("Invalid Argument");
  }

  return s;
}

std::ostream &FusedKernel::write_value(std::ostream &s,
                                       FusedKernel::ValueType &val) {
  std::visit(Cerium::Overloaded{
                 [](auto &arg) { throw std::runtime_error("Invalid Type"); },
                 [&](Limb &l) {
                   if (!l.term() || l.is_zero()) {
                     s << "Z";
                     return;
                   }
                   s << l.term()->name() << "(" << l.limb_idx() << ")";
                   if (l.is_dead()) {
                     s << "[X]";
                   }
                 },
                 [&](Register &r) { s << r; }},
             val);
  return s;
}

std::ostream &FusedKernel::write_instruction(std::ostream &s) {
  if (limb_blocks_.empty()) {
    return s;
  }
  auto indent_depth = 0;
  s << indent(indent_depth) << "function" << kernel_idx_ << " {\n";
  ++indent_depth;
  s << indent(indent_depth) << ".id: " << kernel_idx_ << "\n";
  s << indent(indent_depth) << ".type: ";
  switch (kernel_type_) {
  case KernelType::Mod:
  case KernelType::Generic:
    s << "generic";
    break;
  case KernelType::Mov:
    s << "mov";
    break;
  case KernelType::Msd:
  case KernelType::SuD:
  case KernelType::SuD2:
    s << "sud";
    break;
  case KernelType::Ntt:
    s << "ntt";
    break;
  case KernelType::Int:
    s << "int";
    break;
  case KernelType::Rsv:
    s << "rsv";
    break;
  case KernelType::Bco:
    s << "bco";
    break;
  case KernelType::Pmu:
    s << "pmu";
    break;
  case KernelType::Drm:
    s << "drm";
    break;
  case KernelType::Ags:
    s << "ags";
    break;
  case KernelType::Ard:
    s << "ard";
    break;
  case KernelType::Call:
    s << "call";
    break;
  }
  s << "\n";
  s << indent(indent_depth) << ".outputs: ";
  foreach_map_limb_set_valuetype(fused_kernel_outputs_set_,
                                 [&](const size_t pos, ValueType &ii) {
                                   write_value(s, ii);
                                   s << ", ";
                                 });
  s << "\n";
  s << indent(indent_depth) << ".inputs: ";

  foreach_map_limb_set_valuetype(fused_kernel_inputs_set_,
                                 [&](const size_t pos, ValueType &ii) {
                                   write_value(s, ii);
                                   s << ", ";
                                 });
  s << "\n";
  s << indent(indent_depth) << ".scalars: ";
  foreach_map_limb_set_valuetype(
      fused_kernel_inputs_scalar_set_, [&](const size_t pos, ValueType &ii) {
        auto &i = std::get<0>(ii);
        s << i.term()->name() << "(" << i.limb_idx() << ")";
        if (i.is_dead()) {
          s << "[X]";
        }
        s << ", ";
      });
  s << "\n";
  s << indent(indent_depth) << ".reamapables: ";
  foreach_map_limb_set_valuetype(fused_kernel_inputs_remapable_set_,
                                 [&](const size_t pos, ValueType &ii) {
                                   write_value(s, ii);
                                   s << ", ";
                                 });
  s << "\n";
  s << indent(indent_depth) << ".base_conv: ";
  if (kernel_type_ == KernelType::Int || kernel_type_ == KernelType::Generic ||
      kernel_type_ == KernelType::Drm) {
    for (auto &[k, v] : base_conv_input_bases_) {
      s << "<";
      s << "b" << k << ": ";
      for (auto &base : v) {
        s << base << ", ";
      }
      s << ">";
    }
  } else if (kernel_type_ == KernelType::SuD ||
             kernel_type_ == KernelType::Ntt ||
             kernel_type_ == KernelType::SuD2) {
    for (auto &[k, v] : base_conv_output_bases_) {
      s << "<";
      s << "b" << k << ": ";
      for (auto &base : v) {
        s << base << ", ";
      }
      s << ">";
    }
  } else if (kernel_type_ == KernelType::Bco) {
    for (auto &[k, v] : base_conv_input_bases_) {
      s << "<";
      s << "b" << k << ": ";
      for (auto &base : v) {
        s << base << ", ";
      }
      s << ": ";
      for (auto &base : base_conv_output_bases_[k]) {
        s << base << ", ";
      }
      s << ">";
    }
  }
  s << "\n";

  s << indent(indent_depth) << ".rots: ";
  for (auto &r : rot_indices_) {
    s << r << ", ";
  }
  s << "\n";

  s << indent(indent_depth) << ".sud: ";
  if (kernel_type_ == KernelType::SuD || kernel_type_ == KernelType::Msd ||
      kernel_type_ == KernelType::SuD2) {
    for (auto &b : sud_bases_) {
      s << b << ", ";
    }
  }
  s << "\n";
  s << indent(indent_depth) << ".pmu: ";
  if (kernel_type_ == KernelType::Pmu) {
    for (auto &b : pmu_bases_) {
      s << b << ", ";
    }
  }
  s << "\n";

  s << indent(indent_depth) << ".rsv: ";
  if (kernel_type_ == KernelType::Rsv) {
    for (auto &b : rsv_bases_) {
      s << b << ", ";
    }
  }
  s << "\n";

  s << indent(indent_depth) << ".comm_size: ";
  if (kernel_type_ == KernelType::Drm || kernel_type_ == KernelType::Ags ||
      kernel_type_ == KernelType::Ard) {
    s << comm_size_;
  }
  s << "\n";
  s << indent(indent_depth) << ".comm_num_limbs: ";
  if (kernel_type_ == KernelType::Drm || kernel_type_ == KernelType::Ags ||
      kernel_type_ == KernelType::Ard) {
    s << comm_num_limbs_;
  }
  s << "\n";
  s << indent(indent_depth) << ".comm_srcs: ";
  if (kernel_type_ == KernelType::Drm || kernel_type_ == KernelType::Ags ||
      kernel_type_ == KernelType::Ard) {
    for (auto &cc : comm_srcs) {
      write_value(s, cc);
      s << ", ";
    }
  }
  s << "\n";
  s << indent(indent_depth) << ".comm_dests: ";
  if (kernel_type_ == KernelType::Drm || kernel_type_ == KernelType::Ags ||
      kernel_type_ == KernelType::Ard) {
    for (auto &dd : comm_dests) {
      for (auto &d : dd) {
        write_value(s, d);
        s << ", ";
      }
    }
  }
  s << "\n";
  s << indent(indent_depth) << ".function_name: ";
  if (kernel_type_ == KernelType::Call) {
    s << call_function_name_;
  }
  s << "\n";
  s << indent(indent_depth) << ".function_args: ";
  if (kernel_type_ == KernelType::Call) {
    for (auto &[k, v] : call_function_args_) {
      s << "<" << k << "~";
      write_value(s, v);
      s << ">, ";
    }
  }
  s << "\n";

  s << indent(indent_depth) << ".bases: ";
  for (auto &b : bases_) {
    s << b << ", ";
  }

  s << "\n";

  --indent_depth;
  s << indent(indent_depth) << "}";
  return s;
}

void KernelFusionContext::write_code_file_header(std::ofstream &s) {
  s << "// Auto Generated Code. Do not edit.\n";
  s << "#include \"fhe.cuh\"\n";
  s << "namespace Cerium {\nnamespace " << function_name << " {\n\n";
}

void KernelFusionContext::write_code_file_footer(std::ofstream &s) {
  s << "\n\n} // namespace " << function_name << "\n} // namespace Cerium\n";
}

void KernelFusionContext::write_code(const std::string &dir_name) {
  size_t KERNELS_PER_FILE = 50;
  std::size_t file_num = 0;

  auto num_kernels = kernels_.size();

  size_t kernels_written = 0;
  std::string code_file_path =
      dir_name + "/kernels_" + std::to_string(file_num) + ".cu";
  std::ofstream code_file(code_file_path);
  if (!code_file.is_open()) {
    throw std::runtime_error("Failed to open output file: " + code_file_path);
  }
  write_code_file_header(code_file);
  for (size_t i = 0; i < kernels_.size(); i++) {
    auto &kernel = kernels_[i];
    if (kernel->no_code_to_write()) {
      continue;
    }
    if (!code_file.is_open()) {
      throw std::runtime_error("Output file closed unexpectedly: " +
                               code_file_path);
    }
    kernel->write_code(code_file);
    code_file << "\n//================\n";
    kernels_written++;
    if (kernels_written % KERNELS_PER_FILE == 0 && (i + 1) < kernels_.size()) {
      write_code_file_footer(code_file);
      code_file.close();
      file_num++;
      code_file_path =
          dir_name + "/kernels_" + std::to_string(file_num) + ".cu";
      code_file.open(code_file_path);
      if (!code_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " +
                                 code_file_path);
      }
      write_code_file_header(code_file);
    }
  }
  write_code_file_footer(code_file);
  code_file.close();
}

void KernelFusionContext::write_function_map_impl(std::ofstream &file,
                                                  bool use_cugraph) {
  const auto map_name = use_cugraph ? "function_map_cugraph" : "function_map";
  const auto getter_name =
      use_cugraph ? "get_function_map_cugraph" : "get_function_map";
  const auto getter_asm_name =
      use_cugraph ? "__GET_FUNCTION_MAP_CUGRAPH__" : "__GET_FUNCTION_MAP__";
  const auto function_suffix = use_cugraph ? "_cugraph_" : "_";
  const auto bco_name = use_cugraph ? "bco_cugraph" : "bco";

  auto is_communication_kernel = [](const std::shared_ptr<FusedKernel> &k) {
    return k->kernel_type_ == FusedKernel::KernelType::Ard ||
           k->kernel_type_ == FusedKernel::KernelType::Drm ||
           k->kernel_type_ == FusedKernel::KernelType::Ags;
  };

  auto should_declare_function = [&](const std::shared_ptr<FusedKernel> &k) {
    if (k->limb_blocks_.empty()) {
      return false;
    }
    switch (k->kernel_type_) {
    case FusedKernel::KernelType::Int:
    case FusedKernel::KernelType::Ntt:
    case FusedKernel::KernelType::SuD:
    case FusedKernel::KernelType::SuD2:
      return false;
    default:
      return !is_communication_kernel(k);
    }
  };

  auto should_map_function = [&](const std::shared_ptr<FusedKernel> &k) {
    return !k->limb_blocks_.empty() && !is_communication_kernel(k);
  };

  auto get_bco_bases = [](const std::shared_ptr<FusedKernel> &k) {
    assert(!k->limb_blocks_.empty());
    assert(k->limb_blocks_[0]->instructions_.size() == 1);
    auto limb_instr =
        std::dynamic_pointer_cast<Backend::BaseConvLimbInstruction>(
            k->limb_blocks_[0]->instructions_[0]);
    const auto num_input_bases = limb_instr->polynomial_srcs()[0]->num_limbs();
    const auto num_output_bases =
        limb_instr->polynomial_dests()[0]->num_limbs();
    for (size_t i = 0; i < k->limb_blocks_.size(); ++i) {
      auto limb_instr_i =
          std::dynamic_pointer_cast<Backend::BaseConvLimbInstruction>(
              k->limb_blocks_[i]->instructions_[0]);
      assert(num_input_bases ==
             limb_instr_i->polynomial_srcs()[0]->num_limbs());
      assert(num_output_bases ==
             limb_instr_i->polynomial_dests()[0]->num_limbs());
    }
    return std::pair{num_input_bases, num_output_bases};
  };

  for (auto &k : kernels_) {
    if (!should_declare_function(k)) {
      continue;
    }
    file << (use_cugraph ? k->function_definition_cugraph_
                         : k->function_definition_)
         << ";\n\n";
  }
  file << "FunctionMapType " << map_name << ";\n";
  file << "const FunctionMapType & " << getter_name << "() asm(\""
       << getter_asm_name << "\");\n";
  file << "const FunctionMapType & " << getter_name << "() {\n";
  for (auto &k : kernels_) {
    // Transform functions are generated elsewhere, but still need map entries.
    if (!should_map_function(k)) {
      continue;
    }
    file << "\t" << map_name << "[" << k->kernel_index()
         << "] = (void *) &_function_";
    switch (k->kernel_type_) {
    case FusedKernel::KernelType::Int:
      file << "int";
      break;
    case FusedKernel::KernelType::Ntt:
      file << "ntt";
      break;
    case FusedKernel::KernelType::SuD:
      file << "sud";
      break;
    case FusedKernel::KernelType::SuD2:
      file << "sud2";
      break;
    case FusedKernel::KernelType::Bco: {
      const auto [num_input_bases, num_output_bases] = get_bco_bases(k);
      file << bco_name << "_<" << num_input_bases << "," << num_output_bases
           << ">;\n";
      continue;
    } break;
    default:
      file << k->kernel_index();
      break;
    }
    file << function_suffix << ";\n";
  }
  file << "\treturn " << map_name << ";\n";
  file << "}";
}

void KernelFusionContext::write_function_map(const std::string &dir_name) {

  const std::string function_map_path = dir_name + "/function_map.cu";
  std::ofstream file(function_map_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open output file: " +
                             function_map_path);
  }
  file << "#include \"fhe.cuh\"\n";
  file << "#include \"function.cuh\"\n";
  write_code_file_header(file);
  write_function_map_impl(file, false);
  write_code_file_footer(file);
  file.close();
}

void KernelFusionContext::write_function_map_cugraph(
    const std::string &dir_name) {

  const std::string function_map_cugraph_path =
      dir_name + "/function_map_cugraph.cu";
  std::ofstream file(function_map_cugraph_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open output file: " +
                             function_map_cugraph_path);
  }
  file << "#include \"fhe.cuh\"\n";
  file << "#include \"function.cuh\"\n";
  write_code_file_header(file);
  write_function_map_impl(file, true);
  write_code_file_footer(file);
  file.close();
}

void KernelFusionContext::write_instructions(const std::string &dir_name) {

  const std::string instructions_path = dir_name + "/ll.functions";
  std::ofstream file(instructions_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open output file: " +
                             instructions_path);
  }
  for (auto &k : kernels_) {
    if (k->limb_blocks_.empty()) {
      continue;
    }
    k->write_instruction(file);
    file << "\n\n";
  }
  file.close();
}

void KernelFusionContext::allocate_registers_cerium(size_t num_vregs,
                                                      size_t num_bcus) {

  using LimbMapType = Backend::LimbMap<Register>;
  using FreeListType = std::deque<Register>;
  using BCUFreeListType = std::deque<BaseConversionUnit>;

  FreeListType free_list;
  LimbMapType limb_map_table;

  auto is_mapped = [&](const Backend::Limb &limb) {
    auto it = limb_map_table.find(limb);
    return it != limb_map_table.end();
  };

  auto get_register = [&](const Backend::Limb &limb) {
    return limb_map_table.at(limb);
  };

  auto unmap_register = [&](const Backend::Limb &limb) {
    limb_map_table.erase(limb);
  };

  auto update_register_use = [&](const Backend::Limb &limb) {
    auto it = limb_map_table.find(limb);
    if (it == limb_map_table.end()) {
      return;
    }
    auto reg = it->second;
    limb_map_table.erase(limb);
    limb_map_table[limb] = reg;
    CL_LOG3("Found : {} at register : {}", limb, reg.id());
  };

  auto evict_reg = [&]() {
    throw std::runtime_error("Called Evict Reg\n");

    assert(!limb_map_table.empty());
    uint64_t max_next_use = 0;
    Backend::Limb evict_limb;
    Register evict_reg;
    for (auto &i : limb_map_table) {
      assert(!i.first.is_dead());
      auto next_use = i.first.next_use();
      if (next_use > max_next_use) {
        max_next_use = next_use;
        evict_limb = i.first;
        evict_reg = i.second;
      }
    }

    assert(evict_limb.is_input());
    assert(evict_reg.reg_type() != Register::RegType::None);
    limb_map_table.erase(evict_limb);
    CL_LOG3("Evicting: {} with next use : {}", evict_limb, max_next_use);
    free_list.push_back(evict_reg);
    // TODO: Add spill instructions
  };

  uint64_t NUM_REGS = num_vregs;
  uint64_t NUM_BCU = num_bcus;
  for (auto i = 0; i < NUM_REGS; i++) {
    free_list.push_back(Register(i, Register::RegType::Vector));
  }

  std::deque<uint32_t> base_conversion_units_;
  for (auto i = 0; i < NUM_BCU; i++) {
    base_conversion_units_.push_back(i);
  }
  std::unordered_map<TermIndexType, uint32_t> base_conversion_map_;

  for (auto &k : kernels_) {
    CL_LOG3("Kernel: {}", k->kernel_index());

    std::unordered_map<LimbIndexType, LimbMap<Register>> register_stack;
    std::vector<Register> rotate_registers;
    auto &inputs = k->fused_kernel_inputs_set_;
    auto &outputs = k->fused_kernel_outputs_set_;
    auto foreach = FusedKernel::foreach_map_limb_set_valuetype;

    std::set<uint32_t> bcu_read;

    if (k->kernel_type_ == FusedKernel::KernelType::Bco) {
      auto &blocks = k->limb_blocks_;
      for (auto &b : blocks) {
        for (auto &instr : b->instructions_) {
          auto bco_instr =
              std::dynamic_pointer_cast<BaseConvLimbInstruction>(instr);
          assert(bco_instr);
          auto src = bco_instr->polynomial_srcs()[0];
          auto dest = bco_instr->polynomial_dests()[0];
          if (src->limbs().empty()) {
            continue;
          }
          assert(src->is_bcor());
          assert(dest->is_bcor());
          auto bcu_id = base_conversion_map_.at(src->term_idx());
          base_conversion_map_[dest->term_idx()] = bcu_id;
          base_conversion_map_.erase(src->term_idx());
          for (auto &l : src->limbs()) {
            k->base_conv_input_bases_[bcu_id].push_back(l);
          }
          for (auto &l : dest->limbs()) {
            k->base_conv_output_bases_[bcu_id].push_back(l);
          }
          CL_LOG3("BConv mapping BC: {}->{} at register : {}", src->term_idx(),
                  dest->term_idx(), bcu_id);
        }
      }
      continue;
    } else if (k->kernel_type_ == FusedKernel::KernelType::Mov) {
      assert(inputs.size() == outputs.size());
      auto size = inputs.size();
      size_t cont = 0;
      for (auto i = 0; i < size; i++) {
        auto &blk_inputs = inputs[i];
        auto &blk_outputs = outputs[i];
        assert(blk_inputs.size() == blk_outputs.size());
        assert(blk_inputs.size() == 1);
        auto &in = std::get<0>(blk_inputs[0]);
        auto &out = std::get<0>(blk_outputs[0]);
        if (in.is_input()) {
          continue;
        }
        if (!in.is_dead()) {
          continue;
        }
        if (!is_mapped(in)) {
          continue;
        }
        auto reg = get_register(in);
        limb_map_table.erase(in);
        limb_map_table.insert({out, reg});
        blk_inputs[0] = reg;
        blk_outputs[0] = reg;
        cont++;
      }

      // This logic assumes that all values behave similarly. If even one of the
      // values behaves differently, then the exception will be thrown
      if (cont != 0) {
        if (cont != size) {
          throw std::logic_error("Cont != Size");
        }
        assert(cont == size);
        continue;
      } else {
        k->set_kernel_type(FusedKernel::KernelType::Generic);
      }
    }

    std::vector<std::pair<Limb, Register>> limb_register_map_updates;
    auto process_input = [&](const size_t pos, FusedKernel::ValueType &ii) {
      auto &i = std::get<0>(ii);
      if (i.is_input() || i.is_output()) {
        return;
      }
      if (i.is_bcor()) {
        auto bcu_id = base_conversion_map_.at(i.term_idx());
        auto reg = Register(bcu_id, i.limb_idx());
        if (i.is_dead()) {
          bcu_read.insert(bcu_id);
          CL_LOG3("Dead BC: {} with register : {}", i, bcu_id);
        }
        CL_LOG3("Found BC: {} with register : {}", i, bcu_id);
        k->base_conv_output_bases_[bcu_id].push_back(i.limb_idx());
        ii = reg;
        return;
      }
      if (i.is_dead()) {
        if (is_mapped(i)) {
          auto reg = get_register(i);
          CL_LOG3("Got: {} at register: {}", i, reg.id());
          // Keep the rotate source registers separate as these registers
          // cannot be reused within the same kernel for both input and output
          if (k->rot_inputs_.find(i) != k->rot_inputs_.end()) {
            rotate_registers.push_back(reg);
            CL_LOG3("Rotate Input: {} at register: {}", i, reg.id());
          } else {
            register_stack[pos][i] = reg;
          }
          unmap_register(i);
          CL_LOG3("Freeing: {} at register: {}", i, reg.id());
          if (k->kernel_type_ == FusedKernel::KernelType::Call) {
            auto it = k->call_function_args_map_.find(i);
            if (it != k->call_function_args_map_.end()) {
              for (auto &itt : it->second) {
                k->call_function_args_.at(itt) = reg;
              }
            }
          }
          ii = reg;
        }
      } else {
        auto reg = get_register(i);
        CL_LOG3("Got: {} at register: {}", i, reg.id());
        update_register_use(i);
        if (k->kernel_type_ == FusedKernel::KernelType::Call) {
          auto it = k->call_function_args_map_.find(i);
          if (it != k->call_function_args_map_.end()) {
            for (auto &itt : it->second) {
              k->call_function_args_.at(itt) = reg;
            }
          }
        }
        ii = reg;
      }
    };

    auto process_output = [&](const size_t pos, FusedKernel::ValueType &oo) {
      auto &o = std::get<0>(oo);
      if (o.is_output()) {
        CL_LOG3("Got Output: {}", o);
        return;
      }

      if (o.is_bcor()) {
        if (base_conversion_map_.find(o.term_idx()) ==
            base_conversion_map_.end()) {
          if (base_conversion_units_.empty()) {
            throw std::runtime_error("Insufficient Base Conversion Units");
          }
          assert(!base_conversion_units_.empty());
          auto bcu_id = base_conversion_units_.front();
          base_conversion_units_.pop_front();
          base_conversion_map_[o.term_idx()] = bcu_id;
          CL_LOG3("Mapping BC: {} with register : {}", o.term_idx(), bcu_id);
        }
        auto bcu_id = base_conversion_map_[o.term_idx()];
        auto reg = Register(bcu_id, o.limb_idx());
        CL_LOG3("Mapping BC: {} with register : {}", o, bcu_id);
        k->base_conv_input_bases_[bcu_id].push_back(o.limb_idx());
        oo = reg;
        return;
      }
      if (o.is_dead()) {
        CL_LOG3("Dead Output: {}", o);
        return;
      } else if (k->rot_outputs_.find(o) != k->rot_outputs_.end()) {
        CL_LOG3("Rotate Output: {}", o);
        if (free_list.empty()) {
          evict_reg();
        }
        auto reg = free_list.front();
        free_list.pop_front();
        limb_register_map_updates.push_back({o, reg});
        CL_LOG3("Mapping : {} with register : {}", o, reg.id());
        oo = reg;
      } else {
        Register reg;
        bool register_allocated = false;
        if ((k->kernel_type_ != FusedKernel::KernelType::SuD &&
             k->kernel_type_ != FusedKernel::KernelType::SuD2) &&
            !register_stack[pos].empty()) {
          for (const auto &v : k->limb_dependencies_map_[pos][o]) {
            if (register_stack[pos].find(v) != register_stack[pos].end()) {
              reg = register_stack[pos][v];
              register_stack[pos].erase(v);
              register_allocated = true;
              break;
            }
          }
        }
        if (!register_allocated) {
          if (free_list.empty()) {
            evict_reg();
          }
          reg = free_list.front();
          free_list.pop_front();
        }
        limb_register_map_updates.push_back({o, reg});
        CL_LOG3("Mapping : {} with register : {}", o, reg.id());
        if (k->kernel_type_ == FusedKernel::KernelType::Call) {
          auto it = k->call_function_args_map_.find(o);
          if (it != k->call_function_args_map_.end()) {
            for (auto &itt : it->second) {
              k->call_function_args_.at(itt) = reg;
            }
          }
        }
        oo = reg;
      }
    };

    auto handle_comm_inputs = [&]() {
      if (k->kernel_type_ != FusedKernel::KernelType::Drm &&
          k->kernel_type_ != FusedKernel::KernelType::Ags &&
          k->kernel_type_ != FusedKernel::KernelType::Ard) {
        return;
      }
      auto process_elt = [&](FusedKernel::ValueType &cc) {
        auto &c = std::get<0>(cc);
        if (!c.term()) {
          return;
        }
        if (c.is_zero()) {
          return;
        }
        if (c.is_input()) {
          return;
        }
        if (c.is_output()) {
          return;
        }
        if (is_mapped(c)) {
          auto reg = get_register(c);
          cc = reg;
        } else {
          std::stringstream ss;
          ss << "Unimplemented: Comm source not mapped: " << c;
          throw std::runtime_error(ss.str());
        }
      };
      if (k->kernel_type_ == FusedKernel::KernelType::Drm) {
        for (auto &cc : k->comm_srcs) {
          process_elt(cc);
        }
      } else if (k->kernel_type_ == FusedKernel::KernelType::Ags) {
        for (auto &dests : k->comm_dests) {
          for (auto &dd : dests) {
            process_elt(dd);
          }
        }
      } else if (k->kernel_type_ == FusedKernel::KernelType::Ard) {
        for (auto &dests : k->comm_dests) {
          for (auto &dd : dests) {
            process_elt(dd);
          }
        }
      }
    };

    auto handle_comm_outputs = [&]() {
      if (k->kernel_type_ != FusedKernel::KernelType::Drm &&
          k->kernel_type_ != FusedKernel::KernelType::Ags &&
          k->kernel_type_ != FusedKernel::KernelType::Ard) {
        return;
      }
      auto process_elt = [&](FusedKernel::ValueType &dd) {
        auto &d = std::get<0>(dd);
        if (!d.term()) {
          return;
        }
        if (d.is_output()) {
          return;
        }
        if (d.is_dead()) {
          return;
        }
        if (is_mapped(d)) {
          auto reg = get_register(d);
          dd = reg;
        } else if (d.is_bcor()) {
          auto bcu_id = base_conversion_map_.at(d.term_idx());
          auto reg = Register(bcu_id, d.limb_idx());
          dd = reg;
        } else {
          // throw std::runtime_error("Unimplemented: Comm Dest not mapped");
          // dd = Limb();
        }
      };
      if (k->kernel_type_ == FusedKernel::KernelType::Drm) {
        for (auto &dests : k->comm_dests) {
          for (auto &gg : dests) {
            process_elt(gg);
          }
        }
      } else if (k->kernel_type_ == FusedKernel::KernelType::Ags) {
        for (auto &ss : k->comm_srcs) {
          process_elt(ss);
        }
      } else if (k->kernel_type_ == FusedKernel::KernelType::Ard) {
        for (auto &ss : k->comm_srcs) {
          process_elt(ss);
        }
      }
    };

    handle_comm_inputs();
    foreach (inputs, process_input)
      ;
    foreach (outputs, process_output)
      ;
    for (auto &[l, r] : limb_register_map_updates) {
      limb_map_table.erase(l);
      limb_map_table.insert({l, r});
    }
    handle_comm_outputs();

    for (auto &[l, v] : register_stack) {
      for (auto &[k, r] : v) {
        free_list.push_back(r);
      }
    }
    for (auto &r : rotate_registers) {
      free_list.push_back(r);
    }
    for (auto &bcu : bcu_read) {
      base_conversion_units_.push_back(bcu);
      CL_LOG3("Pushing BCU: {}", bcu);
    }
    register_stack.clear();
    limb_register_map_updates.clear();
  }
}
std::ostream &FusedKernel::write_function_cugraph(std::ostream &s,
                                                  int indent_depth) {

  if (kernel_type_ == KernelType::Int || kernel_type_ == KernelType::Ntt ||
      kernel_type_ == KernelType::SuD || kernel_type_ == KernelType::SuD2) {
    return s;
  }
  if (limb_blocks_.empty()) {
    return s;
  }
  if (kernel_type_ == KernelType::Bco) {
    return s;
  }
  if (kernel_type_ == KernelType::Ard || kernel_type_ == KernelType::Drm ||
      kernel_type_ == KernelType::Ags) {
    return s;
  }

  if (kernel_type_ == KernelType::Msd) {
    function_definition_cugraph_ =
        "__host__ void _function_" + std::to_string(kernel_idx_) +
        "_cugraph_(cudaGraph_t & graph, cudaGraphNode_t & node0, "
        "cudaGraphNode_t & node1, LimbDataType ** args_outputs, const "
        "LimbDataType ** args_inputs, const uint32_t * bases, const "
        "LimbDataType * modulii, const LimbDataType2 * power_of_roots, const "
        "LimbDataType * sud_factors, const LimbDataType * barrett_ratios, "
        "const uint32_t * barrett_k)";
    constexpr std::string_view function_end = "}";

    s << indent(indent_depth) << function_definition_cugraph_ << "{ \n";
    indent_depth++;
    s << indent(indent_depth) << "dim3 gridSize0(NTT0_GRID_DIM_X,"
      << limb_blocks_.size() << ",1);\n";
    s << indent(indent_depth) << "dim3 blockSize0(NTT0_BLOCK_DIM_X,1,1);\n";

    s << indent(indent_depth);
    s << "void *args0[7] = {(void *)&args_outputs,(void *)&args_inputs, (void "
         "*)&bases, (void *)&modulii, (void *)&power_of_roots, (void "
         "*)&barrett_ratios, (void *)&barrett_k};\n";
    s << indent(indent_depth);
    s << "node0 = set_kernel_node_params(graph, (void*)_kernel_ntt_"
      << kernel_idx_ << "_0, gridSize0, blockSize0, 0, (void**)args0);\n";
    s << indent(indent_depth) << "CHECK_CUDA_ERROR();\n";

    s << indent(indent_depth) << "dim3 gridSize1(NTT1_GRID_DIM_X,"
      << limb_blocks_.size() << ",1);\n";
    s << indent(indent_depth) << "dim3 blockSize1(NTT1_BLOCK_DIM_X,1,1);\n";
    s << indent(indent_depth);
    s << "void *args1[8] = {(void *)&args_outputs,(void *)&args_inputs, (void "
         "*)&bases, (void *)&modulii, (void *)&power_of_roots, (void "
         "*)&sud_factors, (void *)&barrett_ratios, (void *)&barrett_k};\n";
    s << indent(indent_depth);
    s << "node1 = set_kernel_node_params(graph, (void*)_kernel_ntt_"
      << kernel_idx_ << "_1, gridSize1, blockSize1, 0, (void**)args1);\n";
    s << indent(indent_depth) << "CHECK_CUDA_ERROR();\n";

    s << indent(indent_depth)
      << "cudaGraphAddDependencies(graph, &node0, &node1, nullptr, 1);\n";

    s << indent(indent_depth) << "CHECK_CUDA_ERROR();\n";

    indent_depth--;
    s << indent(indent_depth) << function_end << "\n";
  } else if (kernel_type_ == KernelType::Rsv) {
    function_definition_cugraph_ =
        "__host__ void _function_" + std::to_string(kernel_idx_) +
        "_cugraph_(cudaGraph_t & graph, cudaGraphNode_t & node0, LimbDataType "
        "** args_outputs, const LimbDataType ** args_inputs, const "
        "LimbDataType * factors, const LimbDataType * modulus, const "
        "LimbDataType * barrett_ratio, const uint32_t barrett_k)";
    constexpr std::string_view function_end = "}";

    s << indent(indent_depth) << function_definition_cugraph_ << "{ \n";
    indent_depth++;
    s << indent(indent_depth) << "dim3 gridSize(GRID_DIM_X,"
      << limb_blocks_.size() << ",1);\n";
    s << indent(indent_depth) << "dim3 blockSize(BLOCK_DIM_X,1,1);\n";
    s << indent(indent_depth);
    s << "void *args1[6] = {(void *)&args_outputs,(void *)&args_inputs, (void "
         "*)&factors, (void *)&modulus, (void *)&barrett_ratio, (void "
         "*)&barrett_k};\n";
    s << indent(indent_depth);
    s << "node0 = set_kernel_node_params(graph, (void*)_kernel_rsv_"
      << kernel_idx_ << ", gridSize, blockSize, 0, (void**)args1);\n";
    s << indent(indent_depth) << "CHECK_CUDA_ERROR();\n";

    indent_depth--;
    s << indent(indent_depth) << function_end << "\n";
  } else if (kernel_type_ == KernelType::Call) {
    // Nothing to Do
    function_definition_cugraph_ =
        "__host__ void _function_" + std::to_string(kernel_idx_) +
        "_cugraph_(cudaGraph_t & graph, cudaGraphNode_t & node0, LimbDataType "
        "** args_outputs, const size_t * num_output_bases, const LimbDataType "
        "** args_inputs, const size_t * num_input_bases, const LimbDataType ** "
        "base_conversion_factors, const LimbDataType ** output_modulii, const "
        "LimbDataType ** barrett_ratios, const uint32_t ** barrett_k, const "
        "size_t shared_mem_size)";
    constexpr std::string_view function_end = "}";
    s << indent(indent_depth) << function_definition_cugraph_ << "{ \n";
    indent_depth++;
    indent_depth--;
    s << indent(indent_depth) << function_end << "\n";

  } else {
    function_definition_cugraph_ =
        "__host__ void _function_" + std::to_string(kernel_idx_) +
        "_cugraph_(cudaGraph_t & graph, cudaGraphNode_t & node0, LimbDataType "
        "** args_outputs, const LimbDataType ** args_inputs, const "
        "LimbDataType * args_scalars, const LimbDataType ** remapables_base, "
        "const size_t * args_remapables, const uint32_t * bases, const "
        "LimbDataType * modulii, const LimbDataType * barrett_ratios, const "
        "uint32_t * barrett_k, const uint32_t ** rotation_map)";
    constexpr std::string_view function_end = "}";

    s << indent(indent_depth) << function_definition_cugraph_ << "{ \n";
    indent_depth++;
    s << indent(indent_depth) << "dim3 gridSize(GRID_DIM_X,"
      << limb_blocks_.size() << ",1);\n";
    s << indent(indent_depth) << "dim3 blockSize(BLOCK_DIM_X,1,1);\n";
    s << indent(indent_depth);
    s << "void *args1[10] = {(void *)&args_outputs,(void *)&args_inputs,(void "
         "*)&args_scalars, (void *)&remapables_base, (void *)&args_remapables, "
         "(void *)&bases, (void *)&modulii, (void *)&barrett_ratios, (void "
         "*)&barrett_k, (void *)&rotation_map};\n";
    s << indent(indent_depth);
    s << "node0 = set_kernel_node_params(graph, (void*)_kernel_" << kernel_idx_
      << ", gridSize, blockSize, 0, (void**)args1);\n";
    s << indent(indent_depth) << "CHECK_CUDA_ERROR();\n";

    indent_depth--;
    s << indent(indent_depth) << function_end << "\n";
  }

  return s;
}
std::ostream &FusedKernel::write_kernel_pmu(std::ostream &s, int indent_depth) {

  std::string kernel_header =
      kernel_signature + "_kernel_" + std::to_string(kernel_idx_) +
      "(LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const "
      "LimbDataType * args_pmu, const LimbDataType ** remapables_base, const "
      "size_t * args_remapables, const uint32_t * bases, const LimbDataType * "
      "modulii, const LimbDataType * barrett_ratios, const uint32_t * "
      "barrett_k, const uint32_t ** rotate_map) {";
  constexpr std::string_view kernel_end = "}";

  s << kernel_header << "\n";
  indent_depth++;

  constexpr std::string_view if_header_0 = "if(j == ";
  constexpr std::string_view if_header_1 = ") {";
  constexpr std::string_view if_end = "}";

  std::size_t count = 0;
  count = 0;
  s << indent(indent_depth) << "//-------------------\n";

  s << indent(indent_depth) << "size_t j = blockIdx.y;\n";
  if (!limb_blocks_.empty()) {
    s << indent(indent_depth) << "auto limb_idx = bases[j];\n";
    s << indent(indent_depth) << "auto modulus_ = modulii[limb_idx];\n";
    s << indent(indent_depth)
      << "auto barrett_ratio_ = barrett_ratios[limb_idx];\n";
    s << indent(indent_depth) << "auto barrett_k_ = barrett_k[limb_idx];\n";
    s << indent(indent_depth)
      << "auto modulus_square_ = multiply_64(modulus_ , modulus_);\n";
    s << indent(indent_depth) << "auto pmu_ = args_pmu[limb_idx];\n";

    auto &l = limb_blocks_[0];
    s << indent(indent_depth)
      << "constexpr size_t num_outputs = " << l->kernel_outputs_.size()
      << ";\n";
    s << indent(indent_depth)
      << "constexpr size_t num_inputs = " << l->kernel_inputs_.size() << ";\n";
    s << indent(indent_depth)
      << "constexpr size_t num_scalars = " << l->kernel_inputs_scalar_.size()
      << ";\n";

    size_t count = 0;
    for (auto &inp : l->kernel_inputs_) {
      s << indent(indent_depth) << "auto v_" << inp.term()->name() << " = "
        << "args_inputs[num_inputs*j + " << count << "];\n";
      count++;
    }
    count = 0;
    for (auto &inp_sca : l->kernel_inputs_scalar_) {
      s << indent(indent_depth) << "auto r_" << inp_sca.term()->name() << " = "
        << "args_scalars[num_scalars*j + " << count << "];\n";
      count++;
    }
    count = 0;
    for (auto &out : l->kernel_outputs_) {
      s << indent(indent_depth) << "auto v_" << out.term()->name() << " = "
        << "args_outputs[num_outputs*j + " << count << "];\n";
      count++;
    }

    constexpr std::string_view loop_header =
        "{int i = threadIdx.x + blockDim.x * blockIdx.x;";
    constexpr std::string_view loop_end = "}";

    s << indent(indent_depth) << loop_header << "\n";
    indent_depth++;
    for (const auto &v : l->kernel_inputs_) {
      s << indent(indent_depth) << "auto r_" << v.term()->name() << " = "
        << "v_" << v.term()->name() << "[i];\n";
    }

    for (auto &limb_instr : l->instructions_) {
      s << indent(indent_depth) << "//\t" << limb_instr->ppOp() << "\n";
      s << indent(indent_depth) << l->instr_to_cuda(limb_instr) << "\n";
    }

    for (const auto &v : l->kernel_outputs_) {
      if (l->reduction_required(&v)) {
        s << indent(indent_depth) << "v_" << v.term()->name()
          << "[i] = " << "reduce2(r_" << v.term()->name()
          << ",modulus_,barrett_ratio_,barrett_k_);\n";
      } else {
        s << indent(indent_depth) << "v_" << v.term()->name()
          << "[i] = " << "r_" << v.term()->name() << ";\n";
      }
    }

    indent_depth--;
    s << indent(indent_depth) << loop_end << "\n";
  }

  indent_depth--;
  s << indent(indent_depth) << kernel_end << "\n";
  return s;
}

std::ostream &FusedKernel::write_kernel_rsv(std::ostream &s, int indent_depth) {

  std::string kernel_header =
      kernel_signature + "_kernel_rsv_" + std::to_string(kernel_idx_) +
      "(LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const "
      "LimbDataType * factors, const LimbDataType * modulus, const "
      "LimbDataType * barrett_ratio, const uint32_t barrett_k) {";
  constexpr std::string_view kernel_end = "}";

  s << kernel_header << "\n";
  indent_depth++;

  constexpr std::string_view if_header_0 = "if(j == ";
  constexpr std::string_view if_header_1 = ") {";
  constexpr std::string_view if_end = "}";

  std::size_t count = 0;

  s << indent(indent_depth) << "size_t j = blockIdx.y;\n";
  if (!limb_blocks_.empty()) {
    auto &l = limb_blocks_[0];
    s << indent(indent_depth)
      << "constexpr size_t num_outputs = " << l->kernel_outputs_.size()
      << ";\n";
    s << indent(indent_depth)
      << "constexpr size_t num_inputs = " << l->kernel_inputs_.size() << ";\n";
    s << indent(indent_depth)
      << "constexpr size_t num_scalars = " << l->kernel_inputs_scalar_.size()
      << ";\n";

    size_t count = 0;
    for (auto &inp : l->kernel_inputs_) {
      s << indent(indent_depth) << "auto v_" << inp.term()->name() << "_"
        << inp.limb_idx() << " = " << "args_inputs[num_inputs*j + " << count
        << "];\n";
      count++;
    }
    count = 0;
    for (auto &inp_sca : l->kernel_inputs_scalar_) {
      s << indent(indent_depth) << "auto s_" << inp_sca.term()->name() << " = "
        << "args_scalars[num_scalars*j + " << count << "];\n";
      count++;
    }
    count = 0;
    for (auto &out : l->kernel_outputs_) {
      s << indent(indent_depth) << "auto v_" << out.term()->name() << "_"
        << out.limb_idx() << " = " << "args_outputs[num_outputs*j + " << count
        << "];\n";
      count++;
    }

    s << indent(indent_depth) << "auto factors_ = factors;\n";

    constexpr std::string_view loop_header =
        "{int i = threadIdx.x + blockDim.x * blockIdx.x;";
    constexpr std::string_view loop_end = "}";

    s << indent(indent_depth) << loop_header << "\n";
    indent_depth++;
    for (const auto &v : l->kernel_inputs_) {
      s << indent(indent_depth) << "auto r_" << v.term()->name() << "_"
        << v.limb_idx() << " = " << "v_" << v.term()->name() << "_"
        << v.limb_idx() << "[i];\n";
    }

    size_t num_outputs = l->instructions_.size();
    s << indent(indent_depth) << "LimbDataType dest[" << num_outputs << "] = {";
    for (auto i = 0; i < num_outputs - 1; i++) {
      s << "0, ";
    }
    s << "0};\n";

    for (auto &limb_instr : l->instructions_) {
      s << indent(indent_depth) << "//\t" << limb_instr->ppOp() << "\n";
      s << indent(indent_depth) << l->instr_to_cuda(limb_instr) << "\n";
    }

    auto d = 0;
    for (const auto &v : l->kernel_outputs_) {
      s << indent(indent_depth) << "v_" << v.term()->name() << "_"
        << v.limb_idx() << "[i] = " << "dest[" << d++ << "]" << ";\n";
    }

    indent_depth--;
    s << indent(indent_depth) << loop_end << "\n";
  }

  indent_depth--;
  s << indent(indent_depth) << kernel_end << "\n";
  return s;
}

std::ostream &FusedKernel::write_kernel_mod(std::ostream &s, int indent_depth) {

  std::string kernel_header =
      kernel_signature + "_kernel_" + std::to_string(kernel_idx_) +
      "(LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const "
      "LimbDataType * args_scalars, const LimbDataType ** remapables_base, "
      "const size_t * args_remapables, const uint32_t * bases, const "
      "LimbDataType * modulii, const LimbDataType * barrett_ratios, const "
      "uint32_t * barrett_k, const uint32_t ** rotate_map) {";
  constexpr std::string_view kernel_end = "}";

  s << kernel_header << "\n";
  indent_depth++;

  constexpr std::string_view if_header_0 = "if(j == ";
  constexpr std::string_view if_header_1 = ") {";
  constexpr std::string_view if_end = "}";

  std::size_t count = 0;
  s << indent(indent_depth) << "size_t j = blockIdx.y;\n";
  if (!limb_blocks_.empty()) {

    s << indent(indent_depth) << "auto limb_idx = ldu_global_u32(&bases[j]);\n";
    s << indent(indent_depth)
      << "auto modulus_ = ldu_global_u32(&modulii[limb_idx]);\n";
    s << indent(indent_depth)
      << "auto barrett_ratio_ = ldu_global_u32(&barrett_ratios[limb_idx]);\n";
    s << indent(indent_depth)
      << "auto barrett_k_ = ldu_global_u32(&barrett_k[limb_idx]);\n";

    auto &l = limb_blocks_[0];
    s << indent(indent_depth)
      << "constexpr size_t num_outputs = " << l->kernel_outputs_.size()
      << ";\n";
    s << indent(indent_depth)
      << "constexpr size_t num_inputs = " << l->kernel_inputs_.size() << ";\n";
    s << indent(indent_depth)
      << "constexpr size_t num_scalars = " << l->kernel_inputs_scalar_.size()
      << ";\n";

    size_t count = 0;
    for (auto &inp : l->kernel_inputs_) {
      s << indent(indent_depth) << "auto v_" << inp.term()->name() << "_"
        << inp.limb_idx() << " = "
        << "ldu_global_u32_ptr(&args_inputs[num_inputs*j + " << count
        << "]);\n";
      count++;
    }
    count = 0;
    for (auto &out : l->kernel_outputs_) {
      s << indent(indent_depth) << "auto v_" << out.term()->name() << " = "
        << "ldu_global_u32_ptr(&args_outputs[num_outputs*j + " << count
        << "]);\n";
      count++;
    }

    constexpr std::string_view loop_header =
        "{int i = threadIdx.x + blockDim.x * blockIdx.x;";
    constexpr std::string_view loop_end = "}";

    s << indent(indent_depth) << loop_header << "\n";
    indent_depth++;
    for (const auto &v : l->kernel_inputs_) {
      s << indent(indent_depth) << "auto r_" << v.term()->name() << "_"
        // << v.limb_idx() << " = " << "ld_global_u32(&v_" << v.term()->name()
        << v.limb_idx() << " = " << "ld_global_u32_evict_last(&v_" << v.term()->name()
        << "_" << v.limb_idx() << "[i]);\n";
    }

    for (auto &limb_instr : l->instructions_) {
      s << indent(indent_depth) << "//\t" << limb_instr->ppOp() << "\n";
      s << indent(indent_depth) << l->instr_to_cuda(limb_instr) << "\n";
    }

    for (const auto &v : l->kernel_outputs_) {
      if (l->reduction_required(&v)) {
        s << indent(indent_depth) << "st_na_global_utt(&v_" << v.term()->name()
          << "[i], " << "reduce2(r_" << v.term()->name()
          << ",modulus_,barrett_ratio_,barrett_k_));\n";

      } else {
        s << indent(indent_depth) << "st_na_global_u32(&v_" << v.term()->name()
          << "[i], " << "r_" << v.term()->name() << ");\n";
      }
    }

    indent_depth--;
    s << indent(indent_depth) << loop_end << "\n";
  }

  indent_depth--;
  s << indent(indent_depth) << kernel_end << "\n";
  return s;
}

std::ostream &FusedKernel::write_kernel_bco(std::ostream &s, int indent_depth) {

  std::string kernel_header =
      kernel_signature + "_kernel_bco_" + std::to_string(kernel_idx_) +
      "(LimbDataType ** args_outputs, const size_t * num_output_bases, const "
      "LimbDataType ** args_inputs, const size_t * num_input_bases, const "
      "uint32_t ** base_conversion_factors, const LimbDataType ** "
      "output_modulii, const LimbDataType ** barrett_ratios, const uint32_t ** "
      "barrett_k) {";
  constexpr std::string_view kernel_end = "}";

  s << kernel_header << "\n";
  indent_depth++;

  std::size_t count = 0;
  s << indent(indent_depth) << "size_t j = blockIdx.y;\n";
  if (!limb_blocks_.empty()) {
    auto &l = limb_blocks_[0];
    size_t count = 0;
    for (auto &limb_instr : l->instructions_) {
      auto NUM_INPUT_BASES = limb_instr->polynomial_srcs()[0]->num_limbs();
      auto NUM_OUTPUT_BASES = limb_instr->polynomial_dests()[0]->num_limbs();
      s << indent(indent_depth) << "//\t" << limb_instr->ppOp() << "\n";
      if (NUM_INPUT_BASES == 1) {
        s << indent(indent_depth) << "convert1<" << NUM_OUTPUT_BASES
          << ">(args_inputs[j], num_input_bases[j], args_outputs[j], "
             "base_conversion_factors[j], output_modulii[j], "
             "barrett_ratios[j], barrett_k[j], num_output_bases[j], LENGTH);"
          << "\n";
      } else {
        s << indent(indent_depth) << "convert<" << NUM_INPUT_BASES << ", "
          << NUM_OUTPUT_BASES
          << ">(args_inputs[j], num_input_bases[j], args_outputs[j], "
             "base_conversion_factors[j], output_modulii[j], "
             "barrett_ratios[j], barrett_k[j], num_output_bases[j], LENGTH);"
          << "\n";
      }
    }

    indent_depth--;
  }

  indent_depth--;
  s << indent(indent_depth) << kernel_end << "\n";
  return s;
}

std::ostream &FusedKernelLimb::write_rsv(std::ostream &s, int indent_depth,
                                         bool init) {

  for (auto &limb_instr : instructions_) {
    s << indent(indent_depth) << "//\t" << limb_instr->ppOp() << "\n";
    s << indent(indent_depth) << instr_to_cuda(limb_instr) << "\n";
  }

  return s;
}

} // namespace Backend
} // namespace Cerium
