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

#pragma once

#include "fmt/core.h"
#include <memory>
#include <vector>

#include "cerium/compiler/backend/limb.h"
#include "cerium/compiler/backend/terms.h"

namespace Cerium {
namespace Backend {

// Limb IR Instruction in Cerium
class LimbInstruction {
public:
  enum OpCode {
    Add,
    Sub,
    Neg,
    MuP, // Plaintext multiply
    Mul,
    Rot,
    Con,
    Ntt,
    Int,
    Bco,
    Div,
    SuD,  // subtract and divide by the polynomial base
    SuD2, // subtract and divide by the polynomial base
    Msd,  // Modulo, subtract and divide by the polynomial base
    Pmu,  // multiply by the polynomial base
    Dis,  // Distrbute globally
    Rcv,  // Receive from Global Distributor
    Drm,  // Distibute/Receive and Move
    Ags,
    Ard,
    Inp, // Input
    Mov, // Move
    Rec, // Recieve
    Snd, // Recieve
    Rsv,
    Mod,
    RsM, // Resolve Mod
    Mad,
    Call

  };

  virtual std::vector<Limb *> dests() = 0;
  virtual std::vector<Limb *> srcs() = 0;
  virtual std::vector<Polynomial *> polynomial_dests() = 0;
  virtual std::vector<Polynomial *> polynomial_srcs() = 0;
  virtual std::string ppOp() const = 0;

  LimbInstruction::OpCode opcode() const { return op_; };
  virtual bool is_pl_instruction() const { return false; }
  virtual bool is_join_instruction() const { return false; }
  LimbIndexType limb_idx() const { return limb_; }

  virtual Polynomial *base_conversion_dest() { return nullptr; }

protected:
  LimbInstruction(OpCode op, const LimbIndexType limb)
      : op_(op) /*, start_at_time_(-1)*/, limb_(limb){};
  LimbInstruction(){};
  OpCode op_;
  LimbIndexType limb_;
};

class InpLimbInstruction : public LimbInstruction {
  Limb dest_;

public:
  InpLimbInstruction(const Limb &dest, const LimbIndexType limb)
      : dest_(dest), LimbInstruction(OpCode::Inp, limb) {
    assert(dest_.is_input());
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>(); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "inp";
    s << " ";
    s << dest_;
    return s.str();
  }
};
class ReceiveInputLimbInstruction : public LimbInstruction {
  Limb dest_;

public:
  ReceiveInputLimbInstruction(const Limb &dest, const LimbIndexType limb)
      : dest_(dest), LimbInstruction(OpCode::Rec, limb) {
    assert(dest_.is_receive());
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>(); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "rec";
    s << " ";
    s << dest_;
    return s.str();
  }
};

class SendLimbInstruction : public LimbInstruction {
  Limb src_;

public:
  SendLimbInstruction(const Limb &src, const LimbIndexType limb)
      : src_(src), LimbInstruction(OpCode::Snd, limb) {
    assert(src_.is_send());
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>(); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({&src_}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "snd : ";
    s << src_;
    return s.str();
  }
};

class UnOpLimbInstruction : public LimbInstruction {
  Limb dest_;
  Limb src1_;
  int32_t rot_idx_;

public:
  UnOpLimbInstruction(OpCode op, const Limb &dest, const Limb &src1,
                      const LimbIndexType limb)
      : dest_(dest), src1_(src1), rot_idx_(0), LimbInstruction(op, limb) {
    switch (op) {
    case OpCode::Neg:
      break;
    case OpCode::Int:
      break;
    case OpCode::Ntt:
      break;
    case OpCode::Mov:
      break;
    case OpCode::Con:
      break;
    default:
      throw std::runtime_error("Invalid OpCode for Unary op: " + op);
    }
  }

  UnOpLimbInstruction(OpCode op, const Limb &dest, const Limb &src1,
                      const int32_t rot_idx, const LimbIndexType limb)
      : dest_(dest), src1_(src1), rot_idx_(rot_idx),
        LimbInstruction(OpCode::Rot, limb) {
    switch (op) {
    case OpCode::Rot:
      break;
    default:
      throw std::runtime_error("Invalid OpCode for Unary op: " + op);
    }

    if (rot_idx == 0) {
      op_ = OpCode::Mov;
    }
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({&src1_}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  Limb *dest() { return &dest_; }

  Limb *src() { return &src1_; }

  std::string ppOp() const override {
    std::stringstream s;
    switch (op_) {
    case OpCode::Neg:
      s << "neg";
      break;
    case OpCode::Rot:
      s << "rot " << rot_idx_;
      break;
    case OpCode::Con:
      s << "con " << rot_idx_;
      break;
    case OpCode::Int:
      s << "int";
      break;
    case OpCode::Ntt:
      s << "ntt";
      break;
    case OpCode::Mov:
      s << "mov";
      break;
    }
    s << " " << dest_ << ": " << src1_;
    return s.str();
  }

  auto rotate_idx() const { return rot_idx_; }
};

class BinOpLimbInstruction : public LimbInstruction {
  Limb dest_;
  Limb src1_;
  Limb src2_;

public:
  BinOpLimbInstruction(OpCode op, const Limb &dest, const Limb &src1,
                       const Limb &src2, const LimbIndexType limb)
      : dest_(dest), src1_(src1), src2_(src2), LimbInstruction(op, limb) {
    switch (op) {

    case OpCode::Add:
    case OpCode::Sub:
    case OpCode::MuP:;
    case OpCode::Mul:
    case OpCode::SuD:
      break;
    default:
      throw std::runtime_error("Invalid OpCode for Unary op: " + op);
    }
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override {
    return std::vector<Limb *>({&src1_, &src2_});
  }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    switch (op_) {
    case OpCode::Add:
      s << "add";
      break;
    case OpCode::Sub:
      s << "sub";
      break;
    case OpCode::MuP:
      s << "mup";
      break;
    case OpCode::Mul:
      s << "mul";
      break;
    case OpCode::SuD:
      s << "sud";
      break;
    }
    s << " " << dest_ << ": " << src1_ << ", " << src2_;

    return s.str();
  }
};

class MadLimbInstruction : public LimbInstruction {
  Limb dest_;
  Limb src1_;
  Limb src2_;
  Limb src3_;

public:
  MadLimbInstruction(const Limb &dest, const Limb &src1, const Limb &src2,
                     const Limb &src3, const LimbIndexType limb)
      : dest_(dest), src1_(src1), src2_(src2), src3_(src3),
        LimbInstruction(OpCode::Mad, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override {
    return std::vector<Limb *>({&src1_, &src2_, &src3_});
  }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "mad " << dest_ << ": " << src1_ << ", " << src2_ << ", " << src3_;
    return s.str();
  }
};

class SuDLimbInstruction : public LimbInstruction {
  Limb dest_;
  Limb src1_;
  Limb src2_;
  std::set<LimbIndexType> limbs_to_drop_;

public:
  SuDLimbInstruction(const Limb &dest, const Limb &src1, const Limb &src2,
                     const LimbIndexType limb,
                     const std::set<LimbIndexType> &limbs_to_drop)
      : dest_(dest), src1_(src1), src2_(src2), limbs_to_drop_(limbs_to_drop),
        LimbInstruction(OpCode::SuD, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override {
    return std::vector<Limb *>({&src1_, &src2_});
  }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "sud " << dest_ << ": " << src1_ << ", " << src2_;
    s << " / ";
    for (auto &l : limbs_to_drop_) {
      s << l << ", ";
    }

    return s.str();
  }

  const auto &divide_bases() { return limbs_to_drop_; }
};

class SuDLimbInstruction2 : public LimbInstruction {
  Limb dest_;
  Limb src1_;
  Limb src2_;
  Limb src3_;
  Limb src4_;
  std::set<LimbIndexType> limbs_to_drop_;

public:
  SuDLimbInstruction2(const Limb &dest, const Limb &src1, const Limb &src2,
                      const Limb &src3, const Limb &src4,
                      const LimbIndexType limb,
                      const std::set<LimbIndexType> &limbs_to_drop)
      : dest_(dest), src1_(src1), src2_(src2), src3_(src3), src4_(src4),
        limbs_to_drop_(limbs_to_drop), LimbInstruction(OpCode::SuD2, limb) {

    // Ensure that this is the order of the term and limb indicies. If this
    // order is not followed, mapping the inputs and outputs to sets will be
    // wrong
    if (src1_.term_idx() >= src2_.term_idx()) {
      throw std::runtime_error(
          "src1_ term index must be less than src2_ term index");
    }
    if (src2_.term_idx() != src3_.term_idx()) {
      throw std::runtime_error(
          "src2_ term index must be equal to src3_ term index");
    }
    if (src4_.term_idx() >= src3_.term_idx()) {
      throw std::runtime_error(
          "src4_ term index must be less than src3_ term index");
    }
    if (src2_.limb_idx() >= src3_.limb_idx()) {
      throw std::runtime_error(
          "src2_ limb index must be less than src3_ limb index");
    }
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override {
    return std::vector<Limb *>({&src1_, &src2_, &src3_, &src4_});
  }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "sud " << dest_ << ": " << src1_ << ", " << src2_ << ", " << src3_
      << ", " << src4_;
    s << " / ";
    for (auto &l : limbs_to_drop_) {
      s << l << ", ";
    }

    return s.str();
  }

  const auto &divide_bases() { return limbs_to_drop_; }
};

class MsdLimbInstruction : public LimbInstruction {
  Limb dest_;
  Limb src1_;
  Polynomial src2_;
  std::set<LimbIndexType> limbs_to_drop_;

public:
  MsdLimbInstruction(const Limb &dest, const Limb &src1, const Polynomial &src2,
                     const LimbIndexType limb,
                     const std::set<LimbIndexType> &limbs_to_drop)
      : dest_(dest), src1_(src1), src2_(src2), limbs_to_drop_(limbs_to_drop),
        LimbInstruction(OpCode::Msd, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({&src1_}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>({&src2_});
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "msd " << dest_ << ": " << src1_ << ", " << src2_;
    s << " / ";
    for (auto &l : limbs_to_drop_) {
      s << l << ", ";
    }

    return s.str();
  }

  const auto &divide_bases() { return limbs_to_drop_; }
};

class PmuLimbInstruction : public LimbInstruction {
  Limb dest_;
  Limb src1_;
  Limb src2_;
  std::set<LimbIndexType> limbs_to_drop_;

public:
  PmuLimbInstruction(const Limb &dest, const Limb &src1, const Limb &src2,
                     const LimbIndexType limb,
                     const std::set<LimbIndexType> &limbs_to_drop)
      : dest_(dest), src1_(src1), src2_(src2), limbs_to_drop_(limbs_to_drop),
        LimbInstruction(OpCode::Pmu, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override {
    return std::vector<Limb *>({&src1_, &src2_});
  }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "pmu " << dest_ << ": " << src1_ << ", " << src2_;
    s << " x ";
    for (auto &l : limbs_to_drop_) {
      s << l << ", ";
    }

    return s.str();
  }

  const auto &divide_bases() { return limbs_to_drop_; }
};

class BaseConvLimbInstruction : public LimbInstruction {
  Polynomial bco_dest_;
  Polynomial bco_src_;

public:
  BaseConvLimbInstruction(const Polynomial &bco_dest, const Polynomial &bco_src)
      : bco_dest_(bco_dest), bco_src_(bco_src),
        LimbInstruction(OpCode::Bco, -1) {
    assert(bco_dest_.is_bcor());
    assert(bco_src_.is_bcor());
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>({}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>({&bco_dest_});
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>({&bco_src_});
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "bco " << bco_dest_ << ": " << bco_src_;
    return s.str();
  }

  bool is_pl_instruction() const override { return false; }

  Polynomial *base_conversion_dest() override { return &bco_dest_; }

  Polynomial *base_conversion_src() { return &bco_src_; }
};

class DistRecvLimbInstruction : public LimbInstruction {
  Polynomial dest_;
  std::vector<Polynomial> srcs_;
  size_t sync_size_;
  size_t dist_idx_;

public:
  DistRecvLimbInstruction(const Polynomial &dest,
                          const std::vector<Polynomial> &srcs,
                          const size_t sync_size, const size_t dist_idx)
      : dest_(dest), srcs_(srcs), sync_size_(sync_size), dist_idx_(dist_idx),
        LimbInstruction(OpCode::Drm, -1) {
    assert(srcs.size() == sync_size);
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>({}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>({&dest_});
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>({&srcs_[dist_idx_]});
  }

  std::vector<Polynomial *> all_srcs() {
    std::vector<Polynomial *> all;
    for (auto &s : srcs_) {
      all.push_back(&s);
    }
    return all;
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "drm " << sync_size_ << "(" << dist_idx_ << ")" << dest_ << ": ";
    for (auto &src : srcs_) {
      s << src << ", ";
    }
    return s.str();
  }

  size_t sync_size() const { return sync_size_; }

  bool is_pl_instruction() const override { return false; }
};

class AggregateScatterLimbInstruction : public LimbInstruction {
  std::vector<Polynomial> dests_;
  Polynomial src_;
  size_t sync_size_;
  size_t agg_idx_;

public:
  AggregateScatterLimbInstruction(const std::vector<Polynomial> &dests,
                                  const Polynomial &src, const size_t sync_size,
                                  const size_t agg_idx)
      : dests_(dests), src_(src), sync_size_(sync_size), agg_idx_(agg_idx),
        LimbInstruction(OpCode::Ags, -1) {
    assert(dests.size() == sync_size);
    assert(agg_idx_ < sync_size_);
  }

  std::vector<Limb *> dests() override { return std::vector<Limb *>({}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>({&dests_[agg_idx_]});
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>({&src_});
  }

  std::vector<Polynomial *> all_dests() {
    std::vector<Polynomial *> all;
    for (auto &d : dests_) {
      all.push_back(&d);
    }
    return all;
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "ags " << sync_size_ << "(" << agg_idx_ << ") ";
    for (auto &d : dests_) {
      s << d << ", ";
    }
    s << ": " << src_;
    return s.str();
  }

  size_t sync_size() const { return sync_size_; }

  bool is_pl_instruction() const override { return false; }
};

class AllReduceLimbInstruction : public LimbInstruction {
  Polynomial dest_;
  Polynomial src_;
  size_t sync_size_;
  size_t agg_idx_;

public:
  AllReduceLimbInstruction(const Polynomial &dest, const Polynomial &src,
                           const size_t sync_size, const size_t agg_idx)
      : dest_(dest), src_(src), sync_size_(sync_size), agg_idx_(agg_idx),
        LimbInstruction(OpCode::Ard, -1) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>({&dest_});
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>({&src_});
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "ard " << sync_size_ << "(" << agg_idx_ << ") ";
    s << dest_ << ", ";
    s << ": " << src_;
    return s.str();
  }

  size_t sync_size() const { return sync_size_; }

  bool is_pl_instruction() const override { return false; }
};

class RecvLimbInstruction : public LimbInstruction {
  Limb dest_;
  uint64_t sync_id_;
  uint64_t sync_size_;

public:
  RecvLimbInstruction(const Limb &dest, const LimbIndexType limb,
                      const uint64_t sync_id, const uint64_t sync_size)
      : dest_(dest), sync_id_((sync_id << 16) + limb), sync_size_(sync_size),
        LimbInstruction(OpCode::Rcv, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "rcv @ " << sync_id_ << ":" << sync_size_ << " " << dest_ << ": ";
    return s.str();
  }
};

class DistLimbInstruction : public LimbInstruction {
  Limb src1_;
  uint64_t sync_id_;
  uint64_t sync_size_;

public:
  DistLimbInstruction(const Limb &src1, const LimbIndexType limb,
                      const uint64_t sync_id, const uint64_t sync_size)
      : src1_(src1), sync_id_((sync_id << 16) + limb), sync_size_(sync_size),
        LimbInstruction(OpCode::Dis, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({&src1_}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "dis @ " << sync_id_ << ":" << sync_size_ << " " << ": " << src1_;
    return s.str();
  }
};

class ResolveLimbInstruction : public LimbInstruction {
  Polynomial dest_;
  Limb src1_;

public:
  ResolveLimbInstruction(const Polynomial &dest, const Limb &src1,
                         const LimbIndexType limb)
      : dest_(dest), src1_(src1), LimbInstruction(OpCode::Rsv, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({&src1_}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>({&dest_});
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>();
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "rsv " << dest_ << ": " << src1_;
    return s.str();
  }

  bool is_pl_instruction() const override { return false; }

  Polynomial *base_conversion_dest() override { return nullptr; }
};

class ModLimbInstruction : public LimbInstruction {
  Limb dest_;
  Polynomial src1_;

public:
  ModLimbInstruction(const Limb &dest, const Polynomial &src1,
                     const LimbIndexType limb)
      : dest_(dest), src1_(src1), LimbInstruction(OpCode::Mod, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>({&src1_});
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "mod " << dest_ << ": " << src1_;
    return s.str();
  }

  bool is_pl_instruction() const override { return false; }

  Polynomial *base_conversion_dest() override { return nullptr; }
};

class ResolveModLimbInstruction : public LimbInstruction {
  Limb dest_;
  Polynomial src1_;

public:
  ResolveModLimbInstruction(const Limb &dest, const Polynomial &src1,
                            const LimbIndexType limb)
      : dest_(dest), src1_(src1), LimbInstruction(OpCode::RsM, limb) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({&dest_}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({}); }

  std::vector<Polynomial *> polynomial_dests() override {
    return std::vector<Polynomial *>();
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    return std::vector<Polynomial *>({&src1_});
  }

  std::string ppOp() const override {
    std::stringstream s;
    s << "rsm " << dest_ << ": " << src1_;
    return s.str();
  }

  bool is_pl_instruction() const override { return false; }

  Polynomial *base_conversion_dest() override { return nullptr; }
};

class CallLimbInstruction : public LimbInstruction {
  std::unordered_map<std::string, Backend::Polynomial> dests_;
  std::unordered_map<std::string, Backend::Polynomial> srcs_;

  std::string function_name_;

public:
  CallLimbInstruction(
      const std::string &function_name,
      std::unordered_map<std::string, Backend::Polynomial> dests,
      std::unordered_map<std::string, Backend::Polynomial> srcs)
      : function_name_(function_name), dests_(dests), srcs_(srcs),
        LimbInstruction(OpCode::Call, -1) {}

  std::vector<Limb *> dests() override { return std::vector<Limb *>({}); }

  std::vector<Limb *> srcs() override { return std::vector<Limb *>({}); }

  std::vector<Polynomial *> polynomial_dests() override {
    std::vector<Polynomial *> dests__;
    for (auto &[k, v] : dests_) {
      dests__.push_back(&v);
    }
    return dests__;
  }

  std::vector<Polynomial *> polynomial_srcs() override {
    std::vector<Polynomial *> srcs__;
    for (auto &[k, v] : srcs_) {
      srcs__.push_back(&v);
    }
    return srcs__;
  }

  auto &dests_map() { return dests_; }
  auto &srcs_map() { return srcs_; }

  std::string ppOp() const override {
    std::stringstream s;
    s << "call " << function_name_ << " ";
    for (auto &[k, v] : dests_) {
      s << v << ", ";
    }
    for (size_t i = 0; i < dests_.size(); i++) {
    }
    s << " : ";
    for (auto &[k, v] : srcs_) {
      s << v << ", ";
    }
    return s.str();
  }

  bool is_pl_instruction() const override { return false; }

  const std::string &function_name() const { return function_name_; }
};

using TermLimbIndex = std::pair<TermIndexType, LimbIndexType>;
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

} // namespace Backend
} // namespace Cerium

template <> struct fmt::formatter<Cerium::Backend::LimbInstruction> {

  constexpr auto
  parse(format_parse_context &ctx) -> format_parse_context::iterator {

    // Parse the presentation format and store it in the formatter:
    auto it = ctx.begin(), end = ctx.end();

    // Check if reached the end of the range:
    if (it != end && *it != '}') ctx.on_error("invalid format");

    // Return an iterator past the end of the parsed range:
    return it;
  }

  auto format(const Cerium::Backend::LimbInstruction &instr,
              format_context &ctx) const -> format_context::iterator {
    std::stringstream s;
    s << instr.ppOp();
    return fmt::format_to(ctx.out(), "{}", s.str());
  }
};
