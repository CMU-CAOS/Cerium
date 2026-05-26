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

#include "cerium/compiler/backend/limb_instruction.h"
#include "cerium/compiler/backend/terms.h"
#include <memory>

namespace Cerium {
namespace Backend {

class KernelGroup;
class KernelInstruction {

public:
  using OpCode = LimbInstruction::OpCode;

  KernelInstruction::OpCode opcode() const { return op_; };
  void set_kg(KernelGroup *kg) { kg_ = kg; }
  KernelGroup *kg() const { return kg_; }

  virtual std::vector<Polynomial> &dests() = 0;
  virtual std::vector<Polynomial> &srcs() = 0;
  virtual std::string ppOp() const = 0;
  virtual std::shared_ptr<KernelInstruction> clone() const = 0;
  virtual void set_limbs(size_t partition_id, size_t num_partitions) = 0;
  virtual bool is_pl_instruction() const { return false; }
  virtual bool is_mod_instruction() const { return false; }
  virtual bool is_join_instruction() const { return false; }
  virtual bool is_move_instruction() const { return false; }
  virtual std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb) const {
    return nullptr;
  };
  virtual size_t num_destinations() const = 0;

  virtual void
  add_operands(const std::shared_ptr<KernelInstruction> &other) = 0;
  virtual bool
  instructions_mergeable(const std::shared_ptr<KernelInstruction> &other) = 0;

  virtual std::vector<std::shared_ptr<KernelInstruction>> split() const {
    return {};
  }

protected:
  KernelInstruction(OpCode op) : op_(op){};
  KernelInstruction(){};
  OpCode op_;

private:
  KernelGroup *kg_ = nullptr;
};

class InputInstruction : public KernelInstruction {
  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  InputInstruction() : KernelInstruction(OpCode::Inp) {}

  void add_operands(const Polynomial &dest) { dests_.push_back(dest); }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  std::string ppOp() const override {
    std::stringstream s;
    s << "inp ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": ";
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<InputInstruction>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    return nullptr;
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    size_t n = dests_.size();
    for (size_t i = 0; i < dests_.size(); i++) {
      auto instruction = std::make_shared<InputInstruction>();
      instruction->add_operands(dests_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class UnOpInstruction : public KernelInstruction {
  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;
  int32_t rot_idx_;

public:
  UnOpInstruction(KernelInstruction::OpCode op) : KernelInstruction(op) {
    switch (op) {
    case OpCode::Neg:
      break;
    case OpCode::Div:
      break;
    case OpCode::Int:
      break;
    case OpCode::Ntt:
      break;
    case OpCode::Con:
      break;
    case OpCode::Rsv:
      break;
    case OpCode::Mod:
      break;
    case OpCode::Mov:
      break;
    default:
      throw std::runtime_error("Invalid OpCode for Unary op: " + op);
    }
  }
  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    if (op_ == OpCode::Rot && rot_idx_ != other->rot_idx_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  std::string ppOp() const override {
    std::stringstream s;
    switch (op_) {
    case OpCode::Neg:
      s << "neg";
      break;
    case OpCode::Div:
      s << "div";
      break;
    case OpCode::Con:
      s << "con";
      break;
    case OpCode::Int:
      s << "int";
      break;
    case OpCode::Ntt:
      s << "ntt";
      break;
    case OpCode::Rsv:
      s << "rsv";
      break;
    case OpCode::Mod:
      s << "mod";
      break;
    case OpCode::Mov:
      s << "mov";
      break;
    }
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<UnOpInstruction>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[i];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }

    auto &src1_limbs = src1.limbs();

    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }

    auto dest_limb = Limb(dest, limb_idx);
    auto src1_limb = Limb(src1, limb_idx);

    if (op_ == OpCode::Rot) {
      return std::make_shared<UnOpLimbInstruction>(op_, dest_limb, src1_limb,
                                                   rot_idx_, limb_idx);
    }
    return std::make_shared<UnOpLimbInstruction>(op_, dest_limb, src1_limb,
                                                 limb_idx);
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    size_t n = dests_.size();
    for (size_t i = 0; i < dests_.size(); i++) {
      auto instruction = std::make_shared<UnOpInstruction>(op_);
      instruction->add_operands(dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class RotationInstruction : public KernelInstruction {
  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;
  std::vector<int32_t> rot_idx_;

public:
  RotationInstruction() : KernelInstruction(OpCode::Rot) {}
  void add_operands(int32_t rot_idx, const Polynomial &dest,
                    const Polynomial &src1) {
    rot_idx_.push_back(rot_idx);
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    rot_idx_.insert(rot_idx_.end(), other->rot_idx_.begin(),
                    other->rot_idx_.end());
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  std::string ppOp() const override {
    std::stringstream s;
    s << "rot ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << rot_idx_[i] << " " << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<RotationInstruction>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[i];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }

    auto &src1_limbs = src1.limbs();

    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }

    auto dest_limb = Limb(dest, limb_idx);
    auto src1_limb = Limb(src1, limb_idx);

    return std::make_shared<UnOpLimbInstruction>(op_, dest_limb, src1_limb,
                                                 rot_idx_[i], limb_idx);
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    size_t n = dests_.size();
    for (size_t i = 0; i < dests_.size(); i++) {
      auto instruction = std::make_shared<RotationInstruction>();
      instruction->add_operands(rot_idx_[i], dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class BinOpInstruction : public KernelInstruction {

  // dests_[i] = srcs_[2*i] op srcs_[2*i + 1];
  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  BinOpInstruction(KernelInstruction::OpCode op) : KernelInstruction(op) {
    switch (op) {
    case OpCode::Add:
    case OpCode::Sub:
    case OpCode::MuP:;
    case OpCode::Mul:
    case OpCode::SuD:
      break;
    default:
      throw std::runtime_error("Invalid OpCode for Binary op: " + op);
    }
  }

  void add_operands(const Polynomial &dest, const Polynomial &src1,
                    const Polynomial &src2) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
    srcs_.push_back(src2);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    auto n1 = dests_.size();
    auto n2 = other->dests_.size();

    if (n1 == 0) {
      return true;
    }

    if (n2 == 0) {
      return true;
    }

    if (op_ == OpCode::SuD) {
      for (auto i = 0; i < n2; i++) {
        auto &src0 = other->srcs_[2 * i];
        auto &src1 = other->srcs_[2 * i + 1];
        if (src0.level() != srcs_[0].level()) {
          return false;
        }
        if (src1.level() != srcs_[1].level()) {
          return false;
        }
      }
    }

    for (auto i = 0; i < n2; i++) {
      auto &src0 = other->srcs_[2 * i];
      auto &src1 = other->srcs_[2 * i + 1];
      if (src0.is_scalar() != srcs_[0].is_scalar()) {
        return false;
      }
      if (src1.is_scalar() != srcs_[1].is_scalar()) {
        return false;
      }

      if (src0.is_plaintext() != srcs_[0].is_plaintext()) {
        return false;
      }
      if (src1.is_plaintext() != srcs_[1].is_plaintext()) {
        return false;
      }

      if (src0.is_plaintext_remapable() != srcs_[0].is_plaintext_remapable()) {
        return false;
      }
      if (src1.is_plaintext_remapable() != srcs_[1].is_plaintext_remapable()) {
        return false;
      }

      if (src0.plaintext_repeat_size() != srcs_[0].plaintext_repeat_size()) {
        return false;
      }
      if (src1.plaintext_repeat_size() != srcs_[1].plaintext_repeat_size()) {
        return false;
      }
      if (src0.limbtype() != srcs_[0].limbtype()) {
        return false;
      }
      if (src1.limbtype() != srcs_[1].limbtype()) {
        return false;
      }
      if (src0.extension_size() != srcs_[0].extension_size()) {
        return false;
      }
      if (src1.extension_size() != srcs_[1].extension_size()) {
        return false;
      }
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  std::string ppOp() const override {
    size_t n = dests_.size();
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
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[2 * i] << ","
        << srcs_[2 * i + 1];
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<BinOpInstruction>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[2 * i];
    auto &src2 = srcs_[2 * i + 1];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }

    auto &src1_limbs = src1.limbs();
    auto &src2_limbs = src2.limbs();

    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }

    if (op_ == OpCode::SuD) {
      auto dest_limb = Limb(dest, limb_idx);
      auto src1_limb = Limb(src1, limb_idx);
      assert(src2_limbs.size() == 1);
      auto src2_limb = Limb(src2, *src2_limbs.begin());
      return std::make_shared<BinOpLimbInstruction>(op_, dest_limb, src1_limb,
                                                    src2_limb, limb_idx);
    }

    if (src2_limbs.find(limb_idx) == src2_limbs.end()) {
      return nullptr;
    }
    auto dest_limb = Limb(dest, limb_idx);
    auto src1_limb = Limb(src1, limb_idx);
    auto src2_limb = Limb(src2, limb_idx);

    return std::make_shared<BinOpLimbInstruction>(op_, dest_limb, src1_limb,
                                                  src2_limb, limb_idx);
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    size_t n = dests_.size();
    for (size_t i = 0; i < dests_.size(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>(op_);
      instruction->add_operands(dests_[i], srcs_[2 * i], srcs_[2 * i + 1]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class InttInstruction : public KernelInstruction {

  // dests_[i] = srcs_[2*i] intt srcs_[2*i + 1];
  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  InttInstruction() : KernelInstruction(OpCode::Int) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "int ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }
  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;

    auto &dest = dests_[i];
    auto &src1 = srcs_[i];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }

    auto &src1_limbs = src1.limbs();
    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }
    auto dest_limb = Limb(dest, limb_idx);
    auto src1_limb = Limb(src1, limb_idx);

    return std::make_shared<UnOpLimbInstruction>(OpCode::Int, dest_limb,
                                                 src1_limb, limb_idx);
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<InttInstruction>(*this);
  }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    size_t n = dests_.size();
    for (size_t i = 0; i < dests_.size(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};
class NttInstruction : public KernelInstruction {

  // dests_[i] = srcs_[2*i] intt srcs_[2*i + 1];
  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  NttInstruction() : KernelInstruction(OpCode::Ntt) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest1, const Polynomial &dest2,
                    const Polynomial &src1, const Polynomial &src2,
                    const Polynomial &src3) {
    dests_.push_back(dest1);
    dests_.push_back(dest2);
    srcs_.push_back(src1);
    srcs_.push_back(src2);
    srcs_.push_back(src3);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "ntt ";
    for (size_t i = 0; i < dests_.size() / 2; i++) {
      s << "\n\t" << dests_[2 * i] << ", " << dests_[2 * i + 1] << ": "
        << srcs_[3 * i] << ", " << srcs_[3 * i + 1] << ", " << srcs_[3 * i + 2];
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<NttInstruction>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size() / 2) {
      return nullptr;
    }
    // auto i = pos * 2;
    auto &dest1 = dests_[2 * pos];
    auto &dest2 = dests_[2 * pos + 1];
    auto &src1 = srcs_[3 * pos];
    auto &src2 = srcs_[3 * pos + 1];
    auto &src3 = srcs_[3 * pos + 2];

    uint8_t dest_id = 0;
    auto &dest1_limbs = dest1.limbs();
    auto &dest2_limbs = dest2.limbs();
    Limb dest_limb;
    if (dest1_limbs.find(limb_idx) != dest1_limbs.end()) {
      dest_id |= 1;
      dest_limb = Limb(dest1, limb_idx);
    }
    if (dest2_limbs.find(limb_idx) != dest2_limbs.end()) {
      dest_id |= 2;
      dest_limb = Limb(dest2, limb_idx);
    }
    if (dest_id == 0) {
      return nullptr;
    }
    assert(dest_id == 1 || dest_id == 2);

    uint8_t src_id = 0;
    auto &src1_limbs = src1.limbs();
    auto &src2_limbs = src2.limbs();
    auto &src3_limbs = src3.limbs();
    std::shared_ptr<LimbInstruction> limb_instruction = nullptr;
    if (src1_limbs.find(limb_idx) != src1_limbs.end()) {
      src_id |= 1;
      auto src_limb = Limb(src1, limb_idx);
      limb_instruction = std::make_shared<UnOpLimbInstruction>(
          OpCode::Ntt, dest_limb, src_limb, limb_idx);
    } else if (src2_limbs.find(limb_idx) != src2_limbs.end()) {
      src_id |= 2;
      auto src_limb = Limb(src2, limb_idx);
      limb_instruction = std::make_shared<UnOpLimbInstruction>(
          OpCode::Mov, dest_limb, src_limb, limb_idx);
    } else if (src3_limbs.find(limb_idx) != src3_limbs.end()) {
      src_id |= 4;
      auto src_limb = Limb(src3, limb_idx);
      limb_instruction = std::make_shared<UnOpLimbInstruction>(
          OpCode::Ntt, dest_limb, src_limb, limb_idx);
    }
    assert(src_id == 1 || src_id == 2 || src_id == 4);
    return limb_instruction;
  }

  size_t num_destinations() const override { return dests_.size() / 2; }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[2 * i], dests_[2 * i + 1], srcs_[3 * i],
                                srcs_[3 * i + 1], srcs_[3 * i + 2]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class NttInstructionSingle : public KernelInstruction {

  // dests_[i] = ntt srcs_[i];
  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  NttInstructionSingle() : KernelInstruction(OpCode::Ntt) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest1, const Polynomial &src1) {
    dests_.push_back(dest1);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "ntt ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<NttInstructionSingle>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;
    auto &dest1 = dests_[i];
    auto &src1 = srcs_[i];

    auto &dest1_limbs = dest1.limbs();
    if (dest1_limbs.find(limb_idx) == dest1_limbs.end()) {
      return nullptr;
    }
    auto &src1_limbs = src1.limbs();
    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }
    auto dest_limb = Limb(dest1, limb_idx);
    auto src_limb = Limb(src1, limb_idx);
    return std::make_shared<UnOpLimbInstruction>(OpCode::Ntt, dest_limb,
                                                 src_limb, limb_idx);
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class BconvInstruction : public KernelInstruction {

  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  BconvInstruction() : KernelInstruction(OpCode::Bco) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    auto n1 = dests_.size();
    auto n2 = other->dests_.size();

    if (n1 == 0) {
      return true;
    }

    if (n2 == 0) {
      return true;
    }

    for (auto i = 0; i < n2; i++) {
      auto &src0 = other->srcs_[i];
      auto &dest0 = other->dests_[i];
      if (src0.level() != srcs_[0].level()) {
        return false;
      }
      if (src0.limbtype() != srcs_[0].limbtype()) {
        return false;
      }
      if (src0.extension_size() != srcs_[0].extension_size()) {
        return false;
      }
      if (dest0.level() != dests_[0].level()) {
        return false;
      }
      if (dest0.limbtype() != dests_[0].limbtype()) {
        return false;
      }
      if (dest0.extension_size() != dests_[0].extension_size()) {
        return false;
      }
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "bco ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    if (limb_idx != 0) {
      return nullptr;
    }
    auto i = pos;

    auto &dest = dests_[i];
    auto &src = srcs_[i];

    return std::make_shared<BaseConvLimbInstruction>(dest, src);
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<BconvInstruction>(*this);
  }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class MadInstruction : public KernelInstruction {

  bool add_{false};
  // if(!add) {
  // dests_[i] = srcs_[2*i] * srcs_[2*i + 1];
  // } else {
  // dests_[i] = srcs_[3*i] * srcs_[3*i + 1] + srcs_[3*i + 2];
  // }
  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  MadInstruction() : add_(false), KernelInstruction(OpCode::Mad) {}
  MadInstruction(bool add) : add_(add), KernelInstruction(OpCode::Mad) {}

  void add_operands(const Polynomial &dest, const Polynomial &src1,
                    const Polynomial &src2) {
    if (add_) {
      throw std::runtime_error(
          "Can only be called for non add type instruction");
    }
    dests_.push_back(dest);
    srcs_.push_back(src1);
    srcs_.push_back(src2);
  }

  void add_operands(const Polynomial &dest, const Polynomial &src1,
                    const Polynomial &src2, const Polynomial &src3) {
    if (!add_) {
      throw std::runtime_error("Can only be called for add type instruction");
    }
    dests_.push_back(dest);
    srcs_.push_back(src1);
    srcs_.push_back(src2);
    srcs_.push_back(src3);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    if (add_ != other->add_) {
      return false;
    }
    auto n1 = dests_.size();
    auto n2 = other->dests_.size();

    if (n1 == 0) {
      return true;
    }

    if (add_) {
      for (auto i = 0; i < n2; i++) {
        auto &src0 = other->srcs_[2 * i];
        auto &src1 = other->srcs_[2 * i + 1];
        if (src0.is_scalar() != srcs_[0].is_scalar()) {
          return false;
        }
        if (src1.is_scalar() != srcs_[1].is_scalar()) {
          return false;
        }
        if (src0.is_plaintext() != srcs_[0].is_plaintext()) {
          return false;
        }
        if (src1.is_plaintext() != srcs_[1].is_plaintext()) {
          return false;
        }
        if (src0.is_plaintext_remapable() !=
            srcs_[0].is_plaintext_remapable()) {
          return false;
        }
        if (src1.is_plaintext_remapable() !=
            srcs_[1].is_plaintext_remapable()) {
          return false;
        }
        if (src0.plaintext_repeat_size() != srcs_[0].plaintext_repeat_size()) {
          return false;
        }
        if (src1.plaintext_repeat_size() != srcs_[1].plaintext_repeat_size()) {
          return false;
        }
        if (src0.limbtype() != srcs_[0].limbtype()) {
          return false;
        }
        if (src1.limbtype() != srcs_[1].limbtype()) {
          return false;
        }
        if (src0.extension_size() != srcs_[0].extension_size()) {
          return false;
        }
        if (src1.extension_size() != srcs_[1].extension_size()) {
          return false;
        }
      }
    } else {
      for (auto i = 0; i < n2; i++) {
        auto &src0 = other->srcs_[3 * i];
        auto &src1 = other->srcs_[3 * i + 1];
        auto &src2 = other->srcs_[3 * i + 2];
        if (src0.is_scalar() != srcs_[0].is_scalar()) {
          return false;
        }
        if (src1.is_scalar() != srcs_[1].is_scalar()) {
          return false;
        }
        if (src2.is_scalar() != srcs_[2].is_scalar()) {
          return false;
        }

        if (src0.is_plaintext() != srcs_[0].is_plaintext()) {
          return false;
        }
        if (src1.is_plaintext() != srcs_[1].is_plaintext()) {
          return false;
        }
        if (src2.is_plaintext() != srcs_[2].is_plaintext()) {
          return false;
        }

        if (src0.is_plaintext_remapable() !=
            srcs_[0].is_plaintext_remapable()) {
          return false;
        }
        if (src1.is_plaintext_remapable() !=
            srcs_[1].is_plaintext_remapable()) {
          return false;
        }
        if (src2.is_plaintext_remapable() !=
            srcs_[2].is_plaintext_remapable()) {
          return false;
        }

        if (src0.plaintext_repeat_size() != srcs_[0].plaintext_repeat_size()) {
          return false;
        }
        if (src1.plaintext_repeat_size() != srcs_[1].plaintext_repeat_size()) {
          return false;
        }
        if (src2.plaintext_repeat_size() != srcs_[2].plaintext_repeat_size()) {
          return false;
        }
        if (src0.limbtype() != srcs_[0].limbtype()) {
          return false;
        }
        if (src1.limbtype() != srcs_[1].limbtype()) {
          return false;
        }
        if (src2.limbtype() != srcs_[2].limbtype()) {
          return false;
        }

        if (src0.extension_size() != srcs_[0].extension_size()) {
          return false;
        }
        if (src1.extension_size() != srcs_[1].extension_size()) {
          return false;
        }
        if (src2.extension_size() != srcs_[2].extension_size()) {
          return false;
        }
      }
    }

    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    assert(add_ == other->add_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  std::string ppOp() const override {
    std::stringstream s;
    s << "mad ";
    if (!add_) {
      for (size_t i = 0; i < dests_.size(); i++) {
        s << "\n\t" << dests_[i] << ": " << srcs_[2 * i] << ","
          << srcs_[2 * i + 1];
      }
    } else {
      for (size_t i = 0; i < dests_.size(); i++) {
        s << "\n\t" << dests_[i] << ": " << srcs_[3 * i] << ","
          << srcs_[3 * i + 1] << ", " << srcs_[3 * i + 2];
      }
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<MadInstruction>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;

    auto &dest = dests_[i];
    auto &src1 = srcs_[3 * i];
    auto &src2 = srcs_[3 * i + 1];
    auto &src3 = srcs_[3 * i + 2];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }

    auto &src1_limbs = src1.limbs();
    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }
    auto &src2_limbs = src2.limbs();
    if (src2_limbs.find(limb_idx) == src2_limbs.end()) {
      return nullptr;
    }
    auto &src3_limbs = src3.limbs();
    if (src3_limbs.find(limb_idx) == src3_limbs.end()) {
      return nullptr;
    }
    auto dest_limb = Limb(dest, limb_idx);
    auto src1_limb = Limb(src1, limb_idx);
    auto src2_limb = Limb(src2, limb_idx);
    auto src3_limb = Limb(src3, limb_idx);

    return std::make_shared<MadLimbInstruction>(dest_limb, src1_limb, src2_limb,
                                                src3_limb, limb_idx);
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>(add_);
      if (!add_) {
        instruction->add_operands(dests_[i], srcs_[2 * i], srcs_[2 * i + 1]);
      } else {
        instruction->add_operands(dests_[i], srcs_[3 * i], srcs_[3 * i + 1],
                                  srcs_[3 * i + 2]);
      }
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class SudInstruction : public KernelInstruction {

  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  SudInstruction() : KernelInstruction(OpCode::SuD) {}

  void add_operands(const Polynomial &dest, const Polynomial &src1,
                    const Polynomial &src2, const Polynomial &src3) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
    srcs_.push_back(src2);
    srcs_.push_back(src3);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    auto n1 = dests_.size();
    auto n2 = other->dests_.size();

    if (n1 == 0) {
      return true;
    }

    if (n2 == 0) {
      return true;
    }

    for (auto i = 0; i < n2; i++) {
      auto &src0 = other->srcs_[3 * i];
      auto &src1 = other->srcs_[3 * i + 1];
      auto &src2 = other->srcs_[3 * i + 2];
      if (src0.limbtype() != srcs_[0].limbtype()) {
        return false;
      }
      if (src1.limbtype() != srcs_[1].limbtype()) {
        return false;
      }
      if (src2.limbtype() != srcs_[2].limbtype()) {
        return false;
      }

      if (src0.level() != srcs_[0].level()) {
        return false;
      }
      if (src1.level() != srcs_[1].level()) {
        return false;
      }
      if (src2.level() != srcs_[2].level()) {
        return false;
      }

      if (src0.shares() != srcs_[0].shares()) {
        return false;
      }
      if (src1.shares() != srcs_[1].shares()) {
        return false;
      }
      if (src2.shares() != srcs_[2].shares()) {
        return false;
      }

      if (src0.extension_size() != srcs_[0].extension_size()) {
        return false;
      }
      if (src1.extension_size() != srcs_[1].extension_size()) {
        return false;
      }
      if (src2.extension_size() != srcs_[2].extension_size()) {
        return false;
      }
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  std::string ppOp() const override {
    std::stringstream s;
    s << "sud ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[3 * i] << ","
        << srcs_[3 * i + 1] << "{ " << srcs_[3 * i + 2] << " }";
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<SudInstruction>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[3 * i];
    auto &src2 = srcs_[3 * i + 1];
    auto &src3 = srcs_[3 * i + 2];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }

    auto &src1_limbs = src1.limbs();
    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }
    auto &src2_limbs = src2.limbs();
    if (src2_limbs.find(limb_idx) == src2_limbs.end()) {
      return nullptr;
    }
    auto dest_limb = Limb(dest, limb_idx);
    auto src1_limb = Limb(src1, limb_idx);
    auto src2_limb = Limb(src2, limb_idx);

    return std::make_shared<SuDLimbInstruction>(dest_limb, src1_limb, src2_limb,
                                                limb_idx, src3.limbs());
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[3 * i], srcs_[3 * i + 1],
                                srcs_[3 * i + 2]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class SudInstruction2 : public KernelInstruction {

  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  SudInstruction2() : KernelInstruction(OpCode::SuD2) {}

  void add_operands(const Polynomial &dest, const Polynomial &src1,
                    const Polynomial &src2, const Polynomial &src3,
                    const Polynomial &src4, const Polynomial &src5) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
    srcs_.push_back(src2);
    srcs_.push_back(src3);
    srcs_.push_back(src4);
    srcs_.push_back(src5);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  std::string ppOp() const override {
    std::stringstream s;
    s << "sud2 ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[5 * i] << ","
        << srcs_[5 * i + 1] << "," << srcs_[5 * i + 2] << ","
        << srcs_[5 * i + 3] << "{ " << srcs_[5 * i + 4] << " }";
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<SudInstruction2>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[5 * i];
    auto &src2 = srcs_[5 * i + 1];
    auto &src3 = srcs_[5 * i + 2];
    auto &src4 = srcs_[5 * i + 3];
    auto &src5 = srcs_[5 * i + 4];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }

    auto &src1_limbs = src1.limbs();
    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }
    auto &src2_limbs = src2.limbs();
    if (src2_limbs.find(limb_idx) == src2_limbs.end()) {
      return nullptr;
    }
    auto dest_limb = Limb(dest, limb_idx);
    auto src1_limb = Limb(src1, limb_idx);
    auto src2_limb = Limb(src2, limb_idx);
    assert(src3.limbs().size() == 1);
    assert(src4.limbs().size() == 1);
    auto src3_limb = Limb(src3, *src3.limbs().begin());
    auto src4_limb = Limb(src4, *src4.limbs().begin());

    return std::make_shared<SuDLimbInstruction2>(
        dest_limb, src1_limb, src2_limb, src3_limb, src4_limb, limb_idx,
        src5.limbs());
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[5 * i], srcs_[5 * i + 1],
                                srcs_[5 * i + 2], srcs_[5 * i + 3],
                                srcs_[5 * i + 4]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class PmuInstruction : public KernelInstruction {

  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> bases_;

public:
  PmuInstruction() : KernelInstruction(OpCode::Pmu) {}

  void add_operands(const Polynomial &dest, const Polynomial &src1,
                    const Polynomial &src2, const Polynomial &src3) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
    srcs_.push_back(src2);
    // srcs_.push_back(src3);
    bases_.push_back(src3);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }

    auto n1 = dests_.size();
    auto n2 = other->dests_.size();

    if (n1 == 0) {
      return true;
    }

    if (n2 == 0) {
      return true;
    }

    for (auto i = 0; i < n2; i++) {
      auto &src0 = other->srcs_[2 * i];
      auto &src1 = other->srcs_[2 * i + 1];
      auto &src2 = other->bases_[i];
      if (src0.limbtype() != srcs_[0].limbtype()) {
        return false;
      }
      if (src1.limbtype() != srcs_[1].limbtype()) {
        return false;
      }
      if (src2.limbtype() != bases_[0].limbtype()) {
        return false;
      }

      if (src0.level() != srcs_[0].level()) {
        return false;
      }
      if (src1.level() != srcs_[1].level()) {
        return false;
      }
      if (src2.level() != bases_[0].level()) {
        return false;
      }

      if (src0.shares() != srcs_[0].shares()) {
        return false;
      }
      if (src1.shares() != srcs_[1].shares()) {
        return false;
      }
      if (src2.shares() != bases_[0].shares()) {
        return false;
      }
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
    bases_.insert(bases_.end(), other->bases_.begin(), other->bases_.end());
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  std::string ppOp() const override {
    std::stringstream s;
    s << "pmu ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[2 * i] << ", "
        << srcs_[2 * i + 1] << "{ " << bases_[i] << " }";
    }
    return s.str();
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<PmuInstruction>(*this);
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &b : bases_) {
      b.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[2 * i];
    auto &src2 = srcs_[2 * i + 1];
    auto &src3 = bases_[i];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }

    uint8_t both_srcs_have_limb = 0;
    auto &src1_limbs = src1.limbs();
    if (src1_limbs.find(limb_idx) != src1_limbs.end()) {
      both_srcs_have_limb |= 1;
    }
    auto &src2_limbs = src2.limbs();
    if (src2_limbs.find(limb_idx) != src2_limbs.end()) {
      both_srcs_have_limb |= 2;
    }
    auto dest_limb = Limb(dest, limb_idx);
    auto src1_limb = Limb(src1, limb_idx);
    switch (both_srcs_have_limb) {
    case 1: {
      return std::make_shared<UnOpLimbInstruction>(OpCode::Mov, dest_limb,
                                                   src1_limb, limb_idx);
    }; break;
    case 3: {
      auto src2_limb = Limb(src2, limb_idx);
      return std::make_shared<PmuLimbInstruction>(
          dest_limb, src1_limb, src2_limb, limb_idx, src3.limbs());
    }; break;
    default:
      throw std::runtime_error("Invalid instruction Pmu");
    }
    throw std::runtime_error("Invalid instruction Pmu");
    return nullptr;
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[2 * i], srcs_[2 * i + 1],
                                bases_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class ResolveModInstruction : public KernelInstruction {

  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  ResolveModInstruction() : KernelInstruction(OpCode::RsM) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "rsm ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    if (pos >= dests_.size()) {
      return nullptr;
    }

    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[i];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }
    auto dest_limb = Limb(dest, limb_idx);
    return std::make_shared<ResolveModLimbInstruction>(dest_limb, src1,
                                                       limb_idx);
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<ResolveModInstruction>(*this);
  }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class ModInstruction : public KernelInstruction {

  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  ModInstruction() : KernelInstruction(OpCode::Mod) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "mod ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    if (pos >= dests_.size()) {
      return nullptr;
    }

    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[i];

    auto &dest_limbs = dest.limbs();
    if (dest_limbs.find(limb_idx) == dest_limbs.end()) {
      return nullptr;
    }
    auto dest_limb = Limb(dest, limb_idx);
    return std::make_shared<ModLimbInstruction>(dest_limb, src1, limb_idx);
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<ModInstruction>(*this);
  }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class ReceiveInstruction : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  PartitionInfo dest_partition_;
  PartitionInfo src_partition_;

public:
  ReceiveInstruction(const PartitionInfo &dest_partition,
                     const PartitionInfo &src_partition)
      : dest_partition_(dest_partition), src_partition_(src_partition),
        KernelInstruction(OpCode::Drm) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    return false;
#if 0
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if(!other) {
      return false;
    }
    if(op_ != other->op_) {
      return false;
    }
    if(dest_partition_.partition_size != other->dest_partition_.partition_size) {
      return false;
    }
    if(dest_partition_.partition_id != other->dest_partition_.partition_id) {
      return false;
    }
    if(src_partition_.partition_size != other->src_partition_.partition_size) {
      return false;
    }
    if(src_partition_.partition_id != other->src_partition_.partition_id) {
      return false;
    }
    return true;
#endif
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "rec[" << dest_partition_.partition_size << ":"
      << dest_partition_.partition_id << "][" << src_partition_.partition_size
      << ":" << src_partition_.partition_id << "]";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  const PartitionInfo &src_partition() const { return src_partition_; }

  const PartitionInfo &dest_partition() const { return dest_partition_; }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    return nullptr;
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<ReceiveInstruction>(*this);
  }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction =
          std::make_shared<SELF_TYPE>(dest_partition_, src_partition_);
      instruction->add_operands(dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class ReceiveInstruction2 : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  PartitionInfo dest_partition_;
  std::vector<uint32_t> dest_partition_ids_;
  PartitionInfo src_partition_;

public:
  ReceiveInstruction2(const PartitionInfo &dest_partition,
                      const std::vector<uint32_t> &dest_partition_ids,
                      const PartitionInfo &src_partition)
      : dest_partition_(dest_partition),
        dest_partition_ids_(dest_partition_ids), src_partition_(src_partition),
        KernelInstruction(OpCode::Drm) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    return false;
#if 0
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if(!other) {
      return false;
    }
    if(op_ != other->op_) {
      return false;
    }
    if(src_partition_.partition_size != other->src_partition_.partition_size) {
      return false;
    }
    if(src_partition_.partition_id != other->src_partition_.partition_id) {
      return false;
    }
    if(dest_partition_.partition_size != other->dest_partition_.partition_size) {
      return false;
    }
    if(dest_partition_ids_ != other->dest_partition_ids_) {
      return false;
    }
    return true;
#endif
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "rec[" << dest_partition_.partition_size << ":"
      << dest_partition_ids_[0];
    for (int i = 1; i < dest_partition_ids_.size(); i++) {
      s << "," << dest_partition_ids_[i];
    }
    s << "][" << src_partition_.partition_size << ":"
      << src_partition_.partition_id << "]";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  const PartitionInfo &src_partition() const { return src_partition_; }

  const PartitionInfo &dest_partition() const { return dest_partition_; }

  const std::vector<uint32_t> &dest_partition_ids() const {
    return dest_partition_ids_;
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    return nullptr;
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<ReceiveInstruction2>(*this);
  }
};

class DistRecvInstruction : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  PartitionInfo dest_partition_;
  PartitionInfo src_partition_;

public:
  DistRecvInstruction()
      : dest_partition_{0, 0}, src_partition_{0, 0},
        KernelInstruction(OpCode::Drm) {}
  DistRecvInstruction(const PartitionInfo &dest_partition,
                      const PartitionInfo &src_partition)
      : dest_partition_(dest_partition), src_partition_(src_partition),
        KernelInstruction(OpCode::Drm) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    // XXX: Figure out what this false is
    false;
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "dr ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  const PartitionInfo &src_partition() const { return src_partition_; }

  const PartitionInfo &dest_partition() const { return dest_partition_; }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    return nullptr;
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<DistRecvInstruction>(*this);
  }
};

class DistRecvInstruction2 : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  uint64_t sync_id_;
  uint64_t sync_size_;

public:
  DistRecvInstruction2(const uint64_t sync_size, const uint64_t sync_id)
      : sync_size_(sync_size), sync_id_(sync_id),
        KernelInstruction(OpCode::Drm) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest,
                    const std::vector<Polynomial> &srcs) {
    dests_.push_back(dest);
    assert(srcs.size() == sync_size_);
    for (auto &s : srcs) {
      srcs_.push_back(s);
    }
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    if (sync_size_ != other->sync_size_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    assert(sync_size_ == other->sync_size_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "dr " << sync_id_ << "(" << sync_size_ << ")";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": ";
      for (size_t j = i * sync_size_; j < (i + 1) * sync_size_; j++) {
        s << srcs_[j] << ", ";
      }
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    assert(sync_size_ == num_partitions);
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (size_t i = 0; i < dests_.size(); i++) {
      for (size_t j = i * sync_size_; j < (i + 1) * sync_size_; j++) {
        srcs_[j].set_limbs_from_termshare(j % num_partitions, num_partitions);
      }
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    if (limb_idx != 0) {
      return nullptr;
    }

    std::vector<Polynomial> srcs_temp;
    for (size_t i = pos * sync_size_; i < (pos + 1) * sync_size_; i++) {
      srcs_temp.push_back(srcs_[i]);
    }
    return std::make_shared<DistRecvLimbInstruction>(dests_[pos], srcs_temp,
                                                     sync_size_, sync_id_);
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<DistRecvInstruction2>(*this);
  }
};

class AllReduceInstruction : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  uint64_t sync_id_;
  uint64_t sync_size_;

public:
  AllReduceInstruction() : KernelInstruction(OpCode::Ard) {}

  AllReduceInstruction(const uint64_t sync_size, const uint64_t sync_id)
      : sync_size_(sync_size), sync_id_(sync_id),
        KernelInstruction(OpCode::Ard) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    false;
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "ard ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (limb_idx != 0) {
      return nullptr;
    }

    return std::make_shared<AllReduceLimbInstruction>(dests_[pos], srcs_[pos],
                                                      sync_size_, sync_id_);
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<AllReduceInstruction>(*this);
  }
};

class AllReduceInstruction3 : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  uint64_t sync_size_;
  uint64_t src_partition_size_;
  uint64_t num_srcs_per_sync_;

public:
  AllReduceInstruction3(uint64_t sync_size)
      : AllReduceInstruction3(sync_size, 1, sync_size) {}

  AllReduceInstruction3(uint64_t sync_size, uint64_t src_partition_size)
      : AllReduceInstruction3(sync_size, src_partition_size,
                              src_partition_size == 0
                                  ? 0
                                  : sync_size / src_partition_size) {}

  AllReduceInstruction3(uint64_t sync_size, uint64_t src_partition_size,
                        uint64_t num_srcs_per_sync)
      : KernelInstruction(OpCode::Ags), sync_size_(sync_size),
        src_partition_size_(src_partition_size),
        num_srcs_per_sync_(num_srcs_per_sync) {
    if (src_partition_size_ == 0) {
      throw std::runtime_error(
          "AllReduceInstruction3 source partition size must be nonzero");
    }
    if (num_srcs_per_sync_ == 0) {
      throw std::runtime_error(
          "AllReduceInstruction3 source count per sync must be nonzero");
    }
    if (src_partition_size_ * num_srcs_per_sync_ != sync_size_) {
      throw std::runtime_error("AllReduceInstruction3 sync size must match "
                               "source partition size and source count");
    }
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest,
                    const std::vector<Polynomial> &srcs) {
    if (srcs.size() != num_srcs_per_sync_) {
      throw std::runtime_error(
          "AllReduceInstruction3 source count does not match grouped layout");
    }
    dests_.push_back(dest);
    for (auto &s : srcs) {
      srcs_.push_back(s);
    }
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    false;
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    if (sync_size_ != other->sync_size_) {
      return false;
    }
    if (src_partition_size_ != other->src_partition_size_) {
      return false;
    }
    if (num_srcs_per_sync_ != other->num_srcs_per_sync_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    assert(sync_size_ == other->sync_size_);
    assert(src_partition_size_ == other->src_partition_size_);
    assert(num_srcs_per_sync_ == other->num_srcs_per_sync_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "ard[sync_size: " << sync_size_
      << "][src_partition_size: " << src_partition_size_
      << "][num_srcs_per_sync: " << num_srcs_per_sync_ << "]";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": ";
      for (size_t j = 0; j < num_srcs_per_sync_; j++) {
        s << "{" << j << "}" << " "
          << srcs_[i * num_srcs_per_sync_ + j] << ", ";
      }
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    return nullptr;
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<AllReduceInstruction3>(*this);
  }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    size_t n = dests_.size();
    for (size_t i = 0; i < dests_.size(); i++) {
      auto ard_instruction =
          std::make_shared<AllReduceInstruction3>(
              sync_size_, src_partition_size_, num_srcs_per_sync_);
      ard_instruction->add_operands(
          dests_[i],
          std::vector<Polynomial>(
              srcs_.begin() + i * num_srcs_per_sync_,
              srcs_.begin() + (i + 1) * num_srcs_per_sync_));
      res[i] = std::move(ard_instruction);
    }
    return res;
  }

  auto sync_size() const { return sync_size_; }

  auto src_partition_size() const { return src_partition_size_; }

  auto num_srcs_per_sync() const { return num_srcs_per_sync_; }
};

class AggregateScatterInstruction : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  std::vector<uint64_t> sync_id_;
  std::vector<uint64_t> sync_size_;

public:
  AggregateScatterInstruction() : KernelInstruction(OpCode::Ags) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    false;
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "ags ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    return nullptr;
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<AggregateScatterInstruction>(*this);
  }

  std::shared_ptr<KernelInstruction> convert_to_all_reduce_instruction() const {
    auto ard_instruction = std::make_shared<AllReduceInstruction>();
    for (size_t i = 0; i < dests_.size(); i++) {
      ard_instruction->add_operands(dests_[i], srcs_[i]);
    }
    return ard_instruction;
  }
};

class AggregateScatterInstruction3 : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  uint64_t sync_size_;
  uint64_t src_partition_size_;
  uint64_t num_srcs_per_sync_;

public:
  AggregateScatterInstruction3(uint64_t sync_size, uint64_t src_partition_size)
      : KernelInstruction(OpCode::Ags), sync_size_(sync_size),
        src_partition_size_(src_partition_size),
        num_srcs_per_sync_(src_partition_size == 0
                                ? 0
                                : sync_size / src_partition_size) {
    if (src_partition_size_ == 0) {
      throw std::runtime_error("AggregateScatterInstruction3 source partition "
                               "size must be nonzero");
    }
    if (sync_size_ % src_partition_size_ != 0) {
      throw std::runtime_error("AggregateScatterInstruction3 sync size must be "
                               "a multiple of source partition size");
    }
  }

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest,
                    const std::vector<Polynomial> &srcs) {
    if (src_partition_size_ * srcs.size() != sync_size_) {
      throw std::runtime_error(
          "AggregateScatterInstruction3 source count does not match "
          "sync/source partition sizes");
    }
    dests_.push_back(dest);
    for (auto &s : srcs) {
      srcs_.push_back(s);
    }
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    false;
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    if (sync_size_ != other->sync_size_) {
      return false;
    }
    if (src_partition_size_ != other->src_partition_size_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    assert(sync_size_ == other->sync_size_);
    assert(src_partition_size_ == other->src_partition_size_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "ags[sync_size: " << sync_size_
      << "][src_partition_size: " << src_partition_size_ << "]";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": ";
      for (size_t j = 0; j < num_srcs_per_sync_; j++) {
        s << "{" << j << "}" << " "
          << srcs_[i * num_srcs_per_sync_ + j] << ", ";
      }
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    return nullptr;
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<AggregateScatterInstruction3>(*this);
  }

  auto sync_size() const { return sync_size_; }

  auto src_partition_size() const { return src_partition_size_; }

  std::shared_ptr<KernelInstruction> convert_to_all_reduce_instruction() const {
    auto ard_instruction = std::make_shared<AllReduceInstruction3>(
        sync_size_, src_partition_size_, num_srcs_per_sync_);
    for (size_t i = 0; i < dests_.size(); i++) {
      std::vector<Polynomial> srcs_temp(
          srcs_.begin() + i * num_srcs_per_sync_,
          srcs_.begin() + (i + 1) * num_srcs_per_sync_);
      ard_instruction->add_operands(dests_[i], srcs_temp);
    }
    return ard_instruction;
  }
};

class AggregateScatterInstruction2 : public KernelInstruction {
  std::vector<Polynomial> srcs_;
  std::vector<Polynomial> dests_;
  uint64_t sync_id_;
  uint64_t sync_size_;

public:
  AggregateScatterInstruction2(const uint64_t sync_size, const uint64_t sync_id)
      : sync_size_(sync_size), sync_id_(sync_id),
        KernelInstruction(OpCode::Ags) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    if (sync_size_ != other->sync_size_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::vector<Polynomial> &dests,
                    const Polynomial &src) {
    srcs_.push_back(src);
    assert(dests.size() == sync_size_);
    for (auto &d : dests) {
      dests_.push_back(d);
    }
  }
  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    assert(sync_size_ == other->sync_size_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "ags " << sync_id_ << "(" << sync_size_ << ")";
    for (size_t i = 0; i < srcs_.size(); i++) {
      s << "\n\t";
      for (size_t j = i * sync_size_; j < (i + 1) * sync_size_; j++) {
        s << dests_[j] << ", ";
      }
      s << ": " << srcs_[i] << ": ";
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override {
    return dests_.size() / sync_size_;
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {

    if (limb_idx != 0) {
      return nullptr;
    }

    std::vector<Polynomial> dests_temp;
    for (size_t i = pos * sync_size_; i < (pos + 1) * sync_size_; i++) {
      dests_temp.push_back(dests_[i]);
    }
    return std::make_shared<AggregateScatterLimbInstruction>(
        dests_temp, srcs_[pos], sync_size_, sync_id_);
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<AggregateScatterInstruction2>(*this);
  }
};

class ResolveInstruction : public KernelInstruction {

  std::vector<Polynomial> dests_;
  std::vector<Polynomial> srcs_;

public:
  ResolveInstruction() : KernelInstruction(OpCode::Rsv) {}

  std::vector<Polynomial> &dests() override { return dests_; }

  std::vector<Polynomial> &srcs() override { return srcs_; }

  void add_operands(const Polynomial &dest, const Polynomial &src1) {
    dests_.push_back(dest);
    srcs_.push_back(src1);
  }

  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    if (!other) {
      return false;
    }
    if (op_ != other->op_) {
      return false;
    }
    return true;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    assert(instructions_mergeable(other_));
    using SELF_TYPE = std::remove_reference<decltype(*this)>::type;
    auto other = std::dynamic_pointer_cast<SELF_TYPE>(other_);
    assert(other);
    assert(op_ == other->op_);
    dests_.insert(dests_.end(), other->dests_.begin(), other->dests_.end());
    srcs_.insert(srcs_.end(), other->srcs_.begin(), other->srcs_.end());
  }

  std::string ppOp() const override {
    size_t n = dests_.size();
    std::stringstream s;
    s << "rsv ";
    for (size_t i = 0; i < dests_.size(); i++) {
      s << "\n\t" << dests_[i] << ": " << srcs_[i];
    }
    return s.str();
  }

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &d : dests_) {
      d.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &s : srcs_) {
      s.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }

  size_t num_destinations() const override { return dests_.size(); }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    auto i = pos;
    auto &dest = dests_[i];
    auto &src1 = srcs_[i];

    auto &src1_limbs = src1.limbs();
    if (src1_limbs.find(limb_idx) == src1_limbs.end()) {
      return nullptr;
    }
    auto src1_limb = Limb(src1, limb_idx);
    return std::make_shared<ResolveLimbInstruction>(dest, src1_limb, limb_idx);
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<ResolveInstruction>(*this);
  }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    using SELF_TYPE =
        std::remove_const<std::remove_reference<decltype(*this)>::type>::type;
    std::vector<std::shared_ptr<KernelInstruction>> res(num_destinations());
    for (size_t i = 0; i < num_destinations(); i++) {
      auto instruction = std::make_shared<SELF_TYPE>();
      instruction->add_operands(dests_[i], srcs_[i]);
      res[i] = std::move(instruction);
    }
    return res;
  }
};

class CallInstruction : public KernelInstruction {
  std::unordered_map<std::string, Backend::Polynomial> dests_;
  std::unordered_map<std::string, Backend::Polynomial> srcs_;
  std::unordered_map<std::string, Backend::PartitionInfo> args_partition_info_;

  std::vector<Polynomial> dests_vec_;
  std::vector<Polynomial> srcs_vec_;

  std::string function_name;
  std::string remapable_base;

public:
  CallInstruction(
      const std::string &function_name,
      const std::unordered_map<std::string, Backend::Polynomial> &dests,
      const std::unordered_map<std::string, Backend::Polynomial> &srcs,
      const std::unordered_map<std::string, Backend::PartitionInfo>
          &args_partition_info,
      const std::string &remapable_base = "")
      : function_name(function_name), dests_(dests), srcs_(srcs),
        remapable_base(remapable_base),
        args_partition_info_(args_partition_info),
        KernelInstruction(OpCode::Call) {
    for (auto &[k, v] : dests_) {
      dests_vec_.push_back(v);
    }

    for (auto &[k, v] : srcs_) {
      srcs_vec_.push_back(v);
    }
  }

  std::vector<Polynomial> &dests() override { return dests_vec_; }

  std::vector<Polynomial> &srcs() override { return srcs_vec_; }

  auto &dests_map() { return dests_; }

  auto &srcs_map() { return srcs_; }

  auto &args_partition_info() { return args_partition_info_; }

  std::string ppOp() const override {
    std::stringstream s;
    s << "call " << function_name << " ";
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

  void set_limbs(size_t partition_id, size_t num_partitions) override {
    for (auto &[k, v] : dests_) {
      v.set_limbs_from_termshare(partition_id, num_partitions);
    }
    for (auto &[k, v] : srcs_) {
      v.set_limbs_from_termshare(partition_id, num_partitions);
    }
  }
  bool instructions_mergeable(
      const std::shared_ptr<KernelInstruction> &other_) override {
    return false;
  }

  void add_operands(const std::shared_ptr<KernelInstruction> &other_) override {
    throw std::runtime_error("Cannot merge call instructions");
  }

  size_t num_destinations() const override {
    // This is set to 1 since call instructions are not mergeable
    return 1;
  }

  std::shared_ptr<LimbInstruction>
  get_limb_instruction(size_t pos, LimbIndexType limb_idx) const override {
    if (pos >= dests_.size()) {
      return nullptr;
    }
    if (limb_idx != 0) {
      return nullptr;
    }

    if (remapable_base.empty()) {
      return std::make_shared<CallLimbInstruction>(function_name, dests_,
                                                   srcs_);
    } else {
      return std::make_shared<CallLimbInstruction>(
          function_name + "(" + remapable_base + ")", dests_, srcs_);
    }
  }

  std::shared_ptr<KernelInstruction> clone() const override {
    return std::make_shared<CallInstruction>(*this);
  }

  std::vector<std::shared_ptr<KernelInstruction>> split() const override {
    return {clone()};
  }
};

} // namespace Backend
} // namespace Cerium
