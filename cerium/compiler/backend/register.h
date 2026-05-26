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

#include <map>
#include <memory>
#include <optional>

#include "cerium/compiler/backend/limb.h"

namespace Cerium {
namespace Backend {

class Register {
public:
  enum RegType { None, Vector, Scalar, Bcu };

private:
  uint32_t id_;
  uint32_t bcu_id_;
  RegType reg_type_;
  bool is_dead_;
  friend std::ostream &operator<<(std::ostream &s, const Register &reg);

public:
  Register()
      : id_(-1), bcu_id_(-1), reg_type_(RegType::None), is_dead_(false){};
  Register(uint32_t id, RegType type)
      : id_(id), bcu_id_(-1), reg_type_(type), is_dead_(false) {
    assert(reg_type_ == RegType::Vector || reg_type_ == RegType::Scalar);
  };
  Register(uint32_t bcu_id, uint32_t id)
      : id_(id), bcu_id_(bcu_id), reg_type_(RegType::Bcu), is_dead_(false){};

  uint32_t id() const { return id_; }

  RegType reg_type() const { return reg_type_; }

  bool is_bcor() const { return reg_type_ == RegType::Bcu; }

  void mark_dead() { is_dead_ = true; }

  bool is_dead() const { return is_dead_; }
};

struct RegisterHash {
  std::size_t operator()(const Register &reg) const {
    uint32_t hash_id = (reg.reg_type() << 28) | reg.id();
    return std::hash<uint32_t>()(hash_id);
  }
};

struct RegisterCompare {
  bool operator()(const Register &lhs, const Register &rhs) const {
    return lhs.id() == rhs.id() && lhs.reg_type() == rhs.reg_type();
  }
};

std::ostream &operator<<(std::ostream &s, const Register &reg);

class BaseConversionUnit {
  uint32_t id_;
  std::shared_ptr<std::unordered_map<LimbIndexType, Register>>
      converted_bases_map_;
  std::shared_ptr<uint32_t> writes_;
  std::shared_ptr<std::set<LimbIndexType>> source_base_indices_;
  std::shared_ptr<std::set<LimbIndexType>> dest_base_indices_;
  friend std::ostream &operator<<(std::ostream &s,
                                  const BaseConversionUnit &bcu);

public:
  BaseConversionUnit()
      : id_(-1), writes_(nullptr), converted_bases_map_(nullptr) {}
  BaseConversionUnit(uint32_t id)
      : id_(id), writes_(nullptr), converted_bases_map_(nullptr) {}

  void initialise(const std::set<LimbIndexType> &dest_base_indices,
                  const std::set<LimbIndexType> &dest_shares) {
    writes_ = std::make_shared<uint32_t>(0);
    converted_bases_map_ =
        std::make_shared<std::unordered_map<LimbIndexType, Register>>();
    source_base_indices_ = std::make_shared<std::set<LimbIndexType>>();
    dest_base_indices_ = std::make_shared<std::set<LimbIndexType>>();
    int i = 0;
    for (auto &base : dest_base_indices) {
      (*converted_bases_map_)[base] = Register(id_, i);
      i++;
    }
    // *source_base_indices_ = source_base_indices;
  }

  Register read_register(const Limb &limb) const {
    auto limb_idx = limb.limb_idx();
    auto reg = converted_bases_map_->at(limb_idx);
    dest_base_indices_->insert(limb_idx);
    return reg;
  }

  void free_register(const Limb &limb) {
    auto limb_idx = limb.limb_idx();
    converted_bases_map_->erase(limb_idx);
  }

  bool is_free() { return converted_bases_map_->empty(); }

  void write(const LimbIndexType limb_index) {
    *writes_ = *writes_ + 1;
    source_base_indices_->insert(limb_index);
  }

  void clear() {
    writes_ = nullptr;
    converted_bases_map_ = nullptr;
    source_base_indices_ = nullptr;
    dest_base_indices_ = nullptr;
  }

  friend class BaseConversionUnitInitialiseISAInstruction;
};

std::ostream &operator<<(std::ostream &s, const BaseConversionUnit &bcu);

class PolynomialRegisterGroup {
  std::vector<Register> registers_;
  uint64_t next_use_;
  // std::shared_ptr<std::set<LimbIndexType>> source_base_indices_;
  friend std::ostream &operator<<(std::ostream &s,
                                  const BaseConversionUnit &bcu);

public:
  PolynomialRegisterGroup() : next_use_(-1) {}
  PolynomialRegisterGroup(const std::vector<Register> &&registers)
      : registers_(registers), next_use_(-1) {}

  const std::vector<Register> &registers() const { return registers_; }

  std::vector<Register> &registers() { return registers_; }

  void set_next_use(uint64_t next_use) { next_use_ = next_use; }

  friend class ResolveInitialiseISAInstruction;
};

std::ostream &operator<<(std::ostream &s, const PolynomialRegisterGroup &prg);

} // namespace Backend
} // namespace Cerium