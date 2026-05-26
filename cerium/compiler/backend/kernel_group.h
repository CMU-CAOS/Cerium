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

#include "cerium/compiler/backend/kernel_instruction.h"
#include "cerium/compiler/util/environment_variables.h"

#include <algorithm>
#include <iterator>

namespace Cerium {
namespace Backend {

extern uint64_t BLOCK_SIZE;

class KernelGroup {

public:
  using IndexType = uint64_t;
  using SplitType = std::vector<std::set<uint16_t>>;
  enum Type {
    Ewi, // Element Wise
    Bco, // Base Conv
    Int,
    Ntt,
    SuD,
    SuD2,
    Mul,
    Pmu,
    Dist,
    Agg,
    Agg3,
    Rec,
    Rec2,
    Ard,
    Ard3,
    Call,
  };

private:
  IndexType index_;
  std::vector<std::shared_ptr<KernelInstruction>> instructions_;

  bool dont_split_modular_;
  Backend::LimbIndexType level_;
  bool level_initialized_;

  uint8_t partition_size_ = 0;
  uint8_t partition_id_ = 0;
  Type type_;

  std::set<IndexType> parents_;
  std::set<IndexType> children_;

public:
  KernelGroup(uint64_t index, uint8_t partition_size, uint8_t partition_id,
              Type type)
      : index_(index), dont_split_modular_(false), level_(0),
        level_initialized_(false), partition_size_(partition_size),
        partition_id_(partition_id), type_(type){};
  KernelGroup(const std::shared_ptr<KernelGroup> &kg)
      : index_(kg->index_), dont_split_modular_(kg->dont_split_modular_),
        level_(kg->level_), level_initialized_(kg->level_initialized_),
        partition_size_(kg->partition_size_), partition_id_(kg->partition_id_),
        type_(kg->type_){};
  KernelGroup(uint64_t index, const std::shared_ptr<KernelGroup> &kg)
      : index_(index), dont_split_modular_(kg->dont_split_modular_),
        level_(kg->level_), level_initialized_(kg->level_initialized_),
        partition_size_(kg->partition_size_), partition_id_(kg->partition_id_),
        type_(kg->type_){};

  bool is_communication_kg() {
    switch (type_) {
    case Type::Agg:
    case Type::Agg3:
    case Type::Dist:
    case Type::Rec:
    case Type::Rec2:
    case Type::Ard:
    case Type::Ard3:
      return true;
      break;
    default:
      return false;
    }
    return false;
  }

  void clear_instructions() {
    instructions_.clear();
    level_ = 0;
    level_initialized_ = false;
  }

  auto &parents() { return parents_; }

  auto &children() { return children_; }

  void add_parent(std::shared_ptr<KernelGroup> &kg,
                  std::vector<std::shared_ptr<KernelGroup>> &kernel_groups) {
    if ((kg->index_ / BLOCK_SIZE) != (index_ / BLOCK_SIZE)) {
      return;
    }
    if (kg->index_ == index_) {
      return; // Cannot be your own parent and child
    }
    for (auto &c : children_) {
      kernel_groups[c]->parents_.insert(kg->index_);
      kg->children_.insert(c);
    }
    for (auto &p : kg->parents_) {
      kernel_groups[p]->children_.insert(index_);
      parents_.insert(p);
    }
    parents_.insert(kg->index_);
    kg->children_.insert(index_);
  }

  void add_parent_immediate(
      std::shared_ptr<KernelGroup> &kg,
      std::vector<std::shared_ptr<KernelGroup>> &kernel_groups) {
    if (kg->index_ == index_) {
      return; // Cannot be your own parent and child
    }
    parents_.insert(kg->index_);
    kg->children_.insert(index_);
  }

  std::set<IndexType> set_intersection(const std::set<IndexType> &a,
                                       const std::set<IndexType> &b) {
    if (a.empty() || b.empty()) {
      return {};
    }

    auto min_a = *a.begin();
    auto min_b = *b.begin();
    if (min_a > min_b) {
      return set_intersection(b, a);
    }

    auto max_a = *std::prev(a.end());
    auto max_b = *std::prev(b.end());
    if (min_b > max_a || min_a > max_b) {
      return {}; // empty set
    }

    auto it_a = a.begin();
    while (it_a != a.end() && *it_a < min_b) {
      ++it_a;
    }

    auto it_b = std::prev(b.end());
    while (it_b != b.begin() && *it_b > max_a) {
      --it_b;
    }

    std::set<IndexType> intersection;
    std::set_intersection(it_a, a.end(), b.begin(), std::next(it_b),
                          std::inserter(intersection, intersection.begin()));
    return intersection;
  }

  bool are_horizontally_mergeable(std::shared_ptr<KernelGroup> &kg) {

    if ((kg->index_ / BLOCK_SIZE) != (index_ / BLOCK_SIZE)) {
      return false;
    }

    if (kg->type_ != type_) {
      return false;
    }

    if (kg->index_ == index_) {
      return false;
    }

    if (partition_size_ != kg->partition_size_) {
      return false;
    }

    if (partition_id_ != kg->partition_id_) {
      return false;
    }

    if (dont_split_modular_ != kg->dont_split_modular_) {
      return false;
    }

    if (kg->type_ == Type::Ewi || kg->type_ == Type::SuD ||
        kg->type_ == Type::SuD2 || kg->type_ == Type::Bco) {
      if (kg->level_ != level_) {
        return false;
      }
    }

    // Horizontal fusion happens before vertical fusion, so all instruction
    // sizes should be 1
    assert(kg->instructions_.size() == 1 && instructions_.size() == 1);

    auto &kg_instruction = kg->instructions_[0];
    auto &instruction_ = instructions_[0];

    bool instructions_mergeable =
        instruction_->instructions_mergeable(kg_instruction);
    if (!instructions_mergeable) {
      return false;
    }

    if (children_.find(kg->index_) != children_.end()) {
      return false;
    }

    if (parents_.find(kg->index_) != parents_.end()) {
      return false;
    }

    auto intersect1 = set_intersection(parents_, kg->children_);
    if (!intersect1.empty()) {
      return false;
    }

    auto intersect2 = set_intersection(kg->parents_, children_);
    if (!intersect2.empty()) {
      return false;
    }

    return true;
  }

  bool
  horizontal_merge(std::shared_ptr<KernelGroup> &kg,
                   std::vector<std::shared_ptr<KernelGroup>> &kernel_groups) {

    if (!are_horizontally_mergeable(kg)) {
      return false;
    }

    assert(kg->dont_split_modular() == dont_split_modular_);

    auto &kg_instructions = kg->instructions_;

    // Horizontal fusion happens before vertical fusion, so all instruction
    // sizes should be 1
    assert(kg_instructions.size() == 1 && instructions_.size() == 1);

    auto &kg_instruction = kg_instructions[0];
    auto &instruction_ = instructions_[0];

    assert(kg_instruction->opcode() == instruction_->opcode());

    instruction_->add_operands(kg_instruction);

    for (auto &p : kg->parents_) {
      kernel_groups[p]->children_.erase(kg->index_);
      kernel_groups[p]->children_.insert(index_);
    }

    for (auto &c : kg->children_) {
      kernel_groups[c]->parents_.erase(kg->index_);
      kernel_groups[c]->parents_.insert(index_);
    }

    for (auto &p : kg->parents_) {
      for (auto &cc : children_) {
        kernel_groups[p]->children_.insert(cc);
      }
    }

    for (auto &c : kg->children_) {
      for (auto &pp : parents_) {
        kernel_groups[c]->parents_.insert(pp);
      }
    }

    for (auto &p : parents_) {
      for (auto &cc : kg->children_) {
        kernel_groups[p]->children_.insert(cc);
      }
    }

    for (auto &c : children_) {
      for (auto &pp : kg->parents_) {
        kernel_groups[c]->parents_.insert(pp);
      }
    }

    for (auto &c : kg->children_) {
      children_.insert(c);
    }

    for (auto &p : kg->parents_) {
      parents_.insert(p);
    }

    kernel_groups[kg->index_] = kernel_groups[index_];

    return true;
  }

  void refresh_parents_and_children(
      std::vector<std::shared_ptr<KernelGroup>> &kernel_groups) {

    std::set<IndexType> new_parents;
    std::set<IndexType> new_children;

    for (auto &p : parents_) {
      new_parents.insert(kernel_groups[p]->index_);
    }

    for (auto &c : children_) {
      new_children.insert(kernel_groups[c]->index_);
    }

    parents_ = std::move(new_parents);
    children_ = std::move(new_children);
  }

  void set_dont_split_modular(bool val) { dont_split_modular_ = val; }

  auto partition_size() { return partition_size_; }
  auto partition_id() { return partition_id_; }

  void push_instruction(std::shared_ptr<KernelInstruction> &&instr) {
    instr->set_kg(this);
    instructions_.push_back(instr);
    if (type_ == Type::Call) {
      return;
    }
    if (type_ != Type::Ewi) {
      return;
    }
    for (auto &dest : instr->dests()) {
      if (!level_initialized_) {
        level_ = dest.level();
        level_initialized_ = true;
      } else if (level_ != dest.level()) {
        throw std::runtime_error("KernelGroup: Instruction level mismatch");
      }
    }
  }

  IndexType index() { return index_; }

  const std::vector<std::shared_ptr<KernelInstruction>> &instructions() const {
    return instructions_;
  }

  std::vector<std::shared_ptr<KernelInstruction>> &instructions() {
    return instructions_;
  }

  bool empty() const { return instructions_.empty(); }

  bool dont_split_modular() const { return dont_split_modular_; }

  const auto type() { return type_; }

  void set_type(Type type) { type_ = type; }

  std::vector<std::shared_ptr<Backend::LimbInstruction>>
  get_limb_instructions(const Backend::LimbIndexType limb_idx) {
    auto gcd = [](size_t a, size_t b) {
      size_t swap = a;
      a = b;
      b = swap;
      while (b != 0) {
        size_t temp = b;
        b = a % b;
        a = temp;
      }
      return a;
    };

    size_t max_num_destinations = 0;
    size_t gcd_num_destinations = 0;
    for (auto &instr : instructions_) {
      auto num_destinations = instr->num_destinations();
      if (num_destinations > max_num_destinations) {
        max_num_destinations = num_destinations;
      }
      gcd_num_destinations = gcd(num_destinations, gcd_num_destinations);
    }
    std::vector<std::shared_ptr<Backend::LimbInstruction>> ret;
    for (size_t i = 0; i < max_num_destinations; i++) {
      for (auto &instr : instructions_) {
        auto limb_instruction = instr->get_limb_instruction(i, limb_idx);
        if (limb_instruction) {
          ret.push_back(limb_instruction);
        }
      }
    }
    return ret;
  }

  std::shared_ptr<Backend::FusedKernel>
  get_fused_kernel(Backend::KernelFusionContext &context) {
    size_t max_num_destinations = 0;
    size_t gcd_num_destinations = 0;
    std::shared_ptr<Backend::FusedKernel> kernel = nullptr;
    using OpCode = Backend::LimbInstruction::OpCode;
    using KernelType = Backend::FusedKernel::KernelType;
    constexpr size_t INT_KERNEL_FUSE_COUNT = 1024;
    constexpr size_t NTT_KERNEL_FUSE_COUNT = 1024;

    auto gcd = [](size_t a, size_t b) {
      size_t swap = a;
      a = b;
      b = swap;
      while (b != 0) {
        size_t temp = b;
        b = a % b;
        a = temp;
      }
      return a;
    };

    auto update_destination_stats =
        [&](const std::shared_ptr<KernelInstruction> &instr) {
          auto num_destinations = instr->num_destinations();
          max_num_destinations =
              std::max(max_num_destinations, num_destinations);
          gcd_num_destinations = gcd(num_destinations, gcd_num_destinations);
        };

    auto make_kernel = [&](KernelType kernel_type) {
      auto new_kernel = context.create_new_kernel();
      new_kernel->set_kernel_type(kernel_type);
      return new_kernel;
    };

    auto emit_limb_kernels_with_mov = [&](auto &&mov_kernel_for_dest,
                                          auto &&emit) {
      for (size_t i = 0; i < max_num_destinations; ++i) {
        auto mov_kernel = mov_kernel_for_dest(i);
        for (size_t limb_idx = 0; limb_idx < 64; ++limb_idx) {
          auto limb_function = context.create_new_limb_kernel(limb_idx);
          auto mov_limb_function = context.create_new_limb_kernel(limb_idx);
          emit(i, limb_idx, limb_function, mov_limb_function);
          mov_kernel->push_limb_kernel(std::move(mov_limb_function));
          kernel->push_limb_kernel(std::move(limb_function));
        }
      }
    };

    if (instructions_.empty()) {
      return kernel;
    }

    for (auto &instr : instructions_) {
      update_destination_stats(instr);
    }

    const auto opCode = instructions_.front()->opcode();

    if (opCode == OpCode::Int) {
      kernel = make_kernel(KernelType::Int);
      size_t kernel_limb_count = 0;
      for (size_t limb_idx = 0; limb_idx < 64; ++limb_idx) {
        for (size_t i = 0; i < max_num_destinations; ++i) {
          auto limb_function = context.create_new_limb_kernel(limb_idx);
          for (auto &instr : instructions_) {
            auto limb_instruction = instr->get_limb_instruction(i, limb_idx);
            if (limb_instruction) {
              limb_function->push_limb_instruction(std::move(limb_instruction));
              ++kernel_limb_count;
            }
          }
          kernel->push_limb_kernel(std::move(limb_function));
          if (kernel_limb_count > INT_KERNEL_FUSE_COUNT) {
            kernel->complete();
            kernel = make_kernel(KernelType::Int);
            kernel_limb_count = 0;
          }
        }
      }
      kernel->complete();
      return kernel;
    }

    if (opCode == OpCode::SuD || opCode == OpCode::SuD2) {
      const auto kernel_type =
          (opCode == OpCode::SuD2) ? KernelType::SuD2 : KernelType::SuD;
      kernel = make_kernel(kernel_type);
      size_t kernel_limb_count = 0;
      for (size_t limb_idx = 0; limb_idx < 64; ++limb_idx) {
        for (size_t i = 0; i < max_num_destinations; ++i) {
          auto limb_function = context.create_new_limb_kernel(limb_idx);
          for (auto &instr : instructions_) {
            auto limb_instruction = instr->get_limb_instruction(i, limb_idx);
            if (limb_instruction) {
              limb_function->push_limb_instruction(std::move(limb_instruction));
              ++kernel_limb_count;
            }
          }
          kernel->push_limb_kernel(std::move(limb_function));
        }
        if (kernel_limb_count > NTT_KERNEL_FUSE_COUNT) {
          kernel->complete();
          kernel = make_kernel(kernel_type);
          kernel_limb_count = 0;
        }
      }
      kernel->complete();
      return kernel;
    }

    if (opCode == OpCode::Msd) {
      kernel = make_kernel(KernelType::Msd);
      for (size_t i = 0; i < max_num_destinations; ++i) {
        for (size_t limb_idx = 0; limb_idx < 64; ++limb_idx) {
          auto limb_function = context.create_new_limb_kernel(limb_idx);
          for (auto &instr : instructions_) {
            auto limb_instruction = instr->get_limb_instruction(i, limb_idx);
            if (limb_instruction) {
              limb_function->push_limb_instruction(std::move(limb_instruction));
            }
          }
          kernel->push_limb_kernel(std::move(limb_function));
        }
        kernel->complete();
      }
      return kernel;
    }

    if (opCode == OpCode::Ntt) {
      kernel = make_kernel(KernelType::Ntt);
      std::vector<std::shared_ptr<FusedKernel>> mov_kernels(
          max_num_destinations);
      for (size_t i = 0; i < max_num_destinations; ++i) {
        mov_kernels[i] = make_kernel(KernelType::Mov);
      }
      size_t kernel_limb_count = 0;
      for (size_t limb_idx = 0; limb_idx < 64; ++limb_idx) {
        for (size_t i = 0; i < max_num_destinations; ++i) {
          auto limb_function = context.create_new_limb_kernel(limb_idx);
          auto mov_limb_function = context.create_new_limb_kernel(limb_idx);
          for (auto &instr : instructions_) {
            auto limb_instruction = instr->get_limb_instruction(i, limb_idx);
            if (limb_instruction) {
              if (limb_instruction->opcode() == OpCode::Mov) {
                mov_limb_function->push_limb_instruction(
                    std::move(limb_instruction));
              } else {
                limb_function->push_limb_instruction(
                    std::move(limb_instruction));
                ++kernel_limb_count;
              }
            }
          }
          mov_kernels[i]->push_limb_kernel(std::move(mov_limb_function));
          kernel->push_limb_kernel(std::move(limb_function));
        }
        if (kernel_limb_count > NTT_KERNEL_FUSE_COUNT) {
          kernel->complete();
          kernel = make_kernel(KernelType::Ntt);
          kernel_limb_count = 0;
        }
      }
      for (auto &mov : mov_kernels) {
        mov->complete();
      }
      kernel->complete();
      return kernel;
    }

    if (opCode == OpCode::Rsv) {
      kernel = make_kernel(KernelType::Rsv);
      for (size_t i = 0; i < max_num_destinations; ++i) {
        auto limb_function = context.create_new_limb_kernel(i);
        for (size_t limb_idx = 0; limb_idx < 64; ++limb_idx) {
          for (auto &instr : instructions_) {
            auto limb_instruction = instr->get_limb_instruction(i, limb_idx);
            if (limb_instruction) {
              limb_function->push_limb_instruction(std::move(limb_instruction));
            }
          }
        }
        kernel->push_limb_kernel(std::move(limb_function));
      }
      kernel->complete();
      return kernel;
    }

    if (opCode == OpCode::Mod || opCode == OpCode::RsM) {
      kernel = make_kernel(KernelType::Mod);
    }

    if (opCode == OpCode::Pmu) {
      kernel = make_kernel(KernelType::Pmu);
      auto mov_kernel = make_kernel(KernelType::Mov);
      emit_limb_kernels_with_mov(
          [&](size_t) { return mov_kernel; },
          [&](size_t i, size_t limb_idx, auto &limb_function,
              auto &mov_limb_function) {
            for (auto &instr : instructions_) {
              auto limb_instruction = instr->get_limb_instruction(i, limb_idx);
              if (limb_instruction) {
                if (limb_instruction->opcode() == OpCode::Mov) {
                  mov_limb_function->push_limb_instruction(
                      std::move(limb_instruction));
                } else {
                  limb_function->push_limb_instruction(
                      std::move(limb_instruction));
                }
              }
            }
          });
      mov_kernel->complete();
      kernel->complete();
      return kernel;
    }

    if (opCode == OpCode::Drm || opCode == OpCode::Ags ||
        opCode == OpCode::Ard) {
      const auto kernel_type = (opCode == OpCode::Drm)   ? KernelType::Drm
                               : (opCode == OpCode::Ags) ? KernelType::Ags
                                                         : KernelType::Ard;
      kernel = make_kernel(kernel_type);

      for (size_t i = 0; i < max_num_destinations; ++i) {
        for (size_t limb_idx = 0; limb_idx < 64; ++limb_idx) {
          auto limb_function = context.create_new_limb_kernel(limb_idx);
          for (auto &instr : instructions_) {
            auto limb_instruction = instr->get_limb_instruction(i, limb_idx);
            if (limb_instruction) {
              limb_function->push_limb_instruction(std::move(limb_instruction));
            }
          }
          kernel->push_limb_kernel(std::move(limb_function));
        }
        if (i != max_num_destinations - 1) {
          // kernel->complete();
          // auto new_kernel = make_kernel(kernel->kernel_type());
          // kernel = new_kernel;
        }
      }
      kernel->complete();
      return kernel;
    }

    if (opCode == OpCode::Bco) {
      assert(instructions_.size() == 1);
      uint32_t num_srcs = -1;
      uint32_t num_dests = -1;
      for (size_t i = 0; i < max_num_destinations; ++i) {
        auto limb_function = context.create_new_limb_kernel(0);
        for (auto &instr : instructions_) {
          auto limb_instruction = instr->get_limb_instruction(i, 0);
          if (!limb_instruction) {
            continue;
          }
          auto limb_instruction_bco =
              std::dynamic_pointer_cast<BaseConvLimbInstruction>(
                  limb_instruction);
          assert(limb_instruction_bco);
          if (num_srcs !=
                  limb_instruction_bco->polynomial_srcs()[0]->num_limbs() ||
              num_dests !=
                  limb_instruction_bco->polynomial_dests()[0]->num_limbs()) {
            if (kernel) {
              kernel->complete();
            }
            kernel = make_kernel(KernelType::Bco);
            num_srcs = limb_instruction_bco->polynomial_srcs()[0]->num_limbs();
            num_dests =
                limb_instruction_bco->polynomial_dests()[0]->num_limbs();
          }
          limb_function->push_limb_instruction(std::move(limb_instruction));
        }
        kernel->push_limb_kernel(std::move(limb_function));
        kernel->complete();
      }
      kernel->complete();
      return kernel;
    }

    if (opCode == OpCode::Call) {
      for (size_t i = 0; i < max_num_destinations; ++i) {
        kernel = make_kernel(KernelType::Call);
        auto limb_function = context.create_new_limb_kernel(0);
        for (auto &instr : instructions_) {
          auto limb_instruction = instr->get_limb_instruction(i, 0);
          if (limb_instruction) {
            limb_function->push_limb_instruction(std::move(limb_instruction));
          }
        }
        kernel->push_limb_kernel(std::move(limb_function));
        kernel->complete();
      }
      return kernel;
    }

    if (!kernel) {
      kernel = context.create_new_kernel();
    }

    gcd_num_destinations = 1;
    for (size_t i = 0; i < gcd_num_destinations; ++i) {
      for (size_t limb_idx = 0; limb_idx < 64; ++limb_idx) {
        auto limb_function = context.create_new_limb_kernel(limb_idx);
        for (auto &instr : instructions_) {
          for (size_t ii = 0; ii < max_num_destinations;
               ii += gcd_num_destinations) {
            auto limb_instruction =
                instr->get_limb_instruction(i + ii, limb_idx);
            if (limb_instruction) {
              limb_function->push_limb_instruction(std::move(limb_instruction));
            }
          }
        }
        kernel->push_limb_kernel(std::move(limb_function));
      }
    }

    kernel->complete();
    return kernel;
  }
  void merge_kg(std::shared_ptr<KernelGroup> &kg) {
    for (auto &instr : kg->instructions()) {
      instr->set_kg(this);
      instructions_.push_back(instr);
    }
    kg->instructions().clear();
  }

  size_t num_limbs() {
    if (type_ == Type::Call) {
      throw std::runtime_error(
          "KernelGroup: Call kernel does not have a level");
    }
    if (type_ != Type::Ewi) {
      return level_;
    }
    bool set = false;
    size_t expected_num_limbs = 0;
    for (auto &instr : instructions_) {
      for (auto &dest : instr->dests()) {
        if (!set) {
          expected_num_limbs = dest.num_limbs();
          set = true;
        } else {
          assert(expected_num_limbs == dest.num_limbs());
        }
      }
    }
    return expected_num_limbs;
  }

  size_t max_num_limbs() {
    if (type_ == Type::Call) {
      throw std::runtime_error(
          "KernelGroup: Call kernel does not have a level");
    }
    if (type_ != Type::Ewi) {
      return level_;
    }
    size_t max_num_limbs = 0;
    for (auto &instr : instructions_) {
      for (auto &dest : instr->dests()) {
        if (max_num_limbs < dest.num_limbs()) {
          max_num_limbs = dest.num_limbs();
        }
      }
    }
    return max_num_limbs;
  }

  std::set<LimbIndexType> limbs() {
    if (type_ == Type::Call) {
      throw std::runtime_error(
          "KernelGroup: Call kernel does not have a level");
    }
    if (type_ != Type::Ewi) {
      return {};
    }
    std::set<LimbIndexType> limbs;
    for (auto &instr : instructions_) {
      for (auto &dest : instr->dests()) {
        for (auto &l : dest.limbs()) {
          limbs.insert(l);
        }
      }
    }
    return limbs;
  }

  size_t num_operations() {
    if (type_ == Type::Call) {
      throw std::runtime_error(
          "KernelGroup: Call kernel does not have a level");
    }
    if (type_ != Type::Ewi) {
      throw std::runtime_error("Should Only be called on ewi instructions");
    }
    size_t num_dests = 0;
    for (auto &instr : instructions_) {
      num_dests += instr->num_destinations();
    }
    return num_dests;
  }
};

using KernelGroupPtr = std::shared_ptr<KernelGroup>;

// } // namespace Cerium
} // namespace Backend
} // namespace Cerium