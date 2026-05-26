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
#include <queue>

namespace Cerium {
namespace Backend {

namespace {

void require_block_size_for_horizontal_fusion(uint64_t block_size) {
  if (block_size == 0) {
    throw std::runtime_error("CERIUM_BLOCK_SIZE must be greater than zero "
                             "when CERIUM_HORIZONTAL_FUSION is enabled");
  }
}

} // namespace

uint64_t BLOCK_SIZE =
    Cerium::Util::getEnvVarUint32("CERIUM_BLOCK_SIZE", 1024);

void CeriumCompiler::fuse_kernels() {
  using OpCode = Backend::LimbInstruction::OpCode;
  bool NO_VERTICAL_FUSION =
      Cerium::Util::getEnvVarUint32("CERIUM_NO_VERTICAL_FUSION", 0);

  auto is_ewise = [&](OpCode op) {
    if (NO_VERTICAL_FUSION) {
      return false;
    }
    // return false;
    switch (op) {
    case OpCode::Inp:
    case OpCode::Add:
    case OpCode::Sub:
    case OpCode::Mul:
    case OpCode::MuP:
    case OpCode::Neg:
    // case OpCode::Mod:
    case OpCode::Mov:
    case OpCode::Mad:
      return true;
    default:
      return false;
    }
  };

  auto is_rot = [&](OpCode op) {
    switch (op) {
    case OpCode::Rot:
    case OpCode::Con:
      return true;
    default:
      return false;
    }
  };

  std::size_t max_num_limbs_prev = 0;
  std::size_t max_num_limbs = 0;

  std::size_t num_dests_prev = 0;
  std::size_t num_dests = 0;

  std::set<LimbIndexType> limbs;
  std::set<LimbIndexType> limbs_prev;
  size_t MAX_NUM_INSTRUCTIONS =
      Cerium::Util::getEnvVarUint32("CERIUM_MAX_INSTRUCTIONS", 1000);
  std::shared_ptr<KernelGroup> kg_prev = nullptr;

  size_t num_instructions;
  for (auto j = 0; j < mod_partitions; j++) {
    auto &k_groups = kernel_groups_split[j];
    for (auto i = 0; i < k_groups.size(); i++) {
      if (!k_groups[i]) {
        continue;
      }
      auto idx = i; // k_groups[i]->index();
      auto &kg = k_groups[idx];
      bool ewise = true;
      bool rot = false;

      // TODO:: Fix this by splitting across gpus
      for (auto &instr : kg->instructions()) {
        if (is_rot(instr->opcode())) {
#ifndef ROT_WRITE
          kg_prev = kg;
          max_num_limbs_prev = kg->max_num_limbs();
          limbs_prev = kg->limbs();
          ewise = false;
#else
          rot = true;
#endif
          break;
        }
        if (!is_ewise(instr->opcode())) {
          kg_prev = nullptr;
          ewise = false;
          break;
        }
        // instr->set_limbs(current_partition_id,current_partition_size);
      }
      if (!ewise) {
        continue;
      }

      // This is quick check to avoid computing limbs and doing set intersection for kernels that have
      // different number of limbs, which is a common case
      max_num_limbs = kg->max_num_limbs();
      if (max_num_limbs_prev != max_num_limbs) {
        kg_prev = nullptr;
      }
      limbs = kg->limbs();
      if(kg_prev != nullptr) {
        std::set<LimbIndexType> intersect;
        std::set_intersection(limbs.begin(), limbs.end(), limbs_prev.begin(),
                              limbs_prev.end(),
                              std::inserter(intersect, intersect.begin()));

        if (intersect.size() != limbs.size()) {
          kg_prev = nullptr;
        }
      }

      if (kg_prev && kg_prev->num_operations() + kg->num_operations() >
                         MAX_NUM_INSTRUCTIONS) {
        kg_prev = nullptr;
      }
      if (!kg_prev) {
        kg_prev = k_groups[idx];
        max_num_limbs_prev = max_num_limbs;
        limbs_prev = limbs;
        #ifdef ROT_WRITE
        if(rot) {
          kg_prev = nullptr;
        }
        #endif
        continue;
      }
      max_num_limbs_prev = max_num_limbs;
      limbs_prev = limbs;
      kg_prev->merge_kg(kg);
      if (rot) {
        kg_prev = nullptr;
      }
    }
  }
}

void CeriumCompiler::fuse_kernels_vertical() {
  using OpCode = Backend::LimbInstruction::OpCode;
  bool NO_VERTICAL_FUSION =
      Cerium::Util::getEnvVarUint32("CERIUM_NO_VERTICAL_FUSION", 0);

  auto is_ewise = [&](OpCode op) {
    if (NO_VERTICAL_FUSION) {
      return false;
    }
    // return false;
    switch (op) {
    case OpCode::Inp:
    case OpCode::Add:
    case OpCode::Sub:
    case OpCode::Mul:
    case OpCode::MuP:
    case OpCode::Neg:
    // case OpCode::Mod:
    case OpCode::Mov:
    case OpCode::Mad:
      return true;
    default:
      return false;
    }
  };

  auto is_rot = [&](OpCode op) {
    switch (op) {
    case OpCode::Rot:
    case OpCode::Con:
      return true;
    default:
      return false;
    }
  };

  std::size_t num_limbs_prev = 0;
  std::size_t num_limbs = 0;

  std::size_t num_dests_prev = 0;
  std::size_t num_dests = 0;

  std::set<LimbIndexType> limbs;
  std::set<LimbIndexType> limbs_prev;
  size_t MAX_NUM_INSTRUCTIONS =
      Cerium::Util::getEnvVarUint32("CERIUM_MAX_INSTRUCTIONS", 1000);
  std::shared_ptr<KernelGroup> kg_prev = nullptr;

  size_t num_instructions;
  for (auto j = 0; j < mod_partitions; j++) {
    auto &k_groups = kernel_groups_split[j];
    for (auto i = 0; i < k_groups.size(); i++) {
      if (!k_groups[i]) {
        continue;
      }
      auto idx = i; // k_groups[i]->index();
      auto &kg = k_groups[idx];
      bool ewise = true;

      // TODO:: Fix this by splitting across gpus
      for (auto &instr : kg->instructions()) {
        if (is_rot(instr->opcode())) {
          kg_prev = kg;
          num_limbs_prev = kg->num_limbs();
          limbs_prev = kg->limbs();
          ewise = false;
          break;
        }
        if (!is_ewise(instr->opcode())) {
          kg_prev = nullptr;
          ewise = false;
          break;
        }
        // instr->set_limbs(current_partition_id,current_partition_size);
      }
      if (!ewise) {
        continue;
      }
      num_limbs = kg->num_limbs();
      limbs = kg->limbs();
      if (num_limbs_prev != num_limbs) {
        kg_prev = nullptr;
        // num_limbs_prev = num_limbs;
      }
      std::set<LimbIndexType> intersect;
      std::set_intersection(limbs.begin(), limbs.end(), limbs_prev.begin(),
                            limbs_prev.end(),
                            std::inserter(intersect, intersect.begin()));

      if (intersect.size() != limbs.size()) {
        kg_prev = nullptr;
      }

      if (kg_prev && kg_prev->num_operations() + kg->num_operations() >
                         MAX_NUM_INSTRUCTIONS) {
        kg_prev = nullptr;
      }
      if (!kg_prev) {
        kg_prev = k_groups[idx];
        num_limbs_prev = num_limbs;
        limbs_prev = limbs;
        continue;
      }
      num_limbs_prev = num_limbs;
      limbs_prev = limbs;
      kg_prev->merge_kg(kg);
    }
  }
}

void CeriumCompiler::split_horizontal_fusion() {
  auto kernel_groups_old = std::move(kernel_groups);
  kernel_groups.clear();

  auto process_instr = [&](std::shared_ptr<KernelGroup> &kg,
                           std::shared_ptr<KernelInstruction> &instr) {
    if (instr->num_destinations() == 0) {
      return;
    }
    auto split = instr->split();
    assert(!split.empty());
    for (auto i = 0; i < split.size(); i++) {
      auto kg_new = create_kernel_group(kg);
      kg_new->push_instruction(std::move(split[i]));
    }
  };

  for (auto &kg : kernel_groups_old) {
    for (auto &instr : kg->instructions()) {
      process_instr(kg, instr);
    }
  }
}

void CeriumCompiler::make_new_kernels() {
  auto kernel_groups_old = std::move(kernel_groups);
  std::unordered_map<Backend::TermIndexType, uint64_t> created_at_kg;
  kernel_groups.clear();

  auto process_instr = [&](std::shared_ptr<KernelGroup> &kg,
                           std::shared_ptr<KernelInstruction> &instr) {
    auto kg_new = create_kernel_group(kg);
    kg_new->push_instruction(std::move(instr));
    for (auto &dst : instr->dests()) {
      created_at_kg[dst.term_idx()] = kg_new->index();
    }
    for (auto &src : instr->srcs()) {
      if (src.is_eval_key()) {
        continue;
      }
      if (src.is_zero()) {
        continue;
      }
      auto src_term_index = created_at_kg.at(src.term_idx());
      kg_new->add_parent(kernel_groups[src_term_index], kernel_groups);
      // created_at_kg[dst.term_index()] = kg_new->index();
    }
  };

  for (auto &kg : kernel_groups_old) {
    for (auto &instr : kg->instructions()) {
      process_instr(kg, instr);
    }
  }
}

void CeriumCompiler::fuse_kernels_horizontal() {
  if (BLOCK_SIZE == 0) {
    return;
  }

  std::cout << "Performing Horizontal Fusion with BLOCK_SIZE: " << BLOCK_SIZE
            << "\n";

  auto num_b = kernel_groups.size() / BLOCK_SIZE;

  for (int b = 0; b < kernel_groups.size(); b += BLOCK_SIZE) {
    for (int i1 = 0; i1 < BLOCK_SIZE; i1++) {
      int idx1 = b + i1;
      if (idx1 >= kernel_groups.size()) {
        continue;
      }
      auto kg1 = kernel_groups[idx1];
      for (int i2 = i1 + 1; i2 < BLOCK_SIZE; i2++) {
        int idx2 = b + i2;
        if (idx2 >= kernel_groups.size()) {
          continue;
        }
        auto kg2 = kernel_groups[idx2];
        bool merged = kg1->horizontal_merge(kg2, kernel_groups);
      }
    }
    std::cout << "\rCompleted Block: " << b / BLOCK_SIZE << "/" << num_b
              << std::flush;
  }
  std::cout << "\n";
}

void CeriumCompiler::topologically_sort_kernels() {
  std::unordered_set<KernelGroup::IndexType> visited;
  std::vector<std::shared_ptr<KernelGroup>> topologically_sorted;
  size_t num_kernels = 0;

  struct KernelGroupPriorityQueueEntry {
    KernelGroup::IndexType index;
    int priority;

    // KernelGroupPriorityQueueEntry()
  };

  // Custom comparator: higher priority value means higher priority
  struct ComparePriority {
    bool operator()(KernelGroupPriorityQueueEntry const &t1,
                    KernelGroupPriorityQueueEntry const &t2) {
      return t1.priority < t2.priority;
    }
  };

  std::priority_queue<KernelGroupPriorityQueueEntry,
                      std::vector<KernelGroupPriorityQueueEntry>,
                      ComparePriority>
      taskPq;

  for (int b = 0; b < kernel_groups.size(); b += BLOCK_SIZE) {
    std::deque<KernelGroup::IndexType> zero_indegree;
    for (int i1 = 0; i1 < BLOCK_SIZE; i1++) {
      int idx1 = b + i1;
      if (idx1 >= kernel_groups.size()) {
        continue;
      }
      auto kg = kernel_groups[idx1];
      auto idx = kg->index();
      if (visited.find(idx) != visited.end()) {
        continue;
      }
      visited.insert(idx);
      num_kernels++;
      kg->refresh_parents_and_children(kernel_groups);
      if (kg->parents().size() != 0) {
        continue;
      }
      if (kg->is_communication_kg()) {
        topologically_sorted.push_back(kg);
      }
      zero_indegree.push_back(idx);
    }

    while (!zero_indegree.empty()) {
      auto front = zero_indegree.front();
      zero_indegree.pop_front();
      auto &kg = kernel_groups[front];
      if (!kg->is_communication_kg()) {
        topologically_sorted.push_back(kg);
      }
      for (auto &c : kg->children()) {
        auto &parents = kernel_groups[c]->parents();
        parents.erase(front);
        if (!parents.empty()) {
          continue;
        }
        // Bring Communication KGs as early as possible
        if (kernel_groups[c]->is_communication_kg()) {
          topologically_sorted.push_back(kernel_groups[c]);
        }
        zero_indegree.push_front(c);
      }
    }
  }
  assert(topologically_sorted.size() == num_kernels);
  kernel_groups.clear();
  for (auto &kg : topologically_sorted) {
    auto kg_new = create_kernel_group(kg);
    kg_new->instructions() = std::move(kg->instructions());
  }
}

void CeriumCompiler::cross_chip_communication_optimization_pass() {
  if (mod_partitions == 1) {
    return;
  }
  auto run_cross_chip_comm_optimization =
      Cerium::Util::getEnvVarUint32("CERIUM_CROSS_CHIP_OPTIMIZE", 0);
  if (run_cross_chip_comm_optimization == 0) {
    return;
  }

  // Find patterns of Agg -> Dist and Replace it with Ard
  std::unordered_map<Backend::TermIndexType, std::shared_ptr<KernelInstruction>>
      created_at_instruction;
  std::unordered_map<std::shared_ptr<KernelInstruction>,
                     std::vector<std::shared_ptr<KernelInstruction>>>
      instruction_uses;

  using KernelInstructionPtr = std::shared_ptr<KernelInstruction>;
  auto create_uses = [&](std::shared_ptr<KernelGroup> &kg,
                         std::shared_ptr<KernelInstruction> &instr) {
    for (auto &dst : instr->dests()) {
      created_at_instruction[dst.term_idx()] = instr;
    }
    for (auto &src : instr->srcs()) {
      if (src.is_eval_key()) {
        continue;
      }
      if (src.is_zero()) {
        continue;
      }
      auto src_term_index = created_at_instruction.at(src.term_idx());
      instruction_uses[src_term_index].push_back(instr);
    }
  };

  auto rewrite_ags = [&]() {
    for (auto &[instr, uses] : instruction_uses) {
      if (instr->opcode() != KernelInstruction::OpCode::Ags) {
        continue;
      }
      KernelInstructionPtr drm_use = nullptr;
      KernelInstructionPtr intt_use = nullptr;
      for (auto &use : uses) {
        if (use->opcode() == KernelInstruction::OpCode::Drm) {
          drm_use = use;
        }
        if (use->opcode() == KernelInstruction::OpCode::Int) {
          intt_use = use;
        }
      }
      if (!drm_use && !intt_use) {
        continue;
      }
      if (drm_use && intt_use) {
        continue; // TODO: Handle the case where both kind of uses are there
      }
      if (drm_use) {
        if (instr->kg()->partition_size() != drm_use->kg()->partition_size()) {
          continue;
        }
        if (instr->kg()->partition_id() != drm_use->kg()->partition_id()) {
          continue;
        }
        if (instr->dests().size() != drm_use->srcs().size()) {
          continue;
        }
        if (instr->dests().size() != drm_use->srcs().size()) {
          continue;
        }
        if (instr->dests().size() != 1) {
          continue;
        }
        KernelInstructionPtr ard_instr = nullptr;
        auto ags_instr =
            std::dynamic_pointer_cast<AggregateScatterInstruction>(instr);
        auto ags_instr3 =
            std::dynamic_pointer_cast<AggregateScatterInstruction3>(instr);

        auto ard_kernel_type = KernelGroup::Type::Ard;
        if (!ags_instr && !ags_instr3) {
          continue;
        } else if (ags_instr) {
          ard_instr = ags_instr->convert_to_all_reduce_instruction();
          ard_kernel_type = KernelGroup::Type::Ard;
        } else {
          ard_instr = ags_instr3->convert_to_all_reduce_instruction();
          ard_kernel_type = KernelGroup::Type::Ard3;
        }

        // Turn the ags instruction to an ard instruction
        // Then modify the limbtype of the intt instruction
        // Finally, modify the drm instruction to a mov instruction

        auto kg_ags = instr->kg();
        kg_ags->clear_instructions();
        kg_ags->set_type(ard_kernel_type);
        kg_ags->push_instruction(std::move(ard_instr));

        auto kg_drm = drm_use->kg();
        kg_drm->set_type(KernelGroup::Type::Ewi);
        kg_drm->instructions().clear();
        auto mov_instr =
            std::make_shared<UnOpInstruction>(KernelInstruction::OpCode::Mov);
        mov_instr->add_operands(drm_use->dests()[0], instr->dests()[0]);
        instr->dests()[0].set_limbtype(Term::LimbType::Usp);
        drm_use->dests()[0].set_limbtype(Term::LimbType::Usp);
        kg_drm->push_instruction(std::move(mov_instr));

      } else {
        if (instruction_uses[intt_use].size() != 1) {
          continue;
        }
        auto intt_use_use = instruction_uses[intt_use][0];
        if (intt_use_use->opcode() != KernelInstruction::OpCode::Drm) {
          continue;
        }
        if (instr->kg()->partition_size() !=
            intt_use_use->kg()->partition_size()) {
          continue;
        }
        if (instr->kg()->partition_id() != intt_use_use->kg()->partition_id()) {
          continue;
        }
        if (instr->dests().size() != intt_use->srcs().size()) {
          continue;
        }
        if (instr->dests().size() != intt_use_use->srcs().size()) {
          continue;
        }
        if (instr->dests().size() != 1) {
          continue;
        }
        KernelInstructionPtr ard_instr = nullptr;
        auto ags_instr =
            std::dynamic_pointer_cast<AggregateScatterInstruction>(instr);
        auto ags_instr3 =
            std::dynamic_pointer_cast<AggregateScatterInstruction3>(instr);

        auto ard_kernel_type = KernelGroup::Type::Ard;
        if (!ags_instr && !ags_instr3) {
          continue;
        } else if (ags_instr) {
          ard_instr = ags_instr->convert_to_all_reduce_instruction();
          ard_kernel_type = KernelGroup::Type::Ard;
        } else {
          ard_instr = ags_instr3->convert_to_all_reduce_instruction();
          ard_kernel_type = KernelGroup::Type::Ard3;
        }

        // Turn the ags instruction to an ard instruction
        // Then modify the limbtype of the intt instruction
        // Finally, modify the drm instruction to a mov instruction

        auto kg_drm = intt_use_use->kg();
        kg_drm->set_type(KernelGroup::Type::Ewi);
        kg_drm->instructions().clear();

        auto kg_ags = instr->kg();
        kg_ags->clear_instructions();
        kg_ags->set_type(ard_kernel_type);
        kg_ags->push_instruction(std::move(ard_instr));

        auto kg_intt = intt_use->kg();
        kg_intt->set_type(KernelGroup::Type::Int);
        kg_intt->clear_instructions();
        auto intt_instr = std::make_shared<InttInstruction>();
        intt_instr->add_operands(intt_use_use->dests()[0], instr->dests()[0]);
        instr->dests()[0].set_limbtype(Term::LimbType::Usp);
        intt_use_use->dests()[0].set_limbtype(Term::LimbType::Usp);
        kg_intt->push_instruction(std::move(intt_instr));

      }
    }
  };

  for (auto &kg : kernel_groups) {
    for (auto &instr : kg->instructions()) {
      create_uses(kg, instr);
    }
  }
  rewrite_ags();
}

void CeriumCompiler::finish() {
  BLOCK_SIZE = Cerium::Util::getEnvVarUint32("CERIUM_BLOCK_SIZE", 1024);
  print_graph_simple2("simple");

  cross_chip_communication_optimization_pass();

  bool DO_HORIZONTAL_FUSION =
      Cerium::Util::getEnvVarUint32("CERIUM_HORIZONTAL_FUSION", 1);
  if (DO_HORIZONTAL_FUSION) {
    require_block_size_for_horizontal_fusion(BLOCK_SIZE);
    make_new_kernels();
    print_graph_simple2("simple_new");
    fuse_kernels_horizontal();
    topologically_sort_kernels();
  }

  bool UNDO_HORIZONTAL_FUSION =
      Cerium::Util::getEnvVarUint32("CERIUM_UNDO_HORIZONTAL_FUSION", 0);
  if (UNDO_HORIZONTAL_FUSION) {
    split_horizontal_fusion();
  }

  // split_ntt_instructions();
  print_graph_simple2("horizontal");
#ifdef CERIUM_USE_SPLIT_KERNELS_LIMBWISE_V2
  split_kernels_limbwise_v2();
#else
  split_kernels_limbwise();
#endif
  set_instruction_limbs();
  fuse_kernels();
  set_temporary_values();
  print_graph_split();
  print_limb_instructions();
  write_program_inputs();
}

} // namespace Backend
} // namespace Cerium
