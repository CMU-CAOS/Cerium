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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "cerium/compiler/backend/limb_instruction.h"
#include "cerium/compiler/backend/register.h"

// Rotation kernels write their results directly to permuted output locations.
// Keep this visible to fusion so consumers are not fused after those writes.
#define ROT_WRITE

namespace Cerium {
namespace Backend {

class FusedKernel;
class KernelFusionContext;
enum KernelType2 {
  Generic,
  Ntt,
  SuD,
  SuD2,
  Msd,
  Int,
  Bco,
  Mod,
  Rsv,
  RsM,
  Mov,
  Pmu,
  Drm,
  Ags,
  Ard,
  Call,
};

std::string indent(int indent_depth);

class FusedKernelLimb {

  KernelFusionContext &context_;

  Backend::LimbOrderedSet kernel_inputs_;
  Backend::LimbOrderedSet kernel_inputs_scalar_;
  Backend::LimbOrderedSet kernel_inputs_remapable_;
  Backend::LimbOrderedSet kernel_outputs_;

  LimbMap<LimbSet> limb_dependencies_map_;

  Backend::LimbSet rot_inputs_;
  Backend::LimbSet rot_outputs_;

  Backend::LimbSet reduction_reqd_;

  std::vector<std::shared_ptr<Backend::LimbInstruction>> instructions_;
  Backend::LimbIndexType limb_idx_;
  std::set<std::int32_t> rot_indices_;
  std::set<LimbIndexType> pmu_base_;
  std::set<LimbIndexType> sud_base_;
  std::set<LimbIndexType> rsv_base_;

  std::optional<Backend::TermIndexType> base_conv_val_;

  bool is_ntt_kernel_{false};

  uint32_t comm_num_limbs_{0};
  uint32_t comm_size_{0};
  std::vector<Limb> comm_srcs;
  std::vector<std::vector<Limb>> comm_dests;

  std::string call_function_name;

  using ValueType = std::variant<Limb, Register>;
  std::unordered_map<std::string, ValueType> call_function_args;
  LimbMap<std::vector<std::string>> call_function_args_map;

  static std::string limb_to_variable(const Backend::Limb *limb);

  const std::set<LimbIndexType> &pmu_base() const;

  const std::set<LimbIndexType> &sud_base() const;

  bool reduction_required(const Limb *limb);

  bool intt_pre_factors_initialized_ = false;

  std::string
  handle_add(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::string
  handle_mad(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::string
  handle_mul(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::string
  handle_pmu(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::string
  handle_rot(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::string
  handle_rsv(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::string
  handle_mod(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::string
  handle_rsm(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::string
  handle_mov(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::ostream &
  handle_ntt(std::ostream &s,
             const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
             const uint8_t kernel_count, int indent_depth);

  std::ostream &
  handle_intt(std::ostream &s,
              const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
              const uint8_t kernel_count, int indent_depth);

  static std::ostream &
  handle_mov(std::ostream &s,
             const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
             uint8_t kernel_count, int indent_depth = 0);

  static std::ostream &
  handle_bco(std::ostream &s,
             const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
             int indent_depth = 0);

  void move_instructions_closer();

  std::string
  instr_to_cuda(const std::shared_ptr<Backend::LimbInstruction> &limb_instr);

  std::ostream &
  ntt_instr_to_cuda(std::ostream &s,
                    const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
                    uint8_t kernel_count, int indent_depth = 0);

  std::ostream &
  bco_instr_to_cuda(std::ostream &s,
                    const std::shared_ptr<Backend::LimbInstruction> &limb_instr,
                    int indent_depth = 0);

public:
  FusedKernelLimb(Backend::LimbIndexType limb_idx, KernelFusionContext &context)
      : limb_idx_(limb_idx), context_(context) {}
  void
  push_limb_instruction(std::shared_ptr<LimbInstruction> &&limb_instruction);

  std::ostream &write_ntt_kernel(std::ostream &s, uint ntt_kernel_count,
                                 KernelType2 &kernel_type,
                                 int indent_depth = 0);

  std::ostream &write_rsv(std::ostream &s, int indent_depth = 0,
                          bool init = false);
  std::ostream &write(std::ostream &s, int indent_depth = 0);
  std::string prettyPrint();
  std::size_t num_instructions() const { return instructions_.size(); }

  friend std::ostream &operator<<(std::ostream &s, const FusedKernelLimb &fkl);

  void set_ntt_kernel(const bool is_ntt_kernel) {
    is_ntt_kernel_ = is_ntt_kernel;
  }

  friend class FusedKernel;
  friend class KernelFusionContext;
};

class FusedKernel {
public:
  using KernelType = KernelType2;

private:
  KernelFusionContext &context_;
  uint64_t kernel_idx_{0};
  std::vector<std::shared_ptr<Backend::FusedKernelLimb>> limb_blocks_;

  using ValueType = std::variant<Limb, Register>;
  std::vector<std::vector<ValueType>> fused_kernel_inputs_set_;
  std::vector<std::vector<ValueType>> fused_kernel_inputs_scalar_set_;
  std::vector<std::vector<ValueType>> fused_kernel_inputs_remapable_set_;
  std::vector<std::vector<ValueType>> fused_kernel_outputs_set_;

  std::vector<LimbMap<LimbSet>> limb_dependencies_map_;

  std::vector<Backend::LimbIndexType> bases_;

  Backend::LimbSet rot_inputs_;
  Backend::LimbSet rot_outputs_;

  std::set<LimbIndexType> pmu_bases_;
  std::set<LimbIndexType> sud_bases_;
  std::set<LimbIndexType> rsv_bases_;

  std::set<std::int32_t> rot_indices_;
  std::unordered_map<Backend::TermIndexType,
                     std::vector<Backend::LimbIndexType>>
      base_conv_input_bases_;
  std::unordered_map<Backend::TermIndexType,
                     std::vector<Backend::LimbIndexType>>
      base_conv_output_bases_;

  uint32_t comm_num_limbs_{0};
  uint32_t comm_size_{0};
  std::vector<ValueType> comm_srcs;
  std::vector<std::vector<ValueType>> comm_dests;

  const std::string kernel_signature{"__global__ static void "};

  KernelType kernel_type_{KernelType::Generic};
  bool is_ntt_kernel_{false};

  std::string function_definition_;
  std::string function_definition_cugraph_;

  std::string call_function_name_;
  std::unordered_map<std::string, ValueType> call_function_args_;
  LimbMap<std::vector<std::string>> call_function_args_map_;

  static void write_load_register(std::ostream &s, const Limb &v,
                                  int indent_depth = 0);

  static void foreach_map_limb_set_valuetype(
      std::vector<std::vector<ValueType>> &map_limb_set,
      std::function<void(size_t pos, ValueType &)> func);

  static void
  foreach_map_limb_set(std::vector<std::vector<ValueType>> &map_limb_set,
                       std::function<void(size_t pos, Limb &)> func);

  std::ostream &write_value(std::ostream &s, ValueType &val);

public:
  friend class KernelFusionContext;
  FusedKernel(uint64_t kernel_idx, KernelFusionContext &context)
      : kernel_idx_(kernel_idx), context_(context){};

  auto kernel_type() { return kernel_type_; }

  void push_limb_kernel(std::shared_ptr<FusedKernelLimb> &&fkl);

  std::ostream &write_kernel_rsv(std::ostream &s, int indent_depth = 0);
  std::ostream &write_kernel_rsv_old(std::ostream &s, int indent_depth = 0);
  std::ostream &write_kernel_mod(std::ostream &s, int indent_depth = 0);
  std::ostream &write_kernel_int(std::ostream &s, int ntt_kernel_count,
                                 int indent_depth = 0);

  std::ostream &write_kernel_ntt(std::ostream &s, int ntt_kernel_count,
                                 int indent_depth = 0);

  std::ostream &write_kernel_bco(std::ostream &s, int indent_depth = 0);
  std::ostream &write_kernel_pmu(std::ostream &s, int indent_depth = 0);
  std::ostream &write_kernel_empty(std::ostream &s, int indent_depth = 0);
  std::ostream &write_kernel_generic(std::ostream &s, int indent_depth = 0);

  std::ostream &write_kernel(std::ostream &s, int indent_depth = 0);

  std::ostream &write_function_cugraph(std::ostream &s, int indent_depth = 0);
  std::ostream &write_function(std::ostream &s, int indent_depth = 0);

  bool no_code_to_write() const;

  std::ostream &write_code(std::ostream &s, int indent_depth = 0);

  std::ostream &write_instruction(std::ostream &s);

  std::string prettyPrint();
  std::string printInstruction();

  friend std::ostream &operator<<(std::ostream &s, const FusedKernel &fk);

  void set_ntt_kernel(const bool is_ntt_kernel) {
    is_ntt_kernel_ = is_ntt_kernel;
  }

  void set_kernel_type(const KernelType kernel_type) {
    kernel_type_ = kernel_type;
  }

  auto &inputs() { return fused_kernel_inputs_set_; }

  auto &inputs_scalar() { return fused_kernel_inputs_scalar_set_; }

  auto &inputs_remapable() { return fused_kernel_inputs_remapable_set_; }

  auto &outputs() { return fused_kernel_outputs_set_; }

  void complete();

  auto kernel_index() const { return kernel_idx_; }
};

class KernelFusionContext {
public:
  struct BaseConversionMetaData {
    std::set<Backend::LimbIndexType> input_bases_;
    std::set<Backend::LimbIndexType> output_bases_;
  };

private:
  uint64_t kernel_idx_{0};
  std::vector<std::shared_ptr<FusedKernel>> kernels_;
  std::unordered_map<Backend::TermIndexType, BaseConversionMetaData>
      base_conversion_metadata_;

  std::string function_name;

public:
  KernelFusionContext(const std::string &function_name)
      : function_name(function_name), kernel_idx_(0) {
    // TODO: Sanitize Function Name
  }

  std::shared_ptr<FusedKernel> create_new_kernel() {
    auto kernel = std::make_shared<Backend::FusedKernel>(kernel_idx_++, *this);
    kernels_.push_back(kernel);
    return kernel;
  };

  std::shared_ptr<FusedKernelLimb>
  create_new_limb_kernel(const Backend::LimbIndexType limb_idx) {
    return std::make_shared<Backend::FusedKernelLimb>(limb_idx, *this);
  };

  const auto &kernels() { return kernels_; }

  friend class FusedKernel;
  friend class FusedKernelLimb;

  void write_function_map_impl(std::ofstream &file, bool use_cugraph);
  void write_function_map(const std::string &dir_name);
  void write_function_map_cugraph(const std::string &dir_name);
  void write_code(const std::string &dir_name);
  void write_instructions(const std::string &dir_name);
  void allocate_registers_cerium(size_t num_vregs, size_t num_bcus);

  void write_code_file_header(std::ofstream &s);
  void write_code_file_footer(std::ofstream &s);
};
} // namespace Backend
} // namespace Cerium
