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

#include <seal/seal.h>
#include <seal/util/mempool.h>
#include <seal/util/rns.h>

#include "cerium/runtime/context.h"
#include "cerium/runtime/cuda/device_limb.h"
#include "cerium/runtime/cuda/nccl_ops.h"
#include "cerium/runtime/execution/device_utils.h"

namespace Cerium {
namespace Runtime {
class Evaluator;
class DeviceBaseConverter {

public:
  using LimbT = DeviceLimb;
  DeviceBaseConverter(const Context &context)
      : context_(context), pool_(seal::MemoryManager::GetPool()), size_(0),
        coeff_count_(context_.n()), writes_completed_(0), converted_(false) {
    size_t MAX_LIMBS = 64;
    size_t MAX_INPUT_LIMBS = 48;
    size_t MAX_OUTPUT_LIMBS = 64;
    LimbT::Element_t *ptr = nullptr;
    base_change_factors_.resize(MAX_LIMBS * MAX_LIMBS);
    input_limbs_ptr_ =
        DevicePointer<LimbT::Element_t>(MAX_INPUT_LIMBS * coeff_count_);
    output_limbs_ptr_ =
        DevicePointer<LimbT::Element_t>(MAX_OUTPUT_LIMBS * coeff_count_);

    output_modulus_device_ = DevicePointer<LimbT::Element_t>(MAX_LIMBS);
    barrett_k_device_ = DevicePointer<LimbT::Element_t>(MAX_LIMBS);
    barrett_ratios_device_ = DevicePointer<LimbT::Element_t>(MAX_LIMBS);
  }

  void set_input_bases(const std::vector<std::uint64_t> &input_rns_base_ids);
  void append_input_bases(const std::vector<std::uint64_t> &input_rns_base_ids);
  void set_output_bases(const std::vector<std::uint64_t> &output_rns_base_ids);
  LimbT::Element_t *get_pointer(std::uint32_t base);
  inline auto input_bases_product() { return input_rns_bases_->base_prod(); }
  inline auto input_bases_size() { return input_rns_bases_->size(); }

  inline const auto &output_base_ids() { return output_rns_base_ids_; }

private:
  const Context &context_;
  std::vector<std::uint64_t> input_rns_base_ids_;
  seal::util::Pointer<seal::util::RNSBase> input_rns_bases_;
  std::vector<std::uint64_t> output_rns_base_ids_;
  std::uint64_t writes_completed_;
  seal::MemoryPoolHandle pool_;
  DevicePointer<LimbT::Element_t> base_change_matrix_;
  DevicePointer<LimbT::Element_t> barrett_ratios_device_;
  DevicePointer<LimbT::Element_t> barrett_k_device_;
  DevicePointer<LimbT::Element_t> output_modulus_device_;
  std::size_t size_;
  std::size_t coeff_count_;

  std::vector<LimbT::Element_t> base_change_factors_;

  std::vector<std::vector<LimbT::Element_t>> base_change_factors_vec_;
  std::vector<std::vector<LimbT::Element_t>> barrett_ratios_vec_;
  std::vector<std::vector<LimbT::Element_t>> barrett_k_vec_;
  std::vector<std::vector<LimbT::Element_t>> output_modulus_vec_;

  std::vector<DevicePointer<LimbT::Element_t>> base_change_factors_vec_device_;
  std::vector<DevicePointer<LimbT::Element_t>> barrett_ratios_vec_device_;
  std::vector<DevicePointer<LimbT::Element_t>> barrett_k_vec_device_;
  std::vector<DevicePointer<LimbT::Element_t>> output_modulus_vec_device_;

  bool converted_;
  DevicePointer<LimbT::Element_t> input_limbs_ptr_;

  DevicePointer<DeviceLimb::Element_t> output_limbs_ptr_;
  friend class Evaluator;
};

class EvaluatorContext {
public:
  using LimbT = DeviceLimb;
  EvaluatorContext() = delete;
  EvaluatorContext(const Context &context, uint32_t device_id,
                   uint32_t num_devices);
  using LimbDataType = LimbT::Element_t;

private:
  uint32_t device_id{0};
  uint32_t num_devices{1};
  const Context &context_;
  std::size_t coeff_count_;
  std::size_t coeff_count_power_;

  std::vector<Pointer<uint32_t>> rotate_permutations_host_;
  std::vector<DevicePointer<uint32_t>> rotate_permutations_device_;

  DevicePointer<LimbT::Element_t> modulii_;
  DevicePointer<LimbT::Element_t> barrett_ratios_;
  DevicePointer<uint32_t> barrett_k_;

  DevicePointer<LimbT::Element_t2> root_powers_uint2_;
  DevicePointer<LimbT::Element_t2> inv_root_powers_div_two_uint2_;

  std::vector<uint32_t> barrett_k_host_;
  std::vector<LimbT::Element_t> barrett_ratios_host_;
  std::vector<LimbT::Element_t> modulii_host_;
  size_t NCCL_BUFFER_MAX_LIMBS;
  CUDA::NCCL::BufferPtr nccl_buffer___;

  void init_params();
  void init_ntt_params();

  void permute_root_powers(std::vector<LimbT::Element_t> &root_powers_host,
                           const uint32_t m_start, const uint32_t m_end,
                           const size_t coeff_count, const size_t per_thread,
                           const size_t num_rns_bases);

  const uint32_t *get_rotation_map_ptr(const int32_t amount);

  friend class Evaluator;
};

class Evaluator {
public:
  using LimbT = DeviceLimb;
  Evaluator() = delete;
  Evaluator(const Context &context,
            std::shared_ptr<EvaluatorContext> evaluator_context);
  // Evaluator(const Context & context, const CUDA::StreamPtr & stream) : context_(context), coeff_count_(context.n()), coeff_count_power_(log2(coeff_count_)), stream_(stream)

  using LimbDataType = LimbT::Element_t;

  void ntt(std::vector<LimbDataType *> &limbs,
           const std::vector<uint32_t> &bases);

  void fn_ewi(void *function, const CUDA::StreamPtr &stream,
              std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<LimbDataType> &scalars,
              const LimbDataType **remapable_base,
              const std::vector<std::size_t> &args_remapables,
              const std::vector<uint32_t> &bases,
              std::vector<int32_t> rotations);
  void fn_int(void *function, const CUDA::StreamPtr &stream,
              std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<uint32_t> &bases);
  void fn_ntt(void *function, const CUDA::StreamPtr &stream,
              std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<uint32_t> &bases);
  void fn_sud(void *function, const CUDA::StreamPtr &stream,
              std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<uint32_t> &bases,
              const std::vector<LimbDataType> &sud_factors);
  void fn_rsv(void *function, const CUDA::StreamPtr &stream,
              std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<LimbDataType> &factors,
              const std::vector<LimbDataType> &base,
              const uint64_t barrett_ratio, const uint32_t barrett_k);
  void fn_bco(void *function, const CUDA::StreamPtr &stream,
              std::vector<DeviceBaseConverter *> &base_converters);
  void drm(const CUDA::StreamPtr &stream, std::vector<LimbDataType *> &gather,
           const std::vector<LimbDataType *> scatter, const size_t comm_size,
           const size_t num_comm_limbs, const size_t tid);
  void ags(const CUDA::StreamPtr &stream, std::vector<LimbDataType *> &outputs,
           const std::vector<LimbDataType *> inputs, const size_t comm_size,
           const size_t num_comm_limbs, const size_t tid,
           const std::vector<uint32_t> &bases);
  void ard(const CUDA::StreamPtr &stream, std::vector<LimbDataType *> &outputs,
           const std::vector<LimbDataType *> inputs, const size_t comm_size,
           const size_t num_comm_limbs, const size_t tid,
           const std::vector<uint32_t> &bases);

  std::vector<void *>
  fn_ewi_args(void *function, std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<LimbDataType> &scalars,
              const LimbDataType **remapable_base,
              const std::vector<std::size_t> &args_remapables,
              const std::vector<uint32_t> &bases,
              std::vector<int32_t> rotations);
  std::vector<void *> fn_int_args(void *function,
                                  std::vector<LimbDataType *> &outputs,
                                  const std::vector<LimbDataType *> &inputs,
                                  const std::vector<uint32_t> &bases);
  std::vector<void *> fn_ntt_args(void *function,
                                  std::vector<LimbDataType *> &outputs,
                                  const std::vector<LimbDataType *> &inputs,
                                  const std::vector<uint32_t> &bases);
  std::vector<void *> fn_sud_args(void *function,
                                  std::vector<LimbDataType *> &outputs,
                                  const std::vector<LimbDataType *> &inputs,
                                  const std::vector<uint32_t> &bases,
                                  const std::vector<LimbDataType> &sud_factors);
  std::vector<void *> fn_rsv_args(void *function,
                                  std::vector<LimbDataType *> &outputs,
                                  const std::vector<LimbDataType *> &inputs,
                                  const std::vector<LimbDataType> &factors,
                                  const std::vector<LimbDataType> &base,
                                  const uint64_t barrett_ratio,
                                  const uint32_t barrett_k);
  std::vector<void *>
  fn_bco_args(void *function,
              std::vector<DeviceBaseConverter *> &base_converters);
  std::vector<void *> drm_args(std::vector<LimbDataType *> &gather,
                               const std::vector<LimbDataType *> scatter,
                               const size_t comm_size,
                               const size_t num_comm_limbs, const size_t tid);
  std::vector<void *> ags_args(std::vector<LimbDataType *> &outputs,
                               const std::vector<LimbDataType *> inputs,
                               const size_t comm_size,
                               const size_t num_comm_limbs, const size_t tid,
                               const std::vector<uint32_t> &bases);
  std::vector<void *> ard_args(std::vector<LimbDataType *> &outputs,
                               const std::vector<LimbDataType *> inputs,
                               const size_t comm_size,
                               const size_t num_comm_limbs, const size_t tid,
                               const std::vector<uint32_t> &bases);

  void fn_ewi(const CUDA::StreamPtr &stream, std::vector<void *> &args);
  void fn_int(const CUDA::StreamPtr &stream, std::vector<void *> &args);
  void fn_ntt(const CUDA::StreamPtr &stream, std::vector<void *> &args);
  void fn_sud(const CUDA::StreamPtr &stream, std::vector<void *> &args);
  void fn_rsv(const CUDA::StreamPtr &stream, std::vector<void *> &args);
  void fn_bco(const CUDA::StreamPtr &stream, std::vector<void *> &args);
  void drm(const CUDA::StreamPtr &stream, std::vector<void *> &args);
  void ags(const CUDA::StreamPtr &stream, std::vector<void *> &args);
  void ard(const CUDA::StreamPtr &stream, std::vector<void *> &args);

  void fn_ewi(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
              std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<LimbDataType> &scalars,
              const LimbDataType **remapable_base,
              const std::vector<std::size_t> &args_remapables,
              const std::vector<uint32_t> &bases,
              std::vector<int32_t> rotations);
  void fn_int(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
              CUDA::NodePtr &node1, std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<uint32_t> &bases);
  void fn_ntt(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
              CUDA::NodePtr &node1, std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<uint32_t> &bases);
  void fn_sud(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
              CUDA::NodePtr &node1, std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<uint32_t> &bases,
              const std::vector<LimbDataType> &sud_factors);
  void fn_rsv(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
              std::vector<LimbDataType *> &outputs,
              const std::vector<LimbDataType *> &inputs,
              const std::vector<LimbDataType> &factors,
              const std::vector<LimbDataType> &base,
              const uint64_t barrett_ratio, const uint32_t barrett_k);
  void fn_bco(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
              std::vector<DeviceBaseConverter *> &base_converters);
  void drm(CUDA::GraphPtr &graph, CUDA::NodePtr &node0, CUDA::NodePtr &node1,
           std::vector<LimbDataType *> &gather,
           const std::vector<LimbDataType *> scatter, const size_t comm_size,
           const size_t num_comm_limbs, const size_t tid);
  void ags(CUDA::GraphPtr &graph, CUDA::NodePtr &node0, CUDA::NodePtr &node1,
           std::vector<LimbDataType *> &outputs,
           const std::vector<LimbDataType *> inputs, const size_t comm_size,
           const size_t num_comm_limbs, const size_t tid,
           const std::vector<uint32_t> &bases);
  void ard(CUDA::GraphPtr &graph, CUDA::NodePtr &node0, CUDA::NodePtr &node1,
           std::vector<LimbDataType *> &outputs,
           const std::vector<LimbDataType *> inputs, const size_t comm_size,
           const size_t num_comm_limbs, const size_t tid,
           const std::vector<uint32_t> &bases);

  void print_memory_location(const uint32_t **ptr, size_t num_locations);

private:
  const Context &context_;
  std::shared_ptr<EvaluatorContext> evaluator_context_;
  std::size_t coeff_count_;
  std::size_t coeff_count_power_;

  std::vector<DevicePointer<LimbDataType *>> args_vec;
  std::vector<DevicePointer<LimbDataType>> args_sca;
  std::vector<DevicePointer<size_t>> args_size;
  std::vector<DevicePointer<const uint32_t *>> args_rot;

  size_t NCCL_BUFFER_MAX_LIMBS;
  CUDA::NCCL::BufferPtr nccl_buffer___;
  LimbT::Element_t *nccl_buffer_;

  CUDA::StreamPtr stream_;

  void init();
};
} // namespace Runtime
} // namespace Cerium
