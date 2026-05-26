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
#include <cstdlib>

#include "cerium/runtime/cuda/cuda_ops.h"
#include "cerium/runtime/cuda/device_limb.h"

namespace Cerium {
namespace Runtime {

using LimbDataType = DeviceLimb::Element_t;
using LimbDataType2 = DeviceLimb::Element_t2;

void execute_ewi(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const LimbDataType *args_scalars,
                 const LimbDataType **remapable_base,
                 const size_t *args_remapables, const uint32_t *bases,
                 const LimbDataType *modulii,
                 const LimbDataType *barrett_ratios, const uint32_t *barrett_k,
                 const uint32_t **rotation_map);

void execute_int(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const uint32_t *bases, const size_t num_bases,
                 const LimbDataType *modulii,
                 const LimbDataType2 *inverse_power_of_roots_div_two);

void execute_ntt(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const uint32_t *bases, const size_t num_bases,
                 const LimbDataType *modulii,
                 const LimbDataType2 *power_of_roots);

void execute_sud(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const uint32_t *bases, const size_t num_bases,
                 const LimbDataType *modulii,
                 const LimbDataType2 *power_of_roots,
                 const LimbDataType *sud_factors,
                 const LimbDataType *barrett_ratios, const uint32_t *barrett_k);

void execute_rsv(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const LimbDataType *rsv_factors,
                 const LimbDataType *rsv_modulus,
                 const LimbDataType *rsv_barrett_ratio,
                 const uint32_t rsv_barrett_k);

void execute_bco(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const size_t *num_output_bases,
                 const LimbDataType **args_inputs,
                 const size_t *num_input_bases,
                 const uint32_t **base_conversion_factors,
                 const LimbDataType **output_modulii,
                 const LimbDataType **barrett_ratios,
                 const uint32_t **barrett_k, const size_t shared_mem_size,
                 const size_t num_conversions);

void execute_ewi(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const LimbDataType *args_scalars,
                 const LimbDataType **remapable_base,
                 const size_t *args_remapables, const uint32_t *bases,
                 const LimbDataType *modulii,
                 const LimbDataType *barrett_ratios, const uint32_t *barrett_k,
                 const uint32_t **rotation_map);

void execute_int(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 CUDA::NodePtr &node1, LimbDataType **args_outputs,
                 const LimbDataType **args_inputs, const uint32_t *bases,
                 const size_t num_bases, const LimbDataType *modulii,
                 const LimbDataType2 *inverse_power_of_roots_div_two);

void execute_ntt(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 CUDA::NodePtr &node1, LimbDataType **args_outputs,
                 const LimbDataType **args_inputs, const uint32_t *bases,
                 const size_t num_bases, const LimbDataType *modulii,
                 const LimbDataType2 *power_of_roots);

void execute_sud(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 CUDA::NodePtr &node1, LimbDataType **args_outputs,
                 const LimbDataType **args_inputs, const uint32_t *bases,
                 const size_t num_bases, const LimbDataType *modulii,
                 const LimbDataType2 *power_of_roots,
                 const LimbDataType *sud_factors,
                 const LimbDataType *barrett_ratios, const uint32_t *barrett_k);

void execute_rsv(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const LimbDataType *rsv_factors,
                 const LimbDataType *rsv_modulus,
                 const LimbDataType *rsv_barrett_ratio,
                 const uint32_t rsv_barrett_k);

void execute_bco(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 LimbDataType **args_outputs, const size_t *num_output_bases,
                 const LimbDataType **args_inputs,
                 const size_t *num_input_bases,
                 const uint32_t **base_conversion_factors,
                 const LimbDataType **output_modulii,
                 const LimbDataType **barrett_ratios,
                 const uint32_t **barrett_k, const size_t shared_mem_size,
                 const size_t num_conversions);

void copy_to_nccl_buffer(const CUDA::StreamPtr &stream,
                         LimbDataType *nccl_buffer, const LimbDataType **inputs,
                         const size_t num_inputs, const size_t length);
void copy_from_nccl_buffer(const CUDA::StreamPtr &stream,
                           LimbDataType **outputs,
                           const LimbDataType *nccl_buffer,
                           const size_t num_outputs, const size_t length);
void copy_from_nccl_buffer_with_mod(const CUDA::StreamPtr &stream,
                                    LimbDataType **outputs,
                                    const LimbDataType *nccl_buffer,
                                    const size_t num_outputs,
                                    const size_t length, const uint32_t *bases,
                                    const LimbDataType *moduli);
void copy_to_nccl_buffer(CUDA::GraphPtr &graph, CUDA::NodePtr &node,
                         LimbDataType *nccl_buffer, const LimbDataType **inputs,
                         const size_t num_inputs, const size_t length);
void copy_from_nccl_buffer(CUDA::GraphPtr &graph, CUDA::NodePtr &node,
                           LimbDataType **outputs,
                           const LimbDataType *nccl_buffer,
                           const size_t num_outputs, const size_t length);
void copy_from_nccl_buffer_with_mod(CUDA::GraphPtr &graph, CUDA::NodePtr &node,
                                    LimbDataType **outputs,
                                    const LimbDataType *nccl_buffer,
                                    const size_t num_outputs,
                                    const size_t length, const uint32_t *bases,
                                    const LimbDataType *moduli);

void set_pointer(LimbDataType **ptr, LimbDataType *val);
void set_pointer(LimbDataType **ptr, LimbDataType *val,
                 const CUDA::StreamPtr &stream);

} // namespace Runtime
} // namespace Cerium
