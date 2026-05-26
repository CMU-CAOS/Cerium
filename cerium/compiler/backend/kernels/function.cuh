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

#include "fhe.cuh"

namespace Cerium {
namespace __FNAME__ {

// Host-side entry points for the generated NTT, SUD, and inverse-transform stages.
__host__ void _function_ntt_(cudaStream_t &stream, LimbDataType **args_outputs,
                             const LimbDataType **args_inputs,
                             const uint32_t *bases, const size_t num_bases,
                             const LimbDataType *modulii,
                             const LimbDataType2 *power_of_roots);
__host__ void
_function_ntt_cugraph_(cudaGraph_t &graph, cudaGraphNode_t &node0,
                       cudaGraphNode_t &node1, LimbDataType **args_outputs,
                       const LimbDataType **args_inputs, const uint32_t *bases,
                       const size_t num_bases, const LimbDataType *modulii,
                       const LimbDataType2 *power_of_roots);
__host__ void _function_sud_(cudaStream_t &stream, LimbDataType **args_outputs,
                             const LimbDataType **args_inputs,
                             const uint32_t *bases, const size_t num_bases,
                             const LimbDataType *modulii,
                             const LimbDataType2 *power_of_roots,
                             const LimbDataType *sud_factors,
                             const LimbDataType *barrett_ratios,
                             const uint32_t *barrett_k);
__host__ void _function_sud_cugraph_(
    cudaGraph_t &graph, cudaGraphNode_t &node0, cudaGraphNode_t &node1,
    LimbDataType **args_outputs, const LimbDataType **args_inputs,
    const uint32_t *bases, const size_t num_bases, const LimbDataType *modulii,
    const LimbDataType2 *power_of_roots, const LimbDataType *sud_factors,
    const LimbDataType *barrett_ratios, const uint32_t *barrett_k);
__host__ void
_function_int_(cudaStream_t &stream, LimbDataType **args_outputs,
               const LimbDataType **args_inputs, const uint32_t *bases,
               const size_t num_bases, const LimbDataType *modulii,
               const LimbDataType2 *inverse_power_of_roots_div_two);
__host__ void
_function_int_cugraph_(cudaGraph_t &graph, cudaGraphNode_t &node0,
                       cudaGraphNode_t &node1, LimbDataType **args_outputs,
                       const LimbDataType **args_inputs, const uint32_t *bases,
                       const size_t num_bases, const LimbDataType *modulii,
                       const LimbDataType2 *inverse_power_of_roots_div_two);

template <size_t NUM_INPUT_BASES, size_t NUM_OUTPUT_BASES>
// Base conversion kernel used by the bootstrapping pipeline.
__global__ static void __launch_bounds__(BLOCK_DIM_X)
    _kernel_bco(LimbDataType **args_outputs, const size_t *num_output_bases,
                const LimbDataType **args_inputs, const size_t *num_input_bases,
                const uint32_t **base_conversion_factors,
                const LimbDataType **output_modulii,
                const LimbDataType **barrett_ratios,
                const uint32_t **barrett_k) {
  (void)num_output_bases;
  (void)num_input_bases;

  // Each y-dimension block processes one conversion instance.
  const size_t j = blockIdx.y;
  convert<NUM_INPUT_BASES, NUM_OUTPUT_BASES>(
      args_inputs[j], args_outputs[j], base_conversion_factors[j],
      output_modulii[j], barrett_ratios[j], barrett_k[j]);
}

template <size_t NUM_INPUT_BASES, size_t NUM_OUTPUT_BASES>
__host__ void
_function_bco_(cudaStream_t &stream, LimbDataType **args_outputs,
               const size_t *num_output_bases, const LimbDataType **args_inputs,
               const size_t *num_input_bases,
               const LimbDataType **base_conversion_factors,
               const LimbDataType **output_modulii,
               const LimbDataType **barrett_ratios, const uint32_t **barrett_k,
               const size_t shared_mem_size, const size_t num_conversions) {
  // Stream launch wrapper for the generated base-conversion kernel.
  dim3 gridSize(GRID_DIM_X, num_conversions);
  dim3 blockSize(BLOCK_DIM_X, 1, 1);
  _kernel_bco<NUM_INPUT_BASES, NUM_OUTPUT_BASES>
      <<<gridSize, blockSize, shared_mem_size, stream>>>(
          args_outputs, num_output_bases, args_inputs, num_input_bases,
          base_conversion_factors, output_modulii, barrett_ratios, barrett_k);
  CHECK_CUDA_ERROR();
}

template <size_t NUM_INPUT_BASES, size_t NUM_OUTPUT_BASES>
__host__ void _function_bco_cugraph_(
    cudaGraph_t &graph, cudaGraphNode_t &node0, LimbDataType **args_outputs,
    const size_t *num_output_bases, const LimbDataType **args_inputs,
    const size_t *num_input_bases, const LimbDataType **base_conversion_factors,
    const LimbDataType **output_modulii, const LimbDataType **barrett_ratios,
    const uint32_t **barrett_k, const size_t shared_mem_size,
    const size_t num_conversions) {
  // Graph launch wrapper so the same kernel can be captured into a CUDA graph.
  dim3 gridSize(GRID_DIM_X, num_conversions);
  dim3 blockSize(BLOCK_DIM_X, 1, 1);
  void *args1[8] = {(void *)&args_outputs,
                    (void *)&num_output_bases,
                    (void *)&args_inputs,
                    (void *)&num_input_bases,
                    (void *)&base_conversion_factors,
                    (void *)&output_modulii,
                    (void *)&barrett_ratios,
                    (void *)&barrett_k};
  node0 = set_kernel_node_params(
      graph, (void *)_kernel_bco<NUM_INPUT_BASES, NUM_OUTPUT_BASES>, gridSize,
      blockSize, shared_mem_size, (void **)args1);
  CHECK_CUDA_ERROR();
}

} // namespace __FNAME__
} // namespace Cerium
