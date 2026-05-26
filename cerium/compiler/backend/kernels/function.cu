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

#include "fhe.cuh"
#include "function.cuh"

namespace Cerium {
namespace __FNAME__ {

__global__ static void __launch_bounds__(NTT0_BLOCK_DIM_X, 32)
    _kernel_ntt_0(LimbDataType **args_outputs, const LimbDataType **args_inputs,
                  const uint32_t *bases, const LimbDataType *modulii,
                  const LimbDataType2 *power_of_roots) {
  __shared__ LimbDataType shmem[nttShMemSize0];
  const auto batch_idx = blockIdx.y;
  const auto table_idx = ldu_global_u32(&bases[batch_idx]);
  const auto modulus = ldu_global_u32(&modulii[table_idx]);
  const auto roots_table = power_of_roots + (table_idx * NTT_TABLE_SIZE);
  constexpr size_t output_count = 1;
  constexpr size_t input_count = 1;
  auto input_coeffs =
      ldu_global_u32_ptr(&args_inputs[input_count * batch_idx + 0]);
  auto output_coeffs =
      ldu_global_u32_ptr(&args_outputs[output_count * batch_idx + 0]);
  // Stage 0 of the NTT writes a fresh output buffer from the input buffer.
  ntt0(output_coeffs, input_coeffs, roots_table, modulus, shmem);
}

__global__ static void __launch_bounds__(NTT1_BLOCK_DIM_X, 32)
    _kernel_ntt_1(LimbDataType **args_outputs, const LimbDataType **args_inputs,
                  const uint32_t *bases, const LimbDataType *modulii,
                  const LimbDataType2 *power_of_roots) {
  __shared__ LimbDataType shmem[nttShMemSize1];
  const auto batch_idx = blockIdx.y;
  const auto table_idx = ldu_global_u32(&bases[batch_idx]);
  const auto modulus = ldu_global_u32(&modulii[table_idx]);
  const auto roots_table = power_of_roots + (table_idx * NTT_TABLE_SIZE);
  constexpr size_t output_count = 1;
  const auto output_coeffs =
      ldu_global_u32_ptr(&args_outputs[output_count * batch_idx + 0]);
  (void)args_inputs;
  // Stage 1 is in place: it updates the buffer produced by stage 0.
  ntt1(output_coeffs, output_coeffs, roots_table, modulus, shmem);
}

__host__ void _function_ntt_(cudaStream_t &stream, LimbDataType **args_outputs,
                             const LimbDataType **args_inputs,
                             const uint32_t *bases, const size_t num_bases,
                             const LimbDataType *modulii,
                             const LimbDataType2 *power_of_roots) {
  // Launch the two NTT stages back-to-back on the same stream.
  dim3 gridSize0(NTT0_GRID_DIM_X, num_bases, 1);
  dim3 blockSize0(NTT0_BLOCK_DIM_X, 1, 1);
  _kernel_ntt_0<<<gridSize0, blockSize0, 0, stream>>>(
      args_outputs, args_inputs, bases, modulii, power_of_roots);
  dim3 gridSize1(NTT1_GRID_DIM_X, num_bases, 1);
  dim3 blockSize1(NTT1_BLOCK_DIM_X, 1, 1);
  _kernel_ntt_1<<<gridSize1, blockSize1, 0, stream>>>(
      args_outputs, args_inputs, bases, modulii, power_of_roots);
}

__host__ void
_function_ntt_cugraph_(cudaGraph_t &graph, cudaGraphNode_t &node0,
                       cudaGraphNode_t &node1, LimbDataType **args_outputs,
                       const LimbDataType **args_inputs, const uint32_t *bases,
                       const size_t num_bases, const LimbDataType *modulii,
                       const LimbDataType2 *power_of_roots) {
  // Graph nodes mirror the stream launches so the runtime can reuse the same staging logic.
  dim3 gridSize0(NTT0_GRID_DIM_X, num_bases, 1);
  dim3 blockSize0(NTT0_BLOCK_DIM_X, 1, 1);
  dim3 gridSize1(NTT1_GRID_DIM_X, num_bases, 1);
  dim3 blockSize1(NTT1_BLOCK_DIM_X, 1, 1);
  void *kernel_args_stage0[5] = {(void *)&args_outputs, (void *)&args_inputs,
                                 (void *)&bases, (void *)&modulii,
                                 (void *)&power_of_roots};
  node0 = set_kernel_node_params(graph, (void *)_kernel_ntt_0, gridSize0,
                                 blockSize0, 0, (void **)kernel_args_stage0);
  CHECK_CUDA_ERROR();
  void *kernel_args_stage1[5] = {(void *)&args_outputs, (void *)&args_inputs,
                                 (void *)&bases, (void *)&modulii,
                                 (void *)&power_of_roots};
  node1 = set_kernel_node_params(graph, (void *)_kernel_ntt_1, gridSize1,
                                 blockSize1, 0, (void **)kernel_args_stage1);
  CHECK_CUDA_ERROR();
  cudaGraphAddDependencies(graph, &node0, &node1, nullptr, 1);
  CHECK_CUDA_ERROR();
}

__global__ static void __launch_bounds__(NTT0_BLOCK_DIM_X, 32)
    _kernel_sud_0(LimbDataType **args_outputs, const LimbDataType **args_inputs,
                  const uint32_t *bases, const LimbDataType *modulii,
                  const LimbDataType2 *power_of_roots) {
  __shared__ LimbDataType shmem[nttShMemSize0];
  const auto batch_idx = blockIdx.y;
  const auto table_idx = ldu_global_u32(&bases[batch_idx]);
  const auto modulus = ldu_global_u32(&modulii[table_idx]);
  const auto roots_table = power_of_roots + (table_idx * NTT_TABLE_SIZE);
  constexpr size_t output_count = 1;
  constexpr size_t input_count = 2;
  auto first_input_coeffs =
      ldu_global_u32_ptr(&args_inputs[input_count * batch_idx + 0]);
  auto second_input_coeffs =
      ldu_global_u32_ptr(&args_inputs[input_count * batch_idx + 1]);
  auto output_coeffs =
      ldu_global_u32_ptr(&args_outputs[output_count * batch_idx + 0]);
  (void)first_input_coeffs;
  // Stage 0 only consumes the second input tensor and prepares the output buffer.
  ntt0_sud(output_coeffs, second_input_coeffs, roots_table, modulus, shmem);
}

__global__ static void __launch_bounds__(NTT1_BLOCK_DIM_X, 32)
    _kernel_sud_1(LimbDataType **args_outputs, const LimbDataType **args_inputs,
                  const uint32_t *bases, const LimbDataType *modulii,
                  const LimbDataType2 *power_of_roots,
                  const LimbDataType *sud_factors,
                  const LimbDataType *barrett_ratio,
                  const uint32_t *barrett_k) {
  __shared__ LimbDataType shmem[nttShMemSize1];
  const auto batch_idx = blockIdx.y;
  const auto table_idx = ldu_global_u32(&bases[batch_idx]);
  const auto modulus = ldu_global_u32(&modulii[table_idx]);
  const auto roots_table = power_of_roots + (table_idx * NTT_TABLE_SIZE);
  constexpr size_t output_count = 1;
  constexpr size_t input_count = 2;
  auto primary_input_coeffs =
      ldu_global_u32_ptr(&args_inputs[input_count * batch_idx + 0]);
  auto secondary_input_coeffs =
      ldu_global_u32_ptr(&args_inputs[input_count * batch_idx + 1]);
  auto output_coeffs =
      ldu_global_u32_ptr(&args_outputs[output_count * batch_idx + 0]);
  (void)secondary_input_coeffs;
  // Stage 1 updates the stage-0 output buffer in place while using the first input tensor.
  ntt1_sud(output_coeffs, output_coeffs, primary_input_coeffs, roots_table,
           ldu_global_u32(&sud_factors[table_idx]), modulus,
           ldu_global_u32(&barrett_ratio[table_idx]),
           ldu_global_u32(&barrett_k[table_idx]), shmem);
}

__host__ void _function_sud_(cudaStream_t &stream, LimbDataType **args_outputs,
                             const LimbDataType **args_inputs,
                             const uint32_t *bases, const size_t num_bases,
                             const LimbDataType *modulii,
                             const LimbDataType2 *power_of_roots,
                             const LimbDataType *sud_factors,
                             const LimbDataType *barrett_ratios,
                             const uint32_t *barrett_k) {
  // The SUD path has a plain launch and a graph launch that share the same inputs.
  dim3 gridSize0(NTT0_GRID_DIM_X, num_bases, 1);
  dim3 blockSize0(NTT0_BLOCK_DIM_X, 1, 1);
  _kernel_sud_0<<<gridSize0, blockSize0, 0, stream>>>(
      args_outputs, args_inputs, bases, modulii, power_of_roots);
  dim3 gridSize1(NTT1_GRID_DIM_X, num_bases, 1);
  dim3 blockSize1(NTT1_BLOCK_DIM_X, 1, 1);
  _kernel_sud_1<<<gridSize1, blockSize1, 0, stream>>>(
      args_outputs, args_inputs, bases, modulii, power_of_roots, sud_factors,
      barrett_ratios, barrett_k);
}

__host__ void _function_sud_cugraph_(
    cudaGraph_t &graph, cudaGraphNode_t &node0, cudaGraphNode_t &node1,
    LimbDataType **args_outputs, const LimbDataType **args_inputs,
    const uint32_t *bases, const size_t num_bases, const LimbDataType *modulii,
    const LimbDataType2 *power_of_roots, const LimbDataType *sud_factors,
    const LimbDataType *barrett_ratios, const uint32_t *barrett_k) {
  // Match the stream version so graph execution preserves the same two-stage flow.
  dim3 gridSize0(NTT0_GRID_DIM_X, num_bases, 1);
  dim3 blockSize0(NTT0_BLOCK_DIM_X, 1, 1);
  dim3 gridSize1(NTT1_GRID_DIM_X, num_bases, 1);
  dim3 blockSize1(NTT1_BLOCK_DIM_X, 1, 1);
  void *kernel_args_stage0[5] = {(void *)&args_outputs, (void *)&args_inputs,
                                 (void *)&bases, (void *)&modulii,
                                 (void *)&power_of_roots};
  node0 = set_kernel_node_params(graph, (void *)_kernel_sud_0, gridSize0,
                                 blockSize0, 0, (void **)kernel_args_stage0);
  CHECK_CUDA_ERROR();
  void *kernel_args_stage1[8] = {(void *)&args_outputs,   (void *)&args_inputs,
                                 (void *)&bases,          (void *)&modulii,
                                 (void *)&power_of_roots, (void *)&sud_factors,
                                 (void *)&barrett_ratios, (void *)&barrett_k};
  node1 = set_kernel_node_params(graph, (void *)_kernel_sud_1, gridSize1,
                                 blockSize1, 0, (void **)kernel_args_stage1);
  CHECK_CUDA_ERROR();
  cudaGraphAddDependencies(graph, &node0, &node1, nullptr, 1);
  CHECK_CUDA_ERROR();
}

__global__ static void
_kernel_int_0(LimbDataType **args_outputs, const LimbDataType **args_inputs,
              const uint32_t *bases, const LimbDataType *modulii,
              const LimbDataType2 *inverse_power_of_roots_div_two) {
  __shared__ LimbDataType shmem[intShMemSize0];
  const auto batch_idx = blockIdx.y;
  const auto table_idx = ldu_global_u32(&bases[batch_idx]);
  const auto modulus = ldu_global_u32(&modulii[table_idx]);
  const auto inverse_roots_table =
      inverse_power_of_roots_div_two + (table_idx * NTT_TABLE_SIZE);
  constexpr size_t output_count = 1;
  constexpr size_t input_count = 1;
  auto input_coeffs =
      ldu_global_u32_ptr(&args_inputs[input_count * batch_idx + 0]);
  auto output_coeffs =
      ldu_global_u32_ptr(&args_outputs[output_count * batch_idx + 0]);
  // The inverse transform runs from the source buffer into the destination buffer.
  intt0(output_coeffs, input_coeffs, inverse_roots_table, modulus, shmem);
}

__global__ static void
_kernel_int_1(LimbDataType **args_outputs, const LimbDataType **args_inputs,
              const uint32_t *bases, const LimbDataType *modulii,
              const LimbDataType2 *inverse_power_of_roots_div_two) {
  __shared__ LimbDataType shmem[intShMemSize1];
  const auto batch_idx = blockIdx.y;
  const auto table_idx = ldu_global_u32(&bases[batch_idx]);
  const auto modulus = ldu_global_u32(&modulii[table_idx]);
  const auto inverse_roots_table =
      inverse_power_of_roots_div_two + (table_idx * NTT_TABLE_SIZE);
  constexpr size_t output_count = 1;
  const auto output_coeffs =
      ldu_global_u32_ptr(&args_outputs[output_count * batch_idx + 0]);
  (void)args_inputs;
  // Stage 1 is in place for the inverse transform as well.
  intt1(output_coeffs, output_coeffs, inverse_roots_table, modulus, shmem);
}

__host__ void
_function_int_(cudaStream_t &stream, LimbDataType **args_outputs,
               const LimbDataType **args_inputs, const uint32_t *bases,
               const size_t num_bases, const LimbDataType *modulii,
               const LimbDataType2 *inverse_power_of_roots_div_two) {
  // Both inverse-transform stages run sequentially in the same stream.
  dim3 gridSize0(INT0_GRID_DIM_X, num_bases, 1);
  dim3 blockSize0(INT0_BLOCK_DIM_X, 1, 1);
  _kernel_int_0<<<gridSize0, blockSize0, 0, stream>>>(
      args_outputs, args_inputs, bases, modulii,
      inverse_power_of_roots_div_two);
  dim3 gridSize1(INT1_GRID_DIM_X, num_bases, 1);
  dim3 blockSize1(INT1_BLOCK_DIM_X, 1, 1);
  _kernel_int_1<<<gridSize1, blockSize1, 0, stream>>>(
      args_outputs, args_inputs, bases, modulii,
      inverse_power_of_roots_div_two);
}

__host__ void
_function_int_cugraph_(cudaGraph_t &graph, cudaGraphNode_t &node0,
                       cudaGraphNode_t &node1, LimbDataType **args_outputs,
                       const LimbDataType **args_inputs, const uint32_t *bases,
                       const size_t num_bases, const LimbDataType *modulii,
                       const LimbDataType2 *inverse_power_of_roots_div_two) {
  // Graph execution mirrors the stream execution path for the inverse transform.
  dim3 gridSize0(INT0_GRID_DIM_X, num_bases, 1);
  dim3 blockSize0(INT0_BLOCK_DIM_X, 1, 1);
  void *kernel_args_stage0[5] = {(void *)&args_outputs, (void *)&args_inputs,
                                 (void *)&bases, (void *)&modulii,
                                 (void *)&inverse_power_of_roots_div_two};
  node0 = set_kernel_node_params(graph, (void *)_kernel_int_0, gridSize0,
                                 blockSize0, 0, (void **)kernel_args_stage0);
  CHECK_CUDA_ERROR();
  dim3 gridSize1(INT1_GRID_DIM_X, num_bases, 1);
  dim3 blockSize1(INT1_BLOCK_DIM_X, 1, 1);
  void *kernel_args_stage1[5] = {(void *)&args_outputs, (void *)&args_inputs,
                                 (void *)&bases, (void *)&modulii,
                                 (void *)&inverse_power_of_roots_div_two};
  node1 = set_kernel_node_params(graph, (void *)_kernel_int_1, gridSize1,
                                 blockSize1, 0, (void **)kernel_args_stage1);
  CHECK_CUDA_ERROR();
  cudaGraphAddDependencies(graph, &node0, &node1, nullptr, 1);
  CHECK_CUDA_ERROR();
}

} // namespace __FNAME__
} // namespace Cerium
