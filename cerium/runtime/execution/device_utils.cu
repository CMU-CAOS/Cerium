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

#include <cstddef>
#include <cstdint>

#include "cerium/runtime/cuda/cuda_ops.cuh"
#include "cerium/runtime/cuda/mem.cuh"
#include "cerium/runtime/execution/device_utils.h"

namespace Cerium {
namespace Runtime {

// std::size_t GRID_DIM = 2048;
std::size_t BLOCK_DIM = 256;
std::size_t GRID_DIM = 256;

using word32 = uint32_t;
using word64 = uint64_t;
using word128 = unsigned __int128;

struct General {
  typedef const void (*elm_function_t)(
      cudaStream_t &stream, LimbDataType **args_outputs,
      const LimbDataType **args_inputs, const LimbDataType *args_scalars,
      const LimbDataType **remapable_base, const size_t *args_remapables,
      const uint32_t *bases, const LimbDataType *modulii,
      const LimbDataType *barrett_ratios, const uint32_t *barrett_k,
      const uint32_t **rotation_map);
  typedef const void (*int_function_t)(
      cudaStream_t &stream, LimbDataType **args_outputs,
      const LimbDataType **args_inputs, const uint32_t *bases,
      const size_t num_bases, const LimbDataType *modulii,
      const LimbDataType2 *inverse_power_of_roots_div_two);
  typedef const void (*ntt_function_t)(cudaStream_t &stream,
                                       LimbDataType **args_outputs,
                                       const LimbDataType **args_inputs,
                                       const uint32_t *bases,
                                       const size_t num_bases,
                                       const LimbDataType *modulii,
                                       const LimbDataType2 *power_of_roots);
  typedef const void (*sud_function_t)(
      cudaStream_t &stream, LimbDataType **args_outputs,
      const LimbDataType **args_inputs, const uint32_t *bases,
      const size_t num_bases, const LimbDataType *modulii,
      const LimbDataType2 *power_of_roots, const LimbDataType *sud_factors,
      const LimbDataType *barrett_ratios, const uint32_t *barrett_k);
  typedef const void (*rsv_function_t)(cudaStream_t &stream,
                                       LimbDataType **args_outputs,
                                       const LimbDataType **args_inputs,
                                       const LimbDataType *rsv_factors,
                                       const LimbDataType *rsv_modulus,
                                       const LimbDataType *rsv_barrett_ratio,
                                       const uint32_t rsv_barrett_k);
  typedef const void (*bco_function_t)(
      cudaStream_t &stream, LimbDataType **args_outputs,
      const size_t *num_output_bases, const LimbDataType **args_inputs,
      const size_t *num_input_bases, const uint32_t **base_conversion_factors,
      const LimbDataType **output_modulii, const LimbDataType **barrett_ratios,
      const uint32_t **barrett_k, const size_t shared_mem_size,
      const size_t num_conversions);
};

struct CuGraph {
  typedef const void (*elm_function_t)(
      cudaGraph_t &graph, cudaGraphNode_t &node0, LimbDataType **args_outputs,
      const LimbDataType **args_inputs, const LimbDataType *args_scalars,
      const LimbDataType **remapable_base, const size_t *args_remapables,
      const uint32_t *bases, const LimbDataType *modulii,
      const LimbDataType *barrett_ratios, const uint32_t *barrett_k,
      const uint32_t **rotation_map);
  typedef const void (*int_function_t)(
      cudaGraph_t &graph, cudaGraphNode_t &node0, cudaGraphNode_t &node1,
      LimbDataType **args_outputs, const LimbDataType **args_inputs,
      const uint32_t *bases, const size_t num_bases,
      const LimbDataType *modulii,
      const LimbDataType2 *inverse_power_of_roots_div_two);
  typedef const void (*ntt_function_t)(
      cudaGraph_t &graph, cudaGraphNode_t &node0, cudaGraphNode_t &node1,
      LimbDataType **args_outputs, const LimbDataType **args_inputs,
      const uint32_t *bases, const size_t num_bases,
      const LimbDataType *modulii, const LimbDataType2 *power_of_roots);
  typedef const void (*sud_function_t)(
      cudaGraph_t &graph, cudaGraphNode_t &node0, cudaGraphNode_t &node1,
      LimbDataType **args_outputs, const LimbDataType **args_inputs,
      const uint32_t *bases, const size_t num_bases,
      const LimbDataType *modulii, const LimbDataType2 *power_of_roots,
      const LimbDataType *sud_factors, const LimbDataType *barrett_ratios,
      const uint32_t *barrett_k);
  typedef const void (*rsv_function_t)(
      cudaGraph_t &graph, cudaGraphNode_t &node0, LimbDataType **args_outputs,
      const LimbDataType **args_inputs, const LimbDataType *rsv_factors,
      const LimbDataType *rsv_modulus, const LimbDataType *rsv_barrett_ratio,
      const uint32_t rsv_barrett_k);
  typedef const void (*bco_function_t)(
      cudaGraph_t &graph, cudaGraphNode_t &node0, LimbDataType **args_outputs,
      const size_t *num_output_bases, const LimbDataType **args_inputs,
      const size_t *num_input_bases, const uint32_t **base_conversion_factors,
      const LimbDataType **output_modulii, const LimbDataType **barrett_ratios,
      const uint32_t **barrett_k, const size_t shared_mem_size,
      const size_t num_conversions);
};

void execute_ewi(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const LimbDataType *args_scalars,
                 const LimbDataType **remapable_base,
                 const size_t *args_remapables, const uint32_t *bases,
                 const LimbDataType *modulii,
                 const LimbDataType *barrett_ratios, const uint32_t *barrett_k,
                 const uint32_t **rotation_map) {
  auto fn = (General::elm_function_t)(function);
  fn(stream->stream_, args_outputs, args_inputs, args_scalars, remapable_base,
     args_remapables, bases, modulii, barrett_ratios, barrett_k, rotation_map);
  CHECK_CUDA_ERROR();
}

void execute_int(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const uint32_t *bases, const size_t num_bases,
                 const LimbDataType *modulii,
                 const LimbDataType2 *inverse_power_of_roots_div_two) {
  auto fn = (General::int_function_t)(function);
  fn(stream->stream_, args_outputs, args_inputs, bases, num_bases, modulii,
     inverse_power_of_roots_div_two);
  CHECK_CUDA_ERROR();
}

void execute_ntt(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const uint32_t *bases, const size_t num_bases,
                 const LimbDataType *modulii,
                 const LimbDataType2 *power_of_roots) {
  auto fn = (General::ntt_function_t)(function);
  fn(stream->stream_, args_outputs, args_inputs, bases, num_bases, modulii,
     power_of_roots);
  CHECK_CUDA_ERROR();
}

void execute_sud(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const uint32_t *bases, const size_t num_bases,
                 const LimbDataType *modulii,
                 const LimbDataType2 *power_of_roots,
                 const LimbDataType *sud_factors,
                 const LimbDataType *barrett_ratios,
                 const uint32_t *barrett_k) {
  auto fn = (General::sud_function_t)(function);
  fn(stream->stream_, args_outputs, args_inputs, bases, num_bases, modulii,
     power_of_roots, sud_factors, barrett_ratios, barrett_k);
  CHECK_CUDA_ERROR();
}

void execute_rsv(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const LimbDataType *rsv_factors,
                 const LimbDataType *rsv_modulus,
                 const LimbDataType *rsv_barrett_ratio,
                 const uint32_t rsv_barrett_k) {
  auto fn = (General::rsv_function_t)(function);
  fn(stream->stream_, args_outputs, args_inputs, rsv_factors, rsv_modulus,
     rsv_barrett_ratio, rsv_barrett_k);
  CHECK_CUDA_ERROR();
}

void execute_bco(void *function, const CUDA::StreamPtr &stream,
                 LimbDataType **args_outputs, const size_t *num_output_bases,
                 const LimbDataType **args_inputs,
                 const size_t *num_input_bases,
                 const uint32_t **base_conversion_factors,
                 const LimbDataType **output_modulii,
                 const LimbDataType **barrett_ratios,
                 const uint32_t **barrett_k, const size_t shared_mem_size,
                 const size_t num_conversions) {
  auto fn = (General::bco_function_t)(function);
  fn(stream->stream_, args_outputs, num_output_bases, args_inputs,
     num_input_bases, base_conversion_factors, output_modulii, barrett_ratios,
     barrett_k, shared_mem_size, num_conversions);
  CHECK_CUDA_ERROR();
}

void execute_ewi(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const LimbDataType *args_scalars,
                 const LimbDataType **remapable_base,
                 const size_t *args_remapables, const uint32_t *bases,
                 const LimbDataType *modulii,
                 const LimbDataType *barrett_ratios, const uint32_t *barrett_k,
                 const uint32_t **rotation_map) {
  auto fn = (CuGraph::elm_function_t)(function);
  fn(graph->graph_, node0->node_, args_outputs, args_inputs, args_scalars,
     remapable_base, args_remapables, bases, modulii, barrett_ratios, barrett_k,
     rotation_map);
  CHECK_CUDA_ERROR();
}

void execute_int(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 CUDA::NodePtr &node1, LimbDataType **args_outputs,
                 const LimbDataType **args_inputs, const uint32_t *bases,
                 const size_t num_bases, const LimbDataType *modulii,
                 const LimbDataType2 *inverse_power_of_roots_div_two) {
  auto fn = (CuGraph::int_function_t)(function);
  fn(graph->graph_, node0->node_, node1->node_, args_outputs, args_inputs,
     bases, num_bases, modulii, inverse_power_of_roots_div_two);
  CHECK_CUDA_ERROR();
}

void execute_ntt(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 CUDA::NodePtr &node1, LimbDataType **args_outputs,
                 const LimbDataType **args_inputs, const uint32_t *bases,
                 const size_t num_bases, const LimbDataType *modulii,
                 const LimbDataType2 *power_of_roots) {
  auto fn = (CuGraph::ntt_function_t)(function);
  fn(graph->graph_, node0->node_, node1->node_, args_outputs, args_inputs,
     bases, num_bases, modulii, power_of_roots);
  CHECK_CUDA_ERROR();
}

void execute_sud(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 CUDA::NodePtr &node1, LimbDataType **args_outputs,
                 const LimbDataType **args_inputs, const uint32_t *bases,
                 const size_t num_bases, const LimbDataType *modulii,
                 const LimbDataType2 *power_of_roots,
                 const LimbDataType *sud_factors,
                 const LimbDataType *barrett_ratios,
                 const uint32_t *barrett_k) {
  auto fn = (CuGraph::sud_function_t)(function);
  fn(graph->graph_, node0->node_, node1->node_, args_outputs, args_inputs,
     bases, num_bases, modulii, power_of_roots, sud_factors, barrett_ratios,
     barrett_k);
  CHECK_CUDA_ERROR();
}

void execute_rsv(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 LimbDataType **args_outputs, const LimbDataType **args_inputs,
                 const LimbDataType *rsv_factors,
                 const LimbDataType *rsv_modulus,
                 const LimbDataType *rsv_barrett_ratio,
                 const uint32_t rsv_barrett_k) {
  auto fn = (CuGraph::rsv_function_t)(function);
  fn(graph->graph_, node0->node_, args_outputs, args_inputs, rsv_factors,
     rsv_modulus, rsv_barrett_ratio, rsv_barrett_k);
  CHECK_CUDA_ERROR();
}

void execute_bco(void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                 LimbDataType **args_outputs, const size_t *num_output_bases,
                 const LimbDataType **args_inputs,
                 const size_t *num_input_bases,
                 const uint32_t **base_conversion_factors,
                 const LimbDataType **output_modulii,
                 const LimbDataType **barrett_ratios,
                 const uint32_t **barrett_k, const size_t shared_mem_size,
                 const size_t num_conversions) {
  auto fn = (CuGraph::bco_function_t)(function);
  fn(graph->graph_, node0->node_, args_outputs, num_output_bases, args_inputs,
     num_input_bases, base_conversion_factors, output_modulii, barrett_ratios,
     barrett_k, shared_mem_size, num_conversions);
  CHECK_CUDA_ERROR();
}

__host__ cudaGraphNode_t set_kernel_node_params(
    cudaGraph_t graph_, void *fn, const dim3 grid_dim, const dim3 block_dim,
    const unsigned int sharedMemBytes, void **args) {
  cudaGraphNode_t node;
  cudaKernelNodeParams kernelNodeParams{0};
  kernelNodeParams.func = fn;
  kernelNodeParams.gridDim = grid_dim;
  kernelNodeParams.blockDim = block_dim;
  kernelNodeParams.sharedMemBytes = sharedMemBytes;
  kernelNodeParams.kernelParams = args;
  kernelNodeParams.extra = NULL;

  cudaGraphAddKernelNode(&node, graph_, NULL, 0, &kernelNodeParams);
  return node;
}

__global__ void __attribute__((optimize("O3")))
_copy_to_nccl_buffer_(LimbDataType *nccl_buffer, const LimbDataType **inputs,
                      const size_t num_inputs, const size_t length) {
  auto j = blockIdx.y;
  auto input_ptr = PTX::ldu_global_u32_ptr(&inputs[j]);
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;

  if (input_ptr == nullptr) {
    PTX::st_na_global_u32(&nccl_buffer[j * length + i], 0);
  } else {
    PTX::st_na_global_u32(&nccl_buffer[j * length + i],
                          PTX::ld_nc_global_u32(&input_ptr[i]));
  }
  return;
}

__global__ void __attribute__((optimize("O3")))
_copy_from_nccl_buffer_(LimbDataType **outputs, const LimbDataType *nccl_buffer,
                        const size_t num_inputs, const size_t length) {
  auto j = blockIdx.y;
  auto output_ptr = PTX::ldu_global_u32_ptr(&outputs[j]);
  if (output_ptr == nullptr) {
    return;
  }
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  PTX::st_na_global_u32(&output_ptr[i],
                        PTX::ld_nc_global_u32(&nccl_buffer[j * length + i]));
}

__global__ void __attribute__((optimize("O3")))
_copy_from_nccl_buffer_with_mod_(LimbDataType **outputs,
                                 const LimbDataType *nccl_buffer,
                                 const size_t num_inputs, const size_t length,
                                 const uint32_t *bases,
                                 const LimbDataType *moduli) {
  auto j = blockIdx.y;
  auto output_ptr = PTX::ldu_global_u32_ptr(&outputs[j]);
  if (output_ptr == nullptr) {
    return;
  }
  auto modulus = PTX::ldu_global_u32(&moduli[PTX::ldu_global_u32(&bases[j])]);
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  auto t = PTX::ld_nc_global_u32(&nccl_buffer[j * length + i]);
#pragma unroll
  for (int i = 0; i < 9; i++) {
    if (t >= modulus) {
      t -= modulus;
    }
  }
  PTX::st_na_global_u32(&output_ptr[i], t);
}

void copy_to_nccl_buffer(const CUDA::StreamPtr &stream,
                         LimbDataType *nccl_buffer, const LimbDataType **inputs,
                         const size_t num_inputs, const size_t length) {
  dim3 gridDim(GRID_DIM, num_inputs, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  _copy_to_nccl_buffer_<<<gridDim, blockDim, 0, stream->stream_>>>(
      nccl_buffer, inputs, num_inputs, length);
}
void copy_from_nccl_buffer(const CUDA::StreamPtr &stream,
                           LimbDataType **outputs,
                           const LimbDataType *nccl_buffer,
                           const size_t num_outputs, const size_t length) {
  dim3 gridDim(GRID_DIM, num_outputs, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  _copy_from_nccl_buffer_<<<gridDim, blockDim, 0, stream->stream_>>>(
      outputs, nccl_buffer, num_outputs, length);
}
void copy_to_nccl_buffer(CUDA::GraphPtr &graph, CUDA::NodePtr &node,
                         LimbDataType *nccl_buffer, const LimbDataType **inputs,
                         const size_t num_inputs, const size_t length) {
  dim3 gridDim(GRID_DIM, num_inputs, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  void *args[4] = {(void *)&nccl_buffer, (void *)&inputs, (void *)&num_inputs,
                   (void *)&length};
  cudaGraphNode_t new_node =
      set_kernel_node_params(graph->graph_, (void *)_copy_to_nccl_buffer_,
                             gridDim, blockDim, 0, (void **)args);
  node->node_ = new_node;
  CHECK_CUDA_ERROR();
}
void copy_from_nccl_buffer(CUDA::GraphPtr &graph, CUDA::NodePtr &node,
                           LimbDataType **outputs,
                           const LimbDataType *nccl_buffer,
                           const size_t num_outputs, const size_t length) {
  dim3 gridDim(GRID_DIM, num_outputs, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  void *args[4] = {(void *)&outputs, (void *)&nccl_buffer, (void *)&num_outputs,
                   (void *)&length};
  cudaGraphNode_t new_node =
      set_kernel_node_params(graph->graph_, (void *)_copy_from_nccl_buffer_,
                             gridDim, blockDim, 0, (void **)args);
  node->node_ = new_node;
  CHECK_CUDA_ERROR();
}

void copy_from_nccl_buffer_with_mod(const CUDA::StreamPtr &stream,
                                    LimbDataType **outputs,
                                    const LimbDataType *nccl_buffer,
                                    const size_t num_outputs,
                                    const size_t length, const uint32_t *bases,
                                    const LimbDataType *moduli) {
  dim3 gridDim(GRID_DIM, num_outputs, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  _copy_from_nccl_buffer_with_mod_<<<gridDim, blockDim, 0, stream->stream_>>>(
      outputs, nccl_buffer, num_outputs, length, bases, moduli);
}
void copy_from_nccl_buffer_with_mod(CUDA::GraphPtr &graph, CUDA::NodePtr &node,
                                    LimbDataType **outputs,
                                    const LimbDataType *nccl_buffer,
                                    const size_t num_outputs,
                                    const size_t length, const uint32_t *bases,
                                    const LimbDataType *moduli) {
  dim3 gridDim(GRID_DIM, num_outputs, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  void *args[6] = {(void *)&outputs, (void *)&nccl_buffer, (void *)&num_outputs,
                   (void *)&length,  (void *)&bases,       (void *)&moduli};
  cudaGraphNode_t new_node = set_kernel_node_params(
      graph->graph_, (void *)_copy_from_nccl_buffer_with_mod_, gridDim,
      blockDim, 0, (void **)args);
  node->node_ = new_node;
  CHECK_CUDA_ERROR();
}

__global__ void _set_pointer_(LimbDataType **ptr, LimbDataType *val) {
  *ptr = val;
}

void set_pointer(LimbDataType **ptr, LimbDataType *val) {
  _set_pointer_<<<1, 1>>>(ptr, val);
  CHECK_CUDA_ERROR();
}

void set_pointer(LimbDataType **ptr, LimbDataType *val,
                 const CUDA::StreamPtr &stream) {
  _set_pointer_<<<1, 1, 0, stream->stream_>>>(ptr, val);
  CHECK_CUDA_ERROR();
}

} // namespace Runtime
} // namespace Cerium
