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

// Lightweight helper for constructing graph nodes when the real kernel is not needed.
__global__ static void empty_kernel() { return; }

// Shared wrapper for building a CUDA graph kernel node from launch parameters.
static __host__ cudaGraphNode_t set_kernel_node_params_empty(
    cudaGraph_t &graph, void *fn, const dim3 grid_dim, const dim3 block_dim,
    const unsigned int sharedMemBytes, void **args) {
  (void)fn;
  cudaGraphNode_t node;
  cudaKernelNodeParams kernelNodeParams{0};
  kernelNodeParams.func = (void *)empty_kernel;
  kernelNodeParams.gridDim = grid_dim;
  kernelNodeParams.blockDim = block_dim;
  kernelNodeParams.sharedMemBytes = sharedMemBytes;
  kernelNodeParams.kernelParams = args;
  kernelNodeParams.extra = NULL;

  cudaGraphAddKernelNode(&node, graph, NULL, 0, &kernelNodeParams);
  return node;
}

// Attach the requested kernel function to a graph node using the provided launch configuration.
static __host__ cudaGraphNode_t set_kernel_node_params(
    cudaGraph_t &graph, void *fn, const dim3 grid_dim, const dim3 block_dim,
    const unsigned int sharedMemBytes, void **args) {
  // return set_kernel_node_params_empty(graph,fn,grid_dim,block_dim,sharedMemBytes,args);
  cudaGraphNode_t node;
  cudaKernelNodeParams kernelNodeParams{0};
  kernelNodeParams.func = fn;
  kernelNodeParams.gridDim = grid_dim;
  kernelNodeParams.blockDim = block_dim;
  kernelNodeParams.sharedMemBytes = sharedMemBytes;
  kernelNodeParams.kernelParams = args;
  kernelNodeParams.extra = NULL;

  cudaGraphAddKernelNode(&node, graph, NULL, 0, &kernelNodeParams);
  return node;
}
