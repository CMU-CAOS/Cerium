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
#include "fhe_bconv.cuh"
#include "fhe_graph.cuh"
#include "fhe_mem.cuh"
#include "fhe_ntt.cuh"
#include <cstdint>
#include <iostream>
#include <unordered_map>

// Shared CUDA-facing aliases and launch constants used by the generated kernels.
using LimbDataType = uint32_t;
using LimbDataType2 = uint2;
using FunctionMapType = std::unordered_map<uint32_t, void *>;

static constexpr size_t BLOCK_DIM_X = 256;
static constexpr size_t GRID_DIM_X = 256;

// Simple post-launch error check that reports the call site on failure.
static void check_cuda(const char *const file, const int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA Error ");
  }
}
#define CHECK_CUDA_ERROR() check_cuda(__FILE__, __LINE__)
