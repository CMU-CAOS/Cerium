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
#include <cstdint>

// This header is intentionally specialized for one PTX policy configuration.
// The load/store helpers below provide a narrow wrapper layer around the exact
// instructions used by the generated kernels in this release.

// #define DISABLE_AGGRESSIVE_PTX_INSTRS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.cs.global.nc.L2::256B"
#else
#define LD_NC_FUNC "ld.cs.global.L2::256B"
#endif

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_FUNC "ld.cs.global.nc.L2::256B"
#else
#define LD_FUNC "ld.cs.global.L2::256B"
#endif

#define LDU_FUNC "ld.global"

#define ST_NA_FUNC "st.global"

// Store one 32-bit word using the configured global-store policy.
__device__ __forceinline__ void st_na_global_u32(const uint32_t *ptr,
                                                 const uint32_t &value) {
  asm(ST_NA_FUNC ".u32 [%0], %1;" ::"l"(ptr), "r"(value));
}

// Store four packed 32-bit words. Callers are responsible for alignment.
__device__ __forceinline__ void st_na_global_u32x4(const uint32_t *ptr,
                                                   const uint32_t *value) {
  asm(ST_NA_FUNC ".v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value[0]),
      "r"(value[1]), "r"(value[2]), "r"(value[3]));
}

// Load one 32-bit word through the non-coherent/cache-streaming path.
__device__ __forceinline__ uint32_t ld_nc_global_u32(const uint32_t *ptr) {
  uint32_t ret;
  asm(LD_NC_FUNC ".u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

// Load one 32-bit word through the standard configured global path.
__device__ __forceinline__ uint32_t ld_global_u32(const uint32_t *ptr) {
  uint32_t ret;
  asm(LD_FUNC ".u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

// Load one 32-bit word through the standard configured global path.
__device__ __forceinline__ uint32_t ld_global_u32_evict_last(const uint32_t *ptr) {
  uint32_t ret;
  asm("ld.global.nc.L1::evict_last.L2::256B" ".u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint64_t ld_nc_global_u64(const uint64_t *ptr) {
  uint64_t ret;
  asm(LD_NC_FUNC ".u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

// Load four packed 32-bit words through the non-coherent/cache-streaming path.
__device__ __forceinline__ void ld_nc_global_u32x4(uint32_t *value,
                                                   const uint32_t *ptr) {
  asm(LD_NC_FUNC ".v4.u32  {%0, %1, %2, %3}, [%4];"
      : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
      : "l"(ptr));
}

__device__ __forceinline__ uint32_t ld_nc_global(const uint32_t *ptr) {
  uint32_t ret;
  asm("ld.global.nc.L1::evict_last.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ const uint32_t *
ldu_global_u32_ptr(const uint32_t **ptr) {
  const uint32_t *ret;
  asm(LDU_FUNC ".u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint32_t *ldu_global_u32_ptr(uint32_t **ptr) {
  uint32_t *ret;
  asm(LDU_FUNC ".u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint32_t ldu_global_u32(const uint32_t *ptr) {
  uint32_t ret;
  asm(LDU_FUNC ".u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint64_t ldu_global_u64(const uint64_t *ptr) {
  uint64_t ret;
  asm(LDU_FUNC ".u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint2 ldu_global(const uint2 *ptr) {
  uint2 ret;
  asm(LDU_FUNC ".v2.u32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
  return ret;
}

template <typename T> __device__ T ld_nc_global(const T *ptr);

template <>
__device__ __forceinline__ uint32_t ld_nc_global(const uint32_t *ptr) {
  uint32_t ret;
  asm(LD_NC_FUNC ".u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

template <> __device__ __forceinline__ uint4 ld_nc_global(const uint4 *ptr) {
  uint4 ret;
  asm(LD_NC_FUNC ".v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
      : "l"(ptr));
  return ret;
}

template <> __device__ __forceinline__ uint2 ld_nc_global(const uint2 *ptr) {
  uint2 ret;
  asm(LD_NC_FUNC ".v2.u32 {%0, %1}, [%2];"
      : "=r"(ret.x), "=r"(ret.y)
      : "l"(ptr));
  return ret;
}

// Load one packed twiddle pair with the strongest L1/L2 retention hint used here.
__device__ __forceinline__ uint2 ld_nc_global_l1_evict_last(const uint2 *ptr) {
  uint2 ret;
  asm("ld.global.nc.L1::evict_last.L2::256B"
      ".v2.u32 {%0, %1}, [%2];"
      : "=r"(ret.x), "=r"(ret.y)
      : "l"(ptr));
  return ret;
}

// Store four packed 32-bit words into shared memory.
__device__ __forceinline__ void st_na_shared(const uint4 *ptr,
                                             const uint4 &value) {
  asm("st.shared"
      ".v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),
      "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}
