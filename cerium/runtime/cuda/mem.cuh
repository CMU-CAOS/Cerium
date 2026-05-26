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
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

namespace Cerium {
namespace Runtime {
namespace PTX {

__device__ __forceinline__ void st_na_global_u32(const uint32_t *ptr,
                                                 const uint32_t &value) {
  asm volatile(ST_NA_FUNC ".u32 [%0], %1;" ::"l"(ptr), "r"(value));
}

__device__ __forceinline__ uint32_t ld_nc_global_u32(const uint32_t *ptr) {
  uint32_t ret;
  asm volatile(LD_NC_FUNC ".u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint32_t *ldu_global_u32_ptr(const uint32_t **ptr) {
  uint32_t *ret;
  asm volatile("ldu.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint32_t ldu_global_u32(const uint32_t *ptr) {
  uint32_t ret;
  asm volatile("ldu.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

} // namespace PTX
} // namespace Runtime
} // namespace Cerium