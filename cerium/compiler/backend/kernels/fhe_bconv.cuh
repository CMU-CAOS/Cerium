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
#include "fhe_graph.cuh"
#include "fhe_mem.cuh"
#include "fhe_ntt.cuh"
#include <cstdint>
#include <unordered_map>

using FunctionMapType = std::unordered_map<uint32_t, void *>;

// This header is intentionally specialized for the generated base-conversion path
// used by this release. The thread/block geometry and intermediate reduction
// schedule below are hard-coded to the active parameter set.

// Combine two 32-bit limbs into one 64-bit value, then reduce it into the
// active output modulus.
__device__ static uint32_t mod_read_2(const uint32_t src0, const uint32_t src1,
                                      const uint32_t modulus,
                                      const uint32_t barrett_ratio,
                                      const uint32_t barrett_k) {
  (void)barrett_k;
  uint32_t packed_limbs[2] = {src0, src1};
  const uint64_t packed_value = *((uint64_t *)&packed_limbs);
  return barrett_reduce_u64_u32(packed_value, modulus, barrett_ratio);
}

// Native 128-bit device integer used by the widened residue path.
using uint128_word = unsigned __int128;

// Use CUDA's device-side 128-bit integer support for the widened product path.
__inline__ __device__ uint128_word multiply_u64_u64_wide(const uint64_t lhs,
                                                         const uint64_t rhs) {
  return static_cast<uint128_word>(lhs) * static_cast<uint128_word>(rhs);
}

// Reduce a 128-bit product modulo a 64-bit composite modulus. This path is used by
// the 2-limb residue accumulation helper below.
__inline__ __device__ uint64_t barrett_reduce_u128_u64(const uint128_word x,
                                                       const uint64_t mod,
                                                       const uint64_t mu,
                                                       const uint64_t k) {
  const uint64_t x_lo = static_cast<uint64_t>(x);
  const uint64_t x_hi = static_cast<uint64_t>(x >> 64);

  const uint128_word lo_prod = multiply_u64_u64_wide(x_lo, mu);
  const uint128_word hi_prod = multiply_u64_u64_wide(x_hi, mu);

  const uint64_t lo_hi = static_cast<uint64_t>(lo_prod >> 64);
  const uint64_t hi_lo = static_cast<uint64_t>(hi_prod);
  uint64_t hi_hi = static_cast<uint64_t>(hi_prod >> 64);

  const uint64_t mid_sum = lo_hi + hi_lo;
  hi_hi += (mid_sum < lo_hi);

  const uint64_t q = (mid_sum >> (k - 64)) + (hi_hi << (128 - k));
  uint64_t r = x_lo - q * mod;
  if (r >= mod) {
    r -= mod;
  }
  return r;
}

// Accumulate one source residue into a packed 64-bit destination coefficient.
__device__ static void rsv_write_2(uint32_t *dest, const uint32_t src,
                                   const uint32_t *factors,
                                   const uint32_t *modulus,
                                   const uint32_t *barrett_ratio,
                                   const uint32_t barrett_k) {
  uint64_t *dest64 = (uint64_t *)dest;
  const uint64_t *factor64 = (const uint64_t *)factors;
  const uint64_t src64 = static_cast<uint64_t>(src);
  const uint64_t modulus64 = *((const uint64_t *)modulus);
  const uint64_t ratio64 = *((const uint64_t *)barrett_ratio);
  const uint128_word wide_product = multiply_u64_u64_wide(src64, factor64[0]);
  const uint64_t reduced_product =
      barrett_reduce_u128_u64(wide_product, modulus64, ratio64, barrett_k);

  *dest64 += reduced_product;
  if (*dest64 >= modulus64) {
    *dest64 -= modulus64;
  }
}

template <size_t NUM_INPUT_BASES, size_t NUM_OUTPUT_BASES>
__device__ __forceinline__ void
convert(const uint32_t * __restrict__  inputs, uint32_t * __restrict__ output,
        const uint32_t * __restrict__ baseConversionFactors, const uint32_t * __restrict__ outputModuli,
        const uint32_t * __restrict__ barrettRatio, const uint32_t * __restrict__ barrettK) {
  extern __shared__ uint32_t factors[];
  const size_t factor_count = NUM_INPUT_BASES * NUM_OUTPUT_BASES;
  const size_t aligned_factor_count = (factor_count / 32 + 1) * 32;
  uint32_t *input_cache = &factors[aligned_factor_count];

  // Stage the base-conversion table in shared memory once per block.
  for (size_t i = threadIdx.x; i < factor_count; i += blockDim.x) {
    factors[i] = ld_global_u32_evict_last(&baseConversionFactors[i]);
  }

// Cache the coefficient stripe from each input basis for reuse by the inner
// accumulation loop.
#pragma unroll
  for (size_t basis_idx = 0; basis_idx < NUM_INPUT_BASES; basis_idx++) {
    input_cache[threadIdx.x + basis_idx * blockDim.x] = ld_nc_global_u32(
        &inputs[threadIdx.x + blockIdx.x * blockDim.x + basis_idx * LENGTH]);
  }
  __syncthreads();

  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t out_idx = 0; out_idx < NUM_OUTPUT_BASES; out_idx++) {
    const uint32_t mod = outputModuli[out_idx];
    const uint64_t mod_sq = static_cast<uint64_t>(mod) * mod;
    const uint32_t mu = barrettRatio[out_idx];
    uint64_t acc = 0;

// This schedule intentionally trims the widened accumulator after fixed
// checkpoints chosen for the active basis geometry.
#pragma unroll
    for (size_t in_idx = 0; in_idx < NUM_INPUT_BASES; in_idx++) {
      const uint32_t term = input_cache[in_idx * blockDim.x + threadIdx.x];
      const uint32_t factor = factors[out_idx * NUM_INPUT_BASES + in_idx];
      const bool reduce_now = (in_idx == 15) || (in_idx == 30);

      if (in_idx == 0) {
        acc = multiply3(term, factor, 0, 0, 0);
        continue;
      }

      if (reduce_now) {
        acc = static_cast<uint64_t>(barrett_reduce_u64_u32(acc, mod, mu));
      }
      acc = multiply_add3(term, factor, acc, mod, mod_sq);
    }

    st_na_global_u32(&output[out_idx * LENGTH + global_idx],
                     barrett_reduce_u64_u32(acc, mod, mu));
  }
}
