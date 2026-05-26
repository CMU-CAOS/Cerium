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
#include <sm_32_intrinsics.h>

// This header is intentionally specialized for a fixed arithmetic configuration.
// The Barrett parameters and widened helper interfaces are kept stable for the
// generated kernels that include this file, and the active modulus set uses
// 28-bit primes.
using LimbDataType = uint32_t;

static constexpr uint32_t BARRETT_K = 58;
// Reduce a 64-bit value modulo a 28-bit prime using a precomputed Barrett reciprocal.
__device__ __forceinline__ uint32_t barrett_reduce_u64_u32(const uint64_t x,
                                                           const uint32_t mod,
                                                           const uint32_t mu) {
  const uint32_t lo = static_cast<uint32_t>(x);
  const uint32_t hi = static_cast<uint32_t>(x >> 32);

  uint32_t q0 = __umulhi(lo, mu);
  const uint64_t q1 = static_cast<uint64_t>(hi) * mu;

  uint32_t q1_hi = static_cast<uint32_t>(q1 >> 32);
  const uint32_t q1_lo = static_cast<uint32_t>(q1);

  const uint32_t folded = q0 + q1_lo;
  q1_hi += (folded < q0);

  const uint32_t quotient = __funnelshift_l(folded, q1_hi, 64 - BARRETT_K);

  uint32_t reduced = lo - quotient * mod;
  if (reduced >= mod) {
    reduced -= mod;
  }
  return reduced;
}

// Wrapper used by generated code to reduce a widened intermediate.
__device__ __forceinline__ uint32_t reduce2(const uint64_t in,
                                            const uint32_t prime,
                                            const uint32_t barrett_ratio,
                                            const uint32_t barrett_k) {
  (void)barrett_k;
  return barrett_reduce_u64_u32(in, prime, barrett_ratio);
}

// Multiply two residues and immediately reduce the 64-bit product.
__device__ __forceinline__ uint32_t multiply(const uint32_t op1,
                                             const uint32_t op2,
                                             const uint32_t prime,
                                             const uint32_t barrett_ratio,
                                             const uint32_t barrett_k) {
  const uint64_t prod = static_cast<uint64_t>(op1) * op2;
  (void)barrett_k;
  return barrett_reduce_u64_u32(prod, prime, barrett_ratio);
}

// Plain 32x32 -> 64 multiply with no modular reduction.
__device__ __forceinline__ uint64_t multiply_64(const uint32_t op1,
                                                const uint32_t op2) {
  return static_cast<uint64_t>(op1) * static_cast<uint64_t>(op2);
}

// Multiply and keep the unreduced 64-bit product for deferred reduction.
__device__ __forceinline__ uint64_t multiply2(const uint32_t op1,
                                              const uint32_t op2,
                                              const uint32_t, const uint32_t,
                                              const uint32_t) {
  return static_cast<uint64_t>(op1) * op2;
}

// Variant of unreduced multiply used by the wider accumulation path.
__device__ __forceinline__ uint64_t multiply3(const uint32_t op1,
                                              const uint32_t op2,
                                              const uint32_t, const uint32_t,
                                              const uint32_t) {
  return static_cast<uint64_t>(op1) * op2;
}

// Add two residues modulo `modulus`.
__device__ __forceinline__ LimbDataType add(LimbDataType x, LimbDataType y,
                                            LimbDataType modulus) {
  x += y;
  if (x >= modulus) {
    x -= modulus;
  }
  return x;
}

// Add two widened values modulo `modulus_square`.
__device__ __forceinline__ uint64_t add2(uint64_t x, uint64_t y, uint32_t,
                                         uint64_t modulus_square) {
  x += y;
  if (x >= modulus_square) {
    x -= modulus_square;
  }
  return x;
}

// Add two widened values without an immediate modular correction. This is used in
// accumulation paths that postpone the final reduction step.
__device__ __forceinline__ uint64_t add3(uint64_t x, uint64_t y, uint32_t,
                                         uint64_t) {
  x += y;
  return x;
}

// Multiply, then add into a `modulus_square`-bounded accumulator.
__device__ __forceinline__ uint64_t
multiply_add2(const uint32_t op1, const uint32_t op2, const uint64_t op3,
              const uint32_t modulus, const uint64_t modulus_square) {
  const uint64_t prod = multiply2(op1, op2, modulus, 0, 0);
  return add2(prod, op3, modulus, modulus_square);
}

// Multiply, then accumulate without reducing the widened sum.
__device__ __forceinline__ uint64_t
multiply_add3(const uint32_t op1, const uint32_t op2, const uint64_t op3,
              const uint32_t modulus, const uint64_t modulus_square) {
  const uint64_t prod = multiply3(op1, op2, modulus, 0, 0);
  return add3(prod, op3, modulus, modulus_square);
}

// Multiply two residues, add a third, and return the reduced sum.
__device__ __forceinline__ uint32_t multiply_add(const uint32_t op1,
                                                 const uint32_t op2,
                                                 const uint32_t op3,
                                                 const uint32_t prime,
                                                 const uint32_t barrett_ratio,
                                                 const uint32_t barrett_k) {
  const auto prod = multiply(op1, op2, prime, barrett_ratio, barrett_k);
  return add(prod, op3, prime);
}

// Subtract two residues modulo `modulus`.
__device__ __forceinline__ LimbDataType subtract(LimbDataType x, LimbDataType y,
                                                 LimbDataType modulus) {
  if (x >= y) {
    return x - y;
  }
  return modulus - (y - x);
}

// Subtract two widened values modulo `modulus_square`.
__device__ __forceinline__ uint64_t subtract2(uint64_t x, uint64_t y, uint32_t,
                                              uint64_t modulus_square) {
  if (x >= y) {
    return x - y;
  }
  return modulus_square - (y - x);
}

// Return a subtraction result in a deliberately oversized positive range.
__device__ __forceinline__ LimbDataType subtract3(LimbDataType x,
                                                  LimbDataType y,
                                                  LimbDataType modulus) {
  return 4 * modulus + x - y;
}

// Placeholder move helper kept for interface compatibility.
__device__ __forceinline__ void mov(LimbDataType *, const LimbDataType *) {}

// Compute the additive inverse of a residue modulo `modulus`.
__device__ __forceinline__ LimbDataType negate(LimbDataType y,
                                               LimbDataType modulus) {
  if (y == 0) {
    return 0;
  }
  return modulus - y;
}

// Compute the additive inverse in the widened `modulus_square` domain.
__device__ __forceinline__ uint64_t negate2(uint64_t, uint64_t y, uint32_t,
                                            uint64_t modulus_square) {
  if (y == 0) {
    return 0;
  }
  return modulus_square - y;
}
