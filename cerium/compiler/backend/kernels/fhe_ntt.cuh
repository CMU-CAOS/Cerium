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
#include "fhe_arith.cuh"
#include "fhe_mem.cuh"
#include "transpose.cuh"
#include <cstdint>

// This header is intentionally specialized for one concrete parameter set.
// The stage sizes, per-thread work partitioning, and shared-memory layouts below
// are hard-coded for that configuration and are not intended to be generic.
//
// Do not change these.
static constexpr int LENGTH = 64 * 1024;
static constexpr int FIRST_STAGE_NTT = 256;
static constexpr int SECOND_STAGE_NTT = LENGTH / FIRST_STAGE_NTT;
static constexpr int NTT_TABLE_SIZE = LENGTH;

static constexpr int FIRST_STAGE_INT = 512;
static constexpr int SECOND_STAGE_INT = LENGTH / FIRST_STAGE_INT;

static constexpr int PER_THREAD_NTT0_SIZE = 4;
static constexpr int PER_THREAD_NTT1_SIZE = 4;

static constexpr int NTT0_PAD_LENGTH = 8;
static constexpr int NTT0_BLOCK_DIM_X =
    (FIRST_STAGE_NTT / PER_THREAD_NTT0_SIZE) * NTT0_PAD_LENGTH;
static constexpr int NTT0_GRID_DIM_X =
    LENGTH / (PER_THREAD_NTT0_SIZE * NTT0_BLOCK_DIM_X);

static constexpr int NTT1_BLOCK_DIM_X = 64;
static constexpr int NTT1_GRID_DIM_X =
    LENGTH / (PER_THREAD_NTT1_SIZE * NTT1_BLOCK_DIM_X);

static constexpr int PER_THREAD_INT0_SIZE = 8;
static constexpr int PER_THREAD_INT1_SIZE = 16;

static constexpr int INT0_BLOCK_DIM_X = 64;
static constexpr int INT0_GRID_DIM_X =
    LENGTH / (PER_THREAD_INT0_SIZE * INT0_BLOCK_DIM_X);

static constexpr int INT1_PAD_LENGTH = 8;
static constexpr int INT1_BLOCK_DIM_X =
    (SECOND_STAGE_INT / PER_THREAD_INT1_SIZE) * INT1_PAD_LENGTH;
static constexpr int INT1_GRID_DIM_X =
    LENGTH / (PER_THREAD_INT1_SIZE * INT1_BLOCK_DIM_X);

static constexpr int nttShMemSize0 = NTT0_PAD_LENGTH * FIRST_STAGE_NTT;
static constexpr int nttShMemSize1 = NTT1_BLOCK_DIM_X * PER_THREAD_NTT1_SIZE;

static constexpr int intShMemSize0 = INT0_BLOCK_DIM_X * PER_THREAD_INT0_SIZE;
static constexpr int intShMemSize1 =
    INT1_PAD_LENGTH * (SECOND_STAGE_INT + INT1_PAD_LENGTH + 1);

__device__ __forceinline__ uint32_t
shoup_multiply_reduce(const uint32_t x, const uint32_t w,
                      const uint32_t w_scaled, const uint32_t mod) {
  const uint32_t q = __umulhi(w_scaled, x);
  const uint32_t wx = x * w;
  return wx - q * mod;
}

__device__ __forceinline__ void ntt_pair_step(uint32_t &a, uint32_t &b,
                                              const uint2 &tw,
                                              const uint32_t mod,
                                              const uint32_t mod2) {
  const uint32_t t = shoup_multiply_reduce(b, tw.x, tw.y, mod);
  uint32_t lo = a;
  if (lo >= mod2) {
    lo -= mod2;
  }
  const uint32_t hi = mod2 - t;
  b = lo + hi;
  a = lo + t;
}

// Applies a radix-4 NTT butterfly to four values held by a single thread.
__device__ __forceinline__ static void
ntt_butterfly_4(uint32_t *local, const uint2 *W, const size_t m,
                const size_t m_idx, const uint32_t prime,
                const uint32_t two_prime) {
  for (int j = 0; j < 2; j++) {
    ntt_pair_step(local[j], local[j + 2],
                  ld_nc_global_l1_evict_last(&W[m + m_idx]), prime, two_prime);
  }
  for (int j = 0; j < 2; j++) {
    ntt_pair_step(local[2 * j + 0], local[2 * j + 1],
                  ld_nc_global_l1_evict_last(&W[(2 + j) * m + m_idx]), prime,
                  two_prime);
  }
}

// Assumptions for this specialized first-stage NTT path:
// - Called from `_kernel_ntt_0` with `blockDim.x == NTT0_BLOCK_DIM_X == 512`.
// - `kPerThread == 4` and `kPad == 8`, matching the fixed stage-0 transpose layout.
// - `lane_tid` spans `[0, 511]`, so the first butterfly index is always 0.
// - Later butterfly indices use `lane_tid >> 5` and `lane_tid >> 3`, which are
//   equivalent to division by `NTT0_BLOCK_DIM_X / 16` and `NTT0_BLOCK_DIM_X / 64`.
// - `out` and `in` may alias, so this code must not rely on `__restrict__`-style
//   non-aliasing assumptions or reorder loads/stores across the shared-memory stage.
__device__ __forceinline__ void
transpose_ntt0_internal(uint32_t *out, const uint32_t *in, const uint2 *W,
                        const uint32_t prime, uint32_t *shMem) {
  constexpr uint32_t kPerThread = 4;
  constexpr uint32_t kPad = 8;

  uint32_t local[kPerThread];

  const uint32_t two_prime = prime << 1;
  const uint32_t global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t lane_tid = threadIdx.x;
  const uint32_t pad_slot = global_tid & (kPad - 1);
  const uint32_t lane_group = (global_tid >> 3) & 3;
  const uint32_t tile_idx = global_tid >> 5;
  const uint32_t block_slot = tile_idx & 15;
  const uint32_t block_group = tile_idx >> 4;

  const uint32_t read_idx = pad_slot + lane_group * (4 * 1024) +
                            block_slot * 256 + block_group * kPad;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    local[j] = ld_nc_global(&in[read_idx + j * 16 * 1024]);
  }

  ntt_butterfly_4(local, W, 1, 0, prime, two_prime);
  transpose2<kPerThread>(local, kPad);

  ntt_butterfly_4(local, W, 4, (lane_tid >> 3) & 3, prime, two_prime);

  uint32_t sh_idx = lane_tid;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    uint32_t bank_idx = sh_idx + NTT0_BLOCK_DIM_X * j;
    bank_idx = (bank_idx / 32) * 32 + ((bank_idx / 32) * 8 + bank_idx) % 32;
    shMem[bank_idx] = local[j];
  }

  __syncthreads();

  sh_idx = (lane_tid & (kPad - 1)) +
           ((lane_tid >> 3) & (kPerThread - 1)) * (kPerThread * kPad) +
           ((lane_tid >> 5) & 3) * NTT0_BLOCK_DIM_X + (lane_tid >> 7) * kPad;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    uint32_t bank_idx = sh_idx + kPerThread * kPad * kPerThread * j;
    bank_idx = (bank_idx / 32) * 32 + ((bank_idx / 32) * 8 + bank_idx) % 32;
    local[j] = shMem[bank_idx];
  }

  ntt_butterfly_4(local, W, 16, lane_tid >> 5, prime, two_prime);
  transpose2<kPerThread>(local, kPad);

  ntt_butterfly_4(local, W, 64, lane_tid >> 3, prime, two_prime);

  const uint32_t write_idx =
      pad_slot + lane_group * 1024 + block_slot * 4096 + block_group * kPad;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    st_na_global_u32(&out[write_idx + 256 * j], local[j]);
  }
}

// First-stage forward NTT entrypoint specialized to the transposed implementation.
__device__ static void ntt0(uint32_t *out, const uint32_t *in,
                            const uint2 *base_inv, const uint32_t prime,
                            uint32_t *temp) {
  return transpose_ntt0_internal(out, in, base_inv, prime, temp);
}

__device__ static void ntt0_sud(uint32_t *out, const uint32_t *in,
                                const uint2 *base_inv, const uint32_t prime,
                                uint32_t *temp) {
  return ntt0(out, in, base_inv, prime, temp);
}

// Assumptions for this specialized second-stage NTT path:
// - Called from `_kernel_ntt_1` with `blockDim.x == NTT1_BLOCK_DIM_X == 64`.
// - `PER_THREAD == 4` and `PAD == 8`, matching the fixed stage-1 transpose layout.
// - `out` and `in` may alias, so this code must not rely on non-aliasing assumptions.
// - The stage index arithmetic below only replaces division by powers of two with
//   equivalent shifts or identity operations; the dataflow and memory layout are unchanged.
__device__ __forceinline__ void
transpose_ntt1_internal(uint32_t *out, const uint32_t *in, const uint2 *W,
                        const uint32_t prime, uint32_t *shMem) {
  constexpr size_t kPerThread = 4;
  constexpr size_t kPad = 8;
  constexpr size_t kSecondStage = 256;
  constexpr size_t kRadix = kSecondStage / kPerThread;

  uint32_t local[kPerThread];

  const uint32_t two_prime = prime << 1;
  const uint32_t global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t lane_tid = threadIdx.x;
  const uint32_t pad_slot = global_tid & (kPad - 1);
  const uint32_t lane_group = (global_tid >> 3) & 3;
  const uint32_t tile_idx = global_tid >> 5;
  const uint32_t block_slot = tile_idx & 1;
  const uint32_t block_group = tile_idx >> 1;

  const uint32_t lane_pad_slot = lane_tid & (kPad - 1);
  const uint32_t lane_group_slot = (lane_tid >> 3) & 3;
  const uint32_t lane_block_slot = (lane_tid >> 5) & 1;

  const uint32_t read_idx =
      pad_slot + lane_group * (kSecondStage / kPerThread / kPerThread) +
      block_slot * kPad + block_group * kSecondStage;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    local[j] = ld_nc_global(&in[read_idx + j * kRadix]);
  }

  ntt_butterfly_4(local, W, 256, global_tid >> 6, prime, two_prime);
  transpose2<kPerThread>(local, kPad);

  const uint32_t stage_idx = ((global_tid >> 3) & 3) + ((global_tid >> 6) << 2);
  ntt_butterfly_4(local, W, 1024, stage_idx, prime, two_prime);

  uint32_t sh_idx = lane_pad_slot + lane_group_slot * kRadix +
                    lane_block_slot * kPad + (lane_tid >> 8) * kSecondStage;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    auto idx = sh_idx + (kRadix / kPerThread) * j;
    idx = (idx / 32) * 32 + ((idx + (idx / 32) * 4) % 32);
    shMem[idx] = local[j];
  }

  __syncthreads();

  const uint32_t lane_subslot = lane_tid & (kPerThread - 1);
  const uint32_t lane_subgroup = lane_tid >> 2;

  sh_idx = lane_subslot + lane_subgroup * 16;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    auto idx = sh_idx + 4 * j;
    idx = (idx / 32) * 32 + ((idx + (idx / 32) * 4) % 32);
    local[j] = shMem[idx];
  }

  ntt_butterfly_4(local, W, 4 * 1024, global_tid >> 2, prime, two_prime);
  transpose<kPerThread>(local);

  ntt_butterfly_4(local, W, 16 * 1024, global_tid, prime, two_prime);

#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    for (int k = 0; k < 2; k++) {
      if (local[j] >= prime) {
        local[j] -= prime;
      }
    }
  }

#pragma unroll
  for (int j = 0; j < kPerThread / 4; j++) {
    st_na_global_u32x4(&out[kPerThread * global_tid + j],
                       local + kPerThread * j);
  }
}

// Specialized SUD variant of the second-stage NTT.
// This preserves the same indexing and shared-memory layout as `transpose_ntt1_internal`,
// then combines the transformed values with `in2` using the provided SUD factor.
__device__ __forceinline__ void transpose_ntt1_internal_sud(
    uint32_t *out, const uint32_t *in, const uint2 *W, const uint32_t prime,
    uint32_t *shMem, const uint32_t *in2, const uint32_t sud_factor,
    const uint32_t barrett_ratio, const uint32_t barrett_k) {
  constexpr size_t kPerThread = 4;
  constexpr size_t kPad = 8;
  constexpr size_t kTileSpan = kPerThread * kPad;
  constexpr size_t kSecondStage = 256;
  constexpr size_t kTileGroups =
      (kSecondStage / kPerThread / kPerThread) / kPad;
  constexpr size_t kRadix = kSecondStage / kPerThread;

  uint32_t local[kPerThread];

  const uint32_t two_prime = prime << 1;
  const uint32_t global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t lane_tid = threadIdx.x;

  const uint32_t pad_slot = global_tid % kPad;
  const uint32_t lane_group = (global_tid % kTileSpan) / kPad;
  const uint32_t block_slot = (global_tid / kTileSpan) % kTileGroups;
  const uint32_t block_group = (global_tid / kTileSpan) / kTileGroups;

  const uint32_t lane_pad_slot = lane_tid % kPad;
  const uint32_t lane_group_slot = (lane_tid % kTileSpan) / kPad;
  const uint32_t lane_block_slot = (lane_tid / kTileSpan) % kTileGroups;
  const uint32_t lane_block_group = (lane_tid / kTileSpan) / kTileGroups;

  const uint32_t read_idx =
      pad_slot + lane_group * (kSecondStage / kPerThread / kPerThread) +
      block_slot * kPad + block_group * kSecondStage;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    local[j] = ld_nc_global(&in[read_idx + j * kRadix]);
  }

  ntt_butterfly_4(local, W, 256, global_tid / 64, prime, two_prime);
  transpose2<kPerThread>(local, kPad);

  const uint32_t stage_idx = (global_tid / 8) % 4 + (global_tid / 64) * 4;
  ntt_butterfly_4(local, W, 1024, stage_idx, prime, two_prime);

  uint32_t sh_idx = lane_pad_slot + lane_group_slot * kRadix +
                    lane_block_slot * kPad + lane_block_group * kSecondStage;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    uint32_t bank_idx = sh_idx + (kRadix / kPerThread) * j;
    bank_idx = (bank_idx / 32) * 32 + ((bank_idx + (bank_idx / 32) * 4) % 32);
    shMem[bank_idx] = local[j];
  }

  __syncthreads();

  const uint32_t lane_subslot = lane_tid % kPerThread;
  const uint32_t lane_subgroup = lane_tid / kPerThread;

  sh_idx = lane_subslot + lane_subgroup * 16;
#pragma unroll
  for (int j = 0; j < kPerThread; j++) {
    uint32_t bank_idx = sh_idx + 4 * j;
    bank_idx = (bank_idx / 32) * 32 + ((bank_idx + (bank_idx / 32) * 4) % 32);
    local[j] = shMem[bank_idx];
  }

  ntt_butterfly_4(local, W, 4 * 1024, global_tid / 4, prime, two_prime);
  transpose<kPerThread>(local);
  ntt_butterfly_4(local, W, 16 * 1024, global_tid, prime, two_prime);

#pragma unroll
  for (int j = 0; j < kPerThread / 4; j++) {
    uint4 value = ld_nc_global((const uint4 *)&in2[4 * global_tid + j]);
    local[4 * j + 0] = multiply(subtract3(value.x, local[4 * j + 0], prime),
                                sud_factor, prime, barrett_ratio, barrett_k);
    local[4 * j + 1] = multiply(subtract3(value.y, local[4 * j + 1], prime),
                                sud_factor, prime, barrett_ratio, barrett_k);
    local[4 * j + 2] = multiply(subtract3(value.z, local[4 * j + 2], prime),
                                sud_factor, prime, barrett_ratio, barrett_k);
    local[4 * j + 3] = multiply(subtract3(value.w, local[4 * j + 3], prime),
                                sud_factor, prime, barrett_ratio, barrett_k);
    st_na_global_u32x4(&out[4 * global_tid + j], local + 4 * j);
  }
}

// Second-stage forward NTT entrypoint specialized to the transposed implementation.
__device__ static void ntt1(uint32_t *out, const uint32_t *in,
                            const uint2 *base_inv, const uint32_t prime,
                            uint32_t *temp) {
  return transpose_ntt1_internal(out, in, base_inv, prime, temp);
}

__device__ static void ntt1_sud(uint32_t *out, const uint32_t *in,
                                const uint32_t *in2, const uint2 *base_inv,
                                const uint32_t sud_factor, const uint32_t prime,
                                const uint32_t barrett_ratio,
                                const uint32_t barrett_k, uint32_t *temp) {

  return transpose_ntt1_internal_sud(out, in, base_inv, prime, temp, in2,
                                     sud_factor, barrett_ratio, barrett_k);
}

// Inverse butterfly for one pair. The sum path is halved modulo `mod`, while the
// difference path is multiplied by the inverse twiddle in Shoup form.
__device__ __forceinline__ void
intt_pair_step(uint32_t &x, uint32_t &y, const uint2 &tw, const uint32_t mod) {
  const uint32_t mod2 = mod << 1;
  const uint32_t diff = mod2 + x - y;
  uint32_t sum = x + y;
  if (sum >= mod2) {
    sum -= mod2;
  }
  const uint32_t carry = diff & 1u;
  sum += carry * mod;
  x = sum >> 1;
  y = shoup_multiply_reduce(diff, tw.x, tw.y, mod);
}

// Applies a radix-8 inverse butterfly using stage-local indexing.
__device__ __forceinline__ static void
intt_butterfly_8(uint32_t *local, const uint2 *W, const size_t m,
                 const size_t m_idx, const uint32_t prime) {
  for (int j = 0; j < 4; j++) {
    intt_pair_step(local[2 * j + 0], local[2 * j + 1],
                   ld_nc_global_l1_evict_last(&W[(4 + j) * m + m_idx]), prime);
  }
  for (int j = 0; j < 2; j++) {
    const uint2 tw = ld_nc_global_l1_evict_last(&W[(2 + j) * m + m_idx]);
    intt_pair_step(local[4 * j + 0], local[4 * j + 2], tw, prime);
    intt_pair_step(local[4 * j + 1], local[4 * j + 3], tw, prime);
  }
  for (int j = 0; j < 4; j++) {
    intt_pair_step(local[j], local[j + 4],
                   ld_nc_global_l1_evict_last(&W[m + m_idx]), prime);
  }
}

// Applies a radix-16 inverse butterfly to the per-thread 16-value working set.
__device__ __forceinline__ static void intt_butterfly_16(uint32_t *local,
                                                         const uint2 *W,
                                                         const size_t tw_idx,
                                                         const uint32_t prime) {
  for (int j = 0; j < 8; j++) {
    intt_pair_step(local[2 * j + 0], local[2 * j + 1],
                   ld_nc_global_l1_evict_last(&W[8 * tw_idx + j]), prime);
  }
  for (int j = 0; j < 4; j++) {
    const uint2 tw = ld_nc_global_l1_evict_last(&W[4 * tw_idx + j]);
    intt_pair_step(local[4 * j + 0], local[4 * j + 2], tw, prime);
    intt_pair_step(local[4 * j + 1], local[4 * j + 3], tw, prime);
  }
  for (int j = 0; j < 2; j++) {
    const uint2 tw = ld_nc_global_l1_evict_last(&W[2 * tw_idx + j]);
    intt_pair_step(local[8 * j + 0], local[8 * j + 4], tw, prime);
    intt_pair_step(local[8 * j + 1], local[8 * j + 5], tw, prime);
    intt_pair_step(local[8 * j + 2], local[8 * j + 6], tw, prime);
    intt_pair_step(local[8 * j + 3], local[8 * j + 7], tw, prime);
  }
  for (int j = 0; j < 8; j++) {
    intt_pair_step(local[j], local[j + 8],
                   ld_nc_global_l1_evict_last(&W[tw_idx]), prime);
  }
}

// Final two radix-8 layers for the specialized inverse path.
__device__ __forceinline__ static void
intt_butterfly_8x2(uint32_t *local, const uint2 *W, const size_t tw_idx,
                   const uint32_t prime) {
  for (int j = 0; j < 4; j++) {
    const uint2 tw = ld_nc_global_l1_evict_last(&W[4 * tw_idx + j]);
    intt_pair_step(local[4 * j + 0], local[4 * j + 2], tw, prime);
    intt_pair_step(local[4 * j + 1], local[4 * j + 3], tw, prime);
  }
  for (int j = 0; j < 2; j++) {
    const uint2 tw = ld_nc_global_l1_evict_last(&W[2 * tw_idx + j]);
    intt_pair_step(local[8 * j + 0], local[8 * j + 4], tw, prime);
    intt_pair_step(local[8 * j + 1], local[8 * j + 5], tw, prime);
    intt_pair_step(local[8 * j + 2], local[8 * j + 6], tw, prime);
    intt_pair_step(local[8 * j + 3], local[8 * j + 7], tw, prime);
  }
  for (int j = 0; j < 8; j++) {
    intt_pair_step(local[j], local[j + 8],
                   ld_nc_global_l1_evict_last(&W[tw_idx]), prime);
  }
}

// Assumptions for this specialized first-stage inverse NTT path:
// - Called with `LENGTH == 64 * 1024`.
// - `SECOND_STAGE_INT == 128`, so `kBlockSpan == LENGTH / 2 / SECOND_STAGE_INT == 256`.
// - Each thread handles exactly 8 values, so `kPerThread == 8` and `kRadix == FIRST_STAGE_INT / 8 == 32`.
// - The staged scratch layout in `temp` matches this fixed shape, allowing the two
//   post-store inverse stages to be emitted directly as spans 32 and 256.
// - `temp` is shared scratch for the block, so the `__syncthreads()` calls are required
//   and the load/store ordering around them must not be changed.
__device__ static void intt0(uint32_t *out, const uint32_t *in,
                             const uint2 *base_inv, const uint32_t prime,
                             uint32_t *temp) {
  constexpr size_t kLength = LENGTH;
  constexpr int kStage = SECOND_STAGE_INT;
  constexpr int kPerThread = 8;
  constexpr int kRadix = FIRST_STAGE_INT / kPerThread;
  constexpr int kBlockSpan = kLength / 2 / kStage;

  const int lane_group = threadIdx.x / kRadix;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t local[kPerThread];

  const int coeff_idx = global_tid % (kLength / kPerThread);
  const int twiddle_block = coeff_idx / (kBlockSpan / 4);
  const int lane_offset = coeff_idx % (kBlockSpan / 4);
  const int base_idx = 2 * twiddle_block * kBlockSpan + lane_offset;

  const uint2 *twiddles = base_inv;
  for (int l = 0; l < kPerThread; l++) {
    local[l] = ld_nc_global_u32(in + global_tid * kPerThread + l);
  }
  intt_butterfly_8(local, twiddles, kStage * (kBlockSpan / 4), coeff_idx,
                   prime);

  __syncthreads();
  for (int l = 0; l < kPerThread; l++) {
    temp[lane_group * kPerThread * kRadix + kPerThread * lane_offset + l] =
        local[l];
  }
  __syncthreads();

  constexpr int kStage0Span = 32;
  const int stage0_block = lane_offset / (kStage0Span / 4);
  const int stage0_lane = lane_offset % (kStage0Span / 4);
  for (int l = 0; l < kPerThread; l++) {
    local[l] =
        temp[lane_group * kPerThread * kRadix + 2 * stage0_block * kStage0Span +
             stage0_lane + (kStage0Span / 4) * l];
  }
  intt_butterfly_8(local, twiddles, 1024, coeff_idx / 8, prime);
  for (int l = 0; l < kPerThread; l++) {
    temp[lane_group * kPerThread * kRadix + 2 * stage0_block * kStage0Span +
         stage0_lane + (kStage0Span / 4) * l] = local[l];
  }
  __syncthreads();

  constexpr int kStage1Span = 256;
  const int stage1_block = lane_offset / (kStage1Span / 4);
  const int stage1_lane = lane_offset % (kStage1Span / 4);
  for (int l = 0; l < kPerThread; l++) {
    local[l] =
        temp[lane_group * kPerThread * kRadix + 2 * stage1_block * kStage1Span +
             stage1_lane + (kStage1Span / 4) * l];
  }
  intt_butterfly_8(local, twiddles, 128, coeff_idx / 64, prime);
  for (int l = 0; l < kPerThread; l++) {
    temp[lane_group * kPerThread * kRadix + 2 * stage1_block * kStage1Span +
         stage1_lane + (kStage1Span / 4) * l] = local[l];
  }
  __syncthreads();

  for (int j = 0; j < kPerThread; j++) {
    st_na_global_u32(out + base_idx + kBlockSpan / 4 * j, local[j]);
  }
}

// Assumptions for this specialized second-stage inverse NTT path:
// - `PER_THREAD_INT1_SIZE == 16`, so each thread owns a fixed 16-value working set.
// - `INT1_PAD_LENGTH == 8` and `SECOND_STAGE_INT / PER_THREAD_INT1_SIZE == 8`,
//   which define the warp-local scratch layout in `temp`.
// - `kRadix == 8`, so there are no additional radix-16 shared-memory stages after the
//   first `intt_butterfly_16(...)`; the tail is always the `intt_butterfly_8x2(...)` case.
// - `temp` is shared scratch for the block, so the current synchronization and
//   load/store ordering must be preserved.
__device__ void static intt1(uint32_t *out, const uint32_t *in,
                             const uint2 *base_inv, const uint32_t prime,
                             uint32_t *temp) {
  constexpr size_t kPerThread = PER_THREAD_INT1_SIZE;
  constexpr size_t kPerThreadBy2 = kPerThread / 2;
  constexpr size_t kLength = LENGTH;
  constexpr size_t kStage = 1;
  constexpr size_t kPad = INT1_PAD_LENGTH;
  constexpr int kRadix = SECOND_STAGE_INT / kPerThread;

  const int warp_lane = threadIdx.x % kPad;
  const int warp_group = threadIdx.x / kPad;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t local[kPerThread];

  const int block_span = kLength / 2 / kStage;
  const int coeff_idx = global_tid % (kLength / kPerThread);
  const int twiddle_block = coeff_idx / (block_span / kPerThreadBy2);
  const int lane_offset = coeff_idx % (block_span / kPerThreadBy2);
  const uint2 *twiddles = base_inv;

  int read_idx = 2 * block_span / kRadix * warp_group + warp_lane +
                 kPad * (lane_offset / (kRadix * kPad));
  for (int j = 0; j < kPerThread; j++) {
    local[j] = ld_nc_global_u32(in + read_idx +
                                block_span / kPerThreadBy2 / kRadix * j);
  }

  const int expanded_radix = kPerThread * kRadix;
  const int twiddle_idx = kStage + twiddle_block;
  const int twiddle_idx3 = kRadix * twiddle_idx + warp_group;
  intt_butterfly_16(local, twiddles, twiddle_idx3, prime);
  for (int j = 0; j < kPerThread; j++) {
    temp[warp_lane * (expanded_radix + kPad) + kPerThread * warp_group + j] =
        local[j];
  }
  __syncthreads();

  for (int l = 0; l < kPerThread; l++) {
    local[l] =
        temp[warp_lane * (expanded_radix + kPad) + warp_group + kRadix * l];
  }
  intt_butterfly_8x2(local, twiddles, twiddle_idx, prime);
  for (int j = 0; j < kPerThread; j++) {
    if (local[j] >= prime) {
      local[j] -= prime;
    }
  }

  read_idx = block_span / kPerThreadBy2 / kRadix * warp_group + warp_lane +
             kPad * (lane_offset / (kRadix * kPad));
  for (int j = 0; j < kPerThread; j++) {
    st_na_global_u32(out + read_idx + block_span / kPerThreadBy2 * j, local[j]);
  }
}
