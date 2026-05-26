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
// Warp-local shuffle transposes used by the specialized NTT staging code.
// Each specialization assumes a 32-lane warp and a fixed number of values per
// thread in the active kernel configuration.

template <uint32_t per_thread> __device__ void transpose(uint32_t *local);

template <uint32_t per_thread>
__device__ void transpose2(uint32_t *local, const uint32_t pad);

template <> __device__ __forceinline__ void transpose<2>(uint32_t *local) {
  constexpr uint32_t per_thread = 2;
#pragma unroll
  for (int k = 0; k < per_thread; k += 2) {
    int lane_offset = threadIdx.x & 1;
    lane_offset ^= 1;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 1, 32);
  }
}

template <> __device__ __forceinline__ void transpose<8>(uint32_t *local) {
  constexpr uint32_t per_thread = 8;
#pragma unroll
  for (int k = 0; k < per_thread; k += 2) {
    int lane_offset = threadIdx.x & 1;
    lane_offset ^= 1;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 1, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 4) {
    int lane_offset = (threadIdx.x >> 1) << 1;
    lane_offset %= 4;
    lane_offset ^= 2;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 2, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 2, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 8) {
    int lane_offset = threadIdx.x & 4;
    lane_offset ^= 4;
    local[lane_offset + k + 0] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 0], 4, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 4, 32);
    local[lane_offset + k + 2] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 2], 4, 32);
    local[lane_offset + k + 3] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 3], 4, 32);
  }
}

template <> __device__ __forceinline__ void transpose<16>(uint32_t *local) {
  constexpr uint32_t per_thread = 16;
#pragma unroll
  for (int k = 0; k < per_thread; k += 2) {
    int lane_offset = threadIdx.x & 1;
    lane_offset ^= 1;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 1, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 4) {
    int lane_offset = (threadIdx.x >> 1) << 1;
    lane_offset %= 4;
    lane_offset ^= 2;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 2, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 2, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 8) {
    int lane_offset = (threadIdx.x >> 2) << 2;
    lane_offset %= 8;
    lane_offset ^= 4;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 4, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 4, 32);
    local[lane_offset + k + 2] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 2], 4, 32);
    local[lane_offset + k + 3] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 3], 4, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 16) {
    int lane_offset = (threadIdx.x >> 3) << 3;
    lane_offset %= 16;
    lane_offset ^= 8;
    local[lane_offset + k + 0] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 0], 8, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 8, 32);
    local[lane_offset + k + 2] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 2], 8, 32);
    local[lane_offset + k + 3] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 3], 8, 32);
    local[lane_offset + k + 4] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 4], 8, 32);
    local[lane_offset + k + 5] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 5], 8, 32);
    local[lane_offset + k + 6] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 6], 8, 32);
    local[lane_offset + k + 7] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 7], 8, 32);
  }
}

template <> __device__ __forceinline__ void transpose<32>(uint32_t *local) {
  constexpr uint32_t per_thread = 32;
#pragma unroll
  for (int k = 0; k < per_thread; k += 2) {
    int lane_offset = threadIdx.x & 1;
    lane_offset ^= 1;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 1, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 4) {
    int lane_offset = (threadIdx.x >> 1) << 1;
    lane_offset %= 4;
    lane_offset ^= 2;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 2, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 2, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 8) {
    int lane_offset = (threadIdx.x >> 2) << 2;
    lane_offset %= 8;
    lane_offset ^= 4;
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 4, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 4, 32);
    local[lane_offset + k + 2] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 2], 4, 32);
    local[lane_offset + k + 3] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 3], 4, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 16) {
    int lane_offset = (threadIdx.x >> 3) << 3;
    lane_offset %= 16;
    lane_offset ^= 8;
    local[lane_offset + k + 0] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 0], 8, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 8, 32);
    local[lane_offset + k + 2] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 2], 8, 32);
    local[lane_offset + k + 3] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 3], 8, 32);
    local[lane_offset + k + 4] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 4], 8, 32);
    local[lane_offset + k + 5] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 5], 8, 32);
    local[lane_offset + k + 6] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 6], 8, 32);
    local[lane_offset + k + 7] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 7], 8, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 32) {
    int lane_offset = (threadIdx.x >> 4) << 4;
    lane_offset %= 32;
    lane_offset ^= 16;
    local[lane_offset + k + 0] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 0], 16, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 16, 32);
    local[lane_offset + k + 2] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 2], 16, 32);
    local[lane_offset + k + 3] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 3], 16, 32);
    local[lane_offset + k + 4] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 4], 16, 32);
    local[lane_offset + k + 5] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 5], 16, 32);
    local[lane_offset + k + 6] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 6], 16, 32);
    local[lane_offset + k + 7] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 7], 16, 32);
    local[lane_offset + k + 8] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 8], 16, 32);
    local[lane_offset + k + 9] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 9], 16, 32);
    local[lane_offset + k + 10] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 10], 16, 32);
    local[lane_offset + k + 11] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 11], 16, 32);
    local[lane_offset + k + 12] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 12], 16, 32);
    local[lane_offset + k + 13] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 13], 16, 32);
    local[lane_offset + k + 14] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 14], 16, 32);
    local[lane_offset + k + 15] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 15], 16, 32);
  }
}

template <>
__device__ __forceinline__ void transpose2<4>(uint32_t *local,
                                              const uint32_t pad) {
  constexpr uint32_t per_thread = 4;
#pragma unroll
  for (int k = 0; k < per_thread; k += 2) {
    int lane_offset = 1;
    if ((threadIdx.x / pad) % 2 == 0) {
      lane_offset = 1;
    } else {
      lane_offset = 0;
    }
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], pad, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 4) {
    int lane_offset = 1;
    if (((threadIdx.x / pad) >> 1) % 2 == 0) {
      lane_offset = 2;
    } else {
      lane_offset = 0;
    }
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], 2 * pad, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 2 * pad, 32);
  }
}

template <>
__device__ __forceinline__ void transpose2<8>(uint32_t *local,
                                              const uint32_t pad) {
  constexpr uint32_t per_thread = 8;
#pragma unroll
  for (int k = 0; k < per_thread; k += 2) {
    int lane_offset = 1;
    if ((threadIdx.x / pad) % 2 == 0) {
      lane_offset = 1;
    } else {
      lane_offset = 0;
    }
    local[lane_offset + k] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k], pad, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 4) {
    int lane_offset = 1;
    if (((threadIdx.x / pad) >> 1) % 2 == 0) {
      lane_offset = 2;
    } else {
      lane_offset = 0;
    }

    local[lane_offset + k + 0] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 0], 2 * pad, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 2 * pad, 32);
  }

#pragma unroll
  for (int k = 0; k < per_thread; k += 8) {
    int lane_offset = 1;
    if (((threadIdx.x / pad) >> 2) % 2 == 0) {
      lane_offset = 4;
    } else {
      lane_offset = 0;
    }

    local[lane_offset + k + 0] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 0], 4 * pad, 32);
    local[lane_offset + k + 1] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 1], 4 * pad, 32);
    local[lane_offset + k + 2] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 2], 4 * pad, 32);
    local[lane_offset + k + 3] =
        __shfl_xor_sync(0xffffffff, local[lane_offset + k + 3], 4 * pad, 32);
  }
}

template <> __device__ __forceinline__ void transpose<4>(uint32_t *local) {
  transpose2<4>(local, 1);
}
