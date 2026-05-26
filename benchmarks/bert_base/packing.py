# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Torch tensor layouts used by parameter and inference packing."""

import torch

SLOTS = 32768
SEQUENCE_LENGTH = 128
HIDDEN_SIZE = 768


def zeros(shape, reference):
    return torch.zeros(shape, dtype=reference.dtype, device=reference.device)


def rotate(vector, steps):
    return torch.roll(vector, -steps, dims=-1)


def row_pack_768x64(first, second):
    out = zeros((6, SLOTS), first)
    matrices = (first, second)
    for lane in range(2):
        for block in range(3):
            for row in range(128):
                out[block + 3 * lane, 256 * row:256 * row + 64] = matrices[lane][block * 256 + row]
                out[block + 3 * lane, 256 * row + 128:256 * row + 192] = matrices[(lane + 1) % 2][block * 256 + 128 + row]
    return out


def row_pack_64x1(first, second):
    out = zeros(SLOTS, first)
    for row in range(128):
        out[256 * row:256 * row + 64] = first
        out[256 * row + 128:256 * row + 192] = second
    return out


def row_pack_768x768(matrix):
    out = zeros((3, 6, SLOTS), matrix)
    for chunk in range(3):
        matrices = (matrix[:, chunk * 256:chunk * 256 + 128], matrix[:, chunk * 256 + 128:chunk * 256 + 256])
        for lane in range(2):
            for block in range(3):
                for row in range(128):
                    out[chunk, block + 3 * lane, 256 * row:256 * row + 128] = matrices[lane][block * 256 + row]
                    out[chunk, block + 3 * lane, 256 * row + 128:256 * row + 256] = matrices[(lane + 1) % 2][block * 256 + 128 + row]
    return out


def row_pack_768x1(vector):
    out = zeros((3, SLOTS), vector)
    for chunk in range(3):
        first = vector[chunk * 256:chunk * 256 + 128]
        second = vector[chunk * 256 + 128:chunk * 256 + 256]
        for row in range(128):
            out[chunk, 256 * row:256 * row + 128] = first
            out[chunk, 256 * row + 128:256 * row + 256] = second
    return out


def row_pack_768x1_type2(vector):
    out = zeros(SLOTS, vector)
    out[:768] = vector
    return out


def row_pack_768x2(matrix):
    out = zeros(SLOTS, matrix)
    out[:768] = matrix[:, 0]
    out[1024:1792] = matrix[:, 1]
    return out


def pack_2x1(vector):
    out = zeros(SLOTS, vector)
    out[0], out[1024] = vector[0], vector[1]
    return out


def pack_128x768(values):
    if values.shape != (SEQUENCE_LENGTH, HIDDEN_SIZE):
        raise ValueError(f"expected {(SEQUENCE_LENGTH, HIDDEN_SIZE)}, got {tuple(values.shape)}")
    return values.reshape(SEQUENCE_LENGTH, 3, 256).permute(1, 0, 2).reshape(3, SLOTS)


def unpack_128x768(values):
    if values.shape != (3, SLOTS):
        raise ValueError(f"expected {(3, SLOTS)}, got {tuple(values.shape)}")
    return values.reshape(3, SEQUENCE_LENGTH, 256).permute(1, 0, 2).reshape(SEQUENCE_LENGTH, HIDDEN_SIZE)
