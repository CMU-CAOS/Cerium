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

"""Shared polynomial approximations used by BERT inference variants."""

import torch


def exp_approx(x, iterations=6):
    result = 1 + x / (2 ** iterations)
    for _ in range(iterations):
        result = result * result
    return result


def inv_sqrt(x, iterations=6):
    estimate = exp_approx(-(x * 0.5 + 0.2), iterations) * 2 + 0.2
    for _ in range(iterations):
        estimate = estimate * (3 - x * estimate * estimate) * 0.5
    return estimate


def inv_sqrt_no_exp(x, iterations=6):
    estimate = 0.2
    for _ in range(iterations):
        estimate = estimate * (3 - x * estimate * estimate) * 0.5
    return estimate


def reciprocal_approx(x):
    return inv_sqrt(x) ** 2


def reciprocal_approx_no_exp(x):
    return inv_sqrt_no_exp(x) ** 2


def f3(x):
    return -5 / 16 * x**7 + 21 / 16 * x**5 - 35 / 16 * x**3 + 35 / 16 * x


def g3(x):
    return -12860 / 2**10 * x**7 + 25614 / 2**10 * x**5 - 16577 / 2**10 * x**3 + 4589 / 2**10 * x


def sign_approx(x):
    return f3(f3(g3(g3(x))))


def approx_gelu(x):
    lower = -0.5054031199708174 - 0.4222658115198386 * x - 0.1180761295118195 * x**2 - 0.0110341340306157 * x**3
    middle = 0.0085263215410380 + 0.5 * x + 0.3603292692789629 * x**2 - 0.037688200365904 * x**4 + 0.0018067462606141 * x**6
    s0 = sign_approx((x + 5) / 100) * 0.5
    s1 = sign_approx((x + 1.97) / 100) * 0.5
    s2 = sign_approx((x - 3) / 100) * 0.5
    return (s0 - s1) * lower + (s1 - s2) * middle + (s2 + 0.5) * x


def max_values(x):
    """Approximate the maximum over the last power-of-two dimension."""
    size = x.shape[-1]
    result = x
    shift = 1
    while shift < size:
        indices = torch.tensor([(index ^ shift) % size for index in range(size)], device=x.device)
        other = torch.index_select(result, -1, indices)
        difference = result - other
        result = ((result + other) + sign_approx(difference) * difference) / 2
        shift *= 2
    return result[..., :1]


def softmax_approx(scores, mask):
    """Approximate softmax used by the base Torch BERT model."""
    scores = scores.to(torch.float32)
    scores = scores - max_values(scores / 40) * 40
    exponentials = exp_approx(scores) * (mask != -1000)
    denominator = exponentials.sum(dim=-1, keepdim=True) + 1e-8
    return exponentials * reciprocal_approx_no_exp(denominator)


def approx_tanh(x):
    """Piecewise tanh approximation used by the base Torch BERT model."""
    s0 = sign_approx((x + 2.855) / 100) * 0.5
    s1 = sign_approx((x - 2.855) / 100) * 0.5
    return (0.5 - s0) * -1 + (s0 - s1) * (x / 2.855) + (s1 + 0.5)
