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

"""Check a repeated 2x2 plaintext matrix applied to encrypted pairs."""

import numpy as np
from examples.common import run_cli, run_program, sample_vector


def main(gpus):
    x = sample_vector(1)
    diagonal = np.array([2.0, 3.0])
    left_mask = np.array([1.0, 0.0])
    right_mask = np.array([0.0, 1.0])
    # For each pair [a, b], the expected result is [2a+b, a+3b].
    expected = np.empty_like(x)
    expected[0::2] = 2 * x[0::2] + x[1::2]
    expected[1::2] = x[0::2] + 3 * x[1::2]
    scale = 1 << 28
    ciphertext_scale = 1 << 84
    run_program("periodic_plaintext_matrix", gpus,
                {"x": (x, ciphertext_scale), "diagonal": (diagonal, scale),
                 "left_mask": (left_mask, scale), "right_mask": (right_mask, scale)},
                expected, ciphertext_scale * scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
