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

"""Apply the same plaintext 2x2 matrix to every packed ciphertext pair."""

from cerium.dsl import (
    CeriumFunction, CeriumProgram, CiphertextInput, Output, PeriodicPlaintextInput,
)
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("periodic_plaintext_matrix", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            # The runner packs many independent [a, b] pairs into x.
            x = CiphertextInput("x", scale=84, level=LEVEL)
            # Apply M = [[2, 1], [1, 3]] to each adjacent pair. Periodic
            # plaintexts encode just a two-slot pattern and repeat it across
            # the ciphertext: diagonal=[2,3], left_mask=[1,0], and
            # right_mask=[0,1] are supplied by the runner.
            diagonal = PeriodicPlaintextInput("diagonal", scale=28, level=LEVEL, period=2)
            left_mask = PeriodicPlaintextInput("left_mask", scale=28, level=LEVEL, period=2)
            right_mask = PeriodicPlaintextInput("right_mask", scale=28, level=LEVEL, period=2)
            # Left rotation brings b to even positions; right rotation brings
            # a to odd positions. The masks discard unwanted neighbors, so
            # each pair becomes [2a+b, a+3b] without mixing adjacent pairs.
            # Each plaintext multiplication adds 28 to x's scale exponent.
            z = x * diagonal + (x << 1) * left_mask + (x >> 1) * right_mask
            print(f"matrix z: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("periodic_plaintext_matrix", build_program, __doc__)
