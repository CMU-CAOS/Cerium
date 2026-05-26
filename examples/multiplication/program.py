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

"""Multiply an encrypted vector by a plaintext vector."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, PlaintextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("multiplication", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            x = CiphertextInput("x", scale=28, level=LEVEL)
            factor = PlaintextInput("factor", scale=28, level=LEVEL)
            # This is slotwise ciphertext-plaintext multiplication. Its scale
            # exponents add (28 + 28 = 56), while the level stays the same.
            z = x * factor
            print(f"multiplication z: scale={z.scale()} bits, level={z.level()}", flush=True)
            # The runner decrypts the named z output at numeric scale 2^56.
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("multiplication", build_program, __doc__)
