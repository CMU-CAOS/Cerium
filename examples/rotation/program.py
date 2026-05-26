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

"""Rotate the slots of an encrypted vector by one position."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("rotation", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            # A single ciphertext packs many values into slots. The runner
            # encrypts all 32,768 slots at scale 2^84.
            x = CiphertextInput("x", scale=84, level=LEVEL)
            # << is a left rotation. It moves the
            # value from slot i+1 into slot i, wrapping at the last slot.
            # The compiler records the needed rotation evaluation key.
            z = x << 1
            print(f"rotation z: scale={z.scale()} bits, level={z.level()}", flush=True)
            # A rotation changes slot positions, but not the scale/level.
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("rotation", build_program, __doc__)
