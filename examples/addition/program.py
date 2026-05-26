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

"""Addition of two encrypted vectors: z = x + y."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    # CeriumProgram groups the DSL functions and records the RNS bit size and
    # GPU partition count used by the compiler.
    program = CeriumProgram("addition", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            # These are symbolic encrypted inputs. The
            # runner supplies values under the same names at numeric scale
            # 2^56; LEVEL selects the starting RNS level for both inputs.
            x = CiphertextInput("x", scale=56, level=LEVEL)
            y = CiphertextInput("y", scale=56, level=LEVEL)
            # Slotwise addition needs equal input scales and levels. It
            # preserves both, which the metadata print below demonstrates.
            z = x + y
            print(f"addition z: scale={z.scale()} bits, level={z.level()}", flush=True)
            # Output gives the result the name used by the runtime runner.
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("addition", build_program, __doc__)
