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

"""Multiply an encrypted vector by plaintext, then rescale the product."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, PlaintextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("rescale", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            # The input scales are DSL exponents: the runner encodes x at
            # 2^56 and factor at 2^28 before the GPU program starts.
            x = CiphertextInput("x", scale=56, level=LEVEL)
            factor = PlaintextInput("factor", scale=28, level=LEVEL)
            # Multiplication adds the scale exponents but leaves the level
            # unchanged: 56 + 28 gives a product scale exponent of 84.
            product = x * factor
            print(f"product: scale={product.scale()} bits, level={product.level()}", flush=True)
            # Rescale divides by the last active RNS prime and removes that
            # prime from the modulus chain. The level falls from 12 to 11;
            # Cerium records the new scale exponent as 84 - RNS_BITS = 56.
            # The runner uses the exact prime when decoding the output.
            z = product.rescale()
            print(f"after rescale: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("rescale", build_program, __doc__)
