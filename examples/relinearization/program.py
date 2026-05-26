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

"""Multiply two ciphertexts and relinearize the product."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("relinearization", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            x = CiphertextInput("x", scale=56, level=LEVEL)
            y = CiphertextInput("y", scale=56, level=LEVEL)
            # Multiplying two ciphertexts produces a larger ciphertext
            # representation. With both input scales at 2^56, the product
            # scale is 2^112; the level is still LEVEL.
            product = x * y
            print(f"product: scale={product.scale()} bits, level={product.level()}", flush=True)
            # Relinearize uses an evaluation key to return the product to the
            # usual ciphertext representation. It does not rescale here, so
            # the printed scale and level remain the same.
            z = product.relinearize()
            print(f"after relinearize: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("relinearization", build_program, __doc__)
