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

"""Call a reusable encrypted function from the entry function."""

from cerium.dsl import (CeriumFunction, CeriumFunctionCall, CeriumProgram,
                        CiphertextArgument, CiphertextInput, FunctionOutput, Output)
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


# The caller refers to the helper built earlier in the same CeriumProgram.
double_function = None


def build_double_function(gpus):
    """Define a reusable helper that doubles its ciphertext argument."""
    global double_function
    double_function = CeriumFunction(f"double_{gpus}gpus", gpus, 0)
    with double_function:
        # CiphertextArgument comes from another CeriumFunction, not from the
        # external runner. Its scale and level must match the caller's value.
        value = CiphertextArgument("value", scale=56, level=LEVEL)
        doubled = value + value
        print(f"helper doubled: scale={doubled.scale()} bits, level={doubled.level()}", flush=True)
        # FunctionOutput returns a named value to the calling DSL function.
        FunctionOutput("doubled", doubled)


def build_main_function(gpus):
    """Define the entry point that calls the helper and exposes z."""
    if double_function is None:
        raise RuntimeError("Build the double function before the main function")
    main = CeriumFunction(main_name(gpus), gpus, 0)
    with main:
        # The runner supplies x; the call maps it to the helper's value
        # argument and retrieves the named doubled result.
        x = CiphertextInput("x", scale=56, level=LEVEL)
        result = CeriumFunctionCall(double_function, {"value": x})["doubled"]
        print(f"call result: scale={result.scale()} bits, level={result.level()}", flush=True)
        # Output is visible to the runtime runner, unlike FunctionOutput.
        Output("z", result)
    return main


def build_program(gpus):
    """Register both functions in one program, helper before caller."""
    global double_function
    double_function = None  # Avoid carrying a helper over from a previous build.
    program = CeriumProgram("function_calling", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        build_double_function(gpus)
        build_main_function(gpus)
    return program

if __name__ == "__main__":
    compile_cli("function_calling", build_program, __doc__)
