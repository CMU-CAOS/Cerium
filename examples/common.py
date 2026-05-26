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

"""Shared compilation and CKKS runtime setup for the small tutorials."""

import argparse
import random
from pathlib import Path

import numpy as np

from examples.primes import Primes

SLOTS = 32 * 1024
LEVEL = 12
RNS_BITS = 28
SUPPORTED_GPUS = (1, 2, 4, 8)


def output_root(example):
    return Path(__file__).resolve().parent / example / "outputs" / example


def main_name(gpus):
    if gpus not in SUPPORTED_GPUS:
        raise ValueError("GPU count must be 1, 2, 4, or 8")
    return f"main_{gpus}gpus"


def compile_program(program, example, gpus):
    """Lower every DSL function to CUDA and runtime metadata."""
    from cerium.compiler import cerium_compile

    cerium_compile(program, gpus, 1024, 128, str(output_root(example)))


def compile_cli(example, build_program, description):
    """Parse the shared compiler arguments and compile a DSL program."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpus", type=int, choices=SUPPORTED_GPUS, default=1)
    args = parser.parse_args()
    compile_program(build_program(args.gpus), example, args.gpus)


def run_cli(run_example, description):
    """Parse the shared runner arguments and execute one example."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpus", type=int, choices=SUPPORTED_GPUS, default=1)
    run_example(parser.parse_args().gpus)


def sample_vector(seed):
    """Use small values so the expected result is easy to inspect."""
    return np.random.default_rng(seed).integers(0, 5, size=SLOTS).astype(float)


def _secret_key(rng, hamming_weight):
    key = [0] * (2 * SLOTS)
    for position in rng.sample(range(2 * SLOTS), hamming_weight):
        key[position] = rng.choice((-1, 1))
    return key


def run_program(example, gpus, raw_inputs, expected, output_scale, *, atol=1e-4):
    """Run one compiled program and compare its decrypted output with NumPy.

    raw_inputs maps each DSL input name to (values, numeric CKKS scale).
    output_scale is the actual scale after operations such as rescale; it may
    differ from the power-of-two scale recorded in the DSL graph.
    """
    import cerium.runtime as runtime

    name = main_name(gpus)
    compiled = output_root(example) / "compiled"
    if not (compiled / name / "program_inputs").is_file():
        raise FileNotFoundError(f"Compile {example} for {gpus} GPU(s) first: {compiled / name}")

    # 1. Build the CKKS context and key material for this repeatable demo.
    rng = random.Random(10)
    seed = np.random.default_rng(10).integers(0, 2**32, size=8, dtype=np.uint64).tolist()
    context = runtime.Context(SLOTS, Primes)
    encryptor_context = runtime.CKKSEncryptorContext(
        context, _secret_key(rng, SLOTS), _secret_key(rng, 32), seed
    )
    encryptor = runtime.CKKSEncryptor(encryptor_context, seed)
    # 2. Wrap named Python values for Cerium's C++ runtime. The compiler's
    # program_inputs files tell IOGenerator which values and keys to prepare.
    inputs = runtime.RawInputsWrapper(raw_inputs)
    generator = runtime.IOGenerator(context)
    program_inputs = str(compiled / name / "program_inputs")

    # Called functions have their own descriptions and runtime files. Generate
    # those too, so make_program can load the complete call graph.
    for function_dir in sorted(compiled.glob(f"*_{gpus}gpus")):
        function_inputs = function_dir / "program_inputs"
        if not function_inputs.is_file():
            continue
        generator.generate_and_serialize_evalkeys(
            str(function_dir / "evalkeys"), str(function_inputs), encryptor_context
        )
        generator.generate_and_serialize_plaintexts(
            str(function_dir / "plaintexts"), str(function_inputs), inputs
        )

    # 3. Load compiled CUDA libraries and metadata. make_program constructs
    # the entry function and any functions that it calls.
    program = runtime.FusedKernelProgram(context, str(compiled), name)
    functions = program.make_program(
        gpus, {name: {"use_cudagraph": "True"}}, inputs
    )
    # 4. Encrypt ciphertext inputs, then copy them to the entry function's GPUs.
    encrypted_inputs = generator.generate_ciphertext_inputs(
        program_inputs, inputs, encryptor_context
    )
    functions[name].copy_ciphertext_inputs_to_gpu(
        str(compiled / name / "inputs"), gpus, encrypted_inputs
    )
    # 5. Execute the compiled graph. Output z remains encrypted here.
    program.run()

    # 6. Decrypt at the correct output scale and compare every slot.
    encrypted_output = functions[name].get_program_outputs()["z"]
    actual = np.asarray(encryptor.decrypt_and_decode(encrypted_output, output_scale))
    expected = np.asarray(expected)
    error = float(np.max(np.abs(actual - expected)))
    print(f"expected first 8 slots: {expected[:8]}")
    print(f"actual first 8 slots:   {actual[:8]}")
    print(f"maximum absolute error: {error:.3e}")
    if not np.allclose(actual, expected, rtol=1e-5, atol=atol):
        raise AssertionError(f"{example} output differs from its expected result")
