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

import random
import numpy as np
from fftFactors import *
import argparse
from primes import Primes
from bootstrap_values import *
import cerium.runtime as cerium_runtime
import os

random.seed(10)
np.random.seed(10)
np.set_printoptions(threshold=64*1024,linewidth=2000)

def generate_secret_key(Slots,HammingWeight=32):
    secretKey = [0]*(2*Slots)
    count = 0
    while count < HammingWeight:
        pos = random.randint(0,2*Slots-1)
        if secretKey[pos] != 0:
            continue
        val = random.randint(0,1)
        if val == 0:
            secretKey[pos] = -1
        elif val == 1:
            secretKey[pos] = 1
        count += 1
    return secretKey


class CeriumFunctionSetup:
    def __init__(self,context,encryptor_context,base_name,function_name,raw_inputs,gpus,vregs,generate_evalkeys=False,generate_plaintexts=False,use_uvm=False,layer_num=0):
        self.context = context
        self.encryptor_context = encryptor_context
        self.base_name = base_name
        self.function_name = function_name 
        self.raw_inputs = raw_inputs
        self.gpus = gpus 
        self.vregs = vregs
        self.generate_evalkeys = generate_evalkeys
        self.generate_plaintexts = generate_plaintexts
        self.use_uvm = use_uvm
        self.layer_num = layer_num

def generate_inputs(setup: CeriumFunctionSetup):
    base_name = setup.base_name
    function_name = setup.function_name
    generate_evalkeys = setup.generate_evalkeys
    generate_plaintexts = setup.generate_plaintexts
    use_uvm = setup.use_uvm
    io_generator = cerium_runtime.IOGenerator(setup.context)
    if generate_evalkeys:
        io_generator.generate_and_serialize_evalkeys(f"{base_name}/{function_name}/evalkeys",f"{base_name}/{function_name}/program_inputs",setup.encryptor_context)
    if generate_plaintexts:
        io_generator.generate_and_serialize_plaintexts(f"{base_name}/{function_name}/plaintexts",f"{base_name}/{function_name}/program_inputs",setup.raw_inputs)
    # print(f"Created Inputs: {setup.function_name}")


def create_inputs(cerium_function_setup):
    for setup in cerium_function_setup:
        generate_inputs(setup)


SLOTS = 32*1024
def run_bootstrap(use_cudagraph,generate_evalkeys,generate_plaintexts,gpus=1):

    # Keep the historical default while allowing experiments to select a run count.
    os.environ.setdefault("CERIUM_RUNTIME_NUM_ITERS", "10")

    OutScale = {}
    if gpus == 1:
        raw_inputs = get_bootstrap_runner_inputs("b33",18)
    else:
        raw_inputs = get_bootstrap_runner_inputs("b33_multi",18)



    secretKey = generate_secret_key(SLOTS,HammingWeight=SLOTS)
    ephemeralKey = generate_secret_key(SLOTS,HammingWeight=32)

    seed = np.random.randint(0,2**32-1,size=8).tolist()
    context = cerium_runtime.Context(SLOTS,Primes)
    encryptor_context = cerium_runtime.CKKSEncryptorContext(context,secretKey,ephemeralKey,seed)
    encryptor = cerium_runtime.CKKSEncryptor(encryptor_context,seed)

    log_message_ratio = 5
    InpScale = Primes[1] * Primes[0] / (1 << log_message_ratio)
    arr = np.array([np.exp(1j * random.uniform(0,2 * np.pi) ) for i in range(SLOTS)],dtype=np.complex128)

    inputs = {}
    inputs["x"] = (arr,InpScale)
    OutScale["z"] = InpScale


    base_name = "outputs/compiled"


    cerium_function_setups = []


    raw_inputs_wrapper = cerium_runtime.RawInputsWrapper(raw_inputs)
    raw_inputs.clear()

    vregs = 4096

    def create_cerium_function_setup(base_name,function_name,generate_evalkeys,generate_plaintexts,use_uvm=False):
        return CeriumFunctionSetup(context,encryptor_context,base_name,function_name,raw_inputs_wrapper,gpus,vregs,generate_evalkeys,generate_plaintexts,use_uvm)

    bootstrap_name = f"bootstrap_32Kslots_33levels_{gpus}gpus"
    cerium_function_setups.append(create_cerium_function_setup(base_name,bootstrap_name,generate_evalkeys,generate_plaintexts))

    create_inputs(cerium_function_setups)

    program = cerium_runtime.FusedKernelProgram(context,base_name,bootstrap_name)

    config = {
        f"{bootstrap_name}" : {
            "use_cudagraph" : f"{use_cudagraph}"
        }
    }

    inputs_wrapper = cerium_runtime.RawInputsWrapper(inputs)


    cerium_functions_map = program.make_program(gpus,config,raw_inputs_wrapper)
    main_cerium_function = cerium_functions_map[bootstrap_name]


    io_generator = cerium_runtime.IOGenerator(context)
    encrypted_inputs = io_generator.generate_ciphertext_inputs(f"{base_name}/{bootstrap_name}/program_inputs",inputs_wrapper,encryptor_context)
    main_cerium_function.copy_ciphertext_inputs_to_gpu(f"{base_name}/{bootstrap_name}/inputs",gpus,encrypted_inputs)

    program.run()


    del raw_inputs
    del raw_inputs_wrapper

    encrypted_outputs = main_cerium_function.get_program_outputs()

    outputs = {}
    for k,v in encrypted_outputs.items():
        outputs[k] = encryptor.decrypt_and_decode(v,OutScale[k])

    z = np.array(outputs["z"])
    # print(z.reshape(-1,64)[-1])
    err = np.max(np.abs(z - arr))
    print(f"Max Absolute Error: {err:.2e}, Precision Bits: {-np.log2(err):.2f}")
    real_err = np.max(np.abs(z.real - arr.real))
    print(f"Max Absolute Real Error: {real_err:.2e}, Precision Bits: {-np.log2(real_err):.2f}")
    imag_err = np.max(np.abs(z.imag - arr.imag))
    print(f"Max Absolute Imag Error: {imag_err:.2e}, Precision Bits: {-np.log2(imag_err):.2f}")
    abs_err = np.mean(np.abs(z - arr))
    print(f"Mean Absolute Error: {abs_err:.2e}, Precision Bits: {-np.log2(abs_err):.2f}")
    rel_err = np.mean(np.abs(z - arr) / np.abs(arr + 1e-6))
    print(f"Mean Relative Error: {rel_err:.2e}, Precision Bits: {-np.log2(rel_err):.2f}")


def str_to_bool(value):
    """ Convert common string representations of truth values to boolean. """
    if isinstance(value, bool):
        return value  # Already a bool, return as is
    value = value.lower()
    if value in {'true', 't', 'yes', 'y', '1'}:
        return True
    elif value in {'false', 'f', 'no', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


    
if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-cg','--use_cudagraphs', type=str_to_bool,  nargs='?', const=True, default=True, help="Whether to use CUDA graphs")
    parser.add_argument('-k','--generate_evalkeys', type=str_to_bool, nargs='?', const=True, default=True, help="Generate evaluation keys")
    parser.add_argument('-p','--generate_plaintexts', type=str_to_bool, nargs='?', const=True, default=True, help="Generate plaintexts")
    parser.add_argument('-n','--gpus', type=int, default=1, help="Number of GPUs")

    args = parser.parse_args()
    run_bootstrap(args.use_cudagraphs,args.generate_evalkeys,args.generate_plaintexts,args.gpus)
