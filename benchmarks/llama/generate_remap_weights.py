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

import numpy as np
import argparse
import random
from pathlib import Path
from primes import *
import pickle
import cerium.runtime as cerium_runtime


import time

random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=64)

def gen_remap_inputs(block_start,num_blocks=1):

    context = cerium_runtime.Context(SLOTS,Primes)
    io_generator = cerium_runtime.IOGenerator(context)
    
    base_name = "outputs/compiled"
    plaintexts_base_name = "llama_plaintexts"

    
    def generate_remapables(function_name,function_name_dir,remap_key,inputs):
        Path(plaintexts_base_name, function_name_dir).mkdir(parents=True, exist_ok=True)
        io_generator.generate_and_serialize_remapables(f"{plaintexts_base_name}/{function_name_dir}/{remap_key}",f"{base_name}/{function_name}/remapable_inputs",inputs)

    for b in range(block_start,block_start+num_blocks) :
        with (Path('llama_models') / 'params' /  f'params_block{b}.pkl').open('rb') as f:
            block_inputs = pickle.load(f)
        params_inputs_wrapper = cerium_runtime.RawInputsWrapper(block_inputs)
        for fname in ["attention_matmul_qkv","attention_matmul_o","rmsnorm_att","rmsnorm_ffn","ffn_matmul_vw","ffn_matmul_o"]:
            generate_remapables(f"{fname}_1gpu",fname,f"block{b}",params_inputs_wrapper)

def main(block_start,num_blocks):
    
    gen_remap_inputs(block_start,num_blocks)


SLOTS = 32768
LVL = 51
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_start', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)

    args = parser.parse_args()
    main(args.block_start,args.num_blocks)

