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
import pickle
import numpy as np
from pathlib import Path
import argparse
from primes import *
from bootstrap import *
import time
import tqdm


random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=64)
np.set_printoptions(threshold=64*1024)

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
        else:
            raise Exception("")
        count += 1
    return secretKey

ImgScaleDiv = 1<<4

ImgScale = Primes[3]*Primes[2] / ImgScaleDiv

def get_resnet_raw_inputs():

    raw_inputs = {}
    with open(f'resnet20_params.pkl', 'rb') as f:
        params = pickle.load(f)
    raw_inputs.update(params)
    return raw_inputs

SLOTS = 32768

def get_prediction(outputs):
    if "result" not in outputs.keys():
        return -1
    else:
        result = np.array(outputs["result"][0:10])
        return int(np.argmax(result.real))


import cerium.runtime as cerium_runtime

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

def generate_inputs(setup: CeriumFunctionSetup ):
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
    print(f"Created Inputs: {setup.function_name}")


def create_inputs(cerium_function_setup):
    for setup in cerium_function_setup:
        generate_inputs(setup)


def run_resnet(raw_inputs,OutScale,use_cudagraphs,base_dir,data_dir,gpus,num_samples):

    raw_inputs["image"] = ([0]*SLOTS,1 << 20)

    Slots = 32768

    secretKey = generate_secret_key(Slots,HammingWeight=32768)
    ephemeralKey = generate_secret_key(Slots,HammingWeight=32)

    seed = np.random.randint(0,2**32-1,size=8).tolist()
    context = cerium_runtime.Context(SLOTS,Primes)
    encryptor_context = cerium_runtime.CKKSEncryptorContext(context,secretKey,ephemeralKey,seed)
    encryptor = cerium_runtime.CKKSEncryptor(encryptor_context,seed)

    base_name = f"{base_dir}/compiled"
    if gpus in [1,2,4,8]:
        main_name = f"main_{gpus}gpus"
    else:
        raise Exception("Invalid GPU count")

    config = {
        f"{main_name}" : {
            "use_cudagraph" : f"{use_cudagraphs}"
        },
     }


 

    raw_inputs_wrapper = cerium_runtime.RawInputsWrapper(raw_inputs)
    raw_inputs.clear()

    cerium_function_setups = []


    def create_cerium_function_setup(base_name,function_name,generate_evalkeys,generate_plaintexts,use_uvm=False):
        return CeriumFunctionSetup(context,encryptor_context,base_name,function_name,raw_inputs_wrapper,gpus,4096,generate_evalkeys,generate_plaintexts,use_uvm)

    ### CREATE INTERPERTERS #####

    cerium_function_setups.append(create_cerium_function_setup(base_name,f"bootstrap_32K_33_{gpus}gpus",True,True))
    cerium_function_setups.append(create_cerium_function_setup(base_name,"relu",True,True))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"{main_name}",True,True))

    create_inputs(cerium_function_setups)

    program = cerium_runtime.FusedKernelProgram(context,base_name,f"{main_name}")

    cerium_functions_map = program.make_program(gpus,config,raw_inputs_wrapper)
    main_cerium_function = cerium_functions_map[f"{main_name}"]
    
    del raw_inputs
    del raw_inputs_wrapper

    data_values, data_labels = load_dataset(data_dir)
    if num_samples <= 0 or num_samples > len(data_labels):
        raise ValueError(f"--num-samples must be between 1 and {len(data_labels)}")

    samples = [i for i in range(num_samples)]
    Preds = []
    Labels = []
    pbar = tqdm.tqdm(range(len(samples)))
    results = {}
    total = 0
    correct = 0

    io_generator = cerium_runtime.IOGenerator(context)


    for image_id in samples:
        image_inputs, label = load_image(image_id, data_values, data_labels)

        image_inputs_wrapper = cerium_runtime.RawInputsWrapper(image_inputs)

        enc_image = io_generator.generate_ciphertext_inputs(f"{base_name}/{main_name}/program_inputs",image_inputs_wrapper,encryptor_context)
        main_cerium_function.copy_ciphertext_inputs_to_gpu(f"{base_name}/{main_name}/inputs",gpus,enc_image)

        program.run()    
        encrypted_outputs = main_cerium_function.get_program_outputs()

        outputs = {}
        for k,v in encrypted_outputs.items():
            outputs[k] = encryptor.decrypt_and_decode(v,OutScale[k])

        pred = get_prediction(outputs)
        correct += (pred == label)
        print("Image ID: ",image_id)
        print("Prediction: ",pred)
        print("Label: ",label)
        print(f"Image ID: {image_id}, Correct: {pred==label}")
        print("=============================")

        results["accuracy"] = correct / (total + 1)


        pbar.update(1)
        total += 1 

        Preds.append(pred)
        Labels.append(label)

        pbar.set_postfix(results)

    print(results)
    for i in range(len(Preds)):
        print(f"{i}: {Preds[i]} {Labels[i]} {int(Preds[i]==Labels[i])}")
    print(f"Final Accuracy: {correct}/{total} = {correct / total:.6f}")


def load_image_legacy(image_id):
    image_inputs = {}
    assert image_id >= 0 and image_id < 25 , "Image ID must be between 0 and 24"
    with open("testFile/test_values_25.txt", "r") as f:
        lines = f.readlines()
        lines = [float(l) for l in lines]

        image = [0] * (SLOTS)

        # move to image offset in file
        offset = 32 * 32 * 3 * image_id

        image[0:32*32*3] = lines[offset:offset + 32*32*3]

        init_p = 8
        for i in range(SLOTS//init_p, SLOTS):
            image[i] = image[i % (SLOTS//init_p)]

        image = np.array(image, dtype=np.float32)

        B = 40.0
        image = image / B
        image_inputs["image"] = (image.flatten().tolist(), ImgScale)

    with open("testFile/test_label.txt", "r") as f:
        lines = f.readlines()
        lines = [int(l) for l in lines]
        label = lines[image_id]

    return image_inputs, label


def load_dataset(data_dir):
    test_dir = Path(data_dir) / "testFile"
    values_path = test_dir / "test_values.txt"
    if not values_path.is_file():
        values_path = test_dir / "test_values_25.txt"
    labels_path = test_dir / "test_label.txt"
    if not values_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Expected ResNet inputs under {test_dir}")

    values = np.loadtxt(values_path, dtype=np.float32)
    labels = np.atleast_1d(np.loadtxt(labels_path, dtype=np.int64))
    if values.size != labels.size * 32 * 32 * 3:
        raise ValueError("ResNet input values do not match the number of labels")
    return values, labels


def load_image(image_id, values, labels):
    offset = 32 * 32 * 3 * image_id
    image = np.zeros(SLOTS, dtype=np.float32)
    image[:32 * 32 * 3] = values[offset:offset + 32 * 32 * 3]

    init_p = 8
    for i in range(SLOTS // init_p, SLOTS):
        image[i] = image[i % (SLOTS // init_p)]

    image /= 40.0
    return {"image": (image.tolist(), ImgScale)}, int(labels[image_id])


def main(use_cudagraphs,base_dir,data_dir,gpus,num_samples):

    print("Starting Program")

    Slots = 32768
    # set resnet inputs
    raw_inputs = get_resnet_raw_inputs()

    ## Bootstrap Inputs
    raw_inputs.update(get_bootstrap_runner_inputs("b33",19))

    OutScale = {}
    OutScale[f"result"] = Primes[3]*Primes[2] / ImgScaleDiv

    ## Debug Outputs
    OutScale[f"AvgPool"] = Primes[5]*Primes[4] / ImgScaleDiv
    l = 21
    for stage in range(20):
        OutScale[f"batch_{stage}"] = Primes[l-1]*Primes[l-2] / ImgScaleDiv
        OutScale[f"plus_{stage}"] = Primes[l-1]*Primes[l-2] / ImgScaleDiv
        OutScale[f"relu_{stage}_final"] = Primes[3]*Primes[2]/ImgScaleDiv
        OutScale[f"relu_{stage}_final_bootstrap"] = Primes[l-1]*Primes[l-2] / ImgScaleDiv

    run_resnet(raw_inputs,OutScale,use_cudagraphs,base_dir,data_dir,gpus,num_samples)

    return


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
    parser.add_argument('--base-dir', type=str, default="outputs", help="Base directory for compiled outputs")
    parser.add_argument('--data-dir', type=Path, default=Path('.'), help="Directory containing testFile/ inputs")
    parser.add_argument('--gpus','-g', type=int, default=1)
    parser.add_argument('--num-samples', '--num_samples', dest='num_samples', type=int, default=5)


    args = parser.parse_args()
    main(args.use_cudagraphs,args.base_dir,args.data_dir,args.gpus,args.num_samples)
