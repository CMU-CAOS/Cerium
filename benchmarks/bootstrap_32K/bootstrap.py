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

from cerium.dsl import *
import argparse
from cerium.compiler import cerium_compile
import os

bootstrapInputs = {}

def get_bootstrap_inputs(name,scale,level,scalar=False):
    partition_size = CurrentPartitionSize()
    partition_id = CurrentPartitionID()
    key = f"{name}_:p{partition_size}:i{partition_id}"
    if key not in bootstrapInputs.keys():
        bootstrapInputs[key] = PlaintextInput(name,scale,level,scalar)
    return bootstrapInputs[key]

def compute_homomorphic_mod_chebyshev_single_gpu(config,x,HK_deg,SCALE):
    if HK_deg != 30:
        raise Exception("Requires HK_deg to be 30 for the current implementation")
    cheby = [[] for _ in range(HK_deg + 1)]
    cheby_pr = [[] for _ in range(HK_deg + 1)]

    cheby[1] = x
    cheby_pr[1] = x

    coeff0 = get_bootstrap_inputs(f'{config}_c0_1',4*SCALE,cheby[1].level(),scalar=True)
    coeff1 = get_bootstrap_inputs(f'{config}_c1_1',4*SCALE,cheby[1].level(),scalar=True)
    val0 = cheby_pr[1]*coeff0
    val1 = cheby_pr[1]*coeff1

    relinearizeRequired = [True for _ in range(HK_deg + 1)]
    relinearizeRequired[1] = False

    for i in range(2,17,2):
        if i == 2 or i % 4 == 0:
            if relinearizeRequired[i//2]:
                cheby[i//2] = cheby[i//2].relinearize()
                cheby[i//2] = cheby[i//2].doubleRescale()
                relinearizeRequired[i//2] = False
            product = cheby[i//2]*cheby[i//2]
            minus_one = get_bootstrap_inputs(f'{config}_m_one_{i}',product.scale(),product.level(),scalar=True)
            cheby[i] = (product + product) + minus_one
        else:
            if relinearizeRequired[i//2 - 1]:
                cheby[i//2 - 1] = cheby[i//2 - 1].relinearize()
                cheby[i//2 - 1] = cheby[i//2 - 1].doubleRescale()
                relinearizeRequired[i//2] = False
            if relinearizeRequired[i//2+1]:
                cheby[i//2 + 1] = cheby[i//2 + 1].relinearize()
                cheby[i//2 + 1] = cheby[i//2 + 1].doubleRescale()
                relinearizeRequired[i//2 + 1] = False

            level_diff = cheby[i//2 - 1].level() - cheby[i//2 + 1].level()
            c0 = cheby[i//2 - 1]
            while level_diff > 0:
                c0 = c0.modswitch()
                level_diff = level_diff - 1
            product = c0*cheby[i//2 + 1]
            c0x = cheby[2]
            while product.level() - c0x.level() != 0:
                c0x = c0x.modswitch()
            rf = get_bootstrap_inputs(f'{config}_rf{i}',2*SCALE,c0x.level(),scalar=True)
            c0x = c0x*rf
            cheby[i] = (product + product) - c0x
        cheby_pr[i] = cheby[i]
        
    for i in range(2,16,2):
        coeff0 = get_bootstrap_inputs(f'{config}_c0_{i}',2*SCALE,cheby_pr[i].level(),scalar=True)
        coeff1 = get_bootstrap_inputs(f'{config}_c1_{i}',2*SCALE,cheby_pr[i].level(),scalar=True)
        product0 = cheby_pr[i]*coeff0
        product1 = cheby_pr[i]*coeff1
        while cheby_pr[i].level() != val0.level():
            val0 = val0.modswitch()
            val1 = val1.modswitch()
        val0 = product0 + val0
        val1 = product1 + val1

    val0 = BreakVector(val0)
    val1 = BreakVector(val1)
    len_val = len(val0)

    Val = MakeVector(val0 + val1)
    Val = Val.relinearize().doubleRescale()


    Val_break = BreakVector(Val)
    val0 = MakeVector(Val_break[:len_val])
    val1 = MakeVector(Val_break[len_val:])

    coeff0 = get_bootstrap_inputs(f'{config}_c0_0',Val.scale(),Val.level(),scalar=True)
    coeff1 = get_bootstrap_inputs(f'{config}_c1_0',Val.scale(),Val.level(),scalar=True)

    val0 = val0 + coeff0
    val1 = val1 + coeff1

    cheby[16] = cheby[16].relinearize()
    cheby[16] = cheby[16].doubleRescale()
    val1 = cheby[16]*val1
    val1 = val1.relinearize(2)
    val0 = val0.modswitch().modswitch()
    val = val0 + val1
    val = val.doubleRescale()
    return val


def compute_homomorphic_mod_single_gpu(x,x_level,scale,initScale,HK_deg,HK_r,config):

    if CurrentPartitionSize() != 1:
        raise Exception("Partition size should be 1 for single gpu computation")

    level = x.level()
    doubleScale = 2*scale

    f0 = get_bootstrap_inputs(f'{config}_f0',4*scale - initScale,level)
    f1 = get_bootstrap_inputs(f'{config}_f1',4*scale - initScale,level)
    X = MakeVector([x,x]) * MakeVector([f0,f1])
    
    X = X.conjugate() + X
    X = X.doubleRescale()

    f1by4K = get_bootstrap_inputs(f'{config}_f1_4K',doubleScale,X.level(),scalar=True)
    X = X + f1by4K

    val = compute_homomorphic_mod_chebyshev_single_gpu(config,X,HK_deg,scale)

    for i in range(HK_r):
        val = val * val
        val = val.relinearize().doubleRescale()
        m_one = get_bootstrap_inputs(f"{config}_m_one_r_{i}",val.scale(),val.level(),scalar=True)
        val = (val + val) + m_one

    val_extract = BreakVector(val)
    val0 = val_extract[0]
    val1 = val_extract[1]
    piqx0 = get_bootstrap_inputs(f'{config}_piqx0',initScale - val0.scale() + scale,val0.level())
    piqx1 = get_bootstrap_inputs(f'{config}_piqx1',initScale - val1.scale() + scale,val1.level())

    z = (val0*piqx0) + (val1*piqx1)
    return z


def compute_homomorphic_mod_multi_gpu(x,x_level,scale,initScale,HK_deg,HK_r,config,partition_size):

    def compute_homomorphic_mod_streamfn(sid,x,z):
        if sid > 2:
            return
        doubleScale = 2*scale
        f = PlaintextInput(f'{config}_f{sid}',doubleScale,x.level())
        x = x*f

        x = x.conjugate() + x
        x = x.doubleRescale()

        f1by4K = PlaintextInput(f'{config}_f1_4K',doubleScale,x.level(),scalar=True)
        x = x  + f1by4K

        bootstrapInputs.clear()
        val = compute_homomorphic_mod_chebyshev_multi_gpu(config,x,HK_deg,scale)

        bootstrapInputs.clear()

        for i in range(HK_r):
            val = val * val
            val = val.relinearize().doubleRescale()
            m_one = PlaintextInput(f"{config}_m_one_r_{i}",val.scale(),val.level(),scalar=True)
            val = (val + val) + m_one

        piqx = PlaintextInput(f'{config}_piqx{sid}',initScale - val.scale() + scale,val.level())
        z[sid] = val*piqx 

    z = [None for _ in range(2)]

    streamSize = max(1,partition_size//2)
    numStreams = 2
    CeriumStream(streamSize,numStreams,compute_homomorphic_mod_streamfn,x,z)
    z = z[0] + z[1]
    return z

def compute_homomorphic_mod_chebyshev_multi_gpu(config,x,HK_deg,SCALE):
    if HK_deg != 30:
        raise Exception("")
    cheby = [[] for _ in range(HK_deg + 1)]
    cheby_pr = [[] for _ in range(HK_deg + 1)]

    cheby[1] = x
    cheby_pr[1] = x

    coeff0 = PlaintextInput(f'{config}_c0_1',4*SCALE,cheby[1].level(),scalar=True)
    coeff1 = PlaintextInput(f'{config}_c1_1',4*SCALE,cheby[1].level(),scalar=True)
    val0 = cheby_pr[1]*coeff0
    val1 = cheby_pr[1]*coeff1

    relinearizeRequired = [True for _ in range(HK_deg + 1)]
    relinearizeRequired[1] = False

    for i in range(2,17,2):
        if i == 2 or i % 4 == 0:
            if relinearizeRequired[i//2]:
                cheby[i//2] = cheby[i//2].relinearize()
                cheby[i//2] = cheby[i//2].doubleRescale()
                relinearizeRequired[i//2] = False
            product = cheby[i//2]*cheby[i//2]
            minus_one = PlaintextInput(f'{config}_m_one_{i}',product.scale(),product.level(),scalar=True)
            cheby[i] = (product + product) + minus_one
        else:
            if relinearizeRequired[i//2 - 1]:
                cheby[i//2 - 1] = cheby[i//2 - 1].relinearize()
                cheby[i//2 - 1] = cheby[i//2 - 1].doubleRescale()
                relinearizeRequired[i//2] = False
            if relinearizeRequired[i//2+1]:
                cheby[i//2 + 1] = cheby[i//2 + 1].relinearize()
                cheby[i//2 + 1] = cheby[i//2 + 1].doubleRescale()
                relinearizeRequired[i//2 + 1] = False

            level_diff = cheby[i//2 - 1].level() - cheby[i//2 + 1].level()
            c0 = cheby[i//2 - 1]
            while level_diff > 0:
                c0 = c0.modswitch()
                level_diff = level_diff - 1
            product = c0*cheby[i//2 + 1]
            c0x = cheby[2]
            while product.level() - c0x.level() != 0:
                c0x = c0x.modswitch()
            rf = PlaintextInput(f'{config}_rf{i}',2*SCALE,c0x.level(),scalar=True)
            c0x = c0x*rf
            cheby[i] = (product + product) - c0x
        cheby_pr[i] = cheby[i]
        
    for i in range(2,16,2):
        coeff0 = PlaintextInput(f'{config}_c0_{i}',2*SCALE,cheby_pr[i].level(),scalar=True)
        coeff1 = PlaintextInput(f'{config}_c1_{i}',2*SCALE,cheby_pr[i].level(),scalar=True)
        product0 = cheby_pr[i]*coeff0
        product1 = cheby_pr[i]*coeff1
        while cheby_pr[i].level() != val0.level():
            val0 = val0.modswitch()
            val1 = val1.modswitch()
        val0 = product0 + val0
        val1 = product1 + val1


    val0 = val0.relinearize().doubleRescale()
    val1 = val1.relinearize().doubleRescale()

    Val = val0


    coeff0 = PlaintextInput(f'{config}_c0_0',Val.scale(),Val.level(),scalar=True)
    coeff1 = PlaintextInput(f'{config}_c1_0',Val.scale(),Val.level(),scalar=True)

    val0 = val0 + coeff0
    val1 = val1 + coeff1

    cheby[16] = cheby[16].relinearize()
    cheby[16] = cheby[16].doubleRescale()
    val1 = cheby[16]*val1
    if CurrentPartitionSize() == 1:
        val1 = val1.relinearize(2)
    else:
        val1 = val1.relinearize().doubleRescale()
    val0 = val0.modswitch().modswitch()
    val = val0 + val1
    val = val.doubleRescale()
    return val



def compute_homomorphic_mod(x,x_level,scale,initScale,HK_deg,HK_r,config,SetScalar=True):
    if CurrentPartitionSize() == 1:
        return compute_homomorphic_mod_single_gpu(x,x_level,scale,initScale,HK_deg,HK_r,config)
    else:
        return compute_homomorphic_mod_multi_gpu(x,x_level,scale,initScale,HK_deg,HK_r,config,CurrentPartitionSize())

def get_bsgs_plaintexts(prefix, babySteps, giantSteps, scale, level, period=None):
    plaintexts = []
    for gs in giantSteps:
        for bs in babySteps:
            rot_idx = bs+gs
            if period is not None:
                a = PeriodicPlaintextInput(prefix + str(rot_idx),scale,level,period)
            else:
                a = PlaintextInput(prefix + str(rot_idx),scale,level)
            plaintexts.append(a)
    return plaintexts

__bootstrap_32K_33_function__ = None
def bootstrap_32K_33_function_single_gpu_init():
    bootstrapInputs.clear()
    function = CeriumFunction("bootstrap_32Kslots_33levels_1gpus", 1,0)
    with function:
        x = CiphertextInput("x",56,2)
        refreshLevel = 18
        scale = 28
        doubleScale = 2*scale
        initScale = x.scale()
        HK_deg = 30
        HK_r = 3

        x = x.toEphemeral().bootstrapModRaise(refreshLevel + 33)

        prefix = f"b33_{refreshLevel}"

        sf = get_bootstrap_inputs(f"{prefix}_sf",scale,x.level(),True)
        x = x*sf

        babySteps = [0, 2048, 4096, 6144]
        giantSteps = [-16384, -8192, 0, 8192]
        pt = get_bsgs_plaintexts(f"{prefix}_au",babySteps,giantSteps,doubleScale,x.level())
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,rescaleLevels=2)

        babySteps = [0, 128, 256, 384, 512, 640, 768, 896]
        giantSteps = [-1920, -896, 128, 1152]
        pt = get_bsgs_plaintexts(f"{prefix}_bu",babySteps,giantSteps,doubleScale,x.level(),period=2048)
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,rescaleLevels=2)

        babySteps = [0, 8, 16, 24, 32, 40, 48, 56]
        giantSteps = [-120, -56, 8, 72]
        pt = get_bsgs_plaintexts(f"{prefix}_cu",babySteps,giantSteps,doubleScale,x.level(),period=128)
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,rescaleLevels=2)

        babySteps = [0, 1, 2, 3, 4]
        giantSteps = [-7, -2, 3]
        pt = get_bsgs_plaintexts(f"{prefix}_du",babySteps,giantSteps,doubleScale,x.level(),period=8)
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,rescaleLevels=2)
        
        x = x.rescale()

        z = compute_homomorphic_mod(x,x.level(),scale,initScale,HK_deg,HK_r,config=f"{prefix}")

        babySteps = [0, 1, 2, 3, 4, 5, 6, 7]
        giantSteps = [-32, -24, -16, -8, 0, 8, 16, 24]
        pt = get_bsgs_plaintexts(f"{prefix}_ap",babySteps,giantSteps,scale,z.level(),period=32)
        z = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,rescaleLevels=1)
        
        babySteps = [0, 32, 64, 96, 128, 160, 192, 224]
        giantSteps = [-1024, -768, -512, -256, 0, 256, 512, 768]
        pt = get_bsgs_plaintexts(f"{prefix}_bp",babySteps,giantSteps,scale,z.level(),period=1024)
        z = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,rescaleLevels=1)

        babySteps = [0, 1024, 2048, 3072]
        giantSteps = [-16384, -12288, -8192, -4096, 0, 4096, 8192, 12288]
        pt = get_bsgs_plaintexts(f"{prefix}_cp",babySteps,giantSteps,scale,z.level())
        z = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,rescaleLevels=1)
        z = z.rescale()
        Output("z",z)

    global __bootstrap_32K_33_function__
    __bootstrap_32K_33_function__ = function

__bootstrap_32K_33_n_function__ = None
def bootstrap_32K_33_function_multi_gpu_init(gpus):
    os.environ["CERIUM_REDUCTION_REWRITE"] = str(1)
    os.environ["CERIUM_CROSS_CHIP_OPTIMIZE"] = str(1)
    if gpus not in [2,4,8]:
        raise Exception("GPUs must be in [2,4,8]")
    bootstrapInputs.clear()
    function = CeriumFunction(f"bootstrap_32Kslots_33levels_{gpus}gpus", gpus,0)
    with function:
        x = CiphertextInput("x",56,2)
        refreshLevel = 18
        scale = 28
        doubleScale = 2*scale
        initScale = x.scale()
        HK_deg = 30
        HK_r = 3

        x = x.toEphemeral().bootstrapModRaise(refreshLevel + 33)

        prefix = f"b33_multi_{refreshLevel}"

        sf = get_bootstrap_inputs(f"{prefix}_sf",scale,x.level(),True)
        x = x*sf

        babySteps = [0, 2048, 4096, 6144]
        giantSteps = [-16384, -8192, 0, 8192]
        pt = get_bsgs_plaintexts(f"{prefix}_au",babySteps,giantSteps,doubleScale,x.level())
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,rescaleLevels=2)

        babySteps = [0, 128, 256, 384]
        giantSteps = [-1920, -1408, -896, -384, 128, 640, 1152, 1664]
        pt = get_bsgs_plaintexts(f"{prefix}_bu",babySteps,giantSteps,doubleScale,x.level(),period=2048)
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,rescaleLevels=2)

        babySteps = [0, 8, 16, 24]
        giantSteps = [-120, -88, -56, -24, 8, 40, 72, 104]
        pt = get_bsgs_plaintexts(f"{prefix}_cu",babySteps,giantSteps,doubleScale,x.level(),period=128)
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,rescaleLevels=2)

        babySteps = [0, 1, 2]
        giantSteps = [-7, -4, -1, 2, 5]
        pt = get_bsgs_plaintexts(f"{prefix}_du",babySteps,giantSteps,doubleScale,x.level(),period=8)
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,rescaleLevels=2)
        
        x = x.rescale()

        z = compute_homomorphic_mod(x,x.level(),scale,initScale,HK_deg,HK_r,config=f"{prefix}")

        babySteps = [0, 1, 2, 3, 4, 5, 6, 7]
        giantSteps = [-32, -24, -16, -8, 0, 8, 16, 24]
        pt = get_bsgs_plaintexts(f"{prefix}_ap",babySteps,giantSteps,scale,z.level(),period=32)
        z = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,rescaleLevels=1)
        
        babySteps = [0, 32, 64, 96, 128, 160, 192, 224]
        giantSteps = [-1024, -768, -512, -256, 0, 256, 512, 768]
        pt = get_bsgs_plaintexts(f"{prefix}_bp",babySteps,giantSteps,scale,z.level(),period=1024)
        z = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,rescaleLevels=1)

        babySteps = [0, 1024, 2048, 3072]
        giantSteps = [-16384, -12288, -8192, -4096, 0, 4096, 8192, 12288]
        pt = get_bsgs_plaintexts(f"{prefix}_cp",babySteps,giantSteps,scale,z.level())
        z = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,rescaleLevels=1)
        z = z.rescale()
        Output("z",z)

    global __bootstrap_32K_33_n_function__
    __bootstrap_32K_33_n_function__ = function

def bootstrap_single_gpu():
    program = CeriumProgram('bootstrap', rns_bit_size=28,num_gpus=1)
    with program:
        bootstrap_32K_33_function_single_gpu_init()

    return program

def bootstrap_multi_gpu(numGpus):
    program = CeriumProgram('bootstrap', rns_bit_size=28,num_gpus=numGpus)
    with program:
        bootstrap_32K_33_function_multi_gpu_init(numGpus)

    return program

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--vregs', type=int, default=4096)
    parser.add_argument('--prefix', type=str, default="outputs/")
    args = parser.parse_args()

    if args.gpus == 1:
        program = bootstrap_single_gpu()
    else:
        program = bootstrap_multi_gpu(args.gpus)

    os.environ["CERIUM_PRINT_GRAPHS"] = str(0)
    cerium_compile(program, args.gpus, args.vregs, 128, args.prefix)