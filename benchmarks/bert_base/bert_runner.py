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
import tqdm
from primes import Primes
import pickle
import cerium.runtime as cerium_runtime
import math
from bootstrap import *

import time

random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=64)

def generate_secret_key(Slots,HammingWeight=32):
    if not 0 <= HammingWeight <= 2 * Slots:
        raise ValueError("HammingWeight must be between 0 and 2 * Slots")

    secretKey = [0]*(2*Slots)
    count = 0
    while count < HammingWeight:
        pos = random.randint(0,2*Slots-1)
        val = random.randint(0,1)
        if secretKey[pos] != 0:
            continue
        secretKey[pos] = -1 if val == 0 else 1
        count += 1
    return secretKey

def generate_softmax_att_mask(input_mask):

    input_mask = (input_mask != -1000).astype(np.float64)[0]
    att_mask = np.zeros((SLOTS,))
    for k in range(2):
        for j in range(128):
            for i in range(128):
                att_mask[128*k + 256*j + i] = input_mask[(i+j) % 128]
    return att_mask


def row_pack_128x768(M):
    res = np.zeros((3,SLOTS,))
    for j in range(3):
        for i in range(128):
            for l in range(256):
                    res[j][256*i + l] = M[i][l + 256*j]
    return np.array(res)

def get_g3_io(inputScale,finalOutScale,level,prefix):
    coeffs = [None for i in range(8)]
    coeffScales = [None for i in range(8)]
    xPowScales = [None for i in range(8)]
    coeffs[7] = -12860/2**10
    coeffs[5] = 25614/2**10
    coeffs[3] = -16577/2**10
    coeffs[1] = 4589/2**10
    Inputs = {}
    OutputScales = {}
    xPowScales[1] = inputScale
    for i in range(1,3):
        xPowScales[2**i] = (xPowScales[2**(i-1)]**2)/Primes[level-i]
        OutputScales[f"{prefix}_powsx_{2**i}"] = xPowScales[2**i]
    
    OutputScale = finalOutScale
    OutputScale = OutputScale/2
    coeffScales[7] = OutputScale * Primes[level-3] * Primes[level-1] * Primes[level-2] / (xPowScales[4] * xPowScales[2] * xPowScales[1])


    coeffScales[5] = OutputScale * Primes[level-3] * Primes[level-1] * Primes[level-2] / (xPowScales[4] * xPowScales[1])
    coeffScales[3] = OutputScale * Primes[level-3] * Primes[level-1] * Primes[level-2] / (xPowScales[2] * xPowScales[1])
    coeffScales[1] = OutputScale * Primes[level-1] * Primes[level-3] / (xPowScales[1])

    for i in [1,3,5,7]:
        Inputs[f"{prefix}_g3_coeff{i}_{level}"] = (coeffs[i],coeffScales[i]) 
    OutputScale = OutputScale*2
    OutputScales[f"{prefix}_x_{level}"] = OutputScale
    return (Inputs,OutputScale,OutputScales)

def get_g3_io_new(inputScale,finalOutScale,level,prefix):
    coeffs = [None for i in range(8)]
    coeffScales = [None for i in range(8)]
    xPowScales = [None for i in range(8)]
    coeffs[7] = -12860/2**10
    coeffs[5] = 25614/2**10
    coeffs[3] = -16577/2**10
    coeffs[1] = 4589/2**10
    Inputs = {}
    OutputScales = {}
    xPowScales[1] = inputScale
    l = level
    for i in range(1,3):
        xPowScales[2**i] = (xPowScales[2**(i-1)]**2)/(Primes[l-1]*Primes[l-2])
        l -= 2
        OutputScales[f"{prefix}_powsx_{2**i}_{l}"] = xPowScales[2**i]

    OutputScale = finalOutScale
    OutputScale = OutputScale/2
    coeffScales[7] = OutputScale * Primes[level-1] * Primes[level-2] * Primes[level-3]* Primes[level-4] * Primes[level-5] * Primes[level-6] / (xPowScales[4] * xPowScales[2] * xPowScales[1])


    coeffScales[5] = OutputScale * Primes[level-1] * Primes[level-2] * Primes[level-5] * Primes[level-6] / (xPowScales[4] * xPowScales[1])
    coeffScales[3] = OutputScale * Primes[level-1] * Primes[level-2] * Primes[level-5] * Primes[level-6] / (xPowScales[2] * xPowScales[1])
    coeffScales[1] = OutputScale * Primes[level-5] * Primes[level-6] / (xPowScales[1])

    for i in [1,3,5,7]:
        Inputs[f"{prefix}_g3_coeff{i}_{level}"] = (coeffs[i],coeffScales[i]) 
        Inputs[f"{prefix}_g3_coeff{i}_{level}_im"] = (np.array([coeffs[i]*1j]*SLOTS),coeffScales[i]) 
        OutputScales[f"x{i}_{level-4}"] = OutputScale*2
    OutputScale = OutputScale*2
    OutputScales[f"{prefix}_x_{level}"] = OutputScale
    return (Inputs,OutputScale,OutputScales)


    y = - 5/ 16 * x**7 + 21 / 16 * x**5 - 35 / 16 * x**3 + 35 / 16 * x
def get_f3_io(inputScale,finalOutScale,level,prefix):
    coeffs = [None for i in range(8)]
    coeffScales = [None for i in range(8)]
    xPowScales = [None for i in range(8)]
    coeffs[7] = -5/16
    coeffs[5] = 21/16
    coeffs[3] = -35/16
    coeffs[1] = 35/16
    Inputs = {}
    OutputScales = {}
    xPowScales[1] = inputScale
    for i in range(1,3):
        xPowScales[2**i] = (xPowScales[2**(i-1)]**2)/Primes[level-i]
        OutputScales[f"{prefix}_powsx_{2**i}"] = xPowScales[2**i]


    OutputScale = finalOutScale
    OutputScale = OutputScale/2
    coeffScales[7] = OutputScale * Primes[level-3] * Primes[level-1] * Primes[level-2] / (xPowScales[4] * xPowScales[2] * xPowScales[1])

    coeffScales[5] = OutputScale * Primes[level-3] * Primes[level-1] * Primes[level-2] / (xPowScales[4] * xPowScales[1])
    coeffScales[3] = OutputScale * Primes[level-3] * Primes[level-1] * Primes[level-2] / (xPowScales[2] * xPowScales[1])
    coeffScales[1] = OutputScale * Primes[level-3] * Primes[level-1] / (xPowScales[1])

    for i in [1,3,5,7]:
        Inputs[f"{prefix}_f3_coeff{i}_{level}"] = (coeffs[i],coeffScales[i]) 
    OutputScale = OutputScale*2
    # OutputScale = OutputScale*16
    OutputScales[f"{prefix}_x_{level}"] = OutputScale
    return (Inputs,OutputScale,OutputScales)

# y = - 5/ 16 * x**7 + 21 / 16 * x**5 - 35 / 16 * x**3 + 35 / 16 * x
def get_f3_io_new(inputScale,finalOutScale,level,prefix):
    coeffs = [None for i in range(8)]
    coeffScales = [None for i in range(8)]
    xPowScales = [None for i in range(8)]
    coeffs[7] = -5/16
    coeffs[5] = 21/16
    coeffs[3] = -35/16
    coeffs[1] = 35/16
    Inputs = {}
    OutputScales = {}
    xPowScales[1] = inputScale
    l = level
    for i in range(1,3):
        xPowScales[2**i] = (xPowScales[2**(i-1)]**2)/(Primes[l-1]*Primes[l-2])
        OutputScales[f"{prefix}_powsx_{2**i}"] = xPowScales[2**i]
        l -= 2

    OutputScale = finalOutScale
    OutputScale = OutputScale/2
    coeffScales[7] = OutputScale * Primes[level-1] * Primes[level-2] * Primes[level-3]* Primes[level-4] * Primes[level-5] * Primes[level-6] / (xPowScales[4] * xPowScales[2] * xPowScales[1])


    coeffScales[5] = OutputScale * Primes[level-1] * Primes[level-2] * Primes[level-5] * Primes[level-6] / (xPowScales[4] * xPowScales[1])
    coeffScales[3] = OutputScale * Primes[level-1] * Primes[level-2] * Primes[level-5] * Primes[level-6] / (xPowScales[2] * xPowScales[1])
    coeffScales[1] = OutputScale * Primes[level-5] * Primes[level-6] / (xPowScales[1])

    for i in [1,3,5,7]:
        Inputs[f"{prefix}_f3_coeff{i}_{level}"] = (coeffs[i],coeffScales[i]) 
        Inputs[f"{prefix}_f3_coeff{i}_{level}_im"] = (np.array([coeffs[i]*1j]*SLOTS),coeffScales[i]) 

    OutputScale = OutputScale*2
    OutputScales[f"{prefix}_x_{level}"] = OutputScale
    return (Inputs,OutputScale,OutputScales)


def get_exp_io_bootstrap(inputScale,level,BootstrapScaleDiv):

    Inputs = {}
    OutputScales = {}
    BootstrapLevel = 13
    outScale = Primes[BootstrapLevel-1]*Primes[BootstrapLevel-2]*Primes[BootstrapLevel-3]*Primes[BootstrapLevel-4]
    outScale = math.sqrt(outScale)
    BootScalDiv = 1 << 14
    outScale = outScale / BootScalDiv
    ITERS = 6
    iters = 5
    denom = 0
    shift = 2
    for i in range(iters):
        prod = math.log(Primes[level-shift-2*i-1]*Primes[level-shift-2*i-2])
        denom = 2*denom + prod
    denom += math.log(outScale)
    inpScale = np.exp(denom / (2**iters))

    expOneBy2NScale = inpScale*Primes[level-1]*Primes[level-2]/inputScale

    Inputs["exp_oneBy2N"] = (1/(2**ITERS),expOneBy2NScale)
    Inputs["exp_one"] = (1,inpScale)
    Inputs["exp_bootstrap_sf"] = (1,BootScalDiv)

    OutputScales["one_plus_x"] = inpScale
    OutputScales["exp_x_5"] = outScale
    OutputScales["exp_x_5_bs"] = outScale
    
    outScale = Primes[BootstrapLevel-3]*Primes[BootstrapLevel-4]
    OutputScales["exp_x"] = outScale

    return (Inputs,outScale,OutputScales)

def get_isqrt_io(inputScale,level,BootstrapScaleDiv,BootstrapScaleDivIsqrt,finalOutScale,prefix):

    Inputs = {}
    OutputScales = {}
    bootstrapOutLevel = 12
    outScale = Primes[bootstrapOutLevel-1]*Primes[bootstrapOutLevel-2]*Primes[bootstrapOutLevel-3]*Primes[bootstrapOutLevel-4]
    outScale = math.sqrt(outScale)
    BootScalDiv = BootstrapScaleDiv
    outScale = outScale / BootScalDiv
    ITERS = 6
    iters = 5
    denom = 0
    shift = 2
    for i in range(iters):
        prod = math.log(Primes[level-shift-2*i-1]*Primes[level-shift-2*i-2])
        denom = 2*denom + prod
    denom += math.log(outScale)
    inpScale = np.exp(denom / (2**iters))

    expOneBy2NScale = inpScale*Primes[level-1]*Primes[level-2]/inputScale

    Inputs[f"{prefix}_isqrt_minus_zero_pt_4"] = (-0.4,inputScale)
    Inputs[f"{prefix}_isqrt_oneBy2N"] = (1/(2*(2**ITERS)),expOneBy2NScale)
    Inputs[f"{prefix}_isqrt_one"] = (1,inpScale)
    Inputs[f"{prefix}_isqrt_exp_bootstrap_sf"] = (1,BootScalDiv)

    yscale = Primes[bootstrapOutLevel-3]*Primes[bootstrapOutLevel-4]
    Inputs[f"{prefix}_isqrt_zeroPt2"] = (0.2,yscale)

    xScale = inputScale
    yLevel = bootstrapOutLevel - 2
    bsLevel = 10

    for i in range(6):
        ysqScale = yscale * yscale / (Primes[yLevel - 1]*Primes[yLevel - 2])
        ySqLevel = yLevel - 2

        if i == 5:
            yOutScale = finalOutScale / BootstrapScaleDivIsqrt
        elif yLevel == 6:
            yOutScale = Primes[bsLevel-1]*Primes[bsLevel-2] / BootstrapScaleDivIsqrt
        else:
            yOutScale = Primes[ySqLevel-3]*Primes[ySqLevel-4] / BootstrapScaleDivIsqrt
        yOutScale *= (Primes[ySqLevel-1]*Primes[ySqLevel-2])

        xrescaleFactor = yOutScale*Primes[yLevel+1]*Primes[yLevel]*Primes[yLevel-1]*Primes[yLevel-2]/(ysqScale * yscale * xScale)
        yrescaleFactor = yOutScale/(yscale)

        Inputs[f"{prefix}_isqrt_xrescale_{i}"] = (0.5,xrescaleFactor/2)
        Inputs[f"{prefix}_isqrt_yrescale_{i}"] = (1.5,yrescaleFactor/2)
        
        yscale = yOutScale / (Primes[ySqLevel-1] * Primes[ySqLevel-2])
        if yLevel == 6:
            yLevel = bsLevel
        else:
            yLevel = ySqLevel - 2

    Inputs[f"{prefix}_isqrt_y_bootstrap_sf"] = (1, BootstrapScaleDivIsqrt)
    yscale = yscale * BootstrapScaleDivIsqrt
    OutputScales[f"isqrt"] = finalOutScale 


    return (Inputs,finalOutScale,OutputScales)

def get_isqrt_io_no_exp(inputScale,level,BootstrapScaleDiv,BootstrapScaleDivIsqrt,finalOutScale,prefix):

    Inputs = {}
    OutputScales = {}
    yLevel = 10
    yscale = Primes[yLevel-1]*Primes[yLevel-2]
    Inputs[f"{prefix}_isqrt_zeroPt2"] = (0.2,yscale)

    xScale = inputScale
    bsLevel = 10

    for i in range(6):
        ysqScale = yscale * yscale / (Primes[yLevel - 1]*Primes[yLevel - 2])
        ySqLevel = yLevel - 2

        if i == 0:
            Inputs[f"{prefix}_isqrt_zeroPt04"] = (0.04,ysqScale)
        if i == 5:
            yOutScale = finalOutScale / BootstrapScaleDivIsqrt
        elif yLevel == 6:
            yOutScale = Primes[bsLevel-1]*Primes[bsLevel-2] / BootstrapScaleDivIsqrt
        else:
            yOutScale = Primes[ySqLevel-3]*Primes[ySqLevel-4] / BootstrapScaleDivIsqrt
        yOutScale *= (Primes[ySqLevel-1]*Primes[ySqLevel-2])

        xrescaleFactor = yOutScale*Primes[yLevel+1]*Primes[yLevel]*Primes[yLevel-1]*Primes[yLevel-2]/(ysqScale * yscale * xScale)
        yrescaleFactor = yOutScale/(yscale)

        Inputs[f"{prefix}_isqrt_xrescale_{i}"] = (0.5,xrescaleFactor/2)

        if i == 0:
            Inputs[f"{prefix}_isqrt_yrescale_{i}"] = (0.2*1.5,yscale * yrescaleFactor/2)
        else:
            Inputs[f"{prefix}_isqrt_yrescale_{i}"] = (1.5,yrescaleFactor/2)
        
        yscale = yOutScale / (Primes[ySqLevel-1] * Primes[ySqLevel-2])
        if yLevel == 6:
            yLevel = bsLevel
        else:
            yLevel = ySqLevel - 2

    Inputs[f"{prefix}_isqrt_y_bootstrap_sf"] = (1, BootstrapScaleDivIsqrt)
    yscale = yscale * BootstrapScaleDivIsqrt
    OutputScales[f"isqrt"] = finalOutScale 


    return (Inputs,finalOutScale,OutputScales)

def get_sign_io(g3_input_scale,level,BootstrapScaleDiv,prefix,bootstrapAfter=True):
    Inputs = {}
    OutScale = {}
    g3_final_out_scale = Primes[level-4]
    (g3_inputs,g3_output_scale,g3_outscales) = get_g3_io(g3_input_scale,g3_final_out_scale,level,prefix)
    Inputs.update(g3_inputs)
    OutScale.update(g3_outscales)
    g3_final_out_scale = Primes[level-7]
    (g3_inputs,g3_output_scale,g3_outscales) = get_g3_io(g3_output_scale,g3_final_out_scale,level-3,prefix)
    Inputs.update(g3_inputs)
    OutScale.update(g3_outscales)
    if bootstrapAfter:
        f3_final_out_scale = Primes[level-10]
    else:
        f3_final_out_scale = Primes[level-10]
    (f3_inputs,f3_output_scale,f3_outscales) = get_f3_io(g3_output_scale,f3_final_out_scale,level-6,prefix)
    f3_output_scale = f3_output_scale*(1.1)
    Inputs.update(f3_inputs)
    OutScale.update(f3_outscales)
    if bootstrapAfter:
        f3_final_out_scale = Primes[15]*Primes[14]/(BootstrapScaleDiv)
        (f3_inputs,f3_output_scale,f3_outscales) = get_f3_io(f3_output_scale,f3_final_out_scale,level-9,prefix)
        f3_final_out_scale = f3_output_scale*Primes[level-12]
        Inputs.update(f3_inputs)
        OutScale.update(f3_outscales)
    else:
        f3_final_out_scale = Primes[level-13]*Primes[level-14]/(BootstrapScaleDiv)
        (f3_inputs,f3_output_scale,f3_outscales) = get_f3_io(f3_output_scale,f3_final_out_scale,level-9,prefix)
        f3_final_out_scale = f3_output_scale*Primes[level-12]
        Inputs.update(f3_inputs)
        OutScale.update(f3_outscales)
    return(Inputs,f3_final_out_scale,OutScale)

def get_sign_io2(g3_input_scale,level,BootstrapScaleDiv,prefix,bootstrapAfter=True):
    Inputs = {}
    OutScale = {}
    g3_final_out_scale = Primes[level-4]
    (g3_inputs,g3_output_scale,g3_outscales) = get_g3_io(g3_input_scale,g3_final_out_scale,level,prefix)
    Inputs.update(g3_inputs)
    OutScale.update(g3_outscales)

    g3_final_out_scale = Primes[level-7]/ 1.05
    (g3_inputs,g3_output_scale,g3_outscales) = get_g3_io(g3_output_scale,g3_final_out_scale,level-3,prefix)
    Inputs.update(g3_inputs)
    OutScale.update(g3_outscales)

    f3_final_out_scale = Primes[level-10] / 1.05
    (f3_inputs,f3_output_scale,f3_outscales) = get_f3_io(g3_output_scale,f3_final_out_scale,level-6,prefix)
    f3_output_scale = f3_output_scale*Primes[level-9]
    Inputs.update(f3_inputs)
    OutScale.update(f3_outscales)


    OutScale[f"f3_x_{level}"] = f3_output_scale

    Inputs.update(f3_inputs)
    OutScale.update(f3_outscales)

    f3_final_out_scale = Primes[level-15]*Primes[level-16]/BootstrapScaleDiv
    (f3_inputs,f3_output_scale,f3_outscales) = get_f3_io_new(f3_output_scale,f3_final_out_scale,level-8,prefix)
    Inputs.update(f3_inputs)
    OutScale.update(f3_outscales)

    return(Inputs,f3_output_scale,OutScale)

def get_gelu_io(gelu_input_scale,xLevel):

    Inputs = {}
    OutScale = {}
    xScale = gelu_input_scale
    BootstrapScaleDiv = 1<<10
    comp_output_scale = xScale / Primes[xLevel-1]
    comput_output_level = xLevel - 1
    (sign_inputs,sign_output_scale,sign_outscales) = get_sign_io2(comp_output_scale,comput_output_level,BootstrapScaleDiv,"gelu",bootstrapAfter=False)
    Inputs.update(sign_inputs)
    OutScale.update(sign_outscales)

    Inputs["gelu_xRF0"] = (1,10)
    Inputs["gelu_xRF1"] = (1,1000)
    Inputs["gelu_C0"] = (5/100,xScale)
    Inputs["gelu_C1"] = (1.97/100,xScale)
    Inputs["gelu_C2"] = (-3/100,xScale)

    for n in range(12):
        OutScale[f"gelu{n}_comp0"] =  comp_output_scale
        OutScale[f"gelu{n}_comp1"] =  comp_output_scale
        OutScale[f"gelu{n}_comp2"] = comp_output_scale
        OutScale[f"gelu{n}_s0"] = sign_output_scale
        OutScale[f"gelu{n}_s1"] = sign_output_scale
        OutScale[f"gelu{n}_s2"] = sign_output_scale
        OutScale[f"gelu{n}_b1"] = sign_output_scale
        OutScale[f"gelu{n}_b2"] = sign_output_scale
        OutScale[f"gelu{n}_b3"] = sign_output_scale

    Inputs["gelu_one"] = (1,sign_output_scale)

    xLevel = xLevel - 7

    powScale = [None]*7
    powScale[1] = xScale
    powScale[2] = (powScale[1]*powScale[1])/(Primes[xLevel-1]*Primes[xLevel-2])
    powScale[4] = (powScale[2]*powScale[2])/(Primes[xLevel-3]*Primes[xLevel-4])
    powScale[3] = (powScale[1]*powScale[2])/(Primes[xLevel-3]*Primes[xLevel-4])
    powScale[6] = (powScale[2]*powScale[4])/(Primes[xLevel-5]*Primes[xLevel-6])

    # res1 = -0.5054031199708174 - 0.4222658115198386 * x -0.1180761295118195 * x**2 - 0.0110341340306157 * x**3
    # res2 =  + 0.5 * x + 0.3603292692789629 * x**2 - 0.037688200365904 * x**4 + 0.0018067462606141 * x**6

    PCoeffs = [-0.5054031199708174, -0.4222658115198386, -0.1180761295118195, -0.0110341340306157]
    QCoeffs = [0.0085263215410380, 0.5, 0.3603292692789629, None, -0.037688200365904, None , 0.0018067462606141]

    qOutLevel = xLevel - 8
    PQOutScale = Primes[qOutLevel-1]*Primes[qOutLevel-2]*Primes[qOutLevel-3]*Primes[qOutLevel-4]
    PQOutScale = PQOutScale * Primes[qOutLevel+1]*Primes[qOutLevel]
    PQOutScale = PQOutScale / sign_output_scale

    for i in range(1,4):
        Inputs[f"gelu_P{i}"] = (PCoeffs[i]/2,PQOutScale/powScale[i])

    for i in [1,2,4,6]:
        Inputs[f"gelu_Q{i}"] = (QCoeffs[i]/2,PQOutScale/powScale[i])

    PQOutScale = PQOutScale / (Primes[qOutLevel+1]*Primes[qOutLevel])

    Inputs[f"gelu_P0"] = (PCoeffs[0]/2,PQOutScale)
    Inputs[f"gelu_Q0"] = (QCoeffs[0]/2,PQOutScale)

    Inputs[f"gelu_xRF2"] = (1/2,PQOutScale*Primes[xLevel-1]*Primes[xLevel-2]/xScale)

    for n in range(12):
        OutScale[f"gelu{n}_res1"] = PQOutScale
        OutScale[f"gelu{n}_res2"] = PQOutScale
        OutScale[f"gelu{n}_res3"] = PQOutScale

    ResScale = sign_output_scale * PQOutScale
    ResScale = ResScale/Primes[qOutLevel-1]
    qOutLevel = qOutLevel - 1
    for n in range(12):
        OutScale[f"gelu{n}_res"] = ResScale

    return (Inputs,ResScale,OutScale)

def get_layernorm_io(norm_type):
    Inputs = {}
    OutScale = {}
    var_scale = Primes[13]*Primes[12]
    if norm_type not in {"att", "ffn"}:
        raise ValueError(f"Unknown layernorm type: {norm_type}")

    OutScale[f"layernorm_{norm_type}_variance"] = var_scale

    if norm_type == "att":
        layernorm_isqrt_outscale = Primes[7]*Primes[6]
        OutScale[f"layernorm_{norm_type}_mean"] = Primes[5]*Primes[4]
    elif norm_type == "ffn":
        layernorm_isqrt_outscale = Primes[5]*Primes[4]
        OutScale[f"layernorm_{norm_type}_mean"] = Primes[3]*Primes[2]
    (isqrt_inputs,isqrt_output_scale,isqrt_outscales) = get_isqrt_io(var_scale,14,1<<10,1<<15,layernorm_isqrt_outscale,f"layernorm_{norm_type}")
    Inputs.update(isqrt_inputs)
    OutScale.update(isqrt_outscales)

    OutScale[f"layernorm_{norm_type}_isqrt"] = layernorm_isqrt_outscale

    return (Inputs,None,OutScale)

def get_attention_io():
    raw_inputs = {}
    OutScale = {}

    level = 13
    for n in range(6):
        OutScale[f"attention_Q{n}"] = Primes[level-5]*Primes[level-6]*Primes[level-7]
        OutScale[f"attention_K{n}"] = Primes[level-5]*Primes[level-8]*Primes[level-9]
        OutScale[f"attention_V{n}"] = Primes[level-5]*Primes[level-6]*Primes[level-7]
        OutScale[f"attention_scores{n}_bs"] = Primes[15]*Primes[14] / (1<<10)
        OutScale[f"attention_scores{n}"] = Primes[15]*Primes[14]


    ## After Bootstrapping
    level = 16

    BootstrapScaleDiv = 1 << 12
    xScale = Primes[level-1]*Primes[level-2]

    raw_inputs["max_epsilon"] = (-1/(1<<10),xScale)
    g3_input_scale = Primes[level-1]*Primes[level-2]

    (sign_inputs,sign_output_scale,sign_outscales) = get_sign_io(g3_input_scale/Primes[level-1],15,BootstrapScaleDiv,"max",bootstrapAfter=False)
    raw_inputs.update(sign_inputs)
    OutScale.update(sign_outscales)

    max_output_scale = Primes[15]*Primes[14]/(BootstrapScaleDiv)

    raw_inputs["zeroPt5_0"] = ([0.5], max_output_scale * Primes[3] * Primes[2] / xScale)
    raw_inputs["zeroPt5_1"] = ([0.5], max_output_scale * Primes[3] * Primes[2] * Primes[level-1] * Primes[level-2] / (xScale*sign_output_scale))

    raw_inputs["zeroPt5_0_im"] = (np.array([0.5j]), max_output_scale * Primes[3] * Primes[2] / xScale)
    raw_inputs["zeroPt5_1_im"] = (np.array([0.5j]), max_output_scale * Primes[3] * Primes[2] * Primes[level-1] * Primes[level-2] / (xScale*sign_output_scale))

    raw_inputs["array_max_bootstrap_scale_factor"] = (1, BootstrapScaleDiv)

    forty = np.zeros(SLOTS)
    for i in range(0,SLOTS,128):
        forty[i] = 40
    forty[0] = forty[0]*1.1

    forty2 = np.zeros(SLOTS)
    for i in range(0,SLOTS,128):
        forty2[127+i] = 40
    raw_inputs["array_max2_forty"] = (forty2, Primes[15]*Primes[14]*Primes[13]*Primes[12]/(max_output_scale*BootstrapScaleDiv))

    raw_inputs["softmax_40"] = (40, Primes[13]*Primes[12])

    level = 14
    exp_scale = Primes[level-1]*Primes[level-2]
    (exp_inputs,exp_output_scale,exp_outscales) = get_exp_io_bootstrap(exp_scale,14,BootstrapScaleDiv)
    raw_inputs.update(exp_inputs)
    OutScale.update(exp_outscales)

    ## att_mask will be filled in later
    raw_inputs["att_mask"] = ([], Primes[8]*Primes[7])

    one = np.zeros(SLOTS)
    for i in range(0,SLOTS,128):
        one[i] = 1

    softmax_bs_div = 1 << 13
    raw_inputs["softmax_one"] = (one, Primes[13]*Primes[12]/ softmax_bs_div)
    raw_inputs["softmax_one_im"] = (one*1j, Primes[13]*Primes[12]/ softmax_bs_div)
    raw_inputs["softmax_sum_sf"] = (1, softmax_bs_div)

    level = 14
    exp_scale = Primes[level-1]*Primes[level-2]

    softmax_isqrt_out_level = 11
    softmax_isqrt_outscale = math.sqrt(Primes[softmax_isqrt_out_level-1]*Primes[softmax_isqrt_out_level-2]*Primes[softmax_isqrt_out_level-5]*Primes[softmax_isqrt_out_level-6])
    (isqrt_inputs,isqrt_output_scale,isqrt_outscales) = get_isqrt_io_no_exp(exp_scale,14,1<<15,1<<10,softmax_isqrt_outscale,"softmax")
    raw_inputs.update(isqrt_inputs)
    OutScale.update(isqrt_outscales)

    for n in range(6):
        OutScale[f"attention_probs{n}"] = Primes[softmax_isqrt_out_level-4]*Primes[softmax_isqrt_out_level-5]*Primes[softmax_isqrt_out_level-6]

    for n in range(6):
        OutScale[f"attention_context_layer{n}"] = Primes[4]*Primes[3]*Primes[2]

    for n in range(3):
        OutScale[f"attention_context_layer_stacked{n}"] = Primes[4]*Primes[3]*Primes[2]

    raw_inputs["attention_bs_rescale_factor"] = (1, 1 << 10)
    for n in range(3):
        OutScale[f"attention_output{n}"] = Primes[2]*Primes[17]*Primes[16]/(1<<10)
        OutScale[f"attention_residual_output_bs{n}"] = Primes[17]*Primes[16]
        OutScale[f"attention_residual_output{n}"] = Primes[17]*Primes[16]

    return (raw_inputs,OutScale)

def get_block_io():

    raw_inputs = {}
    OutScale = {}
    (attention_ri, attention_os) = get_attention_io()
    raw_inputs.update(attention_ri)
    OutScale.update(attention_os)


    (layernorm_inputs,_,layernorm_outscales) = get_layernorm_io("att")
    raw_inputs.update(layernorm_inputs)
    OutScale.update(layernorm_outscales)

    for n in range(3):
        OutScale[f"layernorm_att_output_bs{n}"] = Primes[4]*Primes[3]*Primes[2]
        OutScale[f"layernorm_att_output{n}"] = Primes[4]*Primes[3]*Primes[2]

    for n in range(12):
        OutScale[f"ffn_up{n}"] = Primes[20]*Primes[19] / (1000)
        OutScale[f"ffn_up_bs{n}"] = Primes[20]*Primes[19] / (1000)


    gelu_input_scale = Primes[20]*Primes[19]
    (gelu_inputs,gelu_output_scale,gelu_outscales) = get_gelu_io(gelu_input_scale,21)
    raw_inputs.update(gelu_inputs)
    OutScale.update(gelu_outscales)

    for n in range(12):
        OutScale[f"ffn_gelu{n}"] = gelu_output_scale


    raw_inputs["ffn_residue_bs_rescale_factor"] = (1, (1 << 10))
    for n in range(3):
        OutScale[f"ffn_down{n}"] = Primes[2]*Primes[17]*Primes[16] / (1<<10)
        OutScale[f"ffn_residue{n}"] = Primes[17]*Primes[16]


    (layernorm_inputs,_,layernorm_outscales) = get_layernorm_io("ffn")
    raw_inputs.update(layernorm_inputs)
    OutScale.update(layernorm_outscales)

    raw_inputs["layernorm_ffn_rescale_factor"] = (1, Primes[10]*(1 << 8))

    for n in range(3):
        OutScale[f"layernorm_ffn_output_bs{n}"] = Primes[12]*Primes[11] / (1<<8)
        OutScale[f"layernorm_ffn_output{n}"] = Primes[12]*Primes[11]*Primes[10]

    return (raw_inputs,OutScale)

def get_tanh_io(gelu_input_scale,level):

    Inputs = {}
    OutScale = {}
    xScale = gelu_input_scale
    BootstrapScaleDiv = 1<<10
    sign_input_scale = xScale
    (sign_inputs,sign_output_scale,sign_outscales) = get_sign_io2(xScale/Primes[level-1],level-1,BootstrapScaleDiv,"tanh",bootstrapAfter=False)
    Inputs.update(sign_inputs)
    OutScale.update(sign_outscales)

    Inputs["tanh_xRF0"] = (1,10)
    Inputs["tanh_xRF1"] = (1,1000)
    Inputs["tanh_C0"] = (2.855/100,xScale)
    Inputs["tanh_C1"] = (-2.855/100,xScale)

    OutScale[f"tanh_comp0"] = Primes[level-1]
    OutScale[f"tanh_comp1"] = Primes[level-1]
    OutScale[f"tanh_comp2"] = Primes[level-1]
    OutScale[f"tanh_s0"] = sign_output_scale
    OutScale[f"tanh_s1"] = sign_output_scale
    OutScale[f"tanh_s2"] = sign_output_scale
    OutScale[f"tanh_b1"] = sign_output_scale
    OutScale[f"tanh_b2"] = sign_output_scale
    OutScale[f"tanh_b3"] = sign_output_scale

    Inputs["tanh_one"] = (1,sign_output_scale)

    level = 6

    ResScale = Primes[level-1]*Primes[level-2]
    Inputs["tanh_Prescale"] = (1/2.855 * 1/2,ResScale * Primes[level] * Primes[level+2]*Primes[level+1]/ (sign_output_scale * xScale))
    Inputs["tanh_res0"] = (1/2,ResScale * Primes[level] / sign_output_scale)
    Inputs["tanh_res2"] = (1/2,ResScale * Primes[level] / sign_output_scale)

    OutScale[f"tanh_res"] = ResScale
    return (Inputs,ResScale,OutScale)

def get_pool_classify_io():

    raw_inputs = {}
    OutScale = {}

    OutScale["pool_linear"] = Primes[20]*Primes[19] / (1000)

    tanh_input_scale = Primes[20]*Primes[19]
    (tanh_inputs,tanh_output_scale,tanh_outscales) = get_tanh_io(tanh_input_scale,21)
    raw_inputs.update(tanh_inputs)
    OutScale.update(tanh_outscales)

    OutScale["prediction"] = Primes[4]*Primes[3]


    return (raw_inputs,OutScale)



def main(num_gpus, use_cudagraphs, num_samples, data_dir):
    secretKey = generate_secret_key(SLOTS,32768)
    ephemeralKey = generate_secret_key(SLOTS,32)
    raw_inputs = {}
    OutScale = {}

    (block_ri,block_os) = get_block_io()
    raw_inputs.update(block_ri)
    OutScale.update(block_os)

    (pool_ri,pool_os) = get_pool_classify_io()
    raw_inputs.update(pool_ri)
    OutScale.update(pool_os)

    LVL = 51
    configureBootstrapLVL(LVL+1)
    bootstrap_inputs = {}
    bootstrap_inputs.update(get_bootstrap_runner_inputs("b35",14))
    bootstrap_inputs.update(get_bootstrap_runner_inputs("b33",16))
    raw_inputs.update(bootstrap_inputs)

    run_bert(raw_inputs,OutScale,secretKey,ephemeralKey,num_gpus,use_cudagraphs,num_samples,data_dir)

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

def print_outputs(outputs, layer, data_sample, prefix=""):

    referenceOutputs = {}
    reference_path = Path(f'reference_outputs_{data_sample}/block{layer}.pkl')
    try:
        with reference_path.open('rb') as f:
            referenceOutputs.update(pickle.load(f))
    except FileNotFoundError:
        print(f"Reference outputs file not found: {reference_path}")


    NUM_OUTPUTS = 128
    ERROR_THRESHOLD = 1/(1 << 10)
    for key in outputs.keys():
        output = np.array(outputs[key])
        print("============================")
        if key in referenceOutputs.keys():
            refr = referenceOutputs[key]
            for i in range(SLOTS):
                err = abs(output[i].real-refr[i])
                # err = err / refr[i]
                if err > ERROR_THRESHOLD:
                    print(f"ERROR: @ {prefix}{key}[{i}]",refr[i],output[i],err)
            print("============================")
            for i in range(NUM_OUTPUTS):
                err = abs(output[i].real-refr[i])
                print(f"{prefix}{key}[{i}]",refr[i],output[i],err)
        else:
            for i in range(NUM_OUTPUTS):
                print(f"{prefix}{key}[{i}]",output[i])
        print("============================")

def get_prediction_scores(outputs):
    if "prediction" not in outputs:
        return None

    res = np.asarray(outputs["prediction"])
    if res.size <= 1024:
        raise ValueError("Prediction output must contain slots 0 and 1024")
    return np.array([res[0].real, res[1024].real])

class CeriumFunctionSetup:
    def __init__(self,context,encryptor_context,base_name,function_name,raw_inputs,generate_evalkeys=False,generate_plaintexts=False,use_uvm=False):
        self.context = context
        self.encryptor_context = encryptor_context
        self.base_name = base_name
        self.function_name = function_name 
        self.raw_inputs = raw_inputs
        self.generate_evalkeys = generate_evalkeys
        self.generate_plaintexts = generate_plaintexts

def generate_inputs(setup: CeriumFunctionSetup ):
    base_name = setup.base_name
    function_name = setup.function_name
    generate_evalkeys = setup.generate_evalkeys
    generate_plaintexts = setup.generate_plaintexts
    io_generator = cerium_runtime.IOGenerator(setup.context)
    if generate_evalkeys:
        io_generator.generate_and_serialize_evalkeys(f"{base_name}/{function_name}/evalkeys",f"{base_name}/{function_name}/program_inputs",setup.encryptor_context)
    if generate_plaintexts:
        io_generator.generate_and_serialize_plaintexts(f"{base_name}/{function_name}/plaintexts",f"{base_name}/{function_name}/program_inputs",setup.raw_inputs)
    print(f"Created Inputs: {setup.function_name}")


def create_inputs(cerium_function_setup):
    for setup in cerium_function_setup:
        generate_inputs(setup)

def run_bert(raw_inputs_main, OutScale_main, secretKey, ephemeralKey,
    gpus, use_cudagraphs, num_samples, data_dir):

    num_blocks = 12
    seed = [0,1,2,3,4,5,6,7]
    context = cerium_runtime.Context(SLOTS,Primes)
    encryptor_context = cerium_runtime.CKKSEncryptorContext(context,secretKey,ephemeralKey,seed)
    encryptor = cerium_runtime.CKKSEncryptor(encryptor_context,seed)

    base_name = "bert_outputs/compiled"
    plaintexts_base_name = "bert_plaintexts"
    ciphertext_inputs_base_name = "bert_ciphertexts"

    if gpus == 1:
        main_name = f"main_1gpu_{num_blocks}"
        softmax_name = "softmax_128x128_vec"
    elif gpus == 2:
        main_name = f"main_2gpu_{num_blocks}"
        softmax_name = "softmax_128x128_vec"
    elif gpus == 4:
        main_name = f"main_4gpu_{num_blocks}"
        softmax_name = "softmax_128x128_vec"
    elif gpus == 8:
        main_name = f"main_8gpu_{num_blocks}"
        softmax_name = "softmax_128x128_vec2"

    config = {
        "pool_classify_layer" : {
            "use_cudagraph" : "True" if use_cudagraphs else "False"
        },
        f"{main_name}" : {
            "use_cudagraph" : "False"
        },
        "default": {
            "use_cudagraph" : "True" if use_cudagraphs else "False"
        },
        f"attention_qkv_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/attention_qkv",
            "remapable_keys" : str({f"block{l}" : f"block{l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : "False",
        },
        f"attention_o_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/attention_o",
            "remapable_keys" : str({f"block{l}" : f"block{l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : "False",
        },
        f"layernorm_att_1gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/layernorm_att",
            "remapable_keys" : str({f"block{l}" : f"block{l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : "False",
        },
        f"ffn_up_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/ffn_up",
            "remapable_keys" : str({f"block{l}" : f"block{l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : "False",
        },
        f"ffn_down_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/ffn_down",
            "remapable_keys" : str({f"block{l}" : f"block{l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : "False",
        },
        f"layernorm_ffn_1gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/layernorm_ffn",
            "remapable_keys" : str({f"block{l}" : f"block{l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : "False",
        },
     }

    raw_inputs = {}
    OutScale = {}

    raw_inputs.update(raw_inputs_main)
    OutScale.update(OutScale_main)

    task_name = "rte"
    with (Path('packed_params') / f'params_{task_name}_block0.pkl').open('rb') as f:
        raw_inputs.update(pickle.load(f))

    ## Update the pool classify layer inputs
    with (Path('packed_params') / f'params_{task_name}_block{12}.pkl').open('rb') as f:
        raw_inputs.update(pickle.load(f))

    raw_inputs_wrapper = cerium_runtime.RawInputsWrapper(raw_inputs)
    raw_inputs.clear()

    cerium_function_setups = []

    def create_cerium_function_setup(base_name,function_name,generate_evalkeys,generate_plaintexts,use_uvm=False):
        return CeriumFunctionSetup(context,encryptor_context,base_name,function_name,raw_inputs_wrapper,generate_evalkeys,generate_plaintexts,use_uvm)

    GENERATE_EVALKEYS = True
    GENERATE_PLAINTEXTS = True
    GENERATE_SAMPLES = True

    GENERATE_EVALKEYS2 = True
    GENERATE_PLAINTEXTS2 = True

    GENERATE_REMAPABLES = True

    cerium_function_setups.append(create_cerium_function_setup(base_name,f"attention_qkv_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"attention_o_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"layernorm_att_1gpu",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"bootstrap_32K_35lvl_t1_vec_1",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"bootstrap_32K_33lvl_t1",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))

    cerium_function_setups.append(create_cerium_function_setup(base_name,f"ffn_up_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"ffn_down_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"layernorm_ffn_1gpu",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))

    if gpus == 8:
        cerium_function_setups.append(create_cerium_function_setup(base_name,f"bootstrap_32K_35lvl_t1_vec_2",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))

    cerium_function_setups.append(create_cerium_function_setup(base_name,"matmul_ct128x128_ct128x128",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,"matmul_ct128x128_ct128x128_transpose",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,softmax_name,GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))

    cerium_function_setups.append(create_cerium_function_setup(base_name,f"gelu_vec",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))

    cerium_function_setups.append(create_cerium_function_setup(base_name,"pool_classify_layer",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))

    cerium_function_setups.append(create_cerium_function_setup(base_name,main_name,GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))

    create_inputs(cerium_function_setups)

    io_generator = cerium_runtime.IOGenerator(context)

    if GENERATE_REMAPABLES == True:
        def generate_remapables(function_name,function_name_dir,remap_key,inputs):
            Path(plaintexts_base_name, function_name_dir).mkdir(parents=True, exist_ok=True)
            io_generator.generate_and_serialize_remapables(f"{plaintexts_base_name}/{function_name_dir}/{remap_key}",f"{base_name}/{function_name}/remapable_inputs",inputs)

        for l in range(num_blocks) :
           with (Path('packed_params') / f'params_{task_name}_block{l}.pkl').open('rb') as f:
               layer_inputs = pickle.load(f)
           params_inputs_wrapper = cerium_runtime.RawInputsWrapper(layer_inputs)
           for fname in [f"attention_qkv", f"attention_o", f"ffn_up", f"ffn_down"]:
            generate_remapables(f"{fname}_{gpus}gpu",fname,f"block{l}",params_inputs_wrapper)
           for fname in [f"layernorm_att", f"layernorm_ffn"]:
            generate_remapables(f"{fname}_1gpu",fname,f"block{l}",params_inputs_wrapper)

    program = cerium_runtime.FusedKernelProgram(context,base_name,f"{main_name}")

    cerium_functions_map = program.make_program(gpus,config,raw_inputs_wrapper)
    main_cerium_function = cerium_functions_map[f"{main_name}"]
    softmax_cerium_function = cerium_functions_map[f"{softmax_name}"]

    data_dir = Path(data_dir).expanduser().resolve()
    labels_path = data_dir / 'inputs' / 'labels.npy'
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Could not find labels at {labels_path}. "
            "--data-dir must contain an inputs/ directory."
        )

    labels_file = np.load(labels_path)
    if num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero")
    if num_samples > len(labels_file):
        raise ValueError(
            f"Requested {num_samples} samples, but {data_dir} contains only "
            f"{len(labels_file)} labels."
        )
    samples = list(range(num_samples))
    Preds = []
    Confs = []
    pbar = tqdm.tqdm(range(len(samples)))
    results = {}
    total = 0
    correct = 0
    
    Labels = [labels_file[s] for s in samples]

    for sample in samples:

        if GENERATE_SAMPLES:

            sample_inputs = {}
            level = 13
            H1 = np.load(data_dir / 'inputs' / f'inputs_{sample}_data.npy')
            H1_packed = row_pack_128x768(H1)
            for i in range(3):
                sample_inputs[f"layernorm_out_output_prev{i}"] = (H1_packed[i],Primes[level-1]*Primes[level-2]*Primes[level-3])

            input_mask = (1 - np.load(data_dir / 'inputs' / f'inputs_{sample}_mask.npy')) * -1000
            att_mask = generate_softmax_att_mask(input_mask)
            sample_inputs["att_mask"] = (att_mask, Primes[8]*Primes[7])

            sample_inputs_wrapper = cerium_runtime.RawInputsWrapper(sample_inputs)

            Path(ciphertext_inputs_base_name).mkdir(parents=True, exist_ok=True)
            io_generator.generate_and_serialize_ciphertexts(f"{ciphertext_inputs_base_name}/enc_data_sample{sample}",f"{base_name}/{main_name}/program_inputs",sample_inputs_wrapper,encryptor_context)
            io_generator.generate_and_serialize_ciphertexts(f"{ciphertext_inputs_base_name}/enc_mask_sample{sample}",f"{base_name}/{softmax_name}/program_inputs",sample_inputs_wrapper,encryptor_context)

            del sample_inputs_wrapper


        enc_text = io_generator.deserialize_ciphertexts(context,f"{ciphertext_inputs_base_name}/enc_data_sample{sample}")
        enc_mask = io_generator.deserialize_ciphertexts(context,f"{ciphertext_inputs_base_name}/enc_mask_sample{sample}")


        start_time = time.time()
        main_cerium_function.copy_ciphertext_inputs_to_gpu(f"{base_name}/{main_name}/inputs",gpus,enc_text)
        softmax_cerium_function.copy_ciphertext_inputs_to_gpu(f"{base_name}/{softmax_name}/inputs",gpus,enc_mask)
        program.run()
        encrypted_outputs = main_cerium_function.get_program_outputs()
        end_time = time.time()

        outputs = {}
        for k,v in encrypted_outputs.items():
            outputs[k] = encryptor.decrypt_and_decode(v,OutScale[k])
        
        execution_time = end_time - start_time
        print(f"Bert Execution time: {execution_time} seconds")

        print("Data Sample: ",sample)
        prediction_scores = get_prediction_scores(outputs)
        label = Labels[total]
        if prediction_scores is not None:
            argmax = int(np.argmax(prediction_scores))
            print("Prediction : ",argmax)
            print("Actual     : ",label)
            correct += (argmax == label)
            accuracy = correct / (total + 1)
            print("Running Accuracy: ",accuracy)

            results["accuracy"] = accuracy
            Preds.append(argmax)


        pbar.update(1)
        total += 1 

        pbar.set_postfix(results)

    print(results)
    for i in range(len(Preds)):
        print(f"{i}: {Preds[i]} {Labels[i]} {(Preds[i]==Labels[i])}")
    print(f"Final Accuracy: {correct}/{total} = {correct / total:.6f}")

SLOTS = 32768
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int,default=1,choices=[1,2,4,8], help="Number of GPUs to use")
    parser.add_argument('--use_cudagraphs', type=str_to_bool, default=True, help="Whether to use CUDA graphs for execution")
    parser.add_argument(
        '--num-samples', '--num_samples',
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)",
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('.'),
        help="Directory containing the inputs/ dataset directory (default: current directory)",
    )

    args = parser.parse_args()


    main(args.gpus, args.use_cudagraphs, args.num_samples, args.data_dir)
