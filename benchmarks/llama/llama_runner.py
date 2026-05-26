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
from primes import *
import pickle
import cerium.runtime as cerium_runtime
import math
from bootstrap import *
import subprocess
import torch


import time

random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=64)

# warnings.filterwarnings("ignore")

def generate_secret_key(Slots,HammingWeight=32):
    secretKey = [0]*(2*Slots)
    count = 0
    while count < HammingWeight:
        pos = random.randint(0,2*Slots-1)
        val = random.randint(0,1)
        if secretKey[pos] != 0:
            continue
        if val == 0:
            secretKey[pos] = -1
        elif val == 1:
            secretKey[pos] = 1
        else:
            raise Exception("")
        count += 1
    return secretKey

def rotate(vec,steps):
    return np.concatenate((vec[steps:],vec[:steps]))

def get_g3_inputs(inputScale,finalOutScale,level,prefix):
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

def get_g3_inputs_new(inputScale,finalOutScale,level,prefix):
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
        l += 2
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

def get_f3_inputs(inputScale,finalOutScale,level,prefix):
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
    OutputScales[f"{prefix}_x_{level}"] = OutputScale
    return (Inputs,OutputScale,OutputScales)

def get_f3_inputs_new(inputScale,finalOutScale,level,prefix):
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
        l += 2

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

def get_exp_inputs_bootstrap(inputScale,level,BootstrapScaleDiv):

    Inputs = {}
    OutputScales = {}

    ITERS = 7

    inputLevel = level - 2

    BootstrapLevel = 16
    outScale = Primes[BootstrapLevel - 5] * Primes[BootstrapLevel - 6]

    BootScaleDiv = 1 << 14

    scale1 = np.log(outScale)

    # Calculate the scale before the bootstrap
    level1 = BootstrapLevel
    itersPostBootstrap = 2
    denom1 = 0
    for i in range(itersPostBootstrap):
        denom1 = denom1 * 2
        denom1 += np.log((Primes[level1 - 1]*Primes[level1 - 2]))
        level1 = level1 - 2
        assert level1 >= 2 , "Level Must be Greater than 2"


    scale2 = np.exp((scale1 + denom1)/(2**itersPostBootstrap))
    scale2 = scale2/BootScaleDiv
    scale2 = np.log(scale2)

    level2 = inputLevel
    denom2 = 0
    for i in range(ITERS - itersPostBootstrap):
        denom2 = denom2 * 2
        denom2 += np.log((Primes[level2 - 1]*Primes[level2 - 2]))
        level2 = level2 - 2
        assert level2 >= 2 , "Level Must be Greater than 2"

    inpScale = np.exp((scale2 + denom2)/(2**(ITERS - itersPostBootstrap)))

    expOneBy2NScale = inpScale*Primes[level-1]*Primes[level-2]/inputScale

    Inputs["exp_oneBy2N"] = (1/(2**ITERS),expOneBy2NScale)
    Inputs["exp_one"] = (1,inpScale)

    Inputs["exp_oneBy2N_re"] = ([1/(2**ITERS)],expOneBy2NScale)
    Inputs["exp_one_re"] = ([1],inpScale)

    im_root = np.exp(0.5 * np.pi * 1j / (2**(ITERS - itersPostBootstrap)))

    Inputs["exp_oneBy2N_im"] = ([im_root/(2**ITERS)],expOneBy2NScale)
    Inputs["exp_one_im"] = ([im_root],inpScale)

    Inputs["exp_bootstrap_sf"] = (1,BootScaleDiv)

    OutputScales["one_plus_x"] = inpScale
    OutputScales["exp_x_5"] = outScale
    OutputScales["exp_x_5_bs"] = outScale
    
    OutputScales["exp_x"] = outScale

    return (Inputs,outScale,OutputScales)

def get_isqrt_inputs_no_exp(inputScale,level,BootstrapScaleDiv,BootstrapScaleDivIsqrt,finalOutScale,prefix):

    Inputs = {}
    OutputScales = {}
    bootstrapOutLevel = 12
 
    oprefix = "attention_0"
    yscale = Primes[bootstrapOutLevel-3]*Primes[bootstrapOutLevel-4]
    OutputScales[f"{oprefix}_isqrt_exp"] = yscale 
    Inputs[f"{prefix}_isqrt_zeroPt2"] = (0.2,yscale)
    OutputScales[f"{oprefix}_isqrt_y"] = yscale 

    xScale = inputScale
    yLevel = bootstrapOutLevel - 2
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
        OutputScales[f"{oprefix}_isqrt_y_{i}"] = yscale 
        OutputScales[f"{oprefix}_isqrt_y_bs{i}"] = yscale 
        OutputScales[f"xy3_{i}"] = yscale 

    Inputs[f"{prefix}_isqrt_y_bootstrap_sf"] = (1, BootstrapScaleDivIsqrt)
    yscale = yscale * BootstrapScaleDivIsqrt
    OutputScales[f"isqrt"] = finalOutScale 


    return (Inputs,finalOutScale,OutputScales)


def get_isqrt_inputs_rmsnorm(inputScale,level,BootstrapScaleDiv,BootstrapScaleDivIsqrt,finalOutScale,prefix):

    Inputs = {}
    OutputScales = {}
    expOutLevel = 10
    expOutScale = Primes[expOutLevel-1]*Primes[expOutLevel-2]

    ITERS = 7
    expBootstrapLevel = 16

    BootScaleDiv = 1 << 14

    scale1 = np.log(expOutScale)

    # Calculate the scale before the bootstrap
    level1 = expBootstrapLevel
    itersPostBootstrap = 3
    denom1 = 0
    for i in range(itersPostBootstrap):
        denom1 = denom1 * 2
        denom1 += np.log((Primes[level1 - 1]*Primes[level1 - 2]))
        level1 = level1 - 2
        assert level1 >= 2 , "Level Must be Greater than 2"


    # Calculate the scale after the bootstrap
    scale2 = np.exp((scale1 + denom1)/(2**itersPostBootstrap))
    scale2 = scale2/BootScaleDiv
    scale2 = np.log(scale2)

    level2 = level-2
    denom2 = 0
    for i in range(ITERS - itersPostBootstrap):
        denom2 = denom2 * 2
        denom2 += np.log((Primes[level2 - 1]*Primes[level2 - 2]))
        level2 = level2 - 2
        assert level2 >= 2 , "Level Must be Greater than 2"

    inpScale = np.exp((scale2 + denom2)/(2**(ITERS - itersPostBootstrap)))

    expOneBy2NScale = inpScale*Primes[level-1]*Primes[level-2]/inputScale


    Inputs[f"{prefix}_isqrt_minus_zero_pt_4"] = (-0.4,inputScale)
    Inputs[f"{prefix}_isqrt_oneBy2N"] = (1/(2*(2**ITERS)),expOneBy2NScale)
    Inputs[f"{prefix}_isqrt_one"] = (1,inpScale)
    Inputs[f"{prefix}_isqrt_exp_bootstrap_sf"] = (1,BootScaleDiv)

    oprefix = "rmsnorm_att"
    OutputScales[f"{oprefix}_isqrt_minus"] = inputScale
    OutputScales[f"{oprefix}_isqrt_one_plus_x"] = inpScale
    

    #####

 
    oprefix = "rmsnorm_att"
    yscale = expOutScale
    OutputScales[f"{oprefix}_isqrt_exp"] = yscale 
    Inputs[f"{prefix}_isqrt_zeroPt2"] = (0.2,yscale)
    OutputScales[f"{oprefix}_isqrt_y"] = yscale 

    xScale = inputScale
    yLevel = expOutLevel
    bsLevel = 10

    Iters = 10

    for i in range(Iters):
        ysqScale = yscale * yscale / (Primes[yLevel - 1]*Primes[yLevel - 2])
        ySqLevel = yLevel - 2

        if i == (Iters -1):
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
        OutputScales[f"{oprefix}_isqrt_y_{i}"] = yscale 
        OutputScales[f"{oprefix}_isqrt_y_bs{i}"] = yscale 
        OutputScales[f"xy3_{i}"] = yscale 

    Inputs[f"{prefix}_isqrt_y_bootstrap_sf"] = (1, BootstrapScaleDivIsqrt)
    yscale = yscale * BootstrapScaleDivIsqrt
    OutputScales[f"isqrt"] = finalOutScale 

    return (Inputs,finalOutScale,OutputScales)


def get_sign_inputs(g3_input_scale,level,BootstrapScaleDiv,prefix,bootstrapAfter=True):
    Inputs = {}
    OutScale = {}
    g3_final_out_scale = Primes[level-4]
    (g3_inputs,g3_output_scale,g3_outscales) = get_g3_inputs(g3_input_scale,g3_final_out_scale,level,prefix)
    Inputs.update(g3_inputs)
    OutScale.update(g3_outscales)
    g3_final_out_scale = Primes[level-7]
    (g3_inputs,g3_output_scale,g3_outscales) = get_g3_inputs(g3_output_scale,g3_final_out_scale,level-3,prefix)
    Inputs.update(g3_inputs)
    OutScale.update(g3_outscales)
    if bootstrapAfter:
        f3_final_out_scale = Primes[level-10]
    else:
        f3_final_out_scale = Primes[level-10]
    (f3_inputs,f3_output_scale,f3_outscales) = get_f3_inputs(g3_output_scale,f3_final_out_scale,level-6,prefix)
    f3_output_scale = f3_output_scale*(1.1)
    Inputs.update(f3_inputs)
    OutScale.update(f3_outscales)
    if bootstrapAfter:
        f3_final_out_scale = Primes[15]*Primes[14]/(BootstrapScaleDiv)
        (f3_inputs,f3_output_scale,f3_outscales) = get_f3_inputs(f3_output_scale,f3_final_out_scale,level-9,prefix)
        f3_final_out_scale = f3_output_scale*Primes[level-12]
        Inputs.update(f3_inputs)
        OutScale.update(f3_outscales)
    else:
        f3_final_out_scale = Primes[level-13]*Primes[level-14]/(BootstrapScaleDiv)
        (f3_inputs,f3_output_scale,f3_outscales) = get_f3_inputs(f3_output_scale,f3_final_out_scale,level-9,prefix)
        f3_final_out_scale = f3_output_scale*Primes[level-12]
        Inputs.update(f3_inputs)
        OutScale.update(f3_outscales)
    return(Inputs,f3_final_out_scale,OutScale)

def get_sign_inputs_new2(g3_input_scale,level,BootstrapScaleDiv,prefix,bootstrapAfter=True):
    Inputs = {}
    OutScale = {}
    g3_final_out_scale = Primes[level-4]
    (g3_inputs,g3_output_scale,g3_outscales) = get_g3_inputs(g3_input_scale,g3_final_out_scale,level,prefix)
    Inputs.update(g3_inputs)
    OutScale.update(g3_outscales)

    g3_final_out_scale = Primes[level-7]/ 1.05
    (g3_inputs,g3_output_scale,g3_outscales) = get_g3_inputs(g3_output_scale,g3_final_out_scale,level-3,prefix)
    Inputs.update(g3_inputs)
    OutScale.update(g3_outscales)

    f3_final_out_scale = Primes[level-10] / 1.05
    (f3_inputs,f3_output_scale,f3_outscales) = get_f3_inputs(g3_output_scale,f3_final_out_scale,level-6,prefix)
    f3_output_scale = f3_output_scale*Primes[level-9]
    Inputs.update(f3_inputs)
    OutScale.update(f3_outscales)


    OutScale[f"f3_x_{level}"] = f3_output_scale

    Inputs.update(f3_inputs)
    OutScale.update(f3_outscales)

    f3_final_out_scale = Primes[level-15]*Primes[level-16]/BootstrapScaleDiv
    (f3_inputs,f3_output_scale,f3_outscales) = get_f3_inputs_new(f3_output_scale,f3_final_out_scale,level-8,prefix)
    Inputs.update(f3_inputs)
    OutScale.update(f3_outscales)

    return(Inputs,f3_output_scale,OutScale)

def get_silu_inputs(v_input_scale,v_level,w_input_scale,w_level):

    Inputs = {}
    OutScale = {}
    vScale = v_input_scale
    wScale = w_input_scale
    BootstrapScaleDiv = 1<<4
    sign_input_scale = vScale
    (sign_inputs,sign_output_scale,sign_outscales) = get_sign_inputs_new2(sign_input_scale/Primes[v_level-1],v_level-1,BootstrapScaleDiv,"silu",bootstrapAfter=False)
    Inputs.update(sign_inputs)
    OutScale.update(sign_outscales)

    Inputs["silu_vRF0"] = (1,20)
    Inputs["silu_vRF1"] = (1,1000)
    Inputs["silu_C0"] = (8/50,vScale)
    Inputs["silu_C1"] = (4/50,vScale)
    Inputs["silu_C2"] = (-4/50,vScale)

    for n in range(56):
        OutScale[f"silu{n}_comp0"] = Primes[16]
        OutScale[f"silu{n}_comp1"] = Primes[16]
        OutScale[f"silu{n}_comp2"] = Primes[16]
        OutScale[f"silu{n}_s0"] = sign_output_scale
        OutScale[f"silu{n}_s1"] = sign_output_scale
        OutScale[f"silu{n}_s2"] = sign_output_scale
        OutScale[f"silu{n}_b1"] = sign_output_scale
        OutScale[f"silu{n}_b2"] = sign_output_scale
        OutScale[f"silu{n}_b3"] = sign_output_scale

    Inputs["silu_one"] = (1,sign_output_scale)

    v_level = v_level - 5

    powScale = [None]*7
    powScale[1] = vScale
    powScale[2] = (powScale[1]*powScale[1])/(Primes[v_level-1]*Primes[v_level-2])
    powScale[4] = (powScale[2]*powScale[2])/(Primes[v_level-3]*Primes[v_level-4])
    powScale[3] = (powScale[1]*powScale[2])/(Primes[v_level-3]*Primes[v_level-4])
    powScale[6] = (powScale[2]*powScale[4])/(Primes[v_level-5]*Primes[v_level-6])

    # res1 = -0.3067541139982155 -0.0819767021525476 * x -0.0055465625580307 * x**2
    # res2 = 0.0085064025895951 + 0.5 * x + 0.2281430841728270 * x ** 2 -0.011113046708173 * x**4  + 0.0002743776353465 * x**6
    # res3 = x


    PCoeffs = [-0.3067541139982155, -0.0819767021525476, -0.0055465625580307]
    QCoeffs = [0.0085064025895951, 0.5, 0.2281430841728270, None, -0.011113046708173, None , 0.0002743776353465]

    qOutLevel = v_level - 10
    PQOutScale = Primes[qOutLevel-1]*Primes[qOutLevel-2]*Primes[qOutLevel-3]
    PQOutScale = PQOutScale * Primes[w_level-1]*Primes[w_level-2]
    PQOutScale = PQOutScale / wScale
    PQOutScale = PQOutScale * Primes[qOutLevel+3]*Primes[qOutLevel+2]
    PQOutScale = PQOutScale / sign_output_scale

    for i in range(1,3):
        Inputs[f"silu_P{i}"] = (PCoeffs[i]/2,PQOutScale/powScale[i])

    for i in [1,2,4,6]:
        Inputs[f"silu_Q{i}"] = (QCoeffs[i]/2,PQOutScale/powScale[i])

    PQOutScale = PQOutScale / (Primes[qOutLevel+3]*Primes[qOutLevel+2])

    Inputs[f"silu_P0"] = (PCoeffs[0]/2,PQOutScale)
    Inputs[f"silu_Q0"] = (QCoeffs[0]/2,PQOutScale)

    Inputs[f"silu_xRF2"] = (1/2,PQOutScale*Primes[v_level-1]*Primes[v_level-2]*Primes[w_level-1]*Primes[w_level-2]/(vScale*wScale))

    for n in range(56):
        OutScale[f"silu{n}_res1"] = PQOutScale
        OutScale[f"silu{n}_res2"] = PQOutScale
        OutScale[f"silu{n}_res3"] = PQOutScale

    PQOutScale = PQOutScale / (Primes[w_level-1]*Primes[w_level-2])

    ResScale = sign_output_scale * PQOutScale * wScale
    for n in range(56):
        OutScale[f"silu{n}"] = ResScale

    return (Inputs,ResScale,OutScale)

def get_matmul_ct128x128_ct128x128_transpose_inputs():

    raw_inputs = {}
    OutScale = {}


    DIV = 80 
    mask = np.zeros((SLOTS,))
    mask[::128] = 1/(DIV*np.sqrt(128))

    bootstrapLevel = 16
    BootstrapScaleDiv = 1 << 3

    raw_inputs[f"mask_re"] = (mask,Primes[bootstrapLevel-1]*Primes[bootstrapLevel-2] / BootstrapScaleDiv)
    raw_inputs[f"mask_im"] = (mask*1j,Primes[bootstrapLevel-1]*Primes[bootstrapLevel-2] / BootstrapScaleDiv)

    raw_inputs["QK_bs_scale_factor"] = (1,BootstrapScaleDiv)

    for i in range(16):
        OutScale[f"attention_QK_bs{i}"] = Primes[15]*Primes[14] / BootstrapScaleDiv
        OutScale[f"attention_QK_bs{i}"] = Primes[15]*Primes[14] / BootstrapScaleDiv
        OutScale[f"attention_QK{i}"] = Primes[15]*Primes[14] 
        OutScale[f"attention_QK{i}"] = Primes[15]*Primes[14]

    return raw_inputs, OutScale


def get_softmax_inputs():
    raw_inputs = {}
    OutScale = {}

    level = 16

    BootstrapScaleDiv = 1 << 12
    xxScale = Primes[level-1]*Primes[level-2]

    raw_inputs["max_epsilon"] = (-1/(1<<10),xxScale)
    g3_input_scale = Primes[level-1]*Primes[level-2]

    OutScale["a"] = g3_input_scale*2
    OutScale["b"] = g3_input_scale*2
    OutScale["ab"] = g3_input_scale
    OutScale["ab_"] = g3_input_scale

    (sign_inputs,sign_output_scale,sign_outscales) = get_sign_inputs(g3_input_scale/Primes[level-1],15,BootstrapScaleDiv,"max",bootstrapAfter=False)
    raw_inputs.update(sign_inputs)
    OutScale.update(sign_outscales)

    max_output_scale = Primes[15]*Primes[14]/(BootstrapScaleDiv)

    raw_inputs["zeroPt5_0"] = ([0.5]*SLOTS, max_output_scale * Primes[3] * Primes[2] / xxScale)
    raw_inputs["zeroPt5_1"] = ([0.5]*SLOTS, max_output_scale * Primes[3] * Primes[2] * Primes[level-1] * Primes[level-2] / (xxScale*sign_output_scale))

    raw_inputs["zeroPt5_0_im"] = (np.array([0.5j]*SLOTS), max_output_scale * Primes[3] * Primes[2] / xxScale)
    raw_inputs["zeroPt5_1_im"] = (np.array([0.5j]*SLOTS), max_output_scale * Primes[3] * Primes[2] * Primes[level-1] * Primes[level-2] / (xxScale*sign_output_scale))

    raw_inputs["array_max_bootstrap_scale_factor"] = (1, BootstrapScaleDiv)

    for i in range(7):
        OutScale[f"array_max_bs{i}"] = max_output_scale 
        for n in range(0,16,2):
            OutScale[f"sign_{i}"] = sign_output_scale 
            OutScale[f"attention_{n}_sign_{i}"] = sign_output_scale 
            OutScale[f"array_max_{i}"] = max_output_scale * BootstrapScaleDiv
            OutScale[f"linear1_{n}_array_max_{i}"] = max_output_scale * BootstrapScaleDiv

    eighty = np.zeros(SLOTS)
    eighty[127::128] = 80
    raw_inputs["array_max2_eighty"] = (eighty, Primes[15]*Primes[14]*Primes[13]*Primes[12]/(max_output_scale*BootstrapScaleDiv))

    for n in range(0,16,2):
        OutScale[f"attention_{n}_max_values0"] = Primes[13]*Primes[12]
        OutScale[f"attention_{n}_max_values1"] = Primes[13]*Primes[12]

    raw_inputs["softmax_80"] = (80, Primes[13]*Primes[12])

    for n in range(0,16,2):
        OutScale[f"attention_{n}_max_values_sub0"] = Primes[13]*Primes[12]
        OutScale[f"attention_{n}_max_values_sub1"] = Primes[13]*Primes[12]

    level = 14
    exp_scale = Primes[level-1]*Primes[level-2]
    (exp_inputs,exp_output_scale,exp_outscales) = get_exp_inputs_bootstrap(exp_scale,14,BootstrapScaleDiv)
    raw_inputs.update(exp_inputs)
    OutScale.update(exp_outscales)

    for n in range(0,16,2):
        OutScale[f"attention_{n}_exp0"] = exp_output_scale 
        OutScale[f"attention_{n}_exp1"] = exp_output_scale 

    for n in range(0,16,2):
        OutScale[f"attention_{n}_exp_att_mask0"] = Primes[9]*Primes[8]
        OutScale[f"attention_{n}_exp_att_mask1"] = Primes[9]*Primes[8]


    ISQRT_DIV = 2
    one = np.zeros(SLOTS)
    for i in range(0,SLOTS,128):
        one[i+127] = 1 / ISQRT_DIV

    softmax_bs_div = 1 << 13
    raw_inputs["softmax_one"] = (one, Primes[13]*Primes[12]/ softmax_bs_div)
    raw_inputs["softmax_one_re"] = (one, Primes[13]*Primes[12]/ softmax_bs_div)
    raw_inputs["softmax_one_im"] = (one*1j, Primes[13]*Primes[12]/ softmax_bs_div)
    raw_inputs["softmax_sum_sf"] = (1, softmax_bs_div)

    for n in range(0,16,2):
        OutScale[f"attention_{n}_softmax_sum_bs0"] = Primes[13]*Primes[12] / softmax_bs_div
        OutScale[f"attention_{n}_softmax_sum_bs1"] = Primes[13]*Primes[12] / softmax_bs_div
        OutScale[f"attention_{n}_softmax_sum0"] = Primes[13]*Primes[12]
        OutScale[f"attention_{n}_softmax_sum1"] = Primes[13]*Primes[12]

    level = 14
    exp_scale = Primes[level-1]*Primes[level-2]
    
    softmax_isqrt_out_level = 12
    softmax_isqrt_outscale = math.sqrt(Primes[softmax_isqrt_out_level-1]*Primes[softmax_isqrt_out_level-2]*Primes[softmax_isqrt_out_level-5]*Primes[softmax_isqrt_out_level-6])
    (isqrt_inputs,isqrt_output_scale,isqrt_outscales) = get_isqrt_inputs_no_exp(exp_scale,14,1<<15,1<<10,softmax_isqrt_outscale,"softmax")
    raw_inputs.update(isqrt_inputs)
    OutScale.update(isqrt_outscales)

    for n in range(0,16,2):
        OutScale[f"attention_{n}_inverse0"] = Primes[softmax_isqrt_out_level-5]*Primes[softmax_isqrt_out_level-6]
        OutScale[f"attention_{n}_inverse1"] = Primes[softmax_isqrt_out_level-5]*Primes[softmax_isqrt_out_level-6]

    for n in range(16):
        OutScale[f"attention_softmax{n}"] = Primes[softmax_isqrt_out_level-4]*Primes[softmax_isqrt_out_level-5]*Primes[softmax_isqrt_out_level-6]*ISQRT_DIV * 1.0001

    return raw_inputs, OutScale

def get_matmul_ct128x128_ct128x128_inputs():
    ISQRT_DIV = 2
    Vlevel = 10
    Inputs = {}
    for gs in range(16):
        for bs in range(8):
            i = bs + 8*gs
            mask = torch.zeros(128,dtype=torch.int32)
            mask[i] = 1

            Inputs[f"mask_128x128_rot_{i*128}"] = (mask.tolist(),Primes[Vlevel-5]*math.sqrt(Primes[Vlevel-6]))

    for gs in range(16):
        for bs in range(8):
            i = bs + 8*gs
            maskL = torch.ones((128),dtype=torch.float64) / (ISQRT_DIV * 1.0001)
            maskR = torch.ones((128),dtype=torch.float64) / (ISQRT_DIV * 1.0001)
            maskL[128 - i : 128] = 0
            maskR[0: 128 - i] = 0
            Inputs[f"maskL_128x128_{i}"] = (maskL.tolist(),math.sqrt(Primes[Vlevel-6])*Primes[Vlevel-7])
            Inputs[f"maskR_128x128_{i}"] = (maskR.tolist(),math.sqrt(Primes[Vlevel-6])*Primes[Vlevel-7])

    return Inputs

def get_rmsnorm_inputs():

    Inputs = {}
    OutScale = {}

    x_level = 16
    x_scale = Primes[x_level - 1]*Primes[x_level -2]


    xBootstrapScaleDiv = 1 << 6
    Inputs["rmsnorm_att_x_rescale_factor"] = (1,xBootstrapScaleDiv)
    Inputs["rmsnorm_ffn_x_rescale_factor"] = (1,xBootstrapScaleDiv)

    xsq_scale = x_scale * x_scale

    OutScale[f"rmsnorm_att_w_sum"] = xsq_scale
    OutScale[f"rmsnorm_ffn_w_sum"] = xsq_scale

    DIV1 = 0.01
    DIV2 = 0.01
    
    oneByN = np.zeros((SLOTS,),dtype=np.float64)
    oneByN[255::256] = 1 / 4096
    
    div_tensor = np.ones((SLOTS,)) * DIV1
    div_tensor[0:256] = DIV2

    oneByN = oneByN / div_tensor

    oneByNscale = Primes[x_level-1] * Primes[x_level-2] * Primes[x_level-3] * Primes[x_level-4] *Primes[x_level-5] * Primes[x_level-6] / xsq_scale

    w_mean_scale = oneByNscale * xsq_scale / (Primes[x_level-1]*Primes[x_level-2]*Primes[x_level-3]*Primes[x_level-4])
    OutScale[f"rmsnorm_att_w_mean"] = w_mean_scale
    OutScale[f"rmsnorm_ffn_w_mean"] = w_mean_scale

    w_mean_level = x_level - 4

    rmsnorm_isqrt_out_level = x_level
    rmsnorm_isqrt_out_scale = Primes[rmsnorm_isqrt_out_level-3]*Primes[rmsnorm_isqrt_out_level-4]
    i,s,o = get_isqrt_inputs_rmsnorm(w_mean_scale,w_mean_level,None,1<<15,rmsnorm_isqrt_out_scale,"rmsnorm")
    Inputs.update(i)
    OutScale.update(o)
    OutScale[f"rmsnorm_att_inv_sqrt"] = rmsnorm_isqrt_out_scale
    OutScale[f"rmsnorm_ffn_inv_sqrt"] = rmsnorm_isqrt_out_scale

    div_isqrt_inv_level = x_level
    div_isqrt_inv_scale = Primes[div_isqrt_inv_level-1]*Primes[div_isqrt_inv_level-2]*Primes[div_isqrt_inv_level-3]*Primes[div_isqrt_inv_level-4]*Primes[div_isqrt_inv_level-5]*Primes[div_isqrt_inv_level-6] / (x_scale * rmsnorm_isqrt_out_scale)

    OutScale[f"rmsnorm_att_inv_sqrt_div"] = rmsnorm_isqrt_out_scale * div_isqrt_inv_scale
    OutScale[f"rmsnorm_ffn_inv_sqrt_div"] = rmsnorm_isqrt_out_scale * div_isqrt_inv_scale

    div_isqrt_inv_level -= 3
    for i in range(16):
        OutScale[f"rmsnorm_att_res{i}"] = Primes[div_isqrt_inv_level-1]*Primes[div_isqrt_inv_level-2]*Primes[div_isqrt_inv_level-3]
        OutScale[f"rmsnorm_ffn_res{i}"] = Primes[div_isqrt_inv_level-1]*Primes[div_isqrt_inv_level-2]*Primes[div_isqrt_inv_level-3]


    return Inputs, OutScale

def get_ffn_inputs():
    Inputs = {}
    OutScale = {}
    wLevel = 12
    vBootstrapOutLevel = 21

    for i in range(56):
        OutScale[f"ffn_V{i}"] = Primes[vBootstrapOutLevel-1]*Primes[vBootstrapOutLevel-2] / (1000)
        OutScale[f"ffn_W{i}"] = Primes[wLevel-5]*Primes[wLevel-6]

    v_input_scale = Primes[20]*Primes[19]
    w_input_scale = Primes[7]*Primes[6]
    (silu_inputs,silu_output_scale,silu_outscales) = get_silu_inputs(v_input_scale,21,w_input_scale,8)
    for i in range(56):
        OutScale[f"ffn_VW{i}"] = silu_output_scale
    Inputs.update(silu_inputs)
    OutScale.update(silu_outscales)

    for i in range(16):
        OutScale[f"ffn_O{i}"] = Primes[15]*Primes[14] / ( 1 << 6) 

    for i in range(16):
        OutScale[f"out{i}"] = Primes[15]*Primes[14] / ( 1 << 6) 

    for b in range(32):
        for i in range(16):
            OutScale[f"block{b}_dbg_out{i}"] = Primes[15]*Primes[14] / (1 << 6) 
    
    return Inputs, OutScale

    
def get_layer_inputs(layer_count):

    raw_inputs = {}
    OutScale = {}

    i,o = get_rmsnorm_inputs()
    raw_inputs.update(i)
    OutScale.update(o)

    i,o = get_matmul_ct128x128_ct128x128_transpose_inputs()
    raw_inputs.update(i)
    OutScale.update(o)

    i,o = get_softmax_inputs()
    raw_inputs.update(i)
    OutScale.update(o)

    raw_inputs.update(get_matmul_ct128x128_ct128x128_inputs())

    i,o = get_ffn_inputs()
    raw_inputs.update(i)
    OutScale.update(o)


    return raw_inputs,OutScale

def main(gpus,data_sample,block_start,cudagraphs,num_blocks):
    secretKey = generate_secret_key(SLOTS,32768)
    ephemeralKey = generate_secret_key(SLOTS,32)

    raw_inputs = {}
    OutScale = {}

    (layer_ri,layer_os) = get_layer_inputs(block_start)
    raw_inputs.update(layer_ri)
    OutScale.update(layer_os)


    configureBootstrapLVL(LVL+1) 
    raw_inputs.update(get_bootstrap_runner_inputs("b35",14))
    raw_inputs.update(get_bootstrap_runner_inputs("b33",19))

    run_llama_blocks(raw_inputs,OutScale,secretKey,ephemeralKey,gpus,cudagraphs,data_sample,block_start,num_blocks)

class CeriumFunctionSetup:
    def __init__(self,context,encryptor_context,base_name,function_name,raw_inputs,generate_evalkeys=False,generate_plaintexts=False,use_uvm=False,layer_num=0):
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
    try:
        with open(f'reference_outputs_{data_sample}/reference_outputs_layer{layer}.pkl', 'rb') as f:
            referenceOutputs.update(pickle.load(f))
    except Exception as e:
        print(f"Error loading reference outputs for layer {layer} in {data_sample}: {e}")
    finally:
        print("")
    print(referenceOutputs.keys())


    NUM_OUTPUTS = 256
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

def run_llama_blocks(raw_inputs_main,OutScale_main,secretKey,ephemeralKey,gpus,cudagraphs,sample,block_start,num_blocks):
    raw_inputs = {}
    OutScale = {}

    raw_inputs.update(raw_inputs_main)
    OutScale.update(OutScale_main)

    with open(f'llama_models/params/params_block{block_start}.pkl', 'rb') as f:
        raw_inputs.update(pickle.load(f))

    referenceInputs = {}
    try:
        with open(f'reference_outputs_{sample}/reference_outputs_layer{block_start}.pkl', 'rb') as f:
            referenceInputs.update(pickle.load(f))
    except Exception as e:
        print(f"Error loading reference inputs for block {block_start} data_sample:{sample}: {e}")
    finally:
        print(f"Loaded reference inputs for block {block_start} data_sample:{sample}")


    referenceOutputs = {}
    try:
        with open(f'reference_outputs_{sample}/reference_outputs_layer{block_start}.pkl', 'rb') as f:
            referenceOutputs.update(pickle.load(f))
    except Exception as e:
        print(f"Error loading reference outputs for block {block_start} data_sample:{sample}: {e}")
    finally:
        print(f"Loaded reference outputs for block {block_start} data_sample:{sample}")


    seed = [0,1,2,3,4,5,6,7]
    context = cerium_runtime.Context(SLOTS,Primes)
    encryptor_context = cerium_runtime.CKKSEncryptorContext(context,secretKey,ephemeralKey,seed)
    encryptor = cerium_runtime.CKKSEncryptor(encryptor_context,seed)

    base_name = "outputs/compiled"
    plaintexts_base_name = "llama_plaintexts"
    ciphertext_inputs_base_name = "llama_ciphertexts"

    for i in range(16):
        OutScale[f"attention_input{i}"] = Primes[11]*Primes[10]#*Primes[9]
        OutScale[f"attention_Q{i}"] = Primes[8]*Primes[7]*Primes[6]
        OutScale[f"attention_K{i}"] = Primes[8]*Primes[5]*Primes[4]
        OutScale[f"attention_V{i}"] = Primes[9]*Primes[8]*Primes[7]

    for n in range(16):
        OutScale[f"attention_attention{n}"] = Primes[5]*Primes[4]*Primes[3]

    for n in range(0,16,4):
        OutScale[f"attention_{n}_diag"] = Primes[5]*np.sqrt(Primes[4])
        for bs in range(8):
            OutScale[f"attention_{n}_diag_bs_{bs}"] = Primes[5]*np.sqrt(Primes[4])

    for n in range(16):
        BootstrapScaleDiv = 1 << 6
        OutScale[f"attention_attention_out{n}"] = Primes[15]*Primes[14] / BootstrapScaleDiv
        OutScale[f"H{n}"] = Primes[15]*Primes[14] / BootstrapScaleDiv

    for b in range(32):
        for i in range(16):
            OutScale[f"block{b}_dbg_H{i}"] = Primes[15]*Primes[14] / BootstrapScaleDiv

    raw_inputs_wrapper = cerium_runtime.RawInputsWrapper(raw_inputs)
    raw_inputs.clear()

    cerium_function_setups = []

    def create_cerium_function_setup(base_name,function_name,generate_evalkeys,generate_plaintexts,use_uvm=False):
        return CeriumFunctionSetup(context,encryptor_context,base_name,function_name,raw_inputs_wrapper,generate_evalkeys,generate_plaintexts,use_uvm)

    GENERATE_EVALKEYS = True
    GENERATE_PLAINTEXTS = True

    GENERATE_EVALKEYS2 = True
    GENERATE_PLAINTEXTS2 = True

    GENERATE_EVALKEYS3 = True
    GENERATE_PLAINTEXTS3 = True

    GENERATE_SAMPLES = True


    softmax_name = "softmax_128x128_vec"

    cerium_function_setups.append(create_cerium_function_setup(base_name,"bootstrap_32K_35lvl_t1_vec",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))

    cerium_function_setups.append(create_cerium_function_setup(base_name,"matmul_ct128x128_ct128x128",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,"matmul_ct128x128_ct128x128_transpose",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"softmax_128x128_vec",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"attention_matmul_qkv_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"attention_matmul_o_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))

    cerium_function_setups.append(create_cerium_function_setup(base_name,f"rmsnorm_att_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"rmsnorm_ffn_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))

    cerium_function_setups.append(create_cerium_function_setup(base_name,"bootstrap_32K_33lvl_t1_vec",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"ffn_matmul_vw_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))
    cerium_function_setups.append(create_cerium_function_setup(base_name,"silu",GENERATE_EVALKEYS,GENERATE_PLAINTEXTS))
    cerium_function_setups.append(create_cerium_function_setup(base_name,f"ffn_matmul_o_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))

    if gpus > 1:
        cerium_function_setups.append(create_cerium_function_setup(base_name,f"attention_score_softmax_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))
        cerium_function_setups.append(create_cerium_function_setup(base_name,f"ffn_silu_{gpus}gpu",GENERATE_EVALKEYS2,GENERATE_PLAINTEXTS2))

    print(f"Running Blocks: [{block_start}: {block_start+num_blocks}]")
    main_name = f"llama_{gpus}gpu_{num_blocks}blocks"

    cerium_function_setups.append(create_cerium_function_setup(base_name,main_name,GENERATE_EVALKEYS3,GENERATE_PLAINTEXTS3))

    create_inputs(cerium_function_setups)

    use_cudagraphs = True if cudagraphs else False
    # config = {"default" : {"use_cudagraph" : f"{use_cudagraphs}"},
    #           f"{main_name}" : {"use_cudagraph" : f"{False}"},
    #           }

    config = {"default" : {"use_cudagraph" : f"{False}"},
              f"{main_name}" : {"use_cudagraph" : f"{False}"},
              f"bootstrap_32K_35lvl_t1_vec" : {"use_cudagraph" : f"{use_cudagraphs}"},
              f"bootstrap_32K_33lvl_t1_vec" : {"use_cudagraph" : f"{use_cudagraphs}"},
              f"matmul_ct128x128_ct128x128_transpose" : {"use_cudagraph" : f"{use_cudagraphs}"},
              f"matmul_ct128x128_ct128x128" : {"use_cudagraph" : f"{use_cudagraphs}"},
            #   f"silu" : {"use_cudagraph" : f"{use_cudagraphs}"},
                f"softmax_128x128_vec" : {"use_cudagraph" : f"{use_cudagraphs}"},
                f"attention_score_softmax_{gpus}gpu" : {"use_cudagraph" : f"{use_cudagraphs}"},
              }


    USE_UVM_REMAPABLES = True
    remapables_config = {
        f"attention_matmul_qkv_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/attention_matmul_qkv",
            "remapable_keys" : str({f"block{l}" : f"block{block_start + l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : f"{USE_UVM_REMAPABLES}",
            "use_cudagraph" : f"{use_cudagraphs}",
        },
        f"attention_matmul_o_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/attention_matmul_o",
            "remapable_keys" : str({f"block{l}" : f"block{block_start + l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : f"{USE_UVM_REMAPABLES}",
            "use_cudagraph" : f"{use_cudagraphs}",
        },
        f"rmsnorm_att_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/rmsnorm_att",
            "remapable_keys" : str({f"block{l}" : f"block{block_start + l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : f"{USE_UVM_REMAPABLES}",
            "use_cudagraph" : f"{use_cudagraphs}",
        },
        f"ffn_matmul_vw_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/ffn_matmul_vw",
            "remapable_keys" : str({f"block{l}" : f"block{block_start + l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : f"{USE_UVM_REMAPABLES}",
            "use_cudagraph" : f"{use_cudagraphs}",
        },
        f"ffn_matmul_o_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/ffn_matmul_o",
            "remapable_keys" : str({f"block{l}" : f"block{block_start + l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : f"{USE_UVM_REMAPABLES}",
            "use_cudagraph" : f"{use_cudagraphs}",
        },
        f"rmsnorm_ffn_{gpus}gpu": {
            "remapable_inputs_base" : f"{plaintexts_base_name}/rmsnorm_ffn",
            "remapable_keys" : str({f"block{l}" : f"block{block_start + l}" for l in range(num_blocks)}),
            "remapables_use_uvm" : f"{USE_UVM_REMAPABLES}",
            "use_cudagraph" : f"{use_cudagraphs}",
        },
    }

    config.update(remapables_config)
    
    program = cerium_runtime.FusedKernelProgram(context,base_name,f"{main_name}")

    cerium_functions_map = program.make_program(gpus,config,raw_inputs_wrapper)
    main_cerium_function = cerium_functions_map[f"{main_name}"]
    softmax_cerium_function = cerium_functions_map[f"{softmax_name}"]



    # data_samples = [0,1,2,3,4]
    data_samples = [sample]
    io_generator = cerium_runtime.IOGenerator(context)


    for data_sample in data_samples:

        if GENERATE_SAMPLES:

            sample_inputs = {}
            fheInputs = {}
            if block_start > 0:
                try:
                    with open(f'fhe_outputs/sample{data_sample}_block{block_start-1}.pkl', 'rb') as f:
                        fheInputs.update(pickle.load(f))
                except Exception as e:
                    print(f"Error loading fhe inputs for block {block_start-1} data_sample:{data_sample}: {e}")
                finally:
                    print(f"Loaded reference fhe for block {block_start-1} data_sample:{data_sample}")
                input_key = "out"
                base = 1
            else:
                try:
                    with open(f'llama_models/llama_inputs/sample{data_sample}_prompt.pkl', 'rb') as f:
                        fheInputs.update(pickle.load(f))
                except Exception as e:
                    print(f"Error loading fhe inputs for block {block_start} data_sample:{data_sample}: {e}")
                finally:
                    print(f"Loaded reference fhe for block {block_start} data_sample:{data_sample}")
                input_key = "rmsnorm_att_input"
                base = 1j



            AttentionMask = {}
            try:
                with open(f'llama_models/llama_inputs/sample{data_sample}_attention_mask.pkl', 'rb') as f:
                    AttentionMask.update(pickle.load(f))
            except Exception as e:
                print(f"Error loading attention mask for block {block_start} data_sample:{data_sample}: {e}")
            finally:
                print(f"Loaded attention mask for block {block_start} data_sample:{data_sample}")


            BootstrapScaleDiv = 1 << 6
            for i in range(16):
                sample_inputs[f"rmsnorm_att_input{i}"] = (np.array(fheInputs[f"{input_key}{i}"])*pow(base,(i%2)),Primes[15]*Primes[14]/BootstrapScaleDiv)

            sample_inputs[f"att_mask"] = (AttentionMask[f"att_mask"],Primes[8]*Primes[7])


            sample_inputs_wrapper = cerium_runtime.RawInputsWrapper(sample_inputs)

            Path(ciphertext_inputs_base_name).mkdir(parents=True, exist_ok=True)
            io_generator.generate_and_serialize_ciphertexts(f"{ciphertext_inputs_base_name}/enc_data_sample{data_sample}_block{block_start}",f"{base_name}/{main_name}/program_inputs",sample_inputs_wrapper,encryptor_context)
            io_generator.generate_and_serialize_ciphertexts(f"{ciphertext_inputs_base_name}/enc_mask_sample{data_sample}",f"{base_name}/{softmax_name}/program_inputs",sample_inputs_wrapper,encryptor_context)

            del sample_inputs_wrapper

        enc_prompt = io_generator.deserialize_ciphertexts(context,f"{ciphertext_inputs_base_name}/enc_data_sample{data_sample}_block{block_start}")
        enc_mask = io_generator.deserialize_ciphertexts(context,f"{ciphertext_inputs_base_name}/enc_mask_sample{data_sample}")

        start_time = time.time()
        main_cerium_function.copy_ciphertext_inputs_to_gpu(f"{base_name}/{main_name}/inputs",gpus,enc_prompt)
        softmax_cerium_function.copy_ciphertext_inputs_to_gpu(f"{base_name}/{softmax_name}/inputs",gpus,enc_mask)
        program.run()
        encrypted_outputs = main_cerium_function.get_program_outputs()
        end_time = time.time()

        outputs = {}
        for k,v in encrypted_outputs.items():
            outputs[k] = encryptor.decrypt_and_decode(v,OutScale[k])


        block_end = block_start + num_blocks - 1
        print_outputs(outputs, block_end, data_sample,prefix=f"block{block_end}_")

        try:
            Path(f"fhe_outputs").mkdir(parents=True, exist_ok=True)
            with open(f'fhe_outputs/sample{data_sample}_block{block_end}.pkl', 'wb') as f:
                pickle.dump(outputs, f)
        except Exception as e:
            print(f"Error dumping outputs for {block_end} data_sample:{data_sample}: {e}")
        finally:
            print("")

SLOTS = 32768
LVL = 51
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_start', type=int, required=True)
    parser.add_argument('--gpus', type=int,default=1)
    parser.add_argument('--cudagraphs', type=str_to_bool, default=True, help="Enable CUDAGraphs")
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--data_sample', type=int, default=0,
                        help="Index of the packed prompt input to run (default: 0)")

    args = parser.parse_args()

    main(args.gpus, args.data_sample, args.block_start, args.cudagraphs,
         args.num_blocks)
