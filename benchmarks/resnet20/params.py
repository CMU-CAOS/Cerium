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

from primes import *
import math 
import pickle
import numpy as np

SLOTS = 32*1024

# import parameters
def import_parameters():
    layer = 20
    end = 2
    dir = "resnet20"

    num_c = 0  # convolution
    num_b = 0  # batch norm bias
    num_m = 0  # batch norm mean
    num_v = 0  # batch norm variance
    num_w = 0  # batch norm weight

    conv_weights = [[] for _ in range(layer-1)]
    bn_bias = [[] for _ in range(layer-1)]
    bn_running_mean = [[] for _ in range(layer-1)]
    bn_running_var = [[] for _ in range(layer-1)]
    bn_weight = [[] for _ in range(layer-1)]
    linear_weight = []
    linear_bias = []

    fh = 3  # filter height
    fw = 3  # filter width
    ci = 0  # number of input channels
    co = 0  # number of output channels

    # convolution parameters
    ci = 3
    co = 16

    # first layer
    f = open(f"pretrained_parameters/{dir}/conv1_weight.txt", "r")
    lines = f.readlines()
    lines = [float(l) for l in lines]
    for i in range(fh * fw * ci * co):
        conv_weights[num_c].append(lines[i])
    f.close()
    num_c += 1

    # convolution parameters
    for j in range(1, 4):
        for k in range(0, end+1):
            # co setting
            if j == 1:
                co = 16
            elif j == 2:
                co = 32
            elif j == 3:
                co = 64

            # ci setting
            if j == 1 or (j == 2 and k == 0):
                ci = 16
            elif (j == 2 and k != 0) or (j == 3 and k == 0):
                ci = 32
            else:
                ci = 64

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_conv1_weight.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(fh * fw * ci * co):
                conv_weights[num_c].append(lines[i])
            f.close()
            num_c += 1

            # ci setting
            if j == 1:
                ci = 16
            elif j == 2:
                ci = 32
            elif j == 3:
                ci = 64

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_conv2_weight.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(fh * fw * ci * co):
                conv_weights[num_c].append(lines[i])
            f.close()
            num_c += 1

    # batch norm parameters
    ci = 16
    f = open(f"pretrained_parameters/{dir}/bn1_bias.txt", "r")
    lines = f.readlines()
    lines = [float(l) for l in lines]
    for i in range(0, ci):
        bn_bias[num_b].append(lines[i])
    f.close()
    num_b += 1

    f = open(f"pretrained_parameters/{dir}/bn1_running_mean.txt", "r")
    lines = f.readlines()
    lines = [float(l) for l in lines]
    for i in range(0, ci):
        bn_running_mean[num_m].append(lines[i])
    f.close()
    num_m += 1

    f = open(f"pretrained_parameters/{dir}/bn1_running_var.txt", "r")
    lines = f.readlines()
    lines = [float(l) for l in lines]
    for i in range(0, ci):
        bn_running_var[num_v].append(lines[i])
    f.close()
    num_v += 1

    f = open(f"pretrained_parameters/{dir}/bn1_weight.txt", "r")
    lines = f.readlines()
    lines = [float(l) for l in lines]
    for i in range(0, ci):
        bn_weight[num_w].append(lines[i])
    f.close()
    num_w += 1

    # batch norm parameters
    for j in range(1, 4):
        # ci setting
        if j == 1:
            ci = 16
        elif j == 2:
            ci = 32
        elif j == 3:
            ci = 64
        for k in range(0, end+1):
            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_bn1_bias.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(0, ci):
                bn_bias[num_b].append(lines[i])
            f.close()
            num_b += 1

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_bn1_running_mean.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(0, ci):
                bn_running_mean[num_m].append(lines[i])
            f.close()
            num_m += 1

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_bn1_running_var.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(0, ci):
                bn_running_var[num_v].append(lines[i])
            f.close()
            num_v += 1

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_bn1_weight.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(0, ci):
                bn_weight[num_w].append(lines[i])
            f.close()
            num_w += 1

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_bn2_bias.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(0, ci):
                bn_bias[num_b].append(lines[i])
            f.close()
            num_b += 1

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_bn2_running_mean.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(0, ci):
                bn_running_mean[num_m].append(lines[i])
            f.close()
            num_m += 1

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_bn2_running_var.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(0, ci):
                bn_running_var[num_v].append(lines[i])
            f.close()
            num_v += 1

            f = open(
                f"pretrained_parameters/{dir}/layer{j}_{k}_bn2_weight.txt", "r")
            lines = f.readlines()
            lines = [float(l) for l in lines]
            for i in range(0, ci):
                bn_weight[num_w].append(lines[i])
            f.close()
            num_w += 1

    # fc parameters
    f = open(f"pretrained_parameters/{dir}/linear_weight.txt", "r")
    lines = f.readlines()
    lines = [float(l) for l in lines]
    for i in range(0, 10 * 64):
        linear_weight.append(lines[i])
    f.close()

    f = open(f"pretrained_parameters/{dir}/linear_bias.txt", "r")
    lines = f.readlines()
    lines = [float(l) for l in lines]
    for i in range(0, 10):
        linear_bias.append(lines[i])
    f.close()

    return {
        "conv": conv_weights,
        "bn_bias": bn_bias,
        "bn_mean": bn_running_mean,
        "bn_var": bn_running_var,
        "bn_weight": bn_weight,
        "linear_weight": linear_weight,
        "linear_bias": linear_bias
    }


LVL=51
AvgPoolBootstrapOutLevel = 6
ReluBootstrapOutLevel = 21

def vectorize_conv_weights(stage, data, running_var, constant_weight, co, st, fh, fw, ki, hi, wi, ci, ti, pi, logn, epsilon):
    # set ho, wo, ko
    if st == 1:
        ho = hi
        wo = wi
        ko = ki
    elif st == 2:
        ho = hi // 2
        wo = wi // 2
        ko = 2 * ki

    # set to, po, q
    n = int(1 << logn)
    to = (co+ko*ko-1) // (ko*ko)
    po = pow2(floor_to_int(math.log(n / (ko*ko*ho*wo*to)) / math.log(2)))
    q = (co+pi-1)//pi


    weight = [[[[0.0 for _ in range(co)] for _ in range(ci)]
               for _ in range(fw)] for _ in range(fh)]
    compact_weight_vec = [[[[0.0 for _ in range(n)] for _ in range(q)]
                           for _ in range(fw)] for _ in range(fh)]
    select_one = [[[[0.0 for _ in range(to)] for _ in range(ko*wo)]
                   for _ in range(ko*ho)] for _ in range(co)]
    select_one_vec = [[0.0 for _ in range(n)] for _ in range(co)]

    # weight setting
    for i1 in range(fh):
        for i2 in range(fw):
            for j3 in range(ci):
                for j4 in range(co):
                    weight[i1][i2][j3][j4] = data[fh *
                                                  fw*ci*j4 + fh*fw*j3 + fw*i1 + i2]

    # compact shifted weight vector setting
    for i1 in range(fh):
        for i2 in range(fw):
            for i9 in range(q):
                for j8 in range(n):
                    j5 = ((j8 % (n//pi)) % (ki*ki*hi*wi))//(ki*wi)
                    j6 = (j8 % (n//pi)) % (ki*wi)
                    i7 = (j8 % (n//pi))//(ki*ki*hi*wi)
                    i8 = j8//(n//pi)

                    if j8 % (n//pi) >= ki*ki*hi*wi*ti or i8+pi*i9 >= co or ki*ki*i7+ki*(j5 % ki)+j6 % ki >= ci or (j6//ki)-(fw-1)//2+i2 < 0 or (j6//ki)-(fw-1)//2+i2 > wi-1 or (j5//ki)-(fh-1)//2+i1 < 0 or (j5//ki)-(fh-1)//2+i1 > hi-1:
                        compact_weight_vec[i1][i2][i9][j8] = 0.0
                    else:
                        compact_weight_vec[i1][i2][i9][j8] = weight[i1][i2][ki *
                                                                            ki*i7+ki*(j5 % ki)+j6 % ki][i8+pi*i9]

    # select one setting
    for j4 in range(co):
        for v1 in range(ko*ho):
            for v2 in range(ko*wo):
                for u3 in range(to):
                    if ko*ko*u3 + ko*(v1 % ko) + v2 % ko == j4:
                        select_one[j4][v1][v2][u3] = constant_weight[j4] / \
                            math.sqrt(running_var[j4]+epsilon)
                    else:
                        select_one[j4][v1][v2][u3] = 0.0

    # select one vector setting
    for j4 in range(co):
        for v1 in range(ko*ho):
            for v2 in range(ko*wo):
                for u3 in range(to):
                    select_one_vec[j4][ko*ko*ho*wo*u3 + ko *
                                       wo*v1 + v2] = select_one[j4][v1][v2][u3]

    pt_weights = {}
    for h_ in range(fh):
        for w_ in range(fw):
            for q_ in range(q):
                pt_weights[f"cn_{stage}_{h_}_{w_}_{q_}"] = compact_weight_vec[h_][w_][q_]

    for i, s_one in enumerate(select_one_vec):
        pt_weights[f"bn_{stage}_{i}"] = s_one
    
    return pt_weights

def vectorize_batch_norm_weights(ki, hi, wi, ci, ti, pi, stage, bias, running_mean, running_var, weight, epsilon, B, logn):
    n = 1 << logn

    # generate g vector
    g = [0.0] * n

    for v4 in range(n):
        v1 = ((v4 % (n//pi)) % (ki*ki*hi*wi))//(ki*wi)
        v2 = (v4 % (n//pi)) % (ki*wi)
        u3 = (v4 % (n//pi))//(ki*ki*hi*wi)

        if ki*ki*u3+ki*(v1 % ki)+v2 % ki >= ci or v4 % (n//pi) >= ki*ki*hi*wi*ti:
            g[v4] = 0.0
        else:
            idx = ki*ki*u3 + ki*(v1 % ki) + v2 % ki
            g[v4] = (running_mean[idx] * weight[idx] /
                     math.sqrt(running_var[idx]+epsilon) - bias[idx])/B
    
    return {f"bn_sub_{stage}": g}

SignCoeffs = [[1.31102, -0.465544, 0.388789, -0.341032, 1.15837, -0.243786, 2.24319, -4.02702, 0, 0, 0, 0, 0, 0],
[0.776581, -0.681841, -0.0584022, -0.560485, 0.133127, -0.459881, 0.936808, -4.11443, 0, 0, 0, 0, 0, 0],
[-0.0428579, -0.774793, -0.181282, -0.245468, -0.789123, -0.727411, -0.325545, -0.186018, -0.197326, -0.141117, -0.0526757, -0.026313, -0.0177327, -0.00908409]]

def sign_raw_inputs_0(inpScale,inpLevel_,r,outScale):

    assert r == 0 or r == 1

    ChebyScales = {}
    ChebyLevels = {}

    Inputs = {}
    inpLevel = inpLevel_
    inpScale *= 4
    outScale /= 2
    ChebyScales[1] = inpScale
    ChebyLevels[1] = inpLevel
    for i in [2,3,4,8]:
        if i % 2 == 0:
            scale = ChebyScales[i//2]*ChebyScales[i//2]
            level = ChebyLevels[i//2]
            Inputs[f"sign_{r}_ones_{i}_{level}"] = (1,scale)
            ChebyScales[i] = scale / Primes[level-1]
            ChebyLevels[i] = level - 1
        else:
            scale = ChebyScales[i//2 + 1]*ChebyScales[i//2]
            level = ChebyLevels[i//2 + 1]
            rescaleFactor = scale * Primes[ChebyLevels[1] - 1] / ChebyScales[1]
            Inputs[f"sign_{r}_rescale_{i}_{ChebyLevels[1]}"] = (1,rescaleFactor)
            ChebyScales[i] = scale / Primes[level-1]
            ChebyLevels[i] = level - 1


    cScale = [None]*8

    cScale[7] = outScale * Primes[ChebyLevels[8] - 1] * Primes[ChebyLevels[4]-1] * Primes[ChebyLevels[2]-1]*Primes[ChebyLevels[1]-1] / (ChebyScales[8]*ChebyScales[4]*ChebyScales[2]*ChebyScales[1])
    cScale[6] = outScale * Primes[ChebyLevels[8] - 1] * Primes[ChebyLevels[4]-1] * Primes[ChebyLevels[2]-1] / (ChebyScales[8]*ChebyScales[4]*ChebyScales[1])
    cScale[5] = outScale * Primes[ChebyLevels[8] - 1] * Primes[ChebyLevels[4]-1] / (ChebyScales[8]*ChebyScales[3])
    cScale[4] = outScale * Primes[ChebyLevels[8] - 1] * Primes[ChebyLevels[4]-1] / (ChebyScales[8]*ChebyScales[1])
    cScale[3] = outScale * Primes[ChebyLevels[8] - 1] * Primes[ChebyLevels[4]-1] / (ChebyScales[4]*ChebyScales[3])
    cScale[2] = outScale * Primes[ChebyLevels[8] - 1] * Primes[ChebyLevels[4]-1] / (ChebyScales[4]*ChebyScales[1])
    cScale[1] = outScale * Primes[ChebyLevels[8] - 1] * Primes[ChebyLevels[4]-1] / (ChebyScales[3])
    cScale[0] = outScale * Primes[ChebyLevels[8] - 1] * Primes[ChebyLevels[4]-1] / (ChebyScales[1])


    Inputs[f"sign_{r}_{7}_{ChebyLevels[1]}"] = (SignCoeffs[r][7],cScale[7])
    Inputs[f"sign_{r}_{6}_{ChebyLevels[1]}"] = (SignCoeffs[r][6],cScale[6])
    Inputs[f"sign_{r}_{5}_{ChebyLevels[3]}"] = (SignCoeffs[r][5],cScale[5])
    Inputs[f"sign_{r}_{4}_{ChebyLevels[1]}"] = (SignCoeffs[r][4],cScale[4])
    Inputs[f"sign_{r}_{3}_{ChebyLevels[3]}"] = (SignCoeffs[r][3],cScale[3])
    Inputs[f"sign_{r}_{2}_{ChebyLevels[1]}"] = (SignCoeffs[r][2],cScale[2])
    Inputs[f"sign_{r}_{1}_{ChebyLevels[3]}"] = (SignCoeffs[r][1],cScale[1])
    Inputs[f"sign_{r}_{0}_{ChebyLevels[1]}"] = (SignCoeffs[r][0],cScale[0])

    outScale *= 2

    return Inputs, outScale, (ChebyLevels[8]-1)

def sign_raw_inputs_1(inpScale,inpLevel_,r,outScale):

    assert r == 2

    ChebyScales = {}
    ChebyLevels = {}

    Inputs = {}
    inpLevel = inpLevel_

    ChebyScales[1] = inpScale
    ChebyLevels[1] = inpLevel
    for i in [2,3,4,5,7,8,16]:
        if i % 2 == 0:
            scale = ChebyScales[i//2]*ChebyScales[i//2]
            level = ChebyLevels[i//2]
            Inputs[f"sign_{r}_ones_{i}_{level}"] = (1,scale)
            ChebyScales[i] = scale / Primes[level-1]
            ChebyLevels[i] = level - 1
        else:
            scale = ChebyScales[i//2 + 1]*ChebyScales[i//2]
            level = ChebyLevels[i//2 + 1]
            rescaleFactor = scale * Primes[ChebyLevels[1] - 1] / ChebyScales[1]
            Inputs[f"sign_{r}_rescale_{i}_{ChebyLevels[1]}"] = (1,rescaleFactor)
            ChebyScales[i] = scale / Primes[level-1]
            ChebyLevels[i] = level - 1

    outScale /= 2
    cScale = [None]*14
    cScale[13] = outScale  * Primes[ChebyLevels[8]-1] * Primes[ChebyLevels[8]] / (ChebyScales[16]  *   ChebyScales[8]*ChebyScales[3])
    cScale[12] = outScale  * Primes[ChebyLevels[8]-1] * Primes[ChebyLevels[8]] / (ChebyScales[16]  *   ChebyScales[8]*ChebyScales[1])
    cScale[11] = outScale  * Primes[ChebyLevels[8]-1] / (ChebyScales[16]*ChebyScales[7])
    cScale[10] = outScale  * Primes[ChebyLevels[8]-1] / (ChebyScales[16]*ChebyScales[5])
    cScale[9]  = outScale  * Primes[ChebyLevels[8]-1] / (ChebyScales[16]*ChebyScales[3])
    cScale[8]  = outScale  * Primes[ChebyLevels[8]-1] / (ChebyScales[16]*ChebyScales[1])
    cScale[7]  = outScale  * Primes[ChebyLevels[8]-1] / (ChebyScales[8]*ChebyScales[7])
    cScale[6]  = outScale  * Primes[ChebyLevels[8]-1] / (ChebyScales[8]*ChebyScales[5])
    cScale[5]  = outScale  * Primes[ChebyLevels[8]-1] / (ChebyScales[8]*ChebyScales[3])
    cScale[4]  = outScale  * Primes[ChebyLevels[8]-1] / (ChebyScales[8]*ChebyScales[1])
    cScale[3]  = outScale   / (ChebyScales[7])
    cScale[2]  = outScale   / (ChebyScales[5])
    cScale[1]  = outScale   / (ChebyScales[3])
    cScale[0]  = outScale   / (ChebyScales[1])

    Inputs[f"sign_{r}_{13}_{ChebyLevels[3]}"] = (SignCoeffs[r][13],cScale[13])
    Inputs[f"sign_{r}_{12}_{ChebyLevels[1]}"] = (SignCoeffs[r][12],cScale[12])
    Inputs[f"sign_{r}_{11}_{ChebyLevels[7]}"] = (SignCoeffs[r][11],cScale[11])
    Inputs[f"sign_{r}_{10}_{ChebyLevels[5]}"] = (SignCoeffs[r][10],cScale[10])
    Inputs[f"sign_{r}_{9}_{ChebyLevels[3]}"] = (SignCoeffs[r][9],cScale[9])
    Inputs[f"sign_{r}_{8}_{ChebyLevels[1]}"] = (SignCoeffs[r][8],cScale[8])
    Inputs[f"sign_{r}_{7}_{ChebyLevels[7]}"] = (SignCoeffs[r][7],cScale[7])
    Inputs[f"sign_{r}_{6}_{ChebyLevels[5]}"] = (SignCoeffs[r][6],cScale[6])
    Inputs[f"sign_{r}_{5}_{ChebyLevels[3]}"] = (SignCoeffs[r][5],cScale[5])
    Inputs[f"sign_{r}_{4}_{ChebyLevels[1]}"] = (SignCoeffs[r][4],cScale[4])
    Inputs[f"sign_{r}_{3}_{ChebyLevels[7]}"] = (SignCoeffs[r][3],cScale[3])
    Inputs[f"sign_{r}_{2}_{ChebyLevels[5]}"] = (SignCoeffs[r][2],cScale[2])
    Inputs[f"sign_{r}_{1}_{ChebyLevels[3]}"] = (SignCoeffs[r][1],cScale[1])
    Inputs[f"sign_{r}_{0}_{ChebyLevels[1]}"] = (SignCoeffs[r][0],cScale[0])

    outScale *= 2

    return Inputs, outScale, ChebyLevels[16]

def get_relu_inputs(ImgScaleDiv):

    Relu0ScaleDiv = 1<<8
    Relu1ScaleDiv = 1 << 8
    Relu2ScaleDiv = 1<<8

    Inputs = {}

    # rescale for bootstrap:
    Inputs[f'rescale_bootstrap_relu_0_{27}'] = (1, Relu0ScaleDiv)
    Inputs[f'rescale_bootstrap_relu_0_{39}'] = (1, Relu0ScaleDiv)
    Inputs[f'rescale_bootstrap_relu_1'] = (1, Relu1ScaleDiv)
    Inputs[f'rescale_bootstrap_relu_final'] = (1, Relu2ScaleDiv)
    Inputs[f'rescale_bootstrap_relu_inp'] = (1, ImgScaleDiv)

    inpLevel = 21
    inpScaleDiv = Primes[inpLevel-1]*Primes[inpLevel-2] / ImgScaleDiv
    inpScale = Primes[inpLevel-1]*Primes[inpLevel-2]
    outLevel= inpLevel - 5
    reluOutScale = Primes[outLevel-1]*Primes[outLevel-2]
    (relu_raw_inp,reluScale,level) = sign_raw_inputs_0(inpScale/Primes[inpLevel-1],inpLevel-1,0,reluOutScale)
    Inputs.update(relu_raw_inp)

    outLevel= inpLevel - 10
    reluOutScale = Primes[outLevel-1]*Primes[outLevel-2]
    (relu_raw_inp,reluScale,level) = sign_raw_inputs_0(reluScale/(Primes[level-1]),level-1,1,reluOutScale)
    Inputs.update(relu_raw_inp)

    outLevel= inpLevel - 15
    reluOutMul = 1
    reluOutScale = Primes[outLevel-1]*Primes[outLevel-2] * reluOutMul
    (relu_raw_inp,reluScale,level) = sign_raw_inputs_1(reluScale/(Primes[level-1]),level-1,2,reluOutScale)
    Inputs.update(relu_raw_inp)
    Inputs[f"relu_half_sign_{level}"] = (0.5,reluOutScale)
    rescale_scale = Primes[inpLevel-1]*Primes[inpLevel-2]*Primes[outLevel-1]*Primes[outLevel-2]*Primes[outLevel-3]*Primes[outLevel-4]/(inpScaleDiv * reluOutScale * ImgScaleDiv)
    Inputs[f"rescale_relu_sign_inp_{inpLevel}"] = (1,rescale_scale)

    return Inputs

def vectorize_downsampling_weights(stage, ki, hi, wi, ci, ti, pi, logn): 
    n = 1 << logn
    ko = 2*ki
    ho = hi//2
    wo = wi//2
    to = ti//2
    co = 2*ci
    po = pow2(floor_to_int(math.log((n)/(ko*ko*ho*wo*to)) / math.log(2.0)))

    # variables
    select_one_vec = [[[0.0] * n for _ in range(ti)] for _ in range(ki)]
    for w1 in range(ki):
        for w2 in range(ti):
            for v4 in range(n):
                j5 = (v4 % (ki*ki*hi*wi))//(ki*wi)
                j6 = v4 % (ki*wi)
                i7 = v4//(ki*ki*hi*wi)
                if v4 < ki*ki*hi*wi*ti and (j5//ki) % 2 == 0 and (j6//ki) % 2 == 0 and (j5 % ki) == w1 and i7 == w2:
                    select_one_vec[w1][w2][v4] = 1.0
                else:
                    select_one_vec[w1][w2][v4] = 0.0

    downsampling_weights = {}
    for w1 in range(ki):
        for w2 in range(ti):
            downsampling_weights[f"dsw_{stage}_{w1}_{w2}"] = (select_one_vec[w1][w2],Primes[ReluBootstrapOutLevel-1]*Primes[ReluBootstrapOutLevel-2])
    return downsampling_weights

def vectorize_avg_pooling(stage, ki, hi, wi, ci, ti, pi, logn, B):
    avg_pooling = {}
    avg_pooling[f"ap_{stage}_scale_factor"] = (1,Primes[AvgPoolBootstrapOutLevel-1])
    n = 1 << logn
    for s in range(ki):
        for u in range(ti):
            select_one = [0.0] * n
            for i in range(ki):
                select_one[(ki*u+s)*ki+i] = B / (hi*wi)
            avg_pooling[f"ap_{stage}_{s}_{u}"] = (select_one,Primes[AvgPoolBootstrapOutLevel-2])
    return avg_pooling

def vectorize_matmul(matrix, bias, logn, q, r):
    n = 1 << logn
    W = [[0.0] * n for _ in range(q+r-1)]
    b = [0.0] * n
    for z in range(q):
        b[z] = bias[z]

    for i in range(q):
        for j in range(r):
            W[i-j+r-1][i] = matrix[i*r+j]

    matmul = {}
    for s in range(q + r - 1):
        # matmul[f"mm_weight_{s}"] = (W[s],Primes[12]*Primes[11])
        matmul[f"w_{s}"] = (W[s],Primes[AvgPoolBootstrapOutLevel-3]*Primes[AvgPoolBootstrapOutLevel-4])
    # matmul["mm_bias"] = b
    return matmul
    
# load all parameters 
def load_all_params():
    end_num=2
    level = 4
    params = import_parameters()

    inputs = {}

    ImgScaleDiv = 1 << 4

    stage = 0

    epsilon = 0.00001
    B = 40.0

    co = 16 
    st = 1 
    fh = 3
    fw = 3 
    ki = 1
    hi = 32
    wi = 32 
    ci = 3 
    ti = 3
    pi = 8 
    logn = 15
    st = 1

    cn_weights = vectorize_conv_weights(stage, params["conv"][stage], params["bn_var"][stage], params["bn_weight"][stage], co, st, fh, fw, ki, hi, wi, ci, ti, pi, logn, epsilon)
    for k,v in cn_weights.items():
        if k.startswith("cn"):
            cn_weights[k] = (v[:SLOTS], Primes[ReluBootstrapOutLevel-1]/2)
        elif k.startswith("bn"):
            cn_weights[k] = (v[:SLOTS], Primes[ReluBootstrapOutLevel-2])
    level -= 2
    inputs.update(cn_weights)

    ci = 16 
    ti = 16
    pi = 2
    bn_sub = vectorize_batch_norm_weights(ki, hi, wi, ci, ti, pi, stage, params["bn_bias"][stage], params["bn_mean"][stage], params["bn_var"][stage], params["bn_weight"][stage], epsilon, B, logn)
    for k,v in bn_sub.items():
        bn_sub[k] = (v[:SLOTS], Primes[ReluBootstrapOutLevel-1]*Primes[ReluBootstrapOutLevel-2]/ImgScaleDiv)
    inputs.update(bn_sub)

    for j in range(3):
        if j == 0:
            co = 16
        elif j == 1:
            co = 32
        elif j == 2:
            co = 64

        if j == 0:
            co = 16
            st = 1
            ki = 1
            hi = 32
            wi = 32
            ci = 16
            ti = 16
            pi = 2

        for k in range(end_num+1):
            if j >= 1 and k == 0:
                st = 2
            else:
                st = 1

            stage = 2*((end_num+1)*j+k)+1
            
            cn_weights = vectorize_conv_weights(stage, params["conv"][stage], params["bn_var"][stage], params["bn_weight"][stage], co, st, fh, fw, ki, hi, wi, ci, ti, pi, logn, epsilon)
            for key,val in cn_weights.items():
                if key.startswith("cn"):
                    cn_weights[key] = (val[:SLOTS], Primes[ReluBootstrapOutLevel-1]/2)
                elif key.startswith("bn"):
                    cn_weights[key] = (val[:SLOTS], Primes[ReluBootstrapOutLevel-2])
            level -= 2
            inputs.update(cn_weights)
            
            if j == 1:
                ki = 2
                hi = 16
                wi = 16
                ci = 32
                ti = 8
                pi = 4
            if j == 2:
                ki = 4
                hi = 8
                wi = 8
                ci = 64
                ti = 4
                pi = 8

            bn_sub = vectorize_batch_norm_weights(ki, hi, wi, ci, ti, pi, stage, params["bn_bias"][stage], params["bn_mean"][stage], params["bn_var"][stage], params["bn_weight"][stage], epsilon, B, logn)
            for key,val in bn_sub.items():
                bn_sub[key] = (val[:SLOTS], Primes[ReluBootstrapOutLevel-1]*Primes[ReluBootstrapOutLevel-2]/ImgScaleDiv)
            inputs.update(bn_sub)

            level = 4

            stage = 2*((end_num+1)*j+k)+2
            st = 1

            cn_weights = vectorize_conv_weights(stage, params["conv"][stage], params["bn_var"][stage], params["bn_weight"][stage], co, st, fh, fw, ki, hi, wi, ci, ti, pi, logn, epsilon)
            for key,val in cn_weights.items():
                if key.startswith("cn"):
                    cn_weights[key] = (val[:SLOTS], Primes[ReluBootstrapOutLevel-1]/2)
                elif key.startswith("bn"):
                    cn_weights[key] = (val[:SLOTS], Primes[ReluBootstrapOutLevel-2])
            inputs.update(cn_weights)

            level = level - 2

            bn_sub = vectorize_batch_norm_weights(ki, hi, wi, ci, ti, pi, stage, params["bn_bias"][stage], params["bn_mean"][stage], params["bn_var"][stage], params["bn_weight"][stage], epsilon, B, logn)
            for key,val in bn_sub.items():
                bn_sub[key] = (val[:SLOTS], Primes[ReluBootstrapOutLevel-1]*Primes[ReluBootstrapOutLevel-2]/ImgScaleDiv)
            inputs.update(bn_sub)



            if j >= 1 and k == 0:
                if j == 1:
                    k_ = 1
                    h_ = 32
                    w_ = 32
                    c_ = 16
                    t_ = 16
                    p_ = 2
                if j == 2:
                    k_ = 2
                    h_ = 16
                    w_ = 16
                    c_ = 32
                    t_ = 8
                    p_ = 4
                ds_weights = vectorize_downsampling_weights(stage, k_, h_, w_, c_, t_, p_, logn)
                inputs.update(ds_weights)
            else:
                inputs[f"rescale_plus_{level}"] = (1,Primes[ReluBootstrapOutLevel-1]*Primes[ReluBootstrapOutLevel-2])
            level = ReluBootstrapOutLevel

    avg_pooling = vectorize_avg_pooling(stage, ki, hi, wi, ci, ti, pi, logn, B)
    inputs.update(avg_pooling)

    matmul = vectorize_matmul(params["linear_weight"], params["linear_bias"], logn, 10, 64)
    inputs.update(matmul)

    relu = get_relu_inputs(ImgScaleDiv)
    inputs.update(relu)
    return inputs 


def floor_to_int(x):
    return int(math.floor(x)+0.5)

def pow2(n):
    prod = 1
    for _ in range(n):
        prod *= 2
    return prod

params = load_all_params()
with open(f'resnet20_params.pkl', 'wb') as f:
    pickle.dump(params, f)