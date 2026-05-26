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

from bootstrap import *
from cerium.dsl import *
from cerium.passes import *
from cerium.compiler import *
import argparse
import argparse

import math
import os

logn = 15
scale = 28
RNS_BIT_SIZE = 28

USE_PARTITIONS=False
TEST=False

def TestOutput(name,val):
    if TEST:
        Output(name,val)

def TestFunctionOutput(name,val):
    if TEST:
        FunctionOutput(name,val)

class Tensor:
    k_ = 0
    h_ = 0
    w_ = 0
    c_ = 0
    t_ = 0
    p_ = 0
    logn_ = 0
    vec_ = []

    def __init__(self, logn, k, h, w, c, t, p, vec):
        self.k_ = k
        self.h_ = h
        self.w_ = w
        self.c_ = c
        self.t_ = t
        self.p_ = p
        self.logn_ = logn
        self.vec_ = vec

    def __repr__(self):
        return f'k: {self.k_}\nh: {self.h_}\nw: {self.w_}\nc: {self.c_}\nt: {self.t_}\np: {self.p_}\n'

    def k(self):
        return self.k_

    def h(self):
        return self.h_

    def w(self):
        return self.w_

    def c(self):
        return self.c_

    def t(self):
        return self.t_

    def p(self):
        return self.p_

    def logn(self):
        return self.logn_

    def n(self):
        return 1 << self.logn_

    def vec(self):
        return self.vec_

def resnet(layer):

    Partition(1,0)

    resnetPlainTextInputs.clear()
    end_num = 2

    co = 0  # channels
    st = 0  # stride
    fh = 3  # filter height
    fw = 3  # filter width
    init_p = 8
    stage = 0
    level = 0

    enc_image = CiphertextInput("image", 56, 4)
    cnn = Tensor(logn, 1, 32, 32, 3, 3, init_p, enc_image)
    level = enc_image.level()
    
    cnn, level = convolution(cnn, 16, 1, fh, fw, stage, level=enc_image.level())
    cnn, level = batch_norm(cnn, stage, level)
    cnn, level = relu(cnn, stage)
    

    for j in range(3):
        if j == 0:
            co = 16
        elif j == 1:
            co = 32
        elif j == 2:
            co = 64

        for k in range(end_num+1):
            stage = 2*((end_num+1)*j+k)+1
            cnnInit = cnn

            # set stride
            if j >= 1 and k == 0:
                st = 2
            else:
                st = 1

            cnn, level = convolution(cnn, co, st, fh, fw, stage, level=cnn.vec().level())
            cnn, level = batch_norm(cnn, stage, level)
            cnn, level = relu(cnn, stage) 


            stage = 2*((end_num+1)*j+k)+2
            st = 1

            cnn, level = convolution(cnn, co, st, fh, fw, stage, level=level)
            cnn, level = batch_norm(cnn, stage, level)

            if j >= 1 and k == 0:
                cnnTemp, temp_level = downsample(cnnInit, stage, cnnInit.vec().level())
                temp = cnnTemp.vec()
            else:
                temp = cnnInit.vec()
                rescaleFactor = getPlaintextInput(f"rescale_plus_{cnn.vec().level()}",scale*2,temp.level(),scalar=True)
                temp = temp * rescaleFactor
                temp = temp.doubleRescale()
            while temp.level() != cnn.vec().level():
                temp = temp.modswitch()
            cnn.vec_ = cnn.vec() + temp
            TestOutput(f"plus_{stage}",cnn.vec())
            cnn, level = relu(cnn, stage) 

    cnn, level = avg_pool(cnn, level, stage)
    cnn.vec_ = bootstrap(cnn.vec_,"b33",4,Gpus)
    cnn, level = mat_mul(cnn, None, 10, 64)
    Output("result", cnn.vec())
    return cnn


resnetPlainTextInputs = {}
def getPlaintextInput(name,scale,level,scalar=False):
    if name not in resnetPlainTextInputs.keys():
        pt = PlaintextInput(name,scale,level,scalar)
        resnetPlainTextInputs[name] = pt
        return pt
    else:
        pt = resnetPlainTextInputs[name]
        if pt.level() != level:
            raise ValueError(f"Previous plaintext for name:{name} exists with differnt level")
        if pt.scale() != scale:
            raise ValueError(f"Previous plaintext for name:{name} exists with differnt scale")
        return pt


def convolution_partitions(cnn, co, st, fh, fw, stage, level):

    ki = cnn.k()
    hi = cnn.h()
    wi = cnn.w()
    ci = cnn.c()
    ti = cnn.t()
    pi = cnn.p()
    logn = cnn.logn()
    ko = 0
    ho = 0
    wo = 0
    to = 0
    po = 0

    if st == 1:
        ho = hi
        wo = wi
        ko = ki
    elif st == 2:
        ho = hi // 2
        wo = wi // 2
        ko = 2 * ki



    n = int(1 << logn)
    to = (co+ko*ko-1) // (ko*ko)
    po = pow2(floor_to_int(math.log(n / (ko*ko*ho*wo*to)) / math.log(2)))
    q = (co+pi-1)//pi

    enc_image = []
    select_one_vec = []
    compact_weight_vec = []
    for q_ in range(q):
        Partition(1,q_ % Gpus)
        enc_image.append(Receive(cnn.vec()))
        compact_weight_vec.append([[
            PlaintextInput(f"cn_{stage}_{h_}_{w_}_{q_}", scale, level) for w_ in range(fw)] for h_ in range(fh)])
        select_one_vec.append([PlaintextInput(f'bn_{stage}_{co_}', scale, level-1) for co_ in range(co)])

    compact_weight_vec2 = [[] for _ in range(q)]
    rot_steps = []
    baby_steps = []
    giant_steps = []
    for i1 in range(fh):
        for i2 in range(fw):
            step = ki*ki*wi*(i1-(fh-1)//2) + ki*(i2-(fw-1)//2)
            rot_steps.append(step)
            for q_ in range(q):
                compact_weight_vec2[q_].append(compact_weight_vec[q_][i1][i2])
    for i1 in range(fh):
        giant_steps.append(ki*ki*wi*(i1-(fh-1)//2))
    for i2 in range(fw):
        baby_steps.append(ki*(i2-(fw-1)//2))
    t_sum = None
    total_sum = [None for _ in range(Gpus)] 
    
    for q_ in range(q):
        Partition(1,q_ % Gpus)
        pts = []
        t_sum = RotateMultiplyAccumulate(enc_image[q_],compact_weight_vec2[q_],rot_steps)

        var = t_sum
        
        d = log2_long(ki)
        c = log2_long(ti)

        for x in range(d):
            temp = var
            temp = rotate(temp, pow2(x), n)
            var = var + temp

        for x in range(d):
            temp = var
            temp = rotate(temp, pow2(x) * ki * wi, n)
            var = var + temp

        if c == -1:
            temp = var
            var = RotateAccumulate(temp,[ki*ki*hi*wi*x for x in range(ti)])
        else:
            for x in range(c):
                temp = var
                temp = rotate(temp, pow2(x) * ki * ki*hi*wi, n)
                var = var + temp

        var = var.rescale()
        i8 = 0
        
        steps = []
        pts = []
        while i8 < pi and pi*q_+i8 < co:
            j4 = pi*q_+i8
            steps.append((n//pi)*(j4 % pi) - j4 %
                                      ko - (j4//(ko*ko))*ko*ko*ho*wo - ((j4 % (ko*ko))//ko)*ko*wo)
            pts.append(select_one_vec[q_][j4])
            i8 += 1
        temp = RotateMultiplyAccumulate(var,pts,steps)
        if q_ // Gpus == 0:
            total_sum[q_ % Gpus ] = temp
        else:
            total_sum[q_ % Gpus ] +=  temp

    Partition(Gpus,0)
    var = Receive(total_sum[0])
    for ch in range(1,Gpus):
        if total_sum[ch]:
            ts = Receive(total_sum[ch])
            var += ts 
    Partition(1,0)
    var = Receive(var)

    var = RotateAccumulate(var,[-u6*(n//po) for u6 in range(po)])
    var = var + var.conjugate()
    var = var.rescale()
    Partition(Gpus,0)
    var = Receive(var)

    return Tensor(logn, ko, ho, wo, co, to, po, var), var.level() 

def convolution_simple(cnn, co, st, fh, fw, stage, level):

    ki = cnn.k()
    hi = cnn.h()
    wi = cnn.w()
    ci = cnn.c()
    ti = cnn.t()
    pi = cnn.p()
    logn = cnn.logn()
    ko = 0
    ho = 0
    wo = 0
    to = 0
    po = 0

    if st == 1:
        ho = hi
        wo = wi
        ko = ki
    elif st == 2:
        ho = hi // 2
        wo = wi // 2
        ko = 2 * ki



    n = int(1 << logn)
    to = (co+ko*ko-1) // (ko*ko)
    po = pow2(floor_to_int(math.log(n / (ko*ko*ho*wo*to)) / math.log(2)))
    q = (co+pi-1)//pi

    enc_image = cnn.vec()
    select_one_vec = []
    compact_weight_vec = []
    for q_ in range(q):
        # Partition(1,q_ % gpus)
        # enc_image.append(cnn.vec())
        compact_weight_vec.append([[
            PlaintextInput(f"cn_{stage}_{h_}_{w_}_{q_}", scale, level) for w_ in range(fw)] for h_ in range(fh)])
    select_one_vec = [PlaintextInput(f'bn_{stage}_{co_}', scale, level-1) for co_ in range(co)]

    compact_weight_vec2 = [[] for _ in range(q)]
    rot_steps = []
    baby_steps = []
    giant_steps = []
    for i1 in range(fh):
        for i2 in range(fw):
            step = ki*ki*wi*(i1-(fh-1)//2) + ki*(i2-(fw-1)//2)
            rot_steps.append(step)
            for q_ in range(q):
                compact_weight_vec2[q_].append(compact_weight_vec[q_][i1][i2])
    for i1 in range(fh):
        giant_steps.append(ki*ki*wi*(i1-(fh-1)//2))
    for i2 in range(fw):
        baby_steps.append(ki*(i2-(fw-1)//2))
    t_sum = None
    total_sum = [None for _ in range(q)] 
    
    for q_ in range(q):
        # Partition(1,q_ % gpus)
        pts = []
        t_sum = RotateMultiplyAccumulate(enc_image,compact_weight_vec2[q_],rot_steps)

        var = t_sum
        
        d = log2_long(ki)
        c = log2_long(ti)

        for x in range(d):
            temp = var
            temp = rotate(temp, pow2(x), n)
            var = var + temp

        for x in range(d):
            temp = var
            temp = rotate(temp, pow2(x) * ki * wi, n)
            var = var + temp

        if c == -1:
            temp = var
            var = RotateAccumulate(temp,[ki*ki*hi*wi*x for x in range(ti)])
        else:
            for x in range(c):
                temp = var
                temp = rotate(temp, pow2(x) * ki * ki*hi*wi, n)
                var = var + temp

        var = var.rescale()
        i8 = 0
        
        steps = []
        pts = []
        while i8 < pi and pi*q_+i8 < co:
            j4 = pi*q_+i8
            steps.append((n//pi)*(j4 % pi) - j4 %
                                      ko - (j4//(ko*ko))*ko*ko*ho*wo - ((j4 % (ko*ko))//ko)*ko*wo)
            pts.append(select_one_vec[j4])
            i8 += 1
        temp = RotateMultiplyAccumulate(var,pts,steps)
        total_sum[q_] = temp
    var = total_sum[0]
    for ch in range(1,q):
        if total_sum[ch]:
            ts = total_sum[ch]
            var += ts 

    var = RotateAccumulate(var,[-u6*(n//po) for u6 in range(po)])
    var = var + var.conjugate()
    var = var.rescale()

    return Tensor(logn, ko, ho, wo, co, to, po, var), var.level() 





def convolution(cnn, co, st, fh, fw, stage, level):
    if Gpus > 3 and USE_PARTITIONS:
        return convolution_partitions(cnn,co,st,fh,fw,stage,level)
    else:
        return convolution_simple(cnn,co,st,fh,fw,stage,level)

def batch_norm(cnn, stage, level=0):
    temp = cnn.vec() # (1,2) 0
    g = PlaintextInput(f'bn_sub_{stage}', scale*2, level)
    temp = temp - g
    TestOutput(f"batch_{stage}",temp)
    return Tensor(logn, cnn.k(), cnn.h(), cnn.w(), cnn.c(), cnn.t(), cnn.p(), temp), temp.level()

def sign_0(x,stage,r):
    ## Increase the scale of the values by adding x to itself
    x = x + x
    x = x + x

    chebyshevTerms = {}
    chebyshevTerms[1] = x
    for i in [2,3,4,8]:
        if i % 2 == 0:
            ct = chebyshevTerms[i//2]*chebyshevTerms[i//2]
            ct = ct.relinearize2()
            ct = ct + ct
            one = getPlaintextInput(f"sign_{r}_ones_{i}_{ct.level()}", ct.scale(), ct.level(),scalar=True)
            ct = ct - one
            ct = ct.rescale()
            chebyshevTerms[i] = ct
        else:
            ct = chebyshevTerms[i//2]
            ct_ = chebyshevTerms[i//2 + 1]
            while ct.level() > ct_.level():
                ct = ct.modswitch()
            ct = ct*ct_
            ct = ct.relinearize2()
            ct = ct + ct

            ct_one = chebyshevTerms[1]
            rescaleFactor = getPlaintextInput(f"sign_{r}_rescale_{i}_{ct_one.level()}",2*scale,ct_one.level(),scalar=True)
            ct_one = ct_one * rescaleFactor
            ct_one = ct_one.rescale()
            while(ct_one.level() != (chebyshevTerms[i//2 + 1].level())):
                ct_one = ct_one.modswitch()
            ct = ct - ct_one
            ct = ct.rescale()
            chebyshevTerms[i] = ct

    ReluCoeffScale = 2*scale
    coeff7 = getPlaintextInput(f"sign_{r}_7_{chebyshevTerms[1].level()}", ReluCoeffScale, chebyshevTerms[1].level(),scalar=True)
    t7 = (chebyshevTerms[1]*coeff7).rescale()
    t7 = (t7 * chebyshevTerms[2]).relinearize()#.rescale()

    coeff6 = getPlaintextInput(f"sign_{r}_6_{chebyshevTerms[1].level()}", ReluCoeffScale, chebyshevTerms[1].level(),scalar=True)
    t6 = (chebyshevTerms[1]*coeff6).modswitch()#.rescale()
    t7 = t7 + t6

    coeff5 = getPlaintextInput(f"sign_{r}_5_{chebyshevTerms[3].level()}", ReluCoeffScale, chebyshevTerms[3].level(),scalar=True)
    t5 = (chebyshevTerms[3]*coeff5)

    coeff4 = getPlaintextInput(f"sign_{r}_4_{chebyshevTerms[1].level()}", ReluCoeffScale, chebyshevTerms[1].level(),scalar=True)
    t4 = (chebyshevTerms[1]*coeff4).modswitch().modswitch()
    t5 = t5 + t4

    coeff3 = getPlaintextInput(f"sign_{r}_3_{chebyshevTerms[3].level()}", ReluCoeffScale, chebyshevTerms[3].level(),scalar=True)
    t3 = (chebyshevTerms[3]*coeff3)

    coeff2 = getPlaintextInput(f"sign_{r}_2_{chebyshevTerms[1].level()}", ReluCoeffScale, chebyshevTerms[1].level(),scalar=True)
    t2 = (chebyshevTerms[1]*coeff2).modswitch().modswitch()
    t3 = t3 + t2

    coeff1 = getPlaintextInput(f"sign_{r}_1_{chebyshevTerms[3].level()}", ReluCoeffScale + scale, chebyshevTerms[3].level(),scalar=True)
    t1 = (chebyshevTerms[3]*coeff1)

    coeff0 = getPlaintextInput(f"sign_{r}_0_{chebyshevTerms[1].level()}", ReluCoeffScale + scale, chebyshevTerms[1].level(),scalar=True)
    t0 = (chebyshevTerms[1]*coeff0).modswitch().modswitch()
    t1 = t1 + t0

    t7 = (t7.rescale() * chebyshevTerms[4]).relinearize()

    t7 = t7 + t5

    t7 = (t7.rescale() * chebyshevTerms[8])

    t3 = (t3 * chebyshevTerms[4]).relinearize()

    t3 = t3 + t1
    t3 = t3.rescale()

    t7 = t7 + t3
    t7 = t7.relinearize()
    t7 = t7 + t7.conjugate()

    return t7


def sign_1(x,stage,r):
    chebyshevTerms = {}
    chebyshevTerms[1] = x
    for i in [2,3,4,5,7,8,16]:
        if i % 2 == 0:
            ct = chebyshevTerms[i//2]*chebyshevTerms[i//2]
            ct = ct.relinearize2()
            ct = ct + ct
            one = getPlaintextInput(f"sign_{r}_ones_{i}_{ct.level()}", ct.scale(), ct.level(),scalar=True)
            ct = ct - one
            ct = ct.rescale()
            chebyshevTerms[i] = ct
        else:
            ct = chebyshevTerms[i//2]
            ct_ = chebyshevTerms[i//2 + 1]
            while ct.level() > ct_.level():
                ct = ct.modswitch()
            ct = ct*ct_
            ct = ct.relinearize2()
            ct = ct + ct

            ct_one = chebyshevTerms[1]
            rescaleFactor = getPlaintextInput(f"sign_{r}_rescale_{i}_{ct_one.level()}",2*scale,ct_one.level(),scalar=True)
            ct_one = ct_one * rescaleFactor
            ct_one = ct_one.rescale()
            while(ct_one.level() != (chebyshevTerms[i//2 + 1].level())):
                ct_one = ct_one.modswitch()
            ct = ct - ct_one
            ct = ct.rescale()
            chebyshevTerms[i] = ct

    ##
    ReluCoeffScale = scale
    coeff13 = getPlaintextInput(f"sign_{r}_13_{chebyshevTerms[3].level()}", ReluCoeffScale, chebyshevTerms[3].level(),scalar=True)
    t13 = (chebyshevTerms[3]*coeff13)

    coeff12 = getPlaintextInput(f"sign_{r}_12_{chebyshevTerms[1].level()}", ReluCoeffScale, chebyshevTerms[1].level(),scalar=True)
    t12 = (chebyshevTerms[1]*coeff12).modswitch().modswitch()

    t13 = t13 + t12


    coeff11 = getPlaintextInput(f"sign_{r}_11_{chebyshevTerms[7].level()}", ReluCoeffScale, chebyshevTerms[7].level(),scalar=True)
    t11 = (chebyshevTerms[7]*coeff11)
    coeff10 = getPlaintextInput(f"sign_{r}_10_{chebyshevTerms[5].level()}", ReluCoeffScale, chebyshevTerms[5].level(),scalar=True)
    t10 = (chebyshevTerms[5]*coeff10)
    coeff9 = getPlaintextInput(f"sign_{r}_9_{chebyshevTerms[3].level()}", ReluCoeffScale, chebyshevTerms[3].level(),scalar=True)
    t9 = (chebyshevTerms[3]*coeff9).modswitch()
    coeff8 = getPlaintextInput(f"sign_{r}_8_{chebyshevTerms[1].level()}", ReluCoeffScale, chebyshevTerms[1].level(),scalar=True)
    t8 = (chebyshevTerms[1]*coeff8).modswitch().modswitch().modswitch()

    t11 = t11 + t10 + t9 + t8

    coeff7 = getPlaintextInput(f"sign_{r}_7_{chebyshevTerms[7].level()}", ReluCoeffScale, chebyshevTerms[7].level(),scalar=True)
    t7 = (chebyshevTerms[7]*coeff7)
    coeff6 = getPlaintextInput(f"sign_{r}_6_{chebyshevTerms[5].level()}", ReluCoeffScale, chebyshevTerms[5].level(),scalar=True)
    t6 = (chebyshevTerms[5]*coeff6)
    coeff5 = getPlaintextInput(f"sign_{r}_5_{chebyshevTerms[3].level()}", ReluCoeffScale, chebyshevTerms[3].level(),scalar=True)
    t5 = (chebyshevTerms[3]*coeff5).modswitch()
    coeff4 = getPlaintextInput(f"sign_{r}_4_{chebyshevTerms[1].level()}", ReluCoeffScale, chebyshevTerms[1].level(),scalar=True)
    t4 = (chebyshevTerms[1]*coeff4).modswitch().modswitch().modswitch()

    t7 = t7 + t6 + t5 + t4

    coeff3 = getPlaintextInput(f"sign_{r}_3_{chebyshevTerms[7].level()}", ReluCoeffScale, chebyshevTerms[7].level(),scalar=True)
    t3 = (chebyshevTerms[7]*coeff3)
    coeff2 = getPlaintextInput(f"sign_{r}_2_{chebyshevTerms[5].level()}", ReluCoeffScale, chebyshevTerms[5].level(),scalar=True)
    t2 = (chebyshevTerms[5]*coeff2)
    coeff1 = getPlaintextInput(f"sign_{r}_1_{chebyshevTerms[3].level()}", ReluCoeffScale, chebyshevTerms[3].level(),scalar=True)
    t1 = (chebyshevTerms[3]*coeff1).modswitch()
    coeff0 = getPlaintextInput(f"sign_{r}_0_{chebyshevTerms[1].level()}", ReluCoeffScale, chebyshevTerms[1].level(),scalar=True)
    t0 = (chebyshevTerms[1]*coeff0).modswitch().modswitch().modswitch()

    t3 = t3 + t2 + t1 + t0
    t3 = t3.modswitch()


    t13 = (t13.rescale() * chebyshevTerms[8]).relinearize()

    t13 = t13 + t11
    t13 = (t13.rescale() * chebyshevTerms[16]).relinearize()

    t7 = t7 * chebyshevTerms[8]
    t7 = t7.relinearize().rescale()

    t13 = t13 + t7 + t3
    t13 = t13 + t13.conjugate()

    return t13


ReluFunction = None
def relu_function_init(gpus):

    relu_function = CeriumFunction("relu",1,0)
    with relu_function:
        x = CiphertextArgument("x",56,21)
        
        rescale_bootstrap = getPlaintextInput(f"rescale_bootstrap_relu_inp",0,x.level(),scalar=True)
        x_rescale = x * rescale_bootstrap

        temp = sign_0(x_rescale.rescale(),1,0)
        temp = sign_0(temp.doubleRescale(),1,1)
        sign = sign_1(temp.doubleRescale(),1,2)
        
        half = getPlaintextInput(f"relu_half_sign_{sign.level()}",sign.scale(),sign.level(),scalar=True)
        pick = sign + half


        rescale_relu = getPlaintextInput(f"rescale_relu_sign_inp_{x.level()}",2*scale,x.level(),scalar=True)
        x_rescale = x * rescale_relu
        x_rescale = x_rescale.doubleRescale()
        while(x_rescale.level() != pick.level()):
            x_rescale = x_rescale.modswitch()
        relu = pick * x_rescale
        relu = relu.relinearize().doubleRescale()

        FunctionOutput("z",relu)


    global ReluFunction
    ReluFunction = relu_function

def relu(cnn, stage):
    global ReluFunction
    assert ReluFunction is not None, "Relu Function Not Initialized. Call relu_function_init first."

    temp = bootstrap(cnn.vec(),"b33",19,Gpus)
    TestOutput(f"relu_{stage}_final_bootstrap", temp)
    outputs = CeriumFunctionCall(ReluFunction,{"x":temp})
    temp = outputs["z"]
    TestOutput(f"relu_{stage}_final", temp)
    return Tensor(logn, cnn.k(), cnn.h(), cnn.w(), cnn.c(), cnn.t(), cnn.p(), temp), temp.level()



def downsample(cnn, stage, level):
    ki = cnn.k()
    hi = cnn.h()
    wi = cnn.w()
    ci = cnn.c()
    ti = cnn.t()
    pi = cnn.p()
    logn = cnn.logn()

    ko = 0
    ho = 0
    wo = 0
    co = 0
    to = 0
    po = 0

    # parameter setting
    n = 1 << logn
    ko = 2*ki
    ho = hi//2
    wo = wi//2
    to = ti//2
    co = 2*ci
    po = pow2(floor_to_int(math.log((n)/(ko*ko*ho*wo*to)) / math.log(2.0)))

    ct = cnn.vec()
    sum_ = None
    select_one_vec = [[None for _ in range(ti)] for _ in range(ki)]
    for w1 in range(ki):
        for w2 in range(ti):
            select_one_vec[w1][w2] = PlaintextInput(f'dsw_{stage}_{w1}_{w2}', 2*scale, ct.level())
    
    pts = []
    steps = []
    for w1 in range(ki):
        for w2 in range(ti):
            pts.append(select_one_vec[w1][w2])
            w3 = ((ki*w2+w1) % (2*ko))//2
            w4 = (ki*w2+w1) % 2
            w5 = (ki*w2+w1)//(2*ko)
            steps.append(ki*ki*hi*wi*w2 + ki*wi*w1 -
                                      ko*ko*ho*wo*w5 - ko*wo*w3 - ki*w4 - ko*ko*ho*wo*(ti//8))

    sum_ = MultiplyRotateAccumulate(ct,pts,steps)
    steps = []
    for u6 in range(po):
        steps.append(-(n//po)*u6)

    sum_ = RotateAccumulate(sum_,steps)
    ct = sum_.doubleRescale()
    TestOutput(f"downsample_{stage}",ct)
    return Tensor(logn, ko, ho, wo, co, to, po, ct), ct.level()

def avg_pool(cnn, level, stage):
    # parameter setting
    ki = cnn.k()
    hi = cnn.h()
    wi = cnn.w()
    ci = cnn.c()
    ti = cnn.t()
    pi = cnn.p()
    logn = cnn.logn()
    ko = 1
    ho = 1
    wo = 1
    co = ci
    to = ti
    n = 1 << logn

    ct = cnn.vec()
    sum_ = None
    temp = None

    scale_factor = getPlaintextInput(f"ap_{stage}_scale_factor", scale, ct.level(),scalar=True) 
    ct = ct * scale_factor

    for x in range(log2_long(wi)):
        temp = ct
        temp = rotate(temp, pow2(x)*ki, n)
        ct = ct + temp

    for x in range(log2_long(hi)):
        temp = ct
        temp = rotate(temp, pow2(x)*ki*ki*wi, n)
        ct = ct + temp

    steps = []
    pts = []
    for s in range(ki):
        for u in range(ti):
            p = ki * u + s
            step = -p*ki + ki*ki*hi*wi*u + ki*wi*s
            steps.append(step)
            select_one = getPlaintextInput(f"ap_{stage}_{s}_{u}", scale, ct.level()) 
            pts.append(select_one)

    sum_ = RotateMultiplyAccumulate(ct,pts,steps)
    sum_ = sum_.doubleRescale()
    # sum_ = sum_.rescale()
    TestOutput("AvgPool",sum_)
    return Tensor(logn, ko, ho, wo, co, to, 1, sum_), sum_.level()


def mat_mul(cnn, level, q, r):
    W = []
    for i in range(q+r-1):
        W.append(PlaintextInput(f"w_{i}", 2*scale, cnn.vec().level()))

    ki = cnn.k()
    hi = cnn.h()
    wi = cnn.w()
    ci = cnn.c()
    ti = cnn.t()
    pi = cnn.p()
    logn = cnn.logn()

    ko = ki
    ho = hi
    wo = wi
    co = ci
    to = ti
    po = pi
    n = 1 << logn
    
    ct = cnn.vec()
    sum_ = None

    steps = []
    pts = []
    for s in range(q + r - 1):
        pts.append(W[s])
        steps.append(r-1-s)

    sum_ = RotateMultiplyAccumulate(ct,pts,steps)
    sum_ = sum_.doubleRescale()
    return Tensor(logn, ko, ho, wo, co, to, po, sum_), sum_.level()

def floor_to_int(x):
    return int(math.floor(x)+0.5)

def pow2(n):
    if n < 0:
        return 1
    return 1 << n

def log2_long(n):
    if n > 65536 or n <= 0:
        raise ValueError("n is too large.")
    d = -1
    for i in range(16):
        if n == 2 ** i:
            d = i
            break
    return d

def rotate(cnn_in, steps, n):
    steps = int((steps+n) % n)
    cnn_out = cnn_in << steps
    return cnn_out

def RotateAccumulate(x,steps):
    rots = HoistedRotate(x,steps)
    sum = rots[0]
    for s in rots[1:]:
        sum += s
    return sum

def resnet_program():
    program = CeriumProgram('resnet', rns_bit_size=RNS_BIT_SIZE,num_gpus=Gpus)
    with program:
        bootstrap_32K_33lvl_function_init(Gpus)
        relu_function_init(Gpus)
        function = CeriumFunction(f"main_{Gpus}gpus",Gpus,0)
        with function:
            res = resnet(20)
    return program

def resnet_program_ngpu():
    os.environ["CERIUM_REDUCTION_REWRITE"] = str(1)
    os.environ["CERIUM_CROSS_CHIP_OPTIMIZE"] = str(1)
    program = CeriumProgram('resnet', rns_bit_size=RNS_BIT_SIZE,num_gpus=Gpus)
    with program:
        bootstrap_32K_33lvl_function_init(Gpus)
        relu_function_init(Gpus)
        function = CeriumFunction(f"main_{Gpus}gpus",Gpus,0)
        with function:
            res = resnet(20)
    return program

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--vregs', type=int, default=8192)
    parser.add_argument('--prefix', type=str, default="outputs/")
    args = parser.parse_args()

    global Gpus 
    Gpus = args.gpus


    if Gpus == 1:
        program = resnet_program()
    else:
        program = resnet_program_ngpu()
        # program = bootstrap_test()
    def get_vregs(fname):
        if fname in ["bootstrap_32K_33"]:
            return 4096
        elif fname in ["relu"]:
            return 1536
        elif fname in ["main"]:
            return 4096
        else: 
            return 1024

    def get_block_size(fname):
        if "main" in fname:
            return 256
        else:
            return 1024

    def get_vregs(fname):
        if "main" in fname:
            return 4096
        elif "bootstrap" in fname:
            return 4096*2
        elif "relu" in fname:
            return 2048

    def get_bcus(fname):
        return 128


    functions = program.get_functions()
    skip = []
    # skip += ["main"]
    # skip += ["bootstrap_32K_33"]
    # skip += ["relu_opt"]

    print_graphs = os.environ.get('CERIUM_PRINT_GRAPHS','0')
    os.environ['CERIUM_PRINT_GRAPHS'] = print_graphs

    for name,f in functions.items():
        compile = True
        for s in skip:
            if s in name:
                compile = False
                break
        if not compile:
            continue
        print(f"Compiling: {name}")
        os.environ['CERIUM_BLOCK_SIZE'] = f"{get_block_size(name)}"
        cerium_compile_function(f,args.gpus,get_vregs(name),get_bcus(name),args.prefix)