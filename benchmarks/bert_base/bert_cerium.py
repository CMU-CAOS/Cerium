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
import os

BertPlaintextInputs = {}
BertCiphertextInputs = {}
TEST=False
SCALAR = True

def TestOutput(name,val):
    if TEST:
        Output(name,val)

def TestFunctionOutput(name,val):
    # if TEST:
    #     FunctionOutput(name,val)
        # Output(name,val)
    pass

def getPlaintextInput(name,scale,level,scalar=False):
    partition_size = CurrentPartitionSize()
    partition_id = CurrentPartitionID()
    key = f"{name}_:p{partition_size}:i{partition_id}"
    if key in BertPlaintextInputs.keys():
        pt = BertPlaintextInputs[key]
        if pt.scale() != scale:
            raise Exception("Mismatched Scale")
        if pt.level() != level:
            raise Exception("Mismatched Level")
        return pt
    pt = PlaintextInput(name,scale,level,scalar)
    BertPlaintextInputs[key] = pt
    return pt

def getPeriodicPlaintextInput(name,scale,level,repeatSize):
    partition_size = CurrentPartitionSize()
    partition_id = CurrentPartitionID()
    key = f"{name}_:p{partition_size}:i{partition_id}"
    if key in BertPlaintextInputs.keys():
        pt = BertPlaintextInputs[key]
        if pt.scale() != scale:
            raise Exception("Mismatched Scale")
        if pt.level() != level:
            raise Exception("Mismatched Level")
        return pt
    pt = PeriodicPlaintextInput(name,scale,level,repeatSize)
    BertPlaintextInputs[key] = pt
    return pt


def getCiphertextInput(name,scale,level):
    partition_size = CurrentPartitionSize()
    partition_id = CurrentPartitionID()
    key = f"{name}_:p{partition_size}:i{partition_id}"
    if key in BertCiphertextInputs.keys():
        ct = BertCiphertextInputs[key]
        if ct.scale() != scale:
            raise Exception("Mismatched Scale")
        if ct.level() != level:
            raise Exception("Mismatched Level")
        return ct
    ct = CiphertextInput(name,scale,level)
    BertCiphertextInputs[key] = ct
    return ct

def matmul_ct128x128_pt128x128(ct,pts):
    babySteps = [i for i in range(16)]
    giantSteps = [i*16 for i in range(-8,8)]
    product = BsgsMultiplyAccumulate(ct,pts,babySteps,giantSteps)
    return product

def matmul_ct128x128_pt128x128_new_type3(ct,pts):
    babySteps = [i for i in range(16)]
    giantSteps = [i*16 for i in range(-16,16)]
    product = BsgsMultiplyAccumulate(ct,pts,babySteps,giantSteps)
    return product

__matmul_ct128x128_ct128x128_transpose_function__ = None
def matmul_ct128x128_ct128x128_transpose_function():
    function = CeriumFunction("matmul_ct128x128_ct128x128_transpose", 1,0)
    with function:
        A = CiphertextArgument("A",84,9)
        B = CiphertextArgument("B",84,9)

        Bs = MakeVector(HoistedRotate(B,[256*i for i in range(8)]))
        Bs = Bs.rescale()

        Arot = MakeVector(HoistedRotate(A,[-2048*i for i in range(16)]))
        Arot = Arot.rescale()
        Arot = BreakVector(Arot)

        mask = PeriodicPlaintextInput(f"matmul_ct128x128_mask",2*SCALE,A.level() - 1,128)

        BabyStepsAccumulated = [None for _ in range(16)]
        for gs in range(16):
            Arot_rescale = Arot[gs]
            dot_prods = Bs * Arot_rescale
            dot_prods = dot_prods.relinearize()
            j = 1
            while j < 64:
                dot_prods = dot_prods + (dot_prods << j)
                j *= 2
            temp = dot_prods * mask
            temp = BreakVector(temp)
            sum_bs = RotateAccumulateMany(temp,[-bs for bs in range(8)])
            BabyStepsAccumulated[gs] = sum_bs
        sum = RotateAccumulateMany(BabyStepsAccumulated,[(2048-8)*gs for gs in range(16)])
        sum = sum.doubleRescale()
        sum = sum.doubleRescale()
        FunctionOutput("sum",sum)

    global __matmul_ct128x128_ct128x128_transpose_function__
    if __matmul_ct128x128_ct128x128_transpose_function__ is None:
        __matmul_ct128x128_ct128x128_transpose_function__ = function

def matmul_ct128x128_ct128x128_transpose(A,B):
    assert __matmul_ct128x128_ct128x128_transpose_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(
        __matmul_ct128x128_ct128x128_transpose_function__,{"A":A,"B":B},)
    return outputs["sum"]

def g3_f3(x,coeffs,relinType=2):

    def relineraize(t):
        if relinType==2:
            return t.relinearize2()
        else:
            return t.relinearize()

    pows = [None for _ in range(7)]
    terms = [None for _ in range(8)]
    pows[1] = x
    for i in range(1,3):
        xx = pows[2**(i-1)]
        xx = (xx * xx)
        xx = relineraize(xx)
        xx = xx.rescale()
        pows[2**i] = xx

    x7 = (pows[1] * coeffs[7]).rescale()
    x7 = (x7 * pows[2])
    x7 = relineraize(x7)
    x7 = x7.rescale()
    x7 = (x7 * pows[4])
    x7 = relineraize(x7)

    x5 = (pows[1] * coeffs[5]).doubleRescale()
    x5 = (x5 * pows[4])
    x5 = relineraize(x5)

    x3 = (pows[1] * coeffs[3]).doubleRescale()
    x3 = (x3 * pows[2].modswitch())
    x3 = relineraize(x3)

    x1 = (pows[1] * coeffs[1]).rescale()
    x1 = x1.modswitch()

    x = x1 + x3 + x5 + x7
    x = x + x.conjugate2()
    return x

def g3_f3_new(x,coeffs,relinType=1):

    def relineraize(t):
        if relinType==2:
            return t.relinearize2()
        else:
            return t.relinearize()

    pows = [None for _ in range(7)]
    terms = [None for _ in range(8)]
    pows[1] = x
    for i in range(1,3):
        xx = pows[2**(i-1)]
        xx = (xx * xx)
        xx = relineraize(xx)
        xx = xx.doubleRescale()
        pows[2**i] = xx

    x7 = (pows[1] * coeffs[7]).doubleRescale()
    x7 = (x7 * pows[2])
    x7 = relineraize(x7)
    x7 = x7.doubleRescale()
    x7 = (x7 * pows[4])
    x7 = relineraize(x7)

    x5 = (pows[1] * coeffs[5]).doubleRescale()
    x5 = x5.modswitch().modswitch()
    x5 = (x5 * pows[4])
    x5 = relineraize(x5)

    x3 = (pows[1] * coeffs[3]).doubleRescale()
    x3 = x3.modswitch().modswitch()
    x3 = (x3 * pows[2].modswitch().modswitch())
    x3 = relineraize(x3)

    x1 = (pows[1] * coeffs[1])#.rescale().rescale()
    x1 = x1.modswitch().modswitch().modswitch().modswitch()

    x = x1 + x3 + x5 + x7
    x = x + x.conjugate()
    return x

def max_vec(A,B,i):
    a_plus_b = A + B
    a_minus_b = A - B

    a_minus_b_rescale = BreakVector(a_minus_b.rescale())
    s = [sign_short(a_minus_b_rescale[i],"max") for i in range(2)]
    s = MakeVector(s)

    zeroPt5_0 = MakeVector([getPeriodicPlaintextInput("zeroPt5_0",2*SCALE,a_plus_b.level(),repeatSize=1),getPeriodicPlaintextInput("zeroPt5_0_im",2*SCALE,a_plus_b.level(),repeatSize=1)])
    a_plus_b_half =(a_plus_b * zeroPt5_0)
    zeroPt5_1 = MakeVector([getPeriodicPlaintextInput("zeroPt5_1",2*SCALE,a_minus_b.level(),repeatSize=1),getPeriodicPlaintextInput("zeroPt5_1_im",2*SCALE,a_minus_b.level(),repeatSize=1)])
    a_minus_b_half =(a_minus_b * zeroPt5_1).doubleRescale()
    while a_minus_b_half.level() != s.level():
        a_minus_b_half = a_minus_b_half.modswitch()
    
    result = (a_minus_b_half * s)
    result = BreakVector(result)
    result = [r.relinearize2() for r in result]
    result = MakeVector(result)
    while a_plus_b_half.level() != result.level():
        a_plus_b_half = a_plus_b_half.modswitch()
    result = result + a_plus_b_half
    result = result.doubleRescale()
    return result

def array_max_vec(A):
    x = A
    for i in range(7):
        s = BreakVector(x)
        s = [mm.rotate2(-(2**i)) for mm in s]
        s = MakeVector(s)
        s = max_vec(s,x,i)
        x = bootstrap(s,"b35_2",14)
        scaleFactor = PlaintextInput("array_max_bootstrap_scale_factor",0,x.level(),SCALAR)
        x = x * scaleFactor
        TestFunctionOutput(f"array_max_{i}",x)
    forty = PeriodicPlaintextInput("array_max2_forty",2*SCALE,x.level(),128)
    x = x * forty
    for i in range(7):
        x = x + (x << 2**i)
    x = x.doubleRescale()
    return x

def exp(A,outputDepth):
    iters = 6
    oneBy2N = PlaintextInput("exp_oneBy2N",2*SCALE,A.level(),SCALAR)
    x = A * oneBy2N
    x = x.doubleRescale()
    one = PlaintextInput("exp_one",x.scale(),x.level(),SCALAR)
    x = one + x

    for i in range(5):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()

    x = bootstrap(x,"b35",outputDepth)
    bootstrap_sf = PlaintextInput("exp_bootstrap_sf",0,x.level(),SCALAR)
    x = x * bootstrap_sf
    for _ in range(1):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()

    return x

def inverse_sqrt(x,outDepth,prefix):
    xInit = x
    minusZeroPtFour = PlaintextInput(f"{prefix}_isqrt_minus_zero_pt_4",x.scale(),x.level(),SCALAR)
    x = minusZeroPtFour - x 
    oneBy2N = PlaintextInput(f"{prefix}_isqrt_oneBy2N",2*SCALE,x.level(),SCALAR)
    x = x * oneBy2N
    x = x.doubleRescale()
    one = PlaintextInput(f"{prefix}_isqrt_one",x.scale(),x.level(),SCALAR)
    x = one + x

    for i in range(5):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()

    x = bootstrap(x,"b35",10)
    bootstrap_sf = PlaintextInput(f"{prefix}_isqrt_exp_bootstrap_sf",0,x.level(),SCALAR)
    x = x * bootstrap_sf
    for _ in range(1):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()
    y = x + x
    zeroPt2 = PlaintextInput(f"{prefix}_isqrt_zeroPt2",y.scale(),y.level(),SCALAR)
    y = y + zeroPt2

    x = xInit
    for i in range(6):
        y2 = (y * y).relinearize()
        y2 = y2.doubleRescale()

        x_ = x
        y_ = y
        while x_.level() > y_.level() + 2:
            x_ = x_.modswitch()
        xrescale = PlaintextInput(f"{prefix}_isqrt_xrescale_{i}",2*SCALE,x_.level(),SCALAR)
        x_ = (x_ * xrescale).doubleRescale()
        xy = (x_ * y_).relinearize()
        xy = xy.doubleRescale()
        xy3 = (xy * y2).relinearize()

        yrescale = PlaintextInput(f"{prefix}_isqrt_yrescale_{i}",2*SCALE,y.level(),SCALAR)
        y_ = y * yrescale
        y_ = y_.modswitch().modswitch()

        y = y_ - xy3

        y = y + y.conjugate()
        y = y.doubleRescale()
        if y.level() == 2: 
            TestFunctionOutput(f"isqrt_y_bs{i}",y)
            if i != 5: 
                y = bootstrap(y,"b35",8)
            else:
                y = bootstrap(y,"b35",outDepth)
        else:
            TestFunctionOutput(f"isqrt_y_{i}",y)

    bootstrap_sf = PlaintextInput(f"{prefix}_isqrt_y_bootstrap_sf",0,y.level(),SCALAR)
    y = y * bootstrap_sf
    return y

def inverse_sqrt2(x,outDepth,prefix):
    xInit = x
    zeroPt2 = PlaintextInput(f"{prefix}_isqrt_zeroPt2",2*SCALE,10,SCALAR)
    y = zeroPt2

    x = xInit
    for i in range(6):

        if i == 0: 
            y2 = PlaintextInput(f"{prefix}_isqrt_zeroPt04",2*SCALE,y.level()-2,SCALAR)
        else:
            y2 = (y * y).relinearize()
            y2 = y2.doubleRescale()

        x_ = x
        y_ = y
        while x_.level() > y_.level() + 2:
            x_ = x_.modswitch()
        xrescale = PlaintextInput(f"{prefix}_isqrt_xrescale_{i}",2*SCALE,x_.level(),SCALAR)
        x_ = (x_ * xrescale).doubleRescale()
        xy = (x_ * y_)
        if i != 0:
            xy = xy.relinearize()
        xy = xy.doubleRescale()
        xy3 = (xy * y2)
        if i != 0:
            xy3 = xy3.relinearize()

        if i == 0:
            yrescale = PlaintextInput(f"{prefix}_isqrt_yrescale_{i}",4*SCALE,y.level(),SCALAR)
            y_ = yrescale
        else:
            yrescale = PlaintextInput(f"{prefix}_isqrt_yrescale_{i}",2*SCALE,y.level(),SCALAR)
            y_ = y * yrescale
        y_ = y_.modswitch().modswitch()

        y = y_ - xy3

        y = y + y.conjugate()
        y = y.doubleRescale()
        if y.level() == 2: 
            TestFunctionOutput(f"isqrt_y_bs{i}",y)
            if i != 5: 
                y = bootstrap(y,"b35",8)
            else:
                y = bootstrap(y,"b35",outDepth)
        else:
            TestFunctionOutput(f"isqrt_y_{i}",y)

    bootstrap_sf = PlaintextInput(f"{prefix}_isqrt_y_bootstrap_sf",0,y.level(),SCALAR)
    y = y * bootstrap_sf
    return y

__softmax_128x128_vec_function__ = None
def softmax_128x128_vec_function(suffix=""):
    function = CeriumFunction(f"softmax_128x128_vec{suffix}",1,0)
    with function:
        A0 = CiphertextArgument("A0",2*SCALE,16)
        A1 = CiphertextArgument("A1",2*SCALE,16)
        AVec = [A0,A1]
        mVec = array_max_vec(MakeVector(AVec))
        A = MakeVector(AVec)
        m = mVec
        forty = PlaintextInput("softmax_40",2*SCALE,A.level(),SCALAR)
        A = A*forty
        A = A.doubleRescale()
        A = A - m
        TestFunctionOutput(f"max_values_sub",A)
        Abreak = BreakVector(A)
        e = MakeVector([exp(a,11) for a in Abreak])
        TestFunctionOutput(f"exp",e)
        att_mask = getCiphertextInput("att_mask",2*SCALE,e.level())
        e = (e * att_mask).relinearize()
        sum = e
        e = e.doubleRescale()
        TestFunctionOutput(f"exp_att_mask",e)
        for i in range(7):
            sum = sum + (sum << 2**i)
        sum = sum.rescale()
        one = MakeVector([PeriodicPlaintextInput("softmax_one",2*SCALE,sum.level(),128),PeriodicPlaintextInput("softmax_one_im",2*SCALE,sum.level(),128)])
        sum = sum * one
        for i in range(7):
            sum = sum + (sum >> 2**i)
        sum = sum.doubleRescale().rescale()
        TestFunctionOutput(f"softmax_sum_bs",sum)
        sum = bootstrap(sum,"b35_2",12)
        sum_sf = PlaintextInput("softmax_sum_sf",0,sum.level(),SCALAR)
        sum = sum * sum_sf
        TestFunctionOutput(f"softmax_sum",sum)
        sum = BreakVector(sum)
        isqrt = [inverse_sqrt2(sum_,9,"softmax") for sum_ in sum]
        isqrt = MakeVector(isqrt)
        inv = (isqrt * isqrt).relinearize().doubleRescale()
        TestFunctionOutput(f"inverse",inv)
        s = e * inv
        s = s.relinearize()
        s = s.rescale()
        s = BreakVector(s)
        for n in range(2):
            FunctionOutput(f"s{n}",s[n])

    global __softmax_128x128_vec_function__
    __softmax_128x128_vec_function__ = function



def softmax_128x128_vec(A0,A1):
    assert __softmax_128x128_vec_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__softmax_128x128_vec_function__,{"A0":A0,"A1":A1},)
    s0 = outputs["s0"]
    s1 = outputs["s1"]
    return (s0,s1)

__softmax_128x128_vec2_function__ = None
def softmax_128x128_vec2_function():
    function = CeriumFunction(f"softmax_128x128_vec2",2,0)
    with function:
        Partition(1,1)
        A1 = CiphertextArgument("A1",2*SCALE,16)
        Partition(2,0)
        A1 = Receive(A1)
        Partition(1,0)
        A0 = CiphertextArgument("A0",2*SCALE,16)
        A1 = Receive(A1)
        AVec = [A0,A1]
        mVec = array_max_vec(MakeVector(AVec))
        A = MakeVector(AVec)
        m = mVec
        forty = PlaintextInput("softmax_40",2*SCALE,A.level(),SCALAR)
        A = A*forty
        A = A.doubleRescale()
        A = A - m
        TestFunctionOutput(f"max_values_sub",A)
        Abreak = BreakVector(A)
        e = MakeVector([exp(a,11) for a in Abreak])
        TestFunctionOutput(f"exp",e)
        att_mask = getCiphertextInput("att_mask",2*SCALE,e.level())
        e = (e * att_mask).relinearize()
        sum = e
        e = e.doubleRescale()
        TestFunctionOutput(f"exp_att_mask",e)
        for i in range(7):
            sum = sum + (sum << 2**i)
        sum = sum.rescale()
        one = MakeVector([PeriodicPlaintextInput("softmax_one",2*SCALE,sum.level(),128),PeriodicPlaintextInput("softmax_one_im",2*SCALE,sum.level(),128)])
        sum = sum * one
        for i in range(7):
            sum = sum + (sum >> 2**i)
        sum = sum.doubleRescale().rescale()
        TestFunctionOutput(f"softmax_sum_bs",sum)
        sum = bootstrap(sum,"b35_2",12)
        sum_sf = PlaintextInput("softmax_sum_sf",0,sum.level(),SCALAR)
        sum = sum * sum_sf
        TestFunctionOutput(f"softmax_sum",sum)
        sum = BreakVector(sum)
        isqrt = [inverse_sqrt2(sum_,9,"softmax") for sum_ in sum]
        isqrt = MakeVector(isqrt)
        inv = (isqrt * isqrt).relinearize().doubleRescale()
        TestFunctionOutput(f"inverse",inv)
        s = e * inv
        s = s.relinearize()
        s = BreakVector(s)
        s[0] = s[0].rescale()
        FunctionOutput(f"s{0}",s[0])
        Partition(2,0)
        s[1] = Receive(s[1])
        Partition(1,1)
        s[1] = s[1].rescale()
        FunctionOutput(f"s{1}",Receive(s[1]))

    global __softmax_128x128_vec2_function__
    __softmax_128x128_vec2_function__ = function



def softmax_128x128_vec2(A0,A1):
    assert __softmax_128x128_vec2_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__softmax_128x128_vec2_function__,{"A0":A0,"A1":A1},)
    s0 = outputs["s0"]
    s1 = outputs["s1"]
    return (s0,s1)


__matmul_ct128x128_ct128x128_function__ = None
def matmul_ct128x128_ct128x128_function():
    function = CeriumFunction("matmul_ct128x128_ct128x128",1,0)
    with function:
        ct1 = CiphertextArgument("ct1",3*SCALE,8)
        ct2 = CiphertextArgument("ct2",3*SCALE,9)
        babySteps = [i*256 for i in range(8)]
        giantSteps = [i*256*8 for i in range(0,16)]
        pts = [PeriodicPlaintextInput(f"attention_matmul_mask_128x128_rot_{i*128}",SCALE+SCALE//2,ct2.level(),128) for i in range(0,128)]
        diag = BsgsMultiplyAccumulate(ct2,pts,babySteps,giantSteps,rescaleLevels=1)

        ct1_gs = HoistedRotate(ct1,[(8*gs -2048*gs) for gs in range(16)])
        diag_bs = HoistedRotate(diag, [(256*bs) for bs in range(8)])

        diag_bs_vec = MakeVector(diag_bs)
        diag_bs_vec = diag_bs_vec.doubleRescale()

        babysteps_accumulated = [None for _ in range(16)]

        for gs in range(0,128//8,1):
            l = MakeVector(HoistedRotate(ct1_gs[gs],[bs for bs in range(8)]))
            r = MakeVector(HoistedRotate(ct1_gs[gs],[(-128 + bs) for bs in range(8)]))
            diagRot = diag_bs_vec
            maskL = MakeVector([PeriodicPlaintextInput(f"attention_matmul_maskL_128x128_{8*gs + bs}",SCALE+SCALE//2,l.level(),128) for bs in range(8)])
            maskR = MakeVector([PeriodicPlaintextInput(f"attention_matmul_maskR_128x128_{8*gs + bs}",SCALE+SCALE//2,r.level(),128) for bs in range(8)])
            vec = l*maskL + r*maskR
            vec = vec.doubleRescale()
            vec = vec * diagRot
            vec = BreakVector(vec)
            sumbs = vec[0]
            for bs in range(1,8):
                sumbs += vec[bs]
            babysteps_accumulated[gs] = sumbs
        
        babysteps_accumulated = MakeVector(babysteps_accumulated)
        babysteps_accumulated = babysteps_accumulated.relinearize()
        babysteps_accumulated = BreakVector(babysteps_accumulated)

        sumi = RotateAccumulateMany(babysteps_accumulated,[256*8*gs for gs in range(16)])

        sumi = sumi.rescale()
        FunctionOutput("sumi",sumi)
    
    global __matmul_ct128x128_ct128x128_function__
    __matmul_ct128x128_ct128x128_function__ = function

def matmul_ct128x768_pt768x128_new(ct,pt,mask,maskL,maskR):
    result0 = matmul_ct128x128_pt128x128(ct[0],pt[0])
    result1 = matmul_ct128x128_pt128x128(ct[0],pt[3])
    for i in range(1,3):
        product0 = matmul_ct128x128_pt128x128(ct[i],pt[i])
        result0 += product0
        product1 = matmul_ct128x128_pt128x128(ct[i],pt[i+3])
        result1 += product1
    L = (result1 << 128)*maskL
    R = (result1 >> 128)*maskR
    result1 = L + R
    result0 = result0 * mask
    return result0 + result1

    
    
def matmul_ct128x128_ct128x128(prefix,ct1,ct2):
    assert __matmul_ct128x128_ct128x128_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__matmul_ct128x128_ct128x128_function__,{"ct1":ct1,"ct2":ct2},)
    sumi = outputs["sumi"]
    return sumi

__attention_qkv_function_1gpu__ = None
def attention_qkv_function_1gpu():
    function = CeriumFunction("attention_qkv_1gpu",1,0)
    with function:
        H = [CiphertextArgument(f"H{i}",3*SCALE,13) for i in range(3)]
        NUM = 12
        idx = 0
        Q = [None for _ in range(NUM//2)]
        K = [None for _ in range(NUM//2)]
        V = [None for _ in range(NUM//2)]

        bsgsMask  = PlaintextInput(f"attention_bsgs_Mask_{H[0].level()}",2*SCALE,H[0].level())
        bsgsMaskL = PlaintextInput(f"attention_bsgs_MaskL_{H[0].level()}",2*SCALE,H[0].level())
        bsgsMaskR = PlaintextInput(f"attention_bsgs_MaskR_{H[0].level()}",2*SCALE,H[0].level())

        for num in range(NUM//2):
            n = (NUM//2)*idx + num
            Wq = [[PeriodicPlaintextInput(f"attention_Wq{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
            q = matmul_ct128x768_pt768x128_new(H,Wq,bsgsMask,bsgsMaskL,bsgsMaskR)
            q = q.doubleRescale().doubleRescale()

            Bq = PeriodicPlaintextInput(f"attention_Bq{n}",q.scale(),q.level(),256,remapable=True)
            q = q + Bq
            FunctionOutput(f"Q{n}",q)
            Q[num] = q
        
            Wk = [[PeriodicPlaintextInput(f"attention_Wk{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
            k = matmul_ct128x768_pt768x128_new(H,Wk,bsgsMask,bsgsMaskL,bsgsMaskR)
            k = k.doubleRescale().doubleRescale()
            
            Bk = PeriodicPlaintextInput(f"attention_Bk{n}",k.scale(),k.level(),256,remapable=True)
            k = k + Bk
            FunctionOutput(f"K{n}",k)
            K[num] = k

            Wv = [[PeriodicPlaintextInput(f"attention_Wv{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
            v = matmul_ct128x768_pt768x128_new(H,Wv,bsgsMask,bsgsMaskL,bsgsMaskR)
            v = v.doubleRescale().doubleRescale()

            Bv = PeriodicPlaintextInput(f"attention_Bv{n}",v.scale(),v.level(),256,remapable=True)
            v = v + Bv
            FunctionOutput(f"V{n}",v)
            V[num] = v


    global __attention_qkv_function_1gpu__
    __attention_qkv_function_1gpu__ = function

def attention_qkv_1gpu(H,idx,NUM,block):
    assert __attention_qkv_function_1gpu__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_qkv_function_1gpu__,{f"H{i}":H[i] for i in range(3)},block)
    Q = [outputs[f"Q{(NUM//2)*idx + num}"] for num in range(NUM//2)]
    K = [outputs[f"K{(NUM//2)*idx + num}"] for num in range(NUM//2)]
    V = [outputs[f"V{(NUM//2)*idx + num}"] for num in range(NUM//2)]
    return Q,K,V

__attention_qkv_function_2gpu__ = None
def attention_qkv_function_2gpu():
    function = CeriumFunction("attention_qkv_2gpu",2,0)
    with function:

        H = [None for _ in range(3)]
        def stream_fn1(sid,H):
            if sid > 0: return
            for i in range(3): H[i] = CiphertextArgument(f"H{i}",3*SCALE,13)

        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn1,H)

        H = [Receive(H[i]) for i in range(3)]

        def stream_fn2(sid,Q,K,V):
            NUM = 6
            idx = sid

            bsgsMask  = PlaintextInput(f"attention_bsgs_Mask_{H[0].level()}",2*SCALE,H[0].level())
            bsgsMaskL = PlaintextInput(f"attention_bsgs_MaskL_{H[0].level()}",2*SCALE,H[0].level())
            bsgsMaskR = PlaintextInput(f"attention_bsgs_MaskR_{H[0].level()}",2*SCALE,H[0].level())

            for num in range(NUM//2):
                n = (NUM//2)*idx + num
                Wq = [[PeriodicPlaintextInput(f"attention_Wq{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                q = matmul_ct128x768_pt768x128_new(H,Wq,bsgsMask,bsgsMaskL,bsgsMaskR)
                q = q.doubleRescale().doubleRescale()

                Bq = PeriodicPlaintextInput(f"attention_Bq{n}",q.scale(),q.level(),256,remapable=True)
                q = q + Bq
                FunctionOutput(f"Q{n}",q)
                Q[n] = q
            
                Wk = [[PeriodicPlaintextInput(f"attention_Wk{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                k = matmul_ct128x768_pt768x128_new(H,Wk,bsgsMask,bsgsMaskL,bsgsMaskR)
                k = k.doubleRescale().doubleRescale()
                
                Bk = PeriodicPlaintextInput(f"attention_Bk{n}",k.scale(),k.level(),256,remapable=True)
                k = k + Bk
                FunctionOutput(f"K{n}",k)
                K[n] = k

                Wv = [[PeriodicPlaintextInput(f"attention_Wv{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                v = matmul_ct128x768_pt768x128_new(H,Wv,bsgsMask,bsgsMaskL,bsgsMaskR)
                v = v.doubleRescale().doubleRescale()

                Bv = PeriodicPlaintextInput(f"attention_Bv{n}",v.scale(),v.level(),256,remapable=True)
                v = v + Bv
                FunctionOutput(f"V{n}",v)
                V[n] = v

        Q = [None for _ in range(12)]
        K = [None for _ in range(12)]
        V = [None for _ in range(12)]

        streamSize = 1
        numStreams = 2
        CeriumStream(streamSize,numStreams,stream_fn2,Q,K,V)

    global __attention_qkv_function_2gpu__
    __attention_qkv_function_2gpu__ = function

def attention_qkv_2gpu(H,block):
    assert __attention_qkv_function_2gpu__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_qkv_function_2gpu__,{f"H{i}":H[i] for i in range(3)},block)
    Q = [outputs[f"Q{i}"] for i in range(6)]
    K = [outputs[f"K{i}"] for i in range(6)]
    V = [outputs[f"V{i}"] for i in range(6)]
    return Q,K,V

__attention_qkv_function_4gpu__ = None
def attention_qkv_function_4gpu():
    function = CeriumFunction("attention_qkv_4gpu",4,0)
    with function:

        H = [None for _ in range(3)]
        def stream_fn1(sid,H):
            if sid > 0: return
            for i in range(3): H[i] = CiphertextArgument(f"H{i}",3*SCALE,13)

        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn1,H)

        H = [Receive(H[i]) for i in range(3)]

        def stream_fn2(sid,Q,K,V):
            NUM = 4
            idx = sid

            bsgsMask  = PlaintextInput(f"attention_bsgs_Mask_{H[0].level()}",2*SCALE,H[0].level())
            bsgsMaskL = PlaintextInput(f"attention_bsgs_MaskL_{H[0].level()}",2*SCALE,H[0].level())
            bsgsMaskR = PlaintextInput(f"attention_bsgs_MaskR_{H[0].level()}",2*SCALE,H[0].level())

            for num in range(NUM//2):
                n = (NUM//2)*idx + num
                Wq = [[PeriodicPlaintextInput(f"attention_Wq{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                q = matmul_ct128x768_pt768x128_new(H,Wq,bsgsMask,bsgsMaskL,bsgsMaskR)
                q = q.doubleRescale().doubleRescale()

                Bq = PeriodicPlaintextInput(f"attention_Bq{n}",q.scale(),q.level(),256,remapable=True)
                q = q + Bq
                FunctionOutput(f"Q{n}",q)
                Q[n] = q
            
                Wk = [[PeriodicPlaintextInput(f"attention_Wk{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                k = matmul_ct128x768_pt768x128_new(H,Wk,bsgsMask,bsgsMaskL,bsgsMaskR)
                k = k.doubleRescale().doubleRescale()
                
                Bk = PeriodicPlaintextInput(f"attention_Bk{n}",k.scale(),k.level(),256,remapable=True)
                k = k + Bk
                FunctionOutput(f"K{n}",k)
                K[n] = k

                Wv = [[PeriodicPlaintextInput(f"attention_Wv{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                v = matmul_ct128x768_pt768x128_new(H,Wv,bsgsMask,bsgsMaskL,bsgsMaskR)
                v = v.doubleRescale().doubleRescale()

                Bv = PeriodicPlaintextInput(f"attention_Bv{n}",v.scale(),v.level(),256,remapable=True)
                v = v + Bv
                FunctionOutput(f"V{n}",v)
                V[n] = v

        Q = [None for _ in range(12)]
        K = [None for _ in range(12)]
        V = [None for _ in range(12)]

        streamSize = 1
        numStreams = 3
        CeriumStream(streamSize,numStreams,stream_fn2,Q,K,V)

    global __attention_qkv_function_4gpu__
    __attention_qkv_function_4gpu__ = function

def attention_qkv_4gpu(H,block):
    assert __attention_qkv_function_4gpu__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_qkv_function_4gpu__,{f"H{i}":H[i] for i in range(3)},block)
    Q = [outputs[f"Q{i}"] for i in range(6)]
    K = [outputs[f"K{i}"] for i in range(6)]
    V = [outputs[f"V{i}"] for i in range(6)]
    return Q,K,V


__attention_qkv_function_8gpu__ = None
def attention_qkv_function_8gpu():
    function = CeriumFunction("attention_qkv_8gpu",8,0)
    with function:

        H = [None for _ in range(3)]
        def stream_fn1(sid,H):
            if sid > 0: return
            for i in range(3): H[i] = CiphertextArgument(f"H{i}",3*SCALE,13)

        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn1,H)

        H = [Receive(H[i]) for i in range(3)]

        def stream_fn2(sid,Q,K,V):
            NUM = 2
            idx = sid

            bsgsMask  = PlaintextInput(f"attention_bsgs_Mask_{H[0].level()}",2*SCALE,H[0].level())
            bsgsMaskL = PlaintextInput(f"attention_bsgs_MaskL_{H[0].level()}",2*SCALE,H[0].level())
            bsgsMaskR = PlaintextInput(f"attention_bsgs_MaskR_{H[0].level()}",2*SCALE,H[0].level())

            for num in range(NUM//2):
                n = (NUM//2)*idx + num
                Wq = [[PeriodicPlaintextInput(f"attention_Wq{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                q = matmul_ct128x768_pt768x128_new(H,Wq,bsgsMask,bsgsMaskL,bsgsMaskR)
                q = q.doubleRescale().doubleRescale()

                Bq = PeriodicPlaintextInput(f"attention_Bq{n}",q.scale(),q.level(),256,remapable=True)
                q = q + Bq
                FunctionOutput(f"Q{n}",q)
                Q[n] = q
            
                Wk = [[PeriodicPlaintextInput(f"attention_Wk{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                k = matmul_ct128x768_pt768x128_new(H,Wk,bsgsMask,bsgsMaskL,bsgsMaskR)
                k = k.doubleRescale().doubleRescale()
                
                Bk = PeriodicPlaintextInput(f"attention_Bk{n}",k.scale(),k.level(),256,remapable=True)
                k = k + Bk
                FunctionOutput(f"K{n}",k)
                K[n] = k

                Wv = [[PeriodicPlaintextInput(f"attention_Wv{n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128,128)] for j in range(6)]
                v = matmul_ct128x768_pt768x128_new(H,Wv,bsgsMask,bsgsMaskL,bsgsMaskR)
                v = v.doubleRescale().doubleRescale()

                Bv = PeriodicPlaintextInput(f"attention_Bv{n}",v.scale(),v.level(),256,remapable=True)
                v = v + Bv
                FunctionOutput(f"V{n}",v)
                V[n] = v

        Q = [None for _ in range(12)]
        K = [None for _ in range(12)]
        V = [None for _ in range(12)]

        streamSize = 1
        numStreams = 6
        CeriumStream(streamSize,numStreams,stream_fn2,Q,K,V)

    global __attention_qkv_function_8gpu__
    __attention_qkv_function_8gpu__ = function

def attention_qkv_8gpu(H,block):
    assert __attention_qkv_function_8gpu__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_qkv_function_8gpu__,{f"H{i}":H[i] for i in range(3)},block)
    Q = [outputs[f"Q{i}"] for i in range(6)]
    K = [outputs[f"K{i}"] for i in range(6)]
    V = [outputs[f"V{i}"] for i in range(6)]
    return Q,K,V


__attention_o_function_1gpu__ = None
def attention_o_function_1gpu():
    function = CeriumFunction("attention_o_1gpu",1,0)
    with function:


        ContextLayerStacked = [CiphertextArgument(f"ContextLayerStacked{i}",3*SCALE,5) for i in range(3)]
        Wo = [[[PeriodicPlaintextInput(f"attention_Wo_{n}_{j}_{i}",2*SCALE,ContextLayerStacked[0].level(),256,remapable=True) for i in range(-128 * (1 + (j // 3)),128 * (1 + (j // 3)))] for j in range(6)] for n in range(3)]
        O = matmul_ct128x768_pt768x768_new(ContextLayerStacked,Wo)

        for n in range(3):
            o = O[n]
            o = o.doubleRescale()
            Bo = PeriodicPlaintextInput(f"attention_Bo_{n}",o.scale(),o.level(),256,remapable=True)
            o = o + Bo
            FunctionOutput(f"O{n}",o)


    global __attention_o_function_1gpu__
    __attention_o_function_1gpu__ = function

def attention_o_1gpu(ContextLayerStacked,idx,NUM,block):
    assert __attention_o_function_1gpu__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_o_function_1gpu__,{f"ContextLayerStacked{i}":ContextLayerStacked[i] for i in range(NUM//4)},block)
    O = [outputs[f"O{(NUM//4)*idx + num}"] for num in range(NUM//4)]
    return O

__attention_o_function_2gpu__ = None
def attention_o_function_2gpu():
    function = CeriumFunction("attention_o_2gpu",2,0)
    with function:

        ContextLayerStacked = [None for _ in range(4)]
        def stream_fn(sid,ContextLayerStacked):
            if sid >= 2:
                return
            ContextLayer = [CiphertextArgument(f"ContextLayer{3*sid + i}",3*SCALE,5) for i in range(3)]

            if sid == 0:
                ContextLayerStacked[0] = ContextLayer[0] + (ContextLayer[1] << -64)
                ContextLayerStacked[1] = ContextLayer[2]
            elif sid == 1:
                ContextLayerStacked[2] = ContextLayer[1] + (ContextLayer[2] << -64)
                ContextLayerStacked[3] = (ContextLayer[0] << -64)

        streamSize = 1
        numStreams = 2
        CeriumStream(streamSize,numStreams,stream_fn,ContextLayerStacked)

        ContextLayerStacked[1] += ContextLayerStacked[3]
        Wo = [[[PeriodicPlaintextInput(f"attention_Wo_{n}_{j}_{i}",2*SCALE,ContextLayerStacked[0].level(),256,remapable=True) for i in range(-128 * (1 + (j // 3)),128 * (1 + (j // 3)))] for j in range(6)] for n in range(3)]
        O = matmul_ct128x768_pt768x768_new(ContextLayerStacked,Wo)

        A= [None for _ in range(3)]

        def stream_fn(sid):
            if sid > 0: return
            
            H = [CiphertextArgument(f"H{i}",3*SCALE,13) for i in range(3)]


            for n in range(3):
                o = O[n]
                o = o.doubleRescale()
                Bo = PeriodicPlaintextInput(f"attention_Bo_{n}",o.scale(),o.level(),256,remapable=True)
                O[n] = o + Bo

                h1 = H[n]
                h1_rescale_factor = PlaintextInput("attention_h1_rescale_factor",3*SCALE,h1.level(),SCALAR)
                h1 = h1 * h1_rescale_factor
                h1 = h1.doubleRescale().rescale()

                while h1.level() != o.level():
                    h1 = h1.modswitch()
                h2 = o + h1
                h2 = h2.rescale()
                A[n] = h2

                FunctionOutput(f"O{n}",O[n])
                FunctionOutput(f"A{n}",A[n])
        
        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn)


    global __attention_o_function_2gpu__
    __attention_o_function_2gpu__ = function

def attention_o_2gpu(H,ContextLayer,block):
    assert __attention_o_function_2gpu__ is not None, "Function not defined"
    inputs = {f"ContextLayer{i}": ContextLayer[i] for i in range(6)}
    inputs.update({f"H{i}":H[i] for i in range(3)})
    outputs = CeriumFunctionCall(__attention_o_function_2gpu__,inputs,block)
    O = [outputs[f"O{i}"] for i in range(3)]
    A = [outputs[f"A{i}"] for i in range(3)]
    return O,A


__attention_o_function_4gpu__ = None
def attention_o_function_4gpu():
    function = CeriumFunction("attention_o_4gpu",4,0)
    with function:

        ContextLayerStacked = [None for _ in range(3)]
        def stream_fn(sid,ContextLayerStacked):
            if sid > 2:
                return
            ContextLayer = [CiphertextArgument(f"ContextLayer{2*sid + i}",3*SCALE,5) for i in range(2)]

            ContextLayerStacked[sid] = ContextLayer[0] + (ContextLayer[1] << -64)

        streamSize = 1
        numStreams = 3
        CeriumStream(streamSize,numStreams,stream_fn,ContextLayerStacked)

        Wo = [[[PeriodicPlaintextInput(f"attention_Wo_{n}_{j}_{i}",2*SCALE,ContextLayerStacked[0].level(),256,remapable=True) for i in range(-128 * (1 + (j // 3)),128 * (1 + (j // 3)))] for j in range(6)] for n in range(3)]
        O = matmul_ct128x768_pt768x768_new(ContextLayerStacked,Wo)

        A= [None for _ in range(3)]

        def stream_fn(sid):
            if sid > 0: return
            
            H = [CiphertextArgument(f"H{i}",3*SCALE,13) for i in range(3)]


            for n in range(3):
                o = O[n]
                o = o.doubleRescale()
                Bo = PeriodicPlaintextInput(f"attention_Bo_{n}",o.scale(),o.level(),256,remapable=True)
                O[n] = o + Bo

                h1 = H[n]
                h1_rescale_factor = PlaintextInput("attention_h1_rescale_factor",3*SCALE,h1.level(),SCALAR)
                h1 = h1 * h1_rescale_factor
                h1 = h1.doubleRescale().rescale()

                while h1.level() != o.level():
                    h1 = h1.modswitch()
                h2 = o + h1
                h2 = h2.rescale()
                A[n] = h2

                FunctionOutput(f"O{n}",O[n])
                FunctionOutput(f"A{n}",A[n])
        
        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn)


    global __attention_o_function_4gpu__
    __attention_o_function_4gpu__ = function

def attention_o_4gpu(H,ContextLayer,block):
    assert __attention_o_function_4gpu__ is not None, "Function not defined"
    inputs = {f"ContextLayer{i}": ContextLayer[i] for i in range(6)}
    inputs.update({f"H{i}":H[i] for i in range(3)})
    outputs = CeriumFunctionCall(__attention_o_function_4gpu__,inputs,block)
    O = [outputs[f"O{i}"] for i in range(3)]
    A = [outputs[f"A{i}"] for i in range(3)]
    return O,A


__attention_o_function_8gpu__ = None
def attention_o_function_8gpu():
    function = CeriumFunction("attention_o_8gpu",8,0)
    with function:

        ContextLayer = [None for _ in range(6)]
        def stream_fn(sid,ContextLayer):
            if sid > 2:
                return

            def stream_fn_internal(sidj,ContextLayer):
                if sidj > 1: return
                ContextLayer[2*sid + sidj] = CiphertextArgument(f"ContextLayer{2*sid + sidj}",3*SCALE,5) << (-64 * sidj)
            
            CeriumStream(1,2,stream_fn_internal,ContextLayer)

            # ContextLayerStacked[sid] = ContextLayer[0] + (ContextLayer[1] << -64)

        streamSize = 2
        numStreams = 3
        CeriumStream(streamSize,numStreams,stream_fn,ContextLayer)

        ContextLayerStacked = [ContextLayer[i] + ContextLayer[i + 1] for i in range(0, 6, 2)]
        ContextLayerStacked = [Receive(ContextLayerStacked[i]) for i in range(3)]

        Partition(4,0)



        Wo = [[[PeriodicPlaintextInput(f"attention_Wo_{n}_{j}_{i}",2*SCALE,ContextLayerStacked[0].level(),256,remapable=True) for i in range(-128 * (1 + (j // 3)),128 * (1 + (j // 3)))] for j in range(6)] for n in range(3)]
        O = matmul_ct128x768_pt768x768_new(ContextLayerStacked,Wo)

        A= [None for _ in range(3)]

        def stream_fn(sid):
            if sid > 0: return
            
            H = [CiphertextArgument(f"H{i}",3*SCALE,13) for i in range(3)]


            for n in range(3):
                o = O[n]
                o = o.doubleRescale()
                Bo = PeriodicPlaintextInput(f"attention_Bo_{n}",o.scale(),o.level(),256,remapable=True)
                O[n] = o + Bo

                h1 = H[n]
                h1_rescale_factor = PlaintextInput("attention_h1_rescale_factor",3*SCALE,h1.level(),SCALAR)
                h1 = h1 * h1_rescale_factor
                h1 = h1.doubleRescale().rescale()

                while h1.level() != o.level():
                    h1 = h1.modswitch()
                h2 = o + h1
                h2 = h2.rescale()
                A[n] = h2

                FunctionOutput(f"O{n}",O[n])
                FunctionOutput(f"A{n}",A[n])
        
        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn)


    global __attention_o_function_8gpu__
    __attention_o_function_8gpu__ = function

def attention_o_8gpu(H,ContextLayer,block):
    assert __attention_o_function_8gpu__ is not None, "Function not defined"
    inputs = {f"ContextLayer{i}": ContextLayer[i] for i in range(6)}
    inputs.update({f"H{i}":H[i] for i in range(3)})
    outputs = CeriumFunctionCall(__attention_o_function_8gpu__,inputs,block)
    O = [outputs[f"O{i}"] for i in range(3)]
    A = [outputs[f"A{i}"] for i in range(3)]
    return O,A



def attention_1gpu(H,block):

    NUM = 12
    idx = 0
    Q,K,V = attention_qkv_1gpu(H,idx,NUM,block)
    for num in range(NUM//2):
        TestOutput(f"attention_Q{(NUM//2)*idx + num}",Q[num])
        TestOutput(f"attention_K{(NUM//2)*idx + num}",K[num])
        TestOutput(f"attention_V{(NUM//2)*idx + num}",V[num])

    Scores = [None for _ in range(NUM//2)]
    Probs = [None for _ in range(NUM//2)]
    ContextLayer = [None for _ in range(NUM//2)]

    for num in range(NUM//2):
        qk_t = matmul_ct128x128_ct128x128_transpose(Q[num],K[num])
        Scores[num] = qk_t
        TestOutput(f"attention_scores{num}_bs",qk_t)

    for num in range(0,NUM//2,2):
        n = (NUM//2)*idx + num

        qk_t0 = Scores[num]
        qk_t1 = Scores[num+1]

        qk_t0 = bootstrap(qk_t0,"b35",14)
        qk_t1 = bootstrap(qk_t1,"b35",14)
        qk_t_sf = PlaintextInput("QK_t_bs_scale_factor",0,qk_t0.level(),SCALAR)
        qk_t0 = qk_t0 * qk_t_sf
        qk_t1 = qk_t1 * qk_t_sf

        TestOutput(f"attention_scores{n}",qk_t0)
        TestOutput(f"attention_scores{n+1}",qk_t1)
    
        so = softmax_128x128_vec(qk_t0,qk_t1)
        Probs[num] = so[0]
        Probs[num+1] = so[1]
        TestOutput(f"attention_probs{n}",so[0])
        TestOutput(f"attention_probs{n+1}",so[1])

    for num in range(NUM//2):
        n = (NUM//2)*idx + num
        p = Probs[num]
        context_layer= matmul_ct128x128_ct128x128("linear1",p,V[num])
        TestOutput(f"attention_context_layer{n}",context_layer)

        ContextLayer[num] = context_layer

    ContextLayerStacked = [None for _ in range(NUM//4)]
    for num in range(NUM//2):
        if num % 2 == 1:
            ContextLayerStacked[num // 2] = ContextLayer[num-1] + (ContextLayer[num] << -64)
            TestOutput(f"attention_context_layer_stacked{num // 2}",ContextLayerStacked[num // 2])

    Attention = [None for _ in range(NUM//4)]

    O = attention_o_1gpu(ContextLayerStacked,idx,NUM,block)
    for num in range(NUM//4):
        n = (NUM//2)*idx + num
        o = O[n]
        TestOutput(f"attention_output{n}",o)


        h1 = H[n]
        h1_rescale_factor = PlaintextInput("attention_h1_rescale_factor",3*SCALE,h1.level(),SCALAR)
        h1 = h1 * h1_rescale_factor
        h1 = h1.doubleRescale().rescale()

        while h1.level() != o.level():
            h1 = h1.modswitch()
        h2 = o + h1
        h2 = h2.rescale()
        Attention[n] = h2
        TestOutput(f"attention_residual_output_bs{n}",h2)

    attention_bs_rescale_factor = PlaintextInput(f"attention_bs_rescale_factor",0,18,SCALAR)
    for num in range(NUM//4):
        n = (NUM//2)*idx + num
        Attention[n] = bootstrap(Attention[n],"b33",16) * attention_bs_rescale_factor 
        TestOutput(f"attention_residual_output{n}",Attention[n])

    return Attention

def attention_2gpu(H,block):

    Q,K,V = attention_qkv_2gpu(H,block)

    def stream_fn(sid,ContextLayer):

        idx = sid
        NUM = 6

        Scores = [None for _ in range(NUM//2)]
        Probs = [None for _ in range(NUM//2)]

        for num in range(NUM//2):

            n = (NUM//2)*idx + num

            TestOutput(f"attention_Q{n}",Q[n])
            TestOutput(f"attention_K{n}",K[n])
            TestOutput(f"attention_V{n}",V[n])

            qk_t = matmul_ct128x128_ct128x128_transpose(Q[n],K[n])
            Scores[num] = qk_t
            TestOutput(f"attention_scores{num}_bs",qk_t)

        for num in range(0,NUM//2,2):

            n = (NUM//2)*idx + num

            qk_t_sf = PlaintextInput("QK_t_bs_scale_factor",0,16,SCALAR)
            if num < NUM//2:
                qk_t0 = Scores[num]
                qk_t0 = bootstrap(qk_t0,"b35",14)
                qk_t0 = qk_t0 * qk_t_sf
                TestOutput(f"attention_scores{n}",qk_t0)
            if num + 1 < NUM//2:
                qk_t1 = Scores[num+1]
                qk_t1 = bootstrap(qk_t1,"b35",14)
                qk_t1 = qk_t1 * qk_t_sf
                TestOutput(f"attention_scores{n+1}",qk_t1)
            else:
                qk_t1 = qk_t0
        
            so = softmax_128x128_vec(qk_t0,qk_t1)
            if num < NUM//2:
                Probs[num] = so[0]
                TestOutput(f"attention_probs{n}",so[0])
            if num + 1 < NUM//2:
                Probs[num+1] = so[1]
                TestOutput(f"attention_probs{n+1}",so[1])

        for num in range(NUM//2):
            n = (NUM//2)*idx + num
            p = Probs[num]
            context_layer= matmul_ct128x128_ct128x128("linear1",p,V[n])
            TestOutput(f"attention_context_layer{n}",context_layer)

            ContextLayer[n] = context_layer


    streamSize = 1
    numStreams = 2

    ContextLayer = [None for _ in range(6)]
    CeriumStream(streamSize,numStreams,stream_fn,ContextLayer)

    Attention = [None for _ in range(3)]

    O,A = attention_o_2gpu(H,ContextLayer,block)

    def stream_fn2(sid,Attention):
        if sid != 0:
            return

        for num in range(3):
            n = num
            o = O[n]
            TestOutput(f"attention_output{n}",o)

            TestOutput(f"attention_residual_output_bs{n}",A[n])

        attention_bs_rescale_factor = PlaintextInput(f"attention_bs_rescale_factor",0,18,SCALAR)
        for num in range(3):
            n = num
            Attention[n] = bootstrap(A[n],"b33",16) * attention_bs_rescale_factor 
            TestOutput(f"attention_residual_output{n}",Attention[n])

    streamSize = 1 
    numStreams = 1
    CeriumStream(streamSize,numStreams,stream_fn2,Attention)

    return Attention

def attention_4gpu(H,block):

    Q,K,V = attention_qkv_4gpu(H,block)

    def stream_fn(sid,ContextLayer):

        if sid > 2: return

        idx = sid
        NUM = 4

        Scores = [None for _ in range(NUM//2)]
        Probs = [None for _ in range(NUM//2)]

        for num in range(NUM//2):

            n = (NUM//2)*idx + num

            TestOutput(f"attention_Q{n}",Q[n])
            TestOutput(f"attention_K{n}",K[n])
            TestOutput(f"attention_V{n}",V[n])

            qk_t = matmul_ct128x128_ct128x128_transpose(Q[n],K[n])
            Scores[num] = qk_t
            TestOutput(f"attention_scores{num}_bs",qk_t)

        for num in range(0,NUM//2,2):

            n = (NUM//2)*idx + num

            qk_t_sf = PlaintextInput("QK_t_bs_scale_factor",0,16,SCALAR)
            qk_t0 = Scores[num]
            qk_t0 = bootstrap(qk_t0,"b35",14)
            qk_t0 = qk_t0 * qk_t_sf
            TestOutput(f"attention_scores{n}",qk_t0)
            qk_t1 = Scores[num+1]
            qk_t1 = bootstrap(qk_t1,"b35",14)
            qk_t1 = qk_t1 * qk_t_sf
            TestOutput(f"attention_scores{n+1}",qk_t1)
        
            so = softmax_128x128_vec(qk_t0,qk_t1)
            Probs[num] = so[0]
            TestOutput(f"attention_probs{n}",so[0])
            Probs[num+1] = so[1]
            TestOutput(f"attention_probs{n+1}",so[1])

        for num in range(NUM//2):
            n = (NUM//2)*idx + num
            p = Probs[num]
            context_layer= matmul_ct128x128_ct128x128("linear1",p,V[n])
            TestOutput(f"attention_context_layer{n}",context_layer)

            ContextLayer[n] = context_layer


    streamSize = 1
    numStreams = 3

    ContextLayer = [None for _ in range(6)]
    CeriumStream(streamSize,numStreams,stream_fn,ContextLayer)


    Attention = [None for _ in range(3)]

    O,A = attention_o_4gpu(H,ContextLayer,block)


    def stream_fn2(sid,Attention):
        if sid != 0:
            return

        for num in range(3):
            n = num
            o = O[n]
            TestOutput(f"attention_output{n}",o)

            TestOutput(f"attention_residual_output_bs{n}",A[n])

        attention_bs_rescale_factor = PlaintextInput(f"attention_bs_rescale_factor",0,18,SCALAR)
        for num in range(3):
            n = num
            Attention[n] = bootstrap(A[n],"b33",16) * attention_bs_rescale_factor 
            TestOutput(f"attention_residual_output{n}",Attention[n])

    streamSize = 1 
    numStreams = 1
    CeriumStream(streamSize,numStreams,stream_fn2,Attention)

    return Attention

def attention_8gpu(H,block):

    Q,K,V = attention_qkv_8gpu(H,block)

    def stream_fn(sid,ContextLayer):

        if sid >= 3: return

        idx = sid

        NUM = 4
        Scores = [None for _ in range(NUM//2)]
        Probs = [None for _ in range(NUM//2)]

        def stream_fn2(sid2,Scores):

            if sid2 >= 2: return

            n = (NUM//2)*idx + sid2

            TestOutput(f"attention_Q{n}",Q[n])
            TestOutput(f"attention_K{n}",K[n])
            TestOutput(f"attention_V{n}",V[n])



            qk_t = matmul_ct128x128_ct128x128_transpose(Q[n],K[n])
            TestOutput(f"attention_scores{n}_bs",qk_t)


            qk_t_sf = PlaintextInput("QK_t_bs_scale_factor",0,16,SCALAR)
            qk_t0 = qk_t
            qk_t0 = bootstrap(qk_t0,"b35_1gpu",14)
            qk_t0 = qk_t0 * qk_t_sf
            Scores[sid2] = qk_t0
            TestOutput(f"attention_scores{n}",qk_t0)


        Probs = [None for _ in range(NUM//2)]

        streamSize = 1
        numStreams = 2
        CeriumStream(streamSize,numStreams,stream_fn2,Scores)
        
        so = softmax_128x128_vec2(Scores[0],Scores[1])
        Probs[0] = so[0]
        Probs[1] = so[1]
        
        print(CurrentPartitionSize(), CurrentPartitionID())

        def stream_fn3(sid3,Probs,ContextLayer):

            print(CurrentPartitionSize(), CurrentPartitionID())

            if sid3 >= 2: return
            n = (NUM//2)*idx + sid3
            p = Probs[sid3]
            TestOutput(f"attention_probs{n}",Probs[sid3])
            context_layer = matmul_ct128x128_ct128x128("linear1",p,V[n])
            TestOutput(f"attention_context_layer{n}",context_layer)

            ContextLayer[n] = context_layer

        streamSize = 1
        numStreams = 2
        CeriumStream(streamSize,numStreams,stream_fn3,Probs,ContextLayer)


    streamSize = 2
    numStreams = 3

    ContextLayer = [None for _ in range(6)]
    CeriumStream(streamSize,numStreams,stream_fn,ContextLayer)


    Attention = [None for _ in range(3)]

    O,A = attention_o_8gpu(H,ContextLayer,block)


    def stream_fn2(sid,Attention):
        if sid != 0:
            return

        for num in range(3):
            n = num
            o = O[n]
            TestOutput(f"attention_output{n}",o)

            TestOutput(f"attention_residual_output_bs{n}",A[n])

        attention_bs_rescale_factor = PlaintextInput(f"attention_bs_rescale_factor",0,18,SCALAR)
        for num in range(3):
            n = num
            Attention[n] = bootstrap(A[n],"b33",16) * attention_bs_rescale_factor 
            TestOutput(f"attention_residual_output{n}",Attention[n])

    streamSize = 1 
    numStreams = 1
    CeriumStream(streamSize,numStreams,stream_fn2,Attention)

    return Attention

def matmul_ct128x768_pt768x128_new_type3(ct,pt):
    result0 = matmul_ct128x128_pt128x128(ct[0],pt[0])
    result1 = matmul_ct128x128_pt128x128_new_type3(ct[0],pt[3])
    for i in range(1,3):
        product0 = matmul_ct128x128_pt128x128(ct[i],pt[i])
        result0 += product0
        product1 = matmul_ct128x128_pt128x128_new_type3(ct[i],pt[i+3])
        result1 += product1
    result = result0 + result1
    return result

def matmul_ct128x768_pt768x768_new(ct,pt):
    result = [None for i in range(3)]
    for i in range(3):
        result[i] = matmul_ct128x768_pt768x128_new_type3(ct,pt[i])
    return result


__layernom_att_function_1gpu__ = None
__layernom_ffn_function_1gpu__ = None
def layer_norm_function_1gpu(norm_type):

    function = CeriumFunction(f"layernorm_{norm_type}_1gpu",1,0)
    with function:
        prefix = f"layernorm_{norm_type}"

        H = [CiphertextArgument(f"H{i}",2*SCALE,18) for i in range(3)]

        Hsq = [None for _ in range(3)]
        for i in range(3):
            Hsq[i] = (H[i]*H[i]).relinearize()

        H_sum = H[0]
        Hsq_sum = Hsq[0] 
        for i in range(1,3):
            H_sum = H_sum + H[i]
            Hsq_sum = Hsq_sum + Hsq[i]

        for i in range(8):
            H_sum = H_sum + (H_sum << 2**i)
            Hsq_sum = Hsq_sum + (Hsq_sum << 2**i)

        oneByN_sum = PeriodicPlaintextInput(f"layer_norm_{norm_type}_oneByN_sum",2*SCALE,H_sum.level(),256)
        H_mean = H_sum * oneByN_sum

        oneByN_sq_sum = PeriodicPlaintextInput(f"layer_norm_{norm_type}_oneByN_sq_sum",2*SCALE,Hsq_sum.level(),256)
        H_var = Hsq_sum * oneByN_sq_sum

        for i in range(8):
            H_mean = H_mean + (H_mean >> 2**i)
            H_var = H_var + (H_var >> 2**i)

        H_mean = H_mean.doubleRescale()
        H_var = H_var.doubleRescale()
        H_var = H_var.doubleRescale()

        H_mean_sq = (H_mean * H_mean).relinearize()
        H_mean_sq = H_mean_sq.doubleRescale()

        H_var = H_var - H_mean_sq
        epsilon = PlaintextInput(f"layer_norm_{norm_type}_epsilon",H_var.scale(),H_var.level(),SCALAR)
        H_var = H_var + epsilon

        TestFunctionOutput(f"{prefix}_variance",H_var)

        H_mean_rf = PlaintextInput(f"layer_norm_{norm_type}_H_mean_rescale_factor",2*SCALE,H_mean.level(),SCALAR)
        H_mean = H_mean * H_mean_rf
        H_mean = H_mean.doubleRescale()
        TestOutput(f"{prefix}_mean",H_mean)

        H_rf = PlaintextInput(f"layer_norm_{norm_type}_H_rescale_factor",2*SCALE,H[0].level(),SCALAR)
        for i in range(3):
            H[i] = (H[i]*H_rf).doubleRescale()
            H[i] = H[i].modswitch().modswitch()
            TestFunctionOutput(f"{prefix}_Hrs{i}",H[i])

        if norm_type == "att":
            H_isqrt = inverse_sqrt(H_var,6,"layernorm_att")
        elif norm_type == "ffn":
            H_isqrt = inverse_sqrt(H_var,4,"layernorm_ffn")
        else:
            raise Exception("Invalid layernorm type")

        TestFunctionOutput(f"{prefix}_isqrt",H_isqrt)


        H_norm = [None for i in range(3)]
        for i in range(3):
            H_norm[i] = (H[i] - H_mean)
            while(H_norm[i].level() != H_isqrt.level()):
                H_norm[i] = H_norm[i].modswitch()
            H_norm[i] = H_norm[i] * H_isqrt
            H_norm[i] = (H_norm[i]).relinearize().doubleRescale()
        

        for n in range(3):
            h = H_norm[n]

            W = PeriodicPlaintextInput(f"{prefix}_W_{n}",2*SCALE,h.level(),256,remapable=True)
            h = h * W
            h = h.rescale()
            B = PeriodicPlaintextInput(f"{prefix}_B_{n}",h.scale(),h.level(),256,remapable=True)
            h = h + B

            H_norm[n] = h
        
        for n in range(3):
            FunctionOutput(f"output{n}",H_norm[n])

    if norm_type == "att":
        global __layernom_att_function_1gpu__
        __layernom_att_function_1gpu__ = function
    elif norm_type == "ffn":
        global __layernom_ffn_function_1gpu__
        __layernom_ffn_function_1gpu__ = function


def layer_norm(H,norm_type,block):
    pSize = CurrentPartitionSize()
    pID = CurrentPartitionID()
    Partition(1,0)
    assert norm_type in ["att","ffn"], "Invalid layernorm type"
    if norm_type == "att":
        assert __layernom_att_function_1gpu__ is not None, "Function not defined"
        outputs = CeriumFunctionCall(__layernom_att_function_1gpu__,{f"H{i}":H[i] for i in range(3)},block)
        H_norm = [outputs[f"output{i}"] for i in range(3)]
        for i in range(3):
            TestOutput(f"layernorm_att_output{i}",H_norm[i])
    elif norm_type == "ffn":
        assert __layernom_ffn_function_1gpu__ is not None, "Function not defined"
        outputs = CeriumFunctionCall(__layernom_ffn_function_1gpu__,{f"H{i}":H[i] for i in range(3)},block)
        H_norm = [outputs[f"output{i}"] for i in range(3)]
        H_norm = [h.rescale() for h in H_norm]
        for i in range(3):
            TestOutput(f"layernorm_ffn_output_bs{i}",H_norm[i])
    Partition(pSize,pID)
    return H_norm

def gelu_vec_internal(x):
    xRF0 = PlaintextInput("gelu_xRF0",0,x.level(),SCALAR)
    xRF1 = PlaintextInput("gelu_xRF1",0,x.level(),SCALAR)

    xr = x * xRF0
    x = x * xRF1
    
    x = x.modswitch().modswitch().modswitch()

    C0 = PlaintextInput("gelu_C0",xr.scale(),xr.level(),SCALAR)
    C1 = PlaintextInput("gelu_C1",xr.scale(),xr.level(),SCALAR)
    C2 = PlaintextInput("gelu_C2",xr.scale(),xr.level(),SCALAR)

    comp0 = (xr + C0)
    comp1 = (xr + C1)
    comp2 = (xr + C2)

    comp0 = BreakVector(comp0.rescale())
    comp1 = BreakVector(comp1.rescale())
    comp2 = BreakVector(comp2.rescale())

    s0 = MakeVector([sign(comp0[0],"gelu"),sign(comp0[1],"gelu")])
    s1 = MakeVector([sign(comp1[0],"gelu"),sign(comp1[1],"gelu")])
    s2 = MakeVector([sign(comp2[0],"gelu"),sign(comp2[1],"gelu")])

    TestFunctionOutput(f"comp0",comp0)
    TestFunctionOutput(f"comp1",comp1)
    TestFunctionOutput(f"comp2",comp2)
    TestFunctionOutput(f"s0",s0)
    TestFunctionOutput(f"s1",s1)
    TestFunctionOutput(f"s2",s2)

    one = PlaintextInput("gelu_one",s0.scale(),s0.level(),SCALAR)

    b1 = s0 - s1
    b2 = s1 - s2
    b3 = s2 + one

    TestFunctionOutput(f"b1",b1)
    TestFunctionOutput(f"b2",b2)
    # TestOutput(f"{prefix}_b3",b3)

    for i in range(4):
        x = x.modswitch()

    x2 = (x * x).relinearize().doubleRescale()
    x4 = (x2 * x2).relinearize().doubleRescale()

    x3 = (x.modswitch().modswitch() * x2).relinearize().doubleRescale()
    x6 = (x2.modswitch().modswitch() * x4).relinearize().doubleRescale()

    Q6 = PlaintextInput("gelu_Q6",2*SCALE,x6.level(),SCALAR)
    Q = x6 * Q6

    Q4 = PlaintextInput("gelu_Q4",2*SCALE,x4.level(),SCALAR)
    Qt4 = x4 * Q4
    Qt4 = Qt4.modswitch().modswitch()
    Q = Q + Qt4

    Q2 = PlaintextInput("gelu_Q2",2*SCALE,x2.level(),SCALAR)
    Qt2 = x2 * Q2
    Qt2 = Qt2.modswitch().modswitch().modswitch().modswitch()
    Q = Q + Qt2

    Q1 = PlaintextInput("gelu_Q1",2*SCALE,x.level(),SCALAR)
    Qt1 = x * Q1
    Qt1 = Qt1.modswitch().modswitch().modswitch().modswitch().modswitch().modswitch()
    Q = Q + Qt1

    Q = Q.doubleRescale()

    Q0 = PlaintextInput("gelu_Q0",2*SCALE,Q.level(),SCALAR)
    Q = Q + Q0

    P3 = PlaintextInput("gelu_P3",2*SCALE,x3.level(),SCALAR)
    Pt3 = x3 * P3
    P = Pt3

    P2 = PlaintextInput("gelu_P2",2*SCALE,x2.level(),SCALAR)
    Pt2 = x2 * P2
    Pt2 = Pt2.modswitch().modswitch()
    P = P + Pt2

    P1 = PlaintextInput("gelu_P1",2*SCALE,x.level(),SCALAR)
    Pt1 = x * P1
    Pt1 = Pt1.modswitch().modswitch().modswitch().modswitch()
    P = P + Pt1

    P = P.modswitch().modswitch()

    P = P.doubleRescale()

    P0 = PlaintextInput("gelu_P0",2*SCALE,P.level(),SCALAR)
    P = P + P0

    xRF2 = PlaintextInput("gelu_xRF2",2*SCALE,x.level(),SCALAR)
    x = x * xRF2
    x = x.doubleRescale()

    for i in range(6):
        x = x.modswitch()

    res = (b1 * P) + (b2 * Q) + (b3 * x)
    res = res.relinearize()

    res = res.rescale()
    return res


__gelu_vec__function__ = None
def gelu_vec_function():
    function = CeriumFunction("gelu_vec",1,0)
    with function:
        x0 = CiphertextArgument("x0",2*SCALE,21)
        x1 = CiphertextArgument("x1",2*SCALE,21)
        x = MakeVector([x0,x1])
        res = gelu_vec_internal(x)
        res = BreakVector(res)
        FunctionOutput("res0",res[0])
        FunctionOutput("res1",res[1])
    global __gelu_vec__function__
    __gelu_vec__function__ = function

def gelu_vec(x,prefix=None):
    assert __gelu_vec__function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__gelu_vec__function__,{"x0":x[0],"x1":x[1]})
    res0 = outputs["res0"]
    res1 = outputs["res1"]
    return [res0,res1]


__ffn_up_1gpu_function__ = None
def ffn_up_1gpu_function():

    function = CeriumFunction("ffn_up_1gpu",1,0)
    with function:
        H = [CiphertextArgument(f"H{i}",3*SCALE,5) for i in range(3)]

        Wi = [[[[PeriodicPlaintextInput(f"ffn_Wi_{3*k + n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128*(1 + j//3),128*(1 + j//3))] for j in range(6)] for n in range(3)] for k in range(4)]
        res_ = [matmul_ct128x768_pt768x768_new(H,Wi[k]) for k in range(4)]

        for k in range(4):
            for n in range(3):
                r = res_[k][n]
                r = r.doubleRescale()
                Bi = PeriodicPlaintextInput(f"ffn_Bi_{3*k + n}",r.scale(),r.level(),256,remapable=True)
                r = r + Bi
                r = r.rescale()
                FunctionOutput(f"output{3*k + n}",r)
    global __ffn_up_1gpu_function__
    __ffn_up_1gpu_function__ = function

def ffn_up_1gpu(H,block):
    outputs = CeriumFunctionCall(__ffn_up_1gpu_function__,{f"H{i}":H[i] for i in range(3)},block)
    return [outputs[f"output{i}"] for i in range(12)]

__ffn_up_2gpu_function__ = None
def ffn_up_2gpu_function():

    function = CeriumFunction("ffn_up_2gpu",2,0)
    with function:


        def stream_fn1(sid,H):
            if sid != 0:
                return
            for i in range(3):
                H[i] = CiphertextArgument(f"H{i}",3*SCALE,5)


        streamSize = 1
        numStreams = 1
        H = [None for _ in range(3)]
        CeriumStream(streamSize,numStreams,stream_fn1,H)

        H = [Receive(H[i]) for i in range(3)]

        def stream_fn2(sid):
            if sid > 1:
                return
            for k in range(2*sid,2*(sid+1)):
                Wi = [[[PeriodicPlaintextInput(f"ffn_Wi_{3*k + n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128*(1 + j//3),128*(1 + j//3))] for j in range(6)] for n in range(3)]
                res_ = matmul_ct128x768_pt768x768_new(H,Wi)

                for n in range(3):
                    r = res_[n]
                    r = r.doubleRescale()
                    Bi = PeriodicPlaintextInput(f"ffn_Bi_{3*k + n}",r.scale(),r.level(),256,remapable=True)
                    r = r + Bi
                    r = r.rescale()
                    FunctionOutput(f"output{3*k + n}",r)

        streamSize = 1
        numStreams = 2
        CeriumStream(streamSize,numStreams,stream_fn2)

    global __ffn_up_2gpu_function__
    __ffn_up_2gpu_function__ = function

def ffn_up_2gpu(H,block):
    outputs = CeriumFunctionCall(__ffn_up_2gpu_function__,{f"H{i}":H[i] for i in range(3)},block)
    return [outputs[f"output{i}"] for i in range(12)]

__ffn_up_4gpu_function__ = None
def ffn_up_4gpu_function():

    function = CeriumFunction("ffn_up_4gpu",4,0)
    with function:


        def stream_fn1(sid,H):
            if sid != 0:
                return
            for i in range(3):
                H[i] = CiphertextArgument(f"H{i}",3*SCALE,5)


        streamSize = 1
        numStreams = 1
        H = [None for _ in range(3)]
        CeriumStream(streamSize,numStreams,stream_fn1,H)

        H = [Receive(H[i]) for i in range(3)]

        def stream_fn2(sid):
            if sid > 3:
                return
            k = sid
            Wi = [[[PeriodicPlaintextInput(f"ffn_Wi_{3*k + n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128*(1 + j//3),128*(1 + j//3))] for j in range(6)] for n in range(3)]
            res_ = matmul_ct128x768_pt768x768_new(H,Wi)

            for n in range(3):
                r = res_[n]
                r = r.doubleRescale()
                Bi = PeriodicPlaintextInput(f"ffn_Bi_{3*k + n}",r.scale(),r.level(),256,remapable=True)
                r = r + Bi
                r = r.rescale()
                FunctionOutput(f"output{3*k + n}",r)

        streamSize = 1
        numStreams = 4
        CeriumStream(streamSize,numStreams,stream_fn2)

    global __ffn_up_4gpu_function__
    __ffn_up_4gpu_function__ = function

def ffn_up_4gpu(H,block):
    outputs = CeriumFunctionCall(__ffn_up_4gpu_function__,{f"H{i}":H[i] for i in range(3)},block)
    return [outputs[f"output{i}"] for i in range(12)]

__ffn_up_8gpu_function__ = None
def ffn_up_8gpu_function():

    function = CeriumFunction("ffn_up_8gpu",8,0)
    with function:


        def stream_fn1(sid,H):
            if sid != 0:
                return
            for i in range(3):
                H[i] = CiphertextArgument(f"H{i}",3*SCALE,5)


        streamSize = 1
        numStreams = 1
        H = [None for _ in range(3)]
        CeriumStream(streamSize,numStreams,stream_fn1,H)

        H = [Receive(H[i]) for i in range(3)]

        outputs = [None for _ in range(12)]

        def stream_fn2(sid,outputs):
            if sid > 3:
                return
            k = sid
            Wi = [[[PeriodicPlaintextInput(f"ffn_Wi_{3*k + n}_{j}_{i}",2*SCALE,H[0].level(),256,remapable=True) for i in range(-128*(1 + j//3),128*(1 + j//3))] for j in range(6)] for n in range(3)]
            res_ = matmul_ct128x768_pt768x768_new(H,Wi)

            for n in range(3):
                r = res_[n]
                r = r.doubleRescale()
                Bi = PeriodicPlaintextInput(f"ffn_Bi_{3*k + n}",r.scale(),r.level(),256,remapable=True)
                r = r + Bi
                r = r.rescale()
                outputs[3*k + n] = r
                # FunctionOutput(f"output{3*k + n}",r)
        

        streamSize = 1
        numStreams = 4
        CeriumStream(streamSize,numStreams,stream_fn2,outputs)
        outputs = [Receive(outputs[i]) for i in range(12)]
        
        def stream_fn3(sid,outputs):
            if sid > 5:
                return
            FunctionOutput(f"output{2*sid}",outputs[2*sid])
            FunctionOutput(f"output{2*sid + 1}",outputs[2*sid + 1])


        streamSize = 1
        numStreams = 6
        CeriumStream(streamSize,numStreams,stream_fn3,outputs)


    global __ffn_up_8gpu_function__
    __ffn_up_8gpu_function__ = function

def ffn_up_8gpu(H,block):
    outputs = CeriumFunctionCall(__ffn_up_8gpu_function__,{f"H{i}":H[i] for i in range(3)},block)
    return [outputs[f"output{i}"] for i in range(12)]



__ffn_down_1gpu_function__ = None
def ffn_down_1gpu_function():

    function = CeriumFunction("ffn_down_1gpu",1,0)
    with function:
        ffn_gelu = [CiphertextArgument(f"ffn_gelu{i}",3*SCALE,5) for i in range(12)] 
        H = [CiphertextArgument(f"H{i}",3*SCALE,5) for i in range(3)]

        Wi = [[[[PeriodicPlaintextInput(f"ffn_Wo_{3*k + n}_{j}_{i}",2*SCALE,ffn_gelu[0].level(),256,remapable=True) for i in range(-128*(1+j//3),128*(1+j//3))] for j in range(6)] for n in range(3)] for k in range(4)]

        for k in range(4):
            if k == 0:
                ffn_down = matmul_ct128x768_pt768x768_new(ffn_gelu[3*k:3*k + 3],Wi[k]) 
            else:
                temp = matmul_ct128x768_pt768x768_new(ffn_gelu[3*k:3*k + 3],Wi[k])
                ffn_down = [ffn_down[i] + temp[i] for i in range(3)]

        h_rf = PlaintextInput("ffn_h_rescale_factor",2*SCALE,H[0].level() - 1,SCALAR)
        for i in range(3):
            r = ffn_down[i]
            r = r.rescale()
            Bi = PeriodicPlaintextInput(f"ffn_Bo_{i}",r.scale(),r.level(),256,remapable=True)
            r = r + Bi
            r = r.rescale()
            FunctionOutput(f"ffn_down{i}",r)
            h = H[i]
            h = h.rescale()
            h = h * h_rf
            h = h.rescale()
            while h.level() != r.level():
                h = h.modswitch()
            r = r + h
            r = r.rescale()
            FunctionOutput(f"ffn_residue{i}",r)
            
    global __ffn_down_1gpu_function__
    __ffn_down_1gpu_function__ = function

def ffn_down_1gpu(H,ffn_gelu,block):
    if __ffn_down_1gpu_function__ is None:
        raise Exception("ffn_down_1gpu_function not defined")
    args = {f"H{i}":H[i] for i in range(3)}
    args.update({f"ffn_gelu{i}":ffn_gelu[i] for i in range(12)})
    outputs = CeriumFunctionCall(__ffn_down_1gpu_function__,args,block)
    [TestOutput(f"ffn_down{i}",outputs[f"ffn_down{i}"]) for i in range(3)]
    return [outputs[f"ffn_residue{i}"] for i in range(3)]

__ffn_down_2gpu_function__ = None
def ffn_down_2gpu_function():

    function = CeriumFunction("ffn_down_2gpu",2,0)
    with function:

        def stream_fn(sid,ffn_down):
            if sid > 1:
                return
            for k in range(2*(sid),2*(sid+1)):
                ffn_gelu = [CiphertextArgument(f"ffn_gelu{3*k+i}",3*SCALE,5) for i in range(3)] 

                Wi = [[[PeriodicPlaintextInput(f"ffn_Wo_{3*k + n}_{j}_{i}",2*SCALE,ffn_gelu[0].level(),256,remapable=True) for i in range(-128*(1+j//3),128*(1+j//3))] for j in range(6)] for n in range(3)]

                if k % 2 == 0:
                    ffn_down[sid] = matmul_ct128x768_pt768x768_new(ffn_gelu,Wi) 
                else:
                    temp = matmul_ct128x768_pt768x768_new(ffn_gelu,Wi) 
                    ffn_down[sid] = [ffn_down[sid][i] + temp[i] for i in range(3)]

        streamSize = 1
        numStreams = 2
        ffn_down = [None for _ in range(numStreams)]
        CeriumStream(streamSize,numStreams,stream_fn,ffn_down)
        ffn_down[0] = [ffn_down[0][i] + ffn_down[1][i] for i in range(3)]
        ffn_down = ffn_down[0]

        def stream_fn2(sid):
            if sid > 0:
                return
            H = [CiphertextArgument(f"H{i}",3*SCALE,5) for i in range(3)]
            h_rf = PlaintextInput("ffn_h_rescale_factor",2*SCALE,H[0].level() - 1,SCALAR)
            for i in range(3):
                r = ffn_down[i]
                r = r.rescale()
                Bi = PeriodicPlaintextInput(f"ffn_Bo_{i}",r.scale(),r.level(),256,remapable=True)
                r = r + Bi
                r = r.rescale()
                FunctionOutput(f"ffn_down{i}",r)
                h = H[i]
                h = h.rescale()
                h = h * h_rf
                h = h.rescale()
                while h.level() != r.level():
                    h = h.modswitch()
                r = r + h
                r = r.rescale()
                FunctionOutput(f"ffn_residue{i}",r)
                # res[i] = r

        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn2)
            
    global __ffn_down_2gpu_function__
    __ffn_down_2gpu_function__ = function

def ffn_down_2gpu(H,ffn_gelu,block):
    if __ffn_down_2gpu_function__ is None:
        raise Exception("ffn_down_2gpu_function not defined")
    args = {f"H{i}":H[i] for i in range(3)}
    args.update({f"ffn_gelu{i}":ffn_gelu[i] for i in range(12)})
    outputs = CeriumFunctionCall(__ffn_down_2gpu_function__,args,block)
    ffn_down = [outputs[f"ffn_down{i}"] for i in range(3)]
    ffn_residue = [outputs[f"ffn_residue{i}"] for i in range(3)]
    # [TestOutput(f"ffn_down{i}",outputs[f"ffn_down{i}"]) for i in range(3)]
    return ffn_down, ffn_residue

__ffn_down_4gpu_function__ = None
def ffn_down_4gpu_function():

    function = CeriumFunction("ffn_down_4gpu",4,0)
    with function:

        def stream_fn(sid,ffn_down):
            if sid > 3:
                return
            k = sid
            ffn_gelu = [CiphertextArgument(f"ffn_gelu{3*k+i}",3*SCALE,5) for i in range(3)] 

            Wi = [[[PeriodicPlaintextInput(f"ffn_Wo_{3*k + n}_{j}_{i}",2*SCALE,ffn_gelu[0].level(),256,remapable=True) for i in range(-128*(1+j//3),128*(1+j//3))] for j in range(6)] for n in range(3)]

            ffn_down[k] = matmul_ct128x768_pt768x768_new(ffn_gelu,Wi) 

        streamSize = 1
        numStreams = 4
        ffn_down = [None for _ in range(numStreams)]
        CeriumStream(streamSize,numStreams,stream_fn,ffn_down)
        ffn_down[0] = [ffn_down[0][i] + ffn_down[1][i] for i in range(3)]
        ffn_down[0] = [ffn_down[0][i] + ffn_down[2][i] for i in range(3)]
        ffn_down[0] = [ffn_down[0][i] + ffn_down[3][i] for i in range(3)]
        ffn_down = ffn_down[0]

        def stream_fn2(sid):
            if sid > 0:
                return
            H = [CiphertextArgument(f"H{i}",3*SCALE,5) for i in range(3)]
            h_rf = PlaintextInput("ffn_h_rescale_factor",2*SCALE,H[0].level() - 1,SCALAR)
            for i in range(3):
                r = ffn_down[i]
                r = r.rescale()
                Bi = PeriodicPlaintextInput(f"ffn_Bo_{i}",r.scale(),r.level(),256,remapable=True)
                r = r + Bi
                r = r.rescale()
                FunctionOutput(f"ffn_down{i}",r)
                h = H[i]
                h = h.rescale()
                h = h * h_rf
                h = h.rescale()
                while h.level() != r.level():
                    h = h.modswitch()
                r = r + h
                r = r.rescale()
                FunctionOutput(f"ffn_residue{i}",r)
                # res[i] = r

        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn2)
            
    global __ffn_down_4gpu_function__
    __ffn_down_4gpu_function__ = function

def ffn_down_4gpu(H,ffn_gelu,block):
    if __ffn_down_4gpu_function__ is None:
        raise Exception("ffn_down_4gpu_function not defined")
    args = {f"H{i}":H[i] for i in range(3)}
    args.update({f"ffn_gelu{i}":ffn_gelu[i] for i in range(12)})
    outputs = CeriumFunctionCall(__ffn_down_4gpu_function__,args,block)
    ffn_down = [outputs[f"ffn_down{i}"] for i in range(3)]
    ffn_residue = [outputs[f"ffn_residue{i}"] for i in range(3)]
    return ffn_down, ffn_residue

__ffn_down_8gpu_function__ = None
def ffn_down_8gpu_function():

    function = CeriumFunction("ffn_down_8gpu",8,0)
    with function:


        ffn_gelu = [None for i in range(12)]
        def stream_fn_input(sid,ffn_gelu):
            if sid > 5: return
            ffn_gelu[2*sid] = CiphertextArgument(f"ffn_gelu{2*sid}",3*SCALE,5)
            ffn_gelu[2*sid + 1] = CiphertextArgument(f"ffn_gelu{2*sid + 1}",3*SCALE,5)
        
        streamSize = 1
        numStreams = 6
        CeriumStream(streamSize,numStreams,stream_fn_input,ffn_gelu)
        ffn_gelu = [Receive(ffn_gelu[i]) for i in range(12)]

        def stream_fn(sid,ffn_gelu,ffn_down):
            if sid > 3:
                return
            k = sid

            Wi = [[[PeriodicPlaintextInput(f"ffn_Wo_{3*k + n}_{j}_{i}",2*SCALE,ffn_gelu[0].level(),256,remapable=True) for i in range(-128*(1+j//3),128*(1+j//3))] for j in range(6)] for n in range(3)]

            ffn_down[k] = matmul_ct128x768_pt768x768_new(ffn_gelu[3*k : 3*k + 3],Wi) 

        streamSize = 1
        numStreams = 4
        ffn_down = [None for _ in range(numStreams)]
        CeriumStream(streamSize,numStreams,stream_fn,ffn_gelu,ffn_down)
        ffn_down[0] = [ffn_down[0][i] + ffn_down[1][i] for i in range(3)]
        ffn_down[0] = [ffn_down[0][i] + ffn_down[2][i] for i in range(3)]
        ffn_down[0] = [ffn_down[0][i] + ffn_down[3][i] for i in range(3)]
        ffn_down = ffn_down[0]

        def stream_fn2(sid):
            if sid > 0:
                return
            H = [CiphertextArgument(f"H{i}",3*SCALE,5) for i in range(3)]
            h_rf = PlaintextInput("ffn_h_rescale_factor",2*SCALE,H[0].level() - 1,SCALAR)
            for i in range(3):
                r = ffn_down[i]
                r = r.rescale()
                Bi = PeriodicPlaintextInput(f"ffn_Bo_{i}",r.scale(),r.level(),256,remapable=True)
                r = r + Bi
                r = r.rescale()
                FunctionOutput(f"ffn_down{i}",r)
                h = H[i]
                h = h.rescale()
                h = h * h_rf
                h = h.rescale()
                while h.level() != r.level():
                    h = h.modswitch()
                r = r + h
                r = r.rescale()
                FunctionOutput(f"ffn_residue{i}",r)
                # res[i] = r

        streamSize = 1
        numStreams = 1
        CeriumStream(streamSize,numStreams,stream_fn2)
            
    global __ffn_down_8gpu_function__
    __ffn_down_8gpu_function__ = function

def ffn_down_8gpu(H,ffn_gelu,block):
    if __ffn_down_8gpu_function__ is None:
        raise Exception("ffn_down_8gpu_function not defined")
    args = {f"H{i}":H[i] for i in range(3)}
    args.update({f"ffn_gelu{i}":ffn_gelu[i] for i in range(12)})
    outputs = CeriumFunctionCall(__ffn_down_8gpu_function__,args,block)
    ffn_down = [outputs[f"ffn_down{i}"] for i in range(3)]
    ffn_residue = [outputs[f"ffn_residue{i}"] for i in range(3)]
    return ffn_down, ffn_residue

def ffn_1gpu(H,block):
    ffn_up = ffn_up_1gpu(H,block)
    for i in range(12):
        ffn_up[i] = bootstrap(ffn_up[i],"b33",19)
        TestOutput(f"ffn_up{i}",ffn_up[i])
    ffn_gelu = [None for i in range(12)]
    for i in range(0,12,2):
        ffn_gelu[i:i+2] = gelu_vec(ffn_up[i:i+2])
        TestOutput(f"ffn_gelu{i}",ffn_gelu[i])
        TestOutput(f"ffn_gelu{i+1}",ffn_gelu[i+1])


    ffn_residue = ffn_down_1gpu(H,ffn_gelu,block)
    for i in range(3):
        ffn_residue[i] = bootstrap(ffn_residue[i],"b33",16) * PlaintextInput("ffn_residue_bs_rescale_factor",0,18,SCALAR)
        TestOutput(f"ffn_residue{i}",ffn_residue[i])

    return ffn_residue

def ffn_2gpu(H,block):
    ffn_up = ffn_up_2gpu(H,block)
    
    ffn_gelu = [None for i in range(12)]
    def stream_fn(sid,ffn_gelu):
        if sid >= 2:
            return
        for i in range(6*sid,6*(sid+1),2):
            ffn_up[i] = bootstrap(ffn_up[i],"b33",19)
            ffn_up[i+1] = bootstrap(ffn_up[i+1],"b33",19)
            TestOutput(f"ffn_up{i}",ffn_up[i])
            TestOutput(f"ffn_up{i+1}",ffn_up[i+1])
            ffn_gelu[i:i+2] = gelu_vec(ffn_up[i:i+2])
            TestOutput(f"ffn_gelu{i}",ffn_gelu[i])
            TestOutput(f"ffn_gelu{i+1}",ffn_gelu[i+1])

    streamSize = 1
    numStreams = 2
    CeriumStream(streamSize,numStreams,stream_fn,ffn_gelu)

    ffn_down, ffn_residue = ffn_down_2gpu(H,ffn_gelu,block)


    def stream_fn2(sid):
        if sid > 0:
            return
        for i in range(3):
            TestOutput(f"ffn_down{i}",ffn_down[i])
            ffn_residue[i] = bootstrap(ffn_residue[i],"b33",16) * PlaintextInput("ffn_residue_bs_rescale_factor",0,18,SCALAR)
            TestOutput(f"ffn_residue{i}",ffn_residue[i])
    
    streamSize = 1
    numStreams = 1
    CeriumStream(streamSize,numStreams,stream_fn2)

    return ffn_residue

def ffn_4gpu(H,block):
    ffn_up = ffn_up_4gpu(H,block)
    
    ffn_gelu = [None for i in range(12)]
    def stream_fn(sid,ffn_gelu):
        if sid > 3:
            return
        i = 3*sid
        ffn_up[i] = bootstrap(ffn_up[i],"b33",19)
        ffn_up[i+1] = bootstrap(ffn_up[i+1],"b33",19)
        ffn_up[i+2] = bootstrap(ffn_up[i+2],"b33",19)
        TestOutput(f"ffn_up{i}",ffn_up[i])
        TestOutput(f"ffn_up{i+1}",ffn_up[i+1])
        TestOutput(f"ffn_up{i+2}",ffn_up[i+2])
        ffn_gelu[i:i+2] = gelu_vec(ffn_up[i:i+2])
        ffn_gelu[i+2] = gelu_vec([ffn_up[i+2]]*2)[0]
        TestOutput(f"ffn_gelu{i}",ffn_gelu[i])
        TestOutput(f"ffn_gelu{i+1}",ffn_gelu[i+1])
        TestOutput(f"ffn_gelu{i+2}",ffn_gelu[i+2])

    streamSize = 1
    numStreams = 4
    CeriumStream(streamSize,numStreams,stream_fn,ffn_gelu)

    ffn_down, ffn_residue = ffn_down_4gpu(H,ffn_gelu,block)


    def stream_fn2(sid):
        if sid > 0:
            return
        for i in range(3):
            TestOutput(f"ffn_down{i}",ffn_down[i])
            ffn_residue[i] = bootstrap(ffn_residue[i],"b33",16) * PlaintextInput("ffn_residue_bs_rescale_factor",0,18,SCALAR)
            TestOutput(f"ffn_residue{i}",ffn_residue[i])
    
    streamSize = 1
    numStreams = 1
    CeriumStream(streamSize,numStreams,stream_fn2)

    return ffn_residue


def ffn_8gpu(H,block):
    ffn_up = ffn_up_8gpu(H,block)
    
    ffn_gelu = [None for i in range(12)]
    def stream_fn(sid,ffn_gelu):
        if sid > 5:
            return
        i = 2*sid
        ffn_up[i] = bootstrap(ffn_up[i],"b33",19)
        ffn_up[i+1] = bootstrap(ffn_up[i+1],"b33",19)
        TestOutput(f"ffn_up{i}",ffn_up[i])
        TestOutput(f"ffn_up{i+1}",ffn_up[i+1])
        ffn_gelu[i:i+2] = gelu_vec(ffn_up[i:i+2])
        TestOutput(f"ffn_gelu{i}",ffn_gelu[i])
        TestOutput(f"ffn_gelu{i+1}",ffn_gelu[i+1])

    streamSize = 1
    numStreams = 6
    CeriumStream(streamSize,numStreams,stream_fn,ffn_gelu)

    ffn_down, ffn_residue = ffn_down_8gpu(H,ffn_gelu,block)


    def stream_fn2(sid):
        if sid > 0:
            return
        for i in range(3):
            TestOutput(f"ffn_down{i}",ffn_down[i])
            ffn_residue[i] = bootstrap(ffn_residue[i],"b33",16) * PlaintextInput("ffn_residue_bs_rescale_factor",0,18,SCALAR)
            TestOutput(f"ffn_residue{i}",ffn_residue[i])
    
    streamSize = 1
    numStreams = 1
    CeriumStream(streamSize,numStreams,stream_fn2)

    return ffn_residue


def sign(x,prefix):
    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_g3_coeff1_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[3] = getPlaintextInput(f"{prefix}_g3_coeff3_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[5] = getPlaintextInput(f"{prefix}_g3_coeff5_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[7] = getPlaintextInput(f"{prefix}_g3_coeff7_{xLevel}",SCALE,x.level(),SCALAR)
    x = g3_f3(x,coeffs)
    x = x.rescale()
    # TestOutput(f"{prefix}_x_{xLevel}",x)

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_g3_coeff1_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[3] = getPlaintextInput(f"{prefix}_g3_coeff3_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[5] = getPlaintextInput(f"{prefix}_g3_coeff5_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[7] = getPlaintextInput(f"{prefix}_g3_coeff7_{xLevel}",SCALE,x.level(),SCALAR)
    x = g3_f3(x,coeffs)
    x = x.rescale()
    # TestOutput(f"{prefix}_x_{xLevel}",x)

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_f3_coeff1_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[3] = getPlaintextInput(f"{prefix}_f3_coeff3_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[5] = getPlaintextInput(f"{prefix}_f3_coeff5_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[7] = getPlaintextInput(f"{prefix}_f3_coeff7_{xLevel}",SCALE,x.level(),SCALAR)
    x = g3_f3(x,coeffs)

 
    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_f3_coeff1_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[3] = getPlaintextInput(f"{prefix}_f3_coeff3_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[5] = getPlaintextInput(f"{prefix}_f3_coeff5_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[7] = getPlaintextInput(f"{prefix}_f3_coeff7_{xLevel}",2*SCALE,x.level(),SCALAR)
    x = g3_f3_new(x,coeffs)
    x = x.doubleRescale()

    return x


def sign_short(x,prefix):
    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_g3_coeff1_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[3] = getPlaintextInput(f"{prefix}_g3_coeff3_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[5] = getPlaintextInput(f"{prefix}_g3_coeff5_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[7] = getPlaintextInput(f"{prefix}_g3_coeff7_{xLevel}",SCALE,x.level(),SCALAR)
    x = g3_f3(x,coeffs)
    x = x.rescale()

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_g3_coeff1_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[3] = getPlaintextInput(f"{prefix}_g3_coeff3_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[5] = getPlaintextInput(f"{prefix}_g3_coeff5_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[7] = getPlaintextInput(f"{prefix}_g3_coeff7_{xLevel}",SCALE,x.level(),SCALAR)
    x = g3_f3(x,coeffs)
    x = x.rescale()

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_f3_coeff1_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[3] = getPlaintextInput(f"{prefix}_f3_coeff3_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[5] = getPlaintextInput(f"{prefix}_f3_coeff5_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[7] = getPlaintextInput(f"{prefix}_f3_coeff7_{xLevel}",SCALE,x.level(),SCALAR)
    x = g3_f3(x,coeffs)
    x = x.rescale()

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_f3_coeff1_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[3] = getPlaintextInput(f"{prefix}_f3_coeff3_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[5] = getPlaintextInput(f"{prefix}_f3_coeff5_{xLevel}",2*SCALE,x.level(),SCALAR)
    coeffs[7] = getPlaintextInput(f"{prefix}_f3_coeff7_{xLevel}",1*SCALE,x.level(),SCALAR)
    x = g3_f3(x,coeffs,relinType=2)
    return x

def matmul_ct1x768_pt768x768_pool(ct,pt,mask,maskL,maskR):
    result = [None for i in range(3)]
    for i in range(3):
        result[i] = matmul_ct128x768_pt768x128_new(ct,pt[i],mask,maskL,maskR)
        if i != 0:
            result[0] += result[i] << -256*i
    return result[0]

def tanh(x):
    xRF0 = PlaintextInput("tanh_xRF0",0,x.level(),SCALAR)
    xRF1 = PlaintextInput("tanh_xRF1",0,x.level(),SCALAR)

    xr = x * xRF0
    x = x * xRF1
    
    C0 = PlaintextInput("tanh_C0",xr.scale(),xr.level(),SCALAR)
    C1 = PlaintextInput("tanh_C1",xr.scale(),xr.level(),SCALAR)


    comp0 = (xr + C0).rescale()
    comp1 = (xr + C1).rescale()

    s0 = sign(comp0,"tanh")
    s1 = sign(comp1,"tanh")

    TestOutput(f"tanh_comp0",comp0)
    TestOutput(f"tanh_comp1",comp1)
    TestOutput(f"tanh_s0",s0)
    TestOutput(f"tanh_s1",s1)


    one = PlaintextInput("tanh_one",s0.scale(),s0.level(),SCALAR)

    b0 = s0 - one
    b1 = s0 - s1
    b2 = s1 + one

    P = x 
    while P.level() != b1.level() + 2: 
        P = P.modswitch()

    Prescale = PlaintextInput("tanh_Prescale",SCALE,P.level(),SCALAR)

    P *= Prescale
    P = P.doubleRescale()

    res0 = PlaintextInput("tanh_res0",SCALE,b0.level(),SCALAR)
    res2 = PlaintextInput("tanh_res2",SCALE,b2.level(),SCALAR)

    res = b0 * res0 + b1 * P + b2 * res2
    res = res.relinearize()

    res = res.rescale()
    return res


def pool_classify_layer(X):

    prefix = f"pool_classify"

    bsgsMask = PlaintextInput(f"{prefix}_bsgs_Mask_{X[0].level()}",2*SCALE,X[0].level())
    bsgsMaskL = PlaintextInput(f"{prefix}_bsgs_MaskL_{X[0].level()}",2*SCALE,X[0].level())
    bsgsMaskR = PlaintextInput(f"{prefix}_bsgs_MaskR_{X[0].level()}",2*SCALE,X[0].level())
    Wp = [[[PeriodicPlaintextInput(f"{prefix}_Wp_{n}_{j}_{i}",2*SCALE,X[0].level(),256) for i in range(-128,128)] for j in range(6)] for n in range(3)]
    pool_linear = matmul_ct1x768_pt768x768_pool(X,Wp,bsgsMask,bsgsMaskL,bsgsMaskR)
    pool_linear = pool_linear.doubleRescale().doubleRescale()

    Bp = PlaintextInput(f"{prefix}_Bp",pool_linear.scale(),pool_linear.level())
    pool_linear += Bp

    pool_linear = pool_linear.rescale()

    pool_linear = bootstrap(pool_linear,"b33",19)

    TestOutput("pool_linear",pool_linear)

    p = tanh(pool_linear)

    TestOutput("tanh_res",p)

    p += p >> 1024

    Wc = PlaintextInput(f"{prefix}_Wc",SCALE,p.level())

    m = p * Wc
    j = 1
    while j < 1024:
        m+= m << j
        j *= 2

    m = m.rescale()
    Bc = PlaintextInput(f"{prefix}_Bc",m.scale(),m.level())
    m += Bc

    FunctionOutput("prediction",m)


__pool_classify_layer_function__ = None
def pool_classify_layer_function():
    function = CeriumFunction("pool_classify_layer",1,0)
    with function:
        X = [CiphertextArgument(f"H{i}",3*SCALE,13) for i in range(3)]
        pool_classify_layer(X)
    global __pool_classify_layer_function__
    __pool_classify_layer_function__ = function

    
def main(num_gpus,prefix):
    global LVL
    # LVL = 51
    LVL = 52
    global SCALE
    SCALE = 28
    global SCALAR
    SCALAR = True

    configureBootstrapLVL(LVL)

    def bert_program_1gpu(num_blocks):
        BertProgram = CeriumProgram('bert', rns_bit_size=SCALE,num_gpus=num_gpus)
        with BertProgram:
            matmul_ct128x128_ct128x128_transpose_function()
            matmul_ct128x128_ct128x128_function()
            bootstrap_32K_33lvl_t1_function()
            bootstrap_32K_35lvl_t1_vec_function()
            softmax_128x128_vec_function()
            attention_qkv_function_1gpu()
            attention_o_function_1gpu()
            layer_norm_function_1gpu("att")
            ffn_up_1gpu_function()
            gelu_vec_function()
            ffn_down_1gpu_function()
            layer_norm_function_1gpu("ffn")
            pool_classify_layer_function()

            function = CeriumFunction(f"main_1gpu_{num_blocks}",1,0)
            with function:
                H = [CiphertextInput(f"layernorm_out_output_prev{i}",3*SCALE,13) for i in range(3)]
                for l in range(num_blocks):
                    attention = attention_1gpu(H,f"block{l}")
                    attention_norm = layer_norm(attention,"att",f"block{l}")
                    ffn = ffn_1gpu(attention_norm,f"block{l}")
                    ffn_norm = layer_norm(ffn,"ffn",f"block{l}")
                    ffn_norm_rf = PlaintextInput("layernorm_ffn_rescale_factor",SCALE,13,SCALAR)
                    for i in range(3):
                        H[i] = bootstrap(ffn_norm[i],f"b35",11) * ffn_norm_rf
                if num_blocks == 12:
                    outputs = CeriumFunctionCall(__pool_classify_layer_function__,{f"H{i}": H[i] for i in range(3)})
                    Output("prediction",outputs["prediction"])
                else:
                    [Output(f"layernorm_ffn_output{i}",H[i]) for i in range(3)]

        return BertProgram

    def bert_program_2gpu(num_blocks):
        BertProgram = CeriumProgram('bert', rns_bit_size=SCALE,num_gpus=num_gpus)
        with BertProgram:
            matmul_ct128x128_ct128x128_transpose_function()
            matmul_ct128x128_ct128x128_function()
            bootstrap_32K_33lvl_t1_function()
            bootstrap_32K_35lvl_t1_vec_function()
            softmax_128x128_vec_function()
            attention_qkv_function_2gpu()
            attention_o_function_2gpu()
            layer_norm_function_1gpu("att")
            ffn_up_2gpu_function()
            gelu_vec_function()
            ffn_down_2gpu_function()
            layer_norm_function_1gpu("ffn")
            pool_classify_layer_function()

            function = CeriumFunction(f"main_2gpu_{num_blocks}",2,0)
            with function:
                H = [None for _ in range(3)]
                def stream_fn1(sid,H):
                    if sid > 0: return
                    for i in range(3): H[i] = CiphertextInput(f"layernorm_out_output_prev{i}",3*SCALE,13)
                CeriumStream(1,1,stream_fn1,H)
                for l in range(num_blocks):
                    attention = attention_2gpu(H,f"block{l}")
                    attention_norm = layer_norm(attention,"att",f"block{l}")
                    ffn = ffn_2gpu(attention_norm,f"block{l}")
                    ffn_norm = layer_norm(ffn,"ffn",f"block{l}")
                    def stream_fn2(sid,H):
                        if sid > 0: return
                        ffn_norm_rf = PlaintextInput("layernorm_ffn_rescale_factor",SCALE,13,SCALAR)
                        for i in range(3):
                            H[i] = bootstrap(ffn_norm[i],f"b35",11) * ffn_norm_rf
                        if num_blocks == 12 and l == 11:
                            outputs = CeriumFunctionCall(__pool_classify_layer_function__,{f"H{i}": H[i] for i in range(3)})
                            Output("prediction",outputs["prediction"])
                        elif l == num_blocks - 1:
                            [Output(f"layernorm_ffn_output{i}",H[i]) for i in range(3)]
                    CeriumStream(1,1,stream_fn2,H)

        return BertProgram

    def bert_program_4gpu(num_blocks):
        BertProgram = CeriumProgram('bert', rns_bit_size=SCALE,num_gpus=num_gpus)
        with BertProgram:
            matmul_ct128x128_ct128x128_transpose_function()
            matmul_ct128x128_ct128x128_function()
            bootstrap_32K_33lvl_t1_function()
            bootstrap_32K_35lvl_t1_vec_function()
            softmax_128x128_vec_function()
            attention_qkv_function_4gpu()
            attention_o_function_4gpu()
            layer_norm_function_1gpu("att")
            ffn_up_4gpu_function()
            gelu_vec_function()
            ffn_down_4gpu_function()
            layer_norm_function_1gpu("ffn")
            pool_classify_layer_function()

            function = CeriumFunction(f"main_4gpu_{num_blocks}",4,0)
            with function:
                H = [None for _ in range(3)]
                def stream_fn1(sid,H):
                    if sid > 0: return
                    for i in range(3): H[i] = CiphertextInput(f"layernorm_out_output_prev{i}",3*SCALE,13)
                CeriumStream(1,1,stream_fn1,H)
                for l in range(num_blocks):
                    attention = attention_4gpu(H,f"block{l}")
                    attention_norm = layer_norm(attention,"att",f"block{l}")
                    ffn = ffn_4gpu(attention_norm,f"block{l}")
                    ffn_norm = layer_norm(ffn,"ffn",f"block{l}")
                    def stream_fn2(sid,H):
                        if sid > 0: return
                        ffn_norm_rf = PlaintextInput("layernorm_ffn_rescale_factor",SCALE,13,SCALAR)
                        for i in range(3):
                            H[i] = bootstrap(ffn_norm[i],f"b35",11) * ffn_norm_rf
                        if num_blocks == 12 and l == 11:
                            outputs = CeriumFunctionCall(__pool_classify_layer_function__,{f"H{i}": H[i] for i in range(3)})
                            Output("prediction",outputs["prediction"])
                        elif l == num_blocks - 1:
                            [Output(f"layernorm_ffn_output{i}",H[i]) for i in range(3)]
                    CeriumStream(1,1,stream_fn2,H)

        return BertProgram

    def bert_program_8gpu(num_blocks):
        BertProgram = CeriumProgram('bert', rns_bit_size=SCALE,num_gpus=num_gpus)
        with BertProgram:

            bootstrap_32K_33lvl_t1_function()
            bootstrap_32K_35lvl_t1_vec_function()
            bootstrap_32K_35lvl_t1_vec_function(gpus=2)

            configureBootstrapGPUs(1)

            layer_norm_function_1gpu("att")
            layer_norm_function_1gpu("ffn")

            configureBootstrapGPUs(2)

            matmul_ct128x128_ct128x128_transpose_function()
            matmul_ct128x128_ct128x128_function()
            softmax_128x128_vec2_function()
            attention_qkv_function_8gpu()
            attention_o_function_8gpu()
            ffn_up_8gpu_function()
            gelu_vec_function()
            ffn_down_8gpu_function()
            pool_classify_layer_function()
        
            function = CeriumFunction(f"main_8gpu_{num_blocks}",8,0)
            with function:
                H = [None for _ in range(3)]
                def stream_fn1(sid,H):
                    if sid > 0: return
                    for i in range(3): H[i] = CiphertextInput(f"layernorm_out_output_prev{i}",3*SCALE,13)
                CeriumStream(1,1,stream_fn1,H)
                for l in range(num_blocks):
                    attention = attention_8gpu(H,f"block{l}")
                    attention_norm = layer_norm(attention,"att",f"block{l}")
                    ffn = ffn_8gpu(attention_norm,f"block{l}")
                    ffn_norm = layer_norm(ffn,"ffn",f"block{l}")
                    def stream_fn2(sid,H):
                        if sid > 0: return
                        ffn_norm_rf = PlaintextInput("layernorm_ffn_rescale_factor",SCALE,13,SCALAR)
                        for i in range(3):
                            H[i] = bootstrap(ffn_norm[i],f"b35",11) * ffn_norm_rf
                        if num_blocks == 12 and l == 11:
                            outputs = CeriumFunctionCall(__pool_classify_layer_function__,{f"H{i}": H[i] for i in range(3)})
                            Output("prediction",outputs["prediction"])
                        elif l == num_blocks - 1:
                            [Output(f"layernorm_ffn_output{i}",H[i]) for i in range(3)]
                    CeriumStream(1,1,stream_fn2,H)

        return BertProgram

    if num_gpus == 1:
        program = bert_program_1gpu(12)
    elif num_gpus == 2:
        program = bert_program_2gpu(12)
    elif num_gpus == 4:
        program = bert_program_4gpu(12)
    elif num_gpus == 8:
        program = bert_program_8gpu(12)

    def get_bcus(fname):
        return 128

    def get_block_size(fname):
        if "main" in fname:
            return 1024
        elif "bootstrap" in fname:
            return 2048
        elif "softmax" in fname:
            return 1024
        elif "matmul_ct128x128_ct128x128_transpose" in fname:
            return 1024
        elif "matmul_ct128x128_ct128x128" in fname:
            return 256
        elif "pool_classify_layer" in fname:
            return 64
        elif "attention_qkv" in fname:
            return 32
        elif "attention_o" in fname:
            return 32
        elif "ffn_up" in fname:
            return 32
        elif "ffn_down" in fname:
            return 32
        else:
            return 1024


    def get_vregs(fname):
        if "main" in fname:
            return 1024
        elif "bootstrap" in fname:
            return 4096
        elif "softmax" in fname:
            return 2048
        elif "matmul_ct128x128_ct128x128" in fname:
            return 3096
        elif "pool_classify_layer" in fname:
            return 4096
        elif "attention_qkv" in fname:
            return 8192
        elif "attention_o" in fname:
            return 8192
        elif "ffn_up" in fname:
            return 8192
        elif "ffn_down" in fname:
            return 8192
        else:
            return 4096
        
    def do_horizontal_fusion(fname):
        if "main" in fname:
            return 0
        else:
            return 1


    functions = program.get_functions()
    skip = []
    skip += ["bootstrap_32K_33lvl_t1"]
    skip += ["bootstrap_32K_35lvl_t1_vec"]
    skip += ["bootstrap_32K_33lvl_t1_vec"]
    skip += ["bootstrap_32K_33lvl_t1_vec2"]
    skip += ["gelu_vec"]
    skip += ["softmax_128x128_vec"]
    skip += ["matmul_ct128x128_ct128x128_transpose"]
    skip += ["matmul_ct128x128_ct128x128"]
    skip += ["pool_classify_layer"]
    skip += ["attention_qkv_1gpu"]
    skip += ["attention_o_1gpu"]

    skip += ["layernorm_att_1gpu"]

    skip += ["ffn_up_1gpu"]
    skip += ["ffn_down_1gpu"]

    skip += ["layernorm_ffn_1gpu"]

    skip += ["attention_qkv_2gpu"]
    skip += ["attention_o_2gpu"]
    skip += ["ffn_up_2gpu"]
    skip += ["ffn_down_2gpu"]

    skip += ["attention_qkv_4gpu"]
    skip += ["attention_o_4gpu"]
    skip += ["ffn_up_4gpu"]
    skip += ["ffn_down_4gpu"]

    skip += ["softmax_128x128_vec2"]
    skip += ["attention_qkv_8gpu"]
    skip += ["attention_o_8gpu"]
    skip += ["ffn_up_8gpu"]
    skip += ["ffn_down_8gpu"]


    skip += ["main"]
    skip = []
    
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
        os.environ['CERIUM_HORIZONTAL_FUSION'] = str(do_horizontal_fusion(name))
        cerium_compile_function(f,num_gpus,get_vregs(name),get_bcus(name),prefix)

 

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str,default="bert_outputs/")
    parser.add_argument('--gpus', type=int,default=1)

    args = parser.parse_args()

    main(args.gpus,args.prefix)


    

    
