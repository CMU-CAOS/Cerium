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

LlamaPlaintextInputs = {}
LlamaCiphertextInputs = {}
TEST=True
SCALE = 28

def TestOutput(name,val,predicate=True):
    if TEST and predicate:
        Output(name,val)

def TestFunctionOutput(name,val):
    return
    if TEST:
        FunctionOutput(name,val)
        # Output(name,val)


def getPlaintextInput(name,scale,level,scalar=False):
    partition_size = CurrentPartitionSize()
    partition_id = CurrentPartitionID()
    key = f"{name}_:p{partition_size}:i{partition_id}"
    if key in LlamaPlaintextInputs.keys():
        pt = LlamaPlaintextInputs[key]
        if pt.scale() != scale:
            raise Exception("Mismatched Scale")
        if pt.level() != level:
            raise Exception("Mismatched Level")
        return pt
    pt = PlaintextInput(name,scale,level,scalar)
    LlamaPlaintextInputs[key] = pt
    return pt

def getPeriodicPlaintextInput(name,scale,level,period):
    partition_size = CurrentPartitionSize()
    partition_id = CurrentPartitionID()
    key = f"{name}_:p{partition_size}:i{partition_id}"
    if key in LlamaPlaintextInputs.keys():
        pt = LlamaPlaintextInputs[key]
        if pt.scale() != scale:
            raise Exception("Mismatched Scale")
        if pt.level() != level:
            raise Exception("Mismatched Level")
        return pt
    pt = PeriodicPlaintextInput(name,scale,level,period)
    LlamaPlaintextInputs[key] = pt
    return pt


def getCiphertextInput(name,scale,level):
    partition_size = CurrentPartitionSize()
    partition_id = CurrentPartitionID()
    key = f"{name}_:p{partition_size}:i{partition_id}"
    if key in LlamaCiphertextInputs.keys():
        ct = LlamaCiphertextInputs[key]
        if ct.scale() != scale:
            raise Exception("Mismatched Scale")
        if ct.level() != level:
            raise Exception("Mismatched Level")
        return ct
    ct = CiphertextInput(name,scale,level)
    LlamaCiphertextInputs[key] = ct
    return ct

def matmul_ct128xHeight_ptHeightxWidth(height,width,ct,pt,masks,use_freqs_cis=False):

    assert width % 256 == 0
    width = width // 256

    assert height % 256 == 0
    height = height // 256


    babysteps = [i for i in range(16)]
    giantsteps = [i for i in range(-128,128,16)]

    num_babysteps = len(babysteps)
    num_giantsteps = len(giantsteps)

    babystep_rotate = [HoistedRotate(ct[i],babysteps) for i in range(height)]
    
    result = [None for j in range(width)]
    for j in range(width):
        acc0 = [None for _ in range(len(giantsteps))]
        acc1 = [None for _ in range(len(giantsteps))]
        for i in range(height):
            plaintexts0 = pt[j][i]
            plaintexts1 = pt[j][height+i]

            for (g,gs) in enumerate(giantsteps):
                for (b,bs) in enumerate(babysteps):
                    temp0 = babystep_rotate[i][b] * plaintexts0[b+g*num_babysteps]
                    temp1 = babystep_rotate[i][b] * plaintexts1[b+g*num_babysteps]
                    if b == 0:
                        bsSum0 = temp0
                        bsSum1 = temp1
                    else:
                        bsSum0 += temp0
                        bsSum1 += temp1

                if i == 0:
                    acc0[g] = bsSum0
                    acc1[g] = bsSum1
                else:
                    acc0[g] += bsSum0
                    acc1[g] += bsSum1
        
        result0 = RotateAccumulateMany(acc0,giantsteps)
        result1 = RotateAccumulateMany(acc1,giantsteps)

        if use_freqs_cis:
            rotation_indices0 = [-1,0,1]
            rotation_indices1 = [-129,-128,-127,127,128,129]
            products = [result0 * masks[j][i] for i in rotation_indices0] + [result1 * masks[j][i] for i in rotation_indices1]
            x = RotateAccumulateMany(products,rotation_indices0 + rotation_indices1)
            result[j] = (x + x.conjugate())
        else:
            products = [result0 * masks[j][0]] + [result1 * masks[j][i] for i in [-128,128]]
            result[j] = RotateAccumulateMany(products,[0,-128,128])

    return result

def matmul_ct128x4096_pt4096x4096(ct,pt,masks,use_freqs_cis=False):
    return matmul_ct128xHeight_ptHeightxWidth(height=4096,width=4096,ct=ct,pt=pt,masks=masks,use_freqs_cis=use_freqs_cis)

def matmul_ct128x4096_pt4096x1024(ct,pt,masks,use_freqs_cis=False):
    return matmul_ct128xHeight_ptHeightxWidth(height=4096,width=1024,ct=ct,pt=pt,masks=masks,use_freqs_cis=use_freqs_cis)

def matmul_ct128x4096_pt4096x14336(ct,pt,masks,use_freqs_cis=False):
    return matmul_ct128xHeight_ptHeightxWidth(height=4096,width=14336,ct=ct,pt=pt,masks=masks,use_freqs_cis=use_freqs_cis)

def matmul_ct128x14336_pt14336x4096(ct,pt,masks,use_freqs_cis=False):
    return matmul_ct128xHeight_ptHeightxWidth(height=14336,width=4096,ct=ct,pt=pt,masks=masks,use_freqs_cis=use_freqs_cis)


def get_plaintext_inputs_matmul(height,width,name,level,scale,period,remapable):
    assert width % 256 == 0
    width = width // 256

    assert height % 128 == 0
    height = height // 128

    plaintexts = [[[PeriodicPlaintextInput(f"{name}_{w}_{h}_{i}",scale,level,period,remapable) for i in range(-128,128)] for h in range(height)] for w in range(width)]
    return plaintexts

def get_plaintext_inputs_matmul_offset(height,heightOffset,width,widthOffset,name,level,scale,period,remapable):
    assert width % 256 == 0
    width = width // 256

    assert height % 128 == 0
    height = height // 128

    assert widthOffset % 256 == 0
    widthOffset = widthOffset // 256

    assert heightOffset % 128 == 0
    heightOffset = heightOffset // 128

    plaintexts = [[[PeriodicPlaintextInput(f"{name}_{w+widthOffset}_{h+heightOffset}_{i}",scale,level,period,remapable) for i in range(-128,128)] for h in range(height)] for w in range(width)]
    return plaintexts


def matmul_ct128x128_pt128x128(ct,pts):
    babySteps = [i for i in range(16)]
    giantSteps = [i*16 for i in range(-8,8)]
    product = BsgsMultiplyAccumulate(ct,pts,babySteps,giantSteps)
    return product


__matmul_ct128x128_ct128x128_transpose_function__ = None
def matmul_ct128x128_ct128x128_transpose_function(prefix):
    function = CeriumFunction("matmul_ct128x128_ct128x128_transpose", 1,0)
    with function:
        B = CiphertextArgument("B",84,9)

        A0 = CiphertextArgument("A0",84,9)
        A1 = CiphertextArgument("A1",84,9)


        mask0 = PeriodicPlaintextInput(f"mask_re",2*SCALE,A0.level() - 1,128)
        mask1 = PeriodicPlaintextInput(f"mask_im",2*SCALE,A1.level() - 1,128)
        mask = [mask0,mask1]
        
        Bs = MakeVector(HoistedRotate(B,[256*i for i in range(8)]))
        Bs = Bs.rescale()

        A0rot = MakeVector(HoistedRotate(A0,[-2048*i for i in range(16)]))
        A0rot = A0rot.rescale()
        A0rot = BreakVector(A0rot)

        A1rot = MakeVector(HoistedRotate(A1,[-2048*i for i in range(16)]))
        A1rot = A1rot.rescale()
        A1rot = BreakVector(A1rot)

        Arot = [A0rot,A1rot]

        for n in range(2):
            BabyStepsAccumulated = [None for _ in range(16)]
            for gs in range(16):
                Arot_rescale = Arot[n][gs]
                dot_prods = Bs * Arot_rescale
                dot_prods = dot_prods.relinearize()
                j = 1
                while j < 128:
                    dot_prods = dot_prods + (dot_prods << j)
                    j *= 2
                temp = dot_prods * mask[n % 2]
                temp = BreakVector(temp)
                sum_bs = RotateAccumulateMany(temp,[-bs for bs in range(8)])
                BabyStepsAccumulated[gs] = sum_bs
            sum = RotateAccumulateMany(BabyStepsAccumulated,[(2048-8)*gs for gs in range(16)])
            sum = sum.doubleRescale()
            sum = sum.doubleRescale()
            FunctionOutput(f"sum{n}",sum)

    global __matmul_ct128x128_ct128x128_transpose_function__
    if __matmul_ct128x128_ct128x128_transpose_function__ is None:
        __matmul_ct128x128_ct128x128_transpose_function__ = function

def matmul_ct128x128_ct128x128_transpose(A,B,prefix):
    A0, A1 = A
    assert __matmul_ct128x128_ct128x128_transpose_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(
        __matmul_ct128x128_ct128x128_transpose_function__,{"A0":A0,"A1":A1, "B":B},)
    return (outputs["sum0"],outputs["sum1"])

def exp(A,outputDepth):
    iters = 7
    oneBy2N = PlaintextInput("exp_oneBy2N",2*SCALE,A.level(),scalar=True)
    x = A * oneBy2N
    x = x.doubleRescale()
    one = PlaintextInput("exp_one",x.scale(),x.level(),scalar=True)
    x = one + x

    for i in range(5):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()

    x = bootstrap(x,"b35",outputDepth+2)
    bootstrap_sf = PlaintextInput("exp_bootstrap_sf",0,x.level(),scalar=True)
    x = x * bootstrap_sf
    for _ in range(2):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()

    return x

def MakePeriodicVectorRealImaginary(name,scale,level,period):
    return MakeVector([
        PeriodicPlaintextInput(f"{name}_re",scale,level,period),
        PeriodicPlaintextInput(f"{name}_im",scale,level,period),
    ])

def exp_vec(A,outputDepth):
    iters = 7
    oneBy2N = MakePeriodicVectorRealImaginary("exp_oneBy2N",2*SCALE,A.level(),1)
    x = A * oneBy2N
    x = x.doubleRescale()
    one = MakePeriodicVectorRealImaginary("exp_one",x.scale(),x.level(),1)
    x = one + x

    for i in range(5):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()

    x = bootstrap(x,"b35_2",outputDepth+2)
    bootstrap_sf = PlaintextInput("exp_bootstrap_sf",0,x.level(),scalar=True)
    x = x * bootstrap_sf
    for _ in range(2):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()

    return x


def inverse_sqrt2(x,outDepth,prefix):
    xInit = x
    zeroPt2 = PlaintextInput(f"{prefix}_isqrt_zeroPt2",2*SCALE,10,scalar=True)
    y = zeroPt2

    x = xInit
    for i in range(6):


        if i == 0: 
            y2 = PlaintextInput(f"{prefix}_isqrt_zeroPt04",2*SCALE,y.level()-2,scalar=True)
        else:
            y2 = (y * y).relinearize()
            y2 = y2.doubleRescale()

        x_ = x
        y_ = y
        while x_.level() > y_.level() + 2:
            x_ = x_.modswitch()
        xrescale = PlaintextInput(f"{prefix}_isqrt_xrescale_{i}",2*SCALE,x_.level(),scalar=True)
        x_ = (x_ * xrescale).doubleRescale()
        xy = (x_ * y_)
        if i != 0:
            xy = xy.relinearize()
        xy = xy.doubleRescale()
        xy3 = (xy * y2)
        if i != 0:
            xy3 = xy3.relinearize()

        if i == 0:
            yrescale = PlaintextInput(f"{prefix}_isqrt_yrescale_{i}",4*SCALE,y.level(),scalar=True)
            y_ = yrescale
        else:
            yrescale = PlaintextInput(f"{prefix}_isqrt_yrescale_{i}",2*SCALE,y.level(),scalar=True)
            y_ = y * yrescale
        y_ = y_.modswitch().modswitch()

        y = y_ - xy3

        y = y + y.conjugate()
        y = y.doubleRescale()
        if y.level() == 2: 
            if i != 5: 
                y = bootstrap(y,"b35",8)
            else:
                y = bootstrap(y,"b35",outDepth)
        else:
            pass

    bootstrap_sf = PlaintextInput(f"{prefix}_isqrt_y_bootstrap_sf",0,y.level(),scalar=True)
    y = y * bootstrap_sf
    return y

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
        # TestOutput(f"powsx_{2**i}",xx)
    xLevel = x.level()

    x7 = (pows[1] * coeffs[7]).rescale()
    x7 = (x7 * pows[2])
    x7 = relineraize(x7)
    x7 = x7.rescale()
    x7 = (x7 * pows[4])
    x7 = relineraize(x7)

    x5 = (pows[1] * coeffs[5]).doubleRescale()
    # x5 = (x5 * pows[4]).relinearize().rescale(SCALE)
    x5 = (x5 * pows[4])
    x5 = relineraize(x5)

    x3 = (pows[1] * coeffs[3]).doubleRescale()
    # x3 = (x3 * pows[2].modswitch()).relinearize().rescale(SCALE)
    x3 = (x3 * pows[2].modswitch())
    x3 = relineraize(x3)

    # x1 = (pows[1] * coeffs[1]).rescale(SCALE).rescale(SCALE)
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
        # TestOutput(f"powsx_{2**i}_{xx.level()}",xx)
    xLevel = x.level()

    x7 = (pows[1] * coeffs[7]).doubleRescale()
    x7 = (x7 * pows[2])
    x7 = relineraize(x7)
    x7 = x7.doubleRescale()
    x7 = (x7 * pows[4])
    x7 = relineraize(x7)

    # TestOutput(f"x7_{x7.level()}",(x7 + x7).doubleRescale())

    x5 = (pows[1] * coeffs[5]).doubleRescale()
    x5 = x5.modswitch().modswitch()
    # x5 = (x5 * pows[4]).relinearize().rescale(SCALE)
    x5 = (x5 * pows[4])
    x5 = relineraize(x5)

    # TestOutput(f"x5_{x5.level()}",(x5 + x5).doubleRescale())

    x3 = (pows[1] * coeffs[3]).doubleRescale()
    x3 = x3.modswitch().modswitch()
    # x3 = (x3 * pows[2].modswitch()).relinearize().rescale(SCALE)
    x3 = (x3 * pows[2].modswitch().modswitch())
    x3 = relineraize(x3)

    # TestOutput(f"x3_{x3.level()}",(x3 + x3).doubleRescale())

    # x1 = (pows[1] * coeffs[1]).rescale(SCALE).rescale(SCALE)
    x1 = (pows[1] * coeffs[1])#.rescale().rescale()
    x1 = x1.modswitch().modswitch().modswitch().modswitch()

    # TestOutput(f"x1_{x1.level()}",(x1 + x1).doubleRescale())

    x = x1 + x3 + x5 + x7
    x = x + x.conjugate()
    return x


def sign_new_short(x,prefix,bootstrapAfter=True):
    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_g3_coeff1_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[3] = getPlaintextInput(f"{prefix}_g3_coeff3_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[5] = getPlaintextInput(f"{prefix}_g3_coeff5_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[7] = getPlaintextInput(f"{prefix}_g3_coeff7_{xLevel}",SCALE,x.level(),scalar=True)
    x = g3_f3(x,coeffs)
    x = x.rescale()
    # TestOutput(f"{prefix}_x_{xLevel}",x)

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_g3_coeff1_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[3] = getPlaintextInput(f"{prefix}_g3_coeff3_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[5] = getPlaintextInput(f"{prefix}_g3_coeff5_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[7] = getPlaintextInput(f"{prefix}_g3_coeff7_{xLevel}",SCALE,x.level(),scalar=True)
    x = g3_f3(x,coeffs)
    x = x.rescale()
    # TestOutput(f"{prefix}_x_{xLevel}",x)

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_f3_coeff1_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[3] = getPlaintextInput(f"{prefix}_f3_coeff3_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[5] = getPlaintextInput(f"{prefix}_f3_coeff5_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[7] = getPlaintextInput(f"{prefix}_f3_coeff7_{xLevel}",SCALE,x.level(),scalar=True)
    x = g3_f3(x,coeffs)
    x = x.rescale()
    # TestOutput(f"{prefix}_x_{xLevel}",x)

    xLevel = x.level()
    if bootstrapAfter:
        BootstrapScaleDiv = 0
    else:
        BootstrapScaleDiv = SCALE
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_f3_coeff1_{xLevel}",3*SCALE - BootstrapScaleDiv,x.level(),scalar=True)
    coeffs[3] = getPlaintextInput(f"{prefix}_f3_coeff3_{xLevel}",3*SCALE - BootstrapScaleDiv,x.level(),scalar=True)
    coeffs[5] = getPlaintextInput(f"{prefix}_f3_coeff5_{xLevel}",3*SCALE - BootstrapScaleDiv,x.level(),scalar=True)
    coeffs[7] = getPlaintextInput(f"{prefix}_f3_coeff7_{xLevel}",2*SCALE - BootstrapScaleDiv,x.level(),scalar=True)
    x = g3_f3(x,coeffs,relinType=2)
    return x

def max_new_vec_short(A,B,i):
    AB = A + B
    AB_ = A - B
    # TestOutput(f"a",A + A)
    # TestOutput(f"b",B + B)
    # TestOutput(f"ab",AB)
    # TestOutput(f"ab_",AB_)
    # s = sign_new(AB_,"max")
    # s = sign_new(AB_,"max_vec")

    AB_rescale = BreakVector(AB_.rescale())
    s = [sign_new_short(AB_rescale[i],"max",bootstrapAfter=False) for i in range(2)]
    s = MakeVector(s)

    # s = sign_new_vec_short(AB_.rescale(),"max",bootstrapAfter=False)
    # Output(f"sign_{i}",s)
    zeroPt5_0 = MakeVector([getPeriodicPlaintextInput("zeroPt5_0",2*SCALE,AB.level(),period=2),getPeriodicPlaintextInput("zeroPt5_0_im",2*SCALE,AB.level(),period=2)])
    AB =(AB * zeroPt5_0)
    zeroPt5_1 = MakeVector([getPeriodicPlaintextInput("zeroPt5_1",2*SCALE,AB_.level(),period=2),getPeriodicPlaintextInput("zeroPt5_1_im",2*SCALE,AB_.level(),period=2)])
    AB_ =(AB_ * zeroPt5_1).doubleRescale()
    # TestOutput("5AB",AB)
    # TestOutput("5AB_",AB_)
    while AB_.level() != s.level():
        AB_ = AB_.modswitch()
    
    # result = (AB_ * s).relinearize()
    AB_ = BreakVector(AB_)
    s = BreakVector(s)
    result = [(AB_[i] * s[i]).relinearize2() for i in range(2)]
    result = MakeVector(result)
    # TestOutput("result",result)
    while AB.level() != result.level():
        AB = AB.modswitch()
    result = result + AB
    result = result.doubleRescale()
    # TestOutput("max",result)
    return result


def array_max_new_vec_short(A):
    x = A
    for i in range(7):
        s = BreakVector(x)
        s = [mm.rotate2(-(2**i)) for mm in s]
        s = MakeVector(s)
        s = max_new_vec_short(s,x,i)
        x = bootstrap(s,"b35_2",14)
        scaleFactor = PlaintextInput("array_max_bootstrap_scale_factor",0,x.level(),scalar=True)
        x = x * scaleFactor
    forty = PeriodicPlaintextInput("array_max2_eighty",2*SCALE,x.level(),128)
    x = x * forty
    for i in range(7):
        x = x + (x << 2**i)
    x = x.doubleRescale()
    return x


def TestFunctionOutputVec2(name,x):
    if not TEST: return
    x = BreakVector(x)
    TestFunctionOutput(f"{name}0",x[0])
    TestFunctionOutput(f"{name}1",x[1])

__softmax_128x128_vec_function__ = None
def softmax_128x128_vec_function(suffix=""):
    function = CeriumFunction(f"softmax_128x128_vec{suffix}",1,0)
    with function:
        A0 = CiphertextArgument("A0",2*SCALE,2)
        A1 = CiphertextArgument("A1",2*SCALE,2)
        A = MakeVector([A0,A1])
        A = bootstrap(A,"b35_2",14)
        A_sf = PlaintextInput("QK_bs_scale_factor",0,A.level(),scalar=True)
        A *= A_sf
        mVec = array_max_new_vec_short(A)
        
        m = mVec
        TestFunctionOutputVec2(f"max_values", m)
        forty = PlaintextInput("softmax_80",2*SCALE,A.level(),scalar=True)
        A = A*forty
        A = A.doubleRescale()
        A = A - m
        TestFunctionOutputVec2(f"max_values_sub", A)
        e = exp_vec(A,12)
        TestFunctionOutputVec2(f"exp", e)
        att_mask = CiphertextInput("att_mask",2*SCALE,e.level())
        e = (e * att_mask).relinearize()
        sum = e
        e = e.doubleRescale()
        TestFunctionOutputVec2(f"exp_att_mask", e)
        for i in range(7):
            sum = sum + (sum >> 2**i)
        sum = sum.rescale()
        one = MakePeriodicVectorRealImaginary("softmax_one",2*SCALE,sum.level(),128)
        sum = sum * one
        for i in range(7):
            sum = sum + (sum << 2**i)
        sum = sum.doubleRescale().rescale()
        TestFunctionOutputVec2(f"softmax_sum_bs", sum)
        sum = bootstrap(sum,"b35_2",12)
        sum_sf = PlaintextInput("softmax_sum_sf",0,sum.level(),scalar=True)
        sum = sum * sum_sf
        TestFunctionOutputVec2(f"softmax_sum", sum)
        sum = BreakVector(sum)
        isqrt = [inverse_sqrt2(sum_,10,"softmax") for sum_ in sum]
        isqrt = MakeVector(isqrt)
        inv = (isqrt * isqrt).relinearize().doubleRescale()
        TestFunctionOutputVec2(f"inverse", inv)
        s = e * inv
        s = s.relinearize()
        s = s.rescale()
        s = BreakVector(s)
        for n in range(2):
            FunctionOutput(f"s{n}",s[n])

    global __softmax_128x128_vec_function__
    __softmax_128x128_vec_function__ = function



def softmax_128x128_vec(A0,A1,prefix):
    assert __softmax_128x128_vec_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__softmax_128x128_vec_function__,{"A0":A0,"A1":A1},)
    s0 = outputs["s0"]
    s1 = outputs["s1"]
    test_outputs = []
    # test_outputs = [f"array_max_{i}" for i in range(7)]
    # test_outputs += [f"sign_{i}" for i in range(7)]
    for i in range(2):
        # test_outputs += [f"max_values{i}", f"max_values_sub{i}", f"exp{i}", f"exp_att_mask{i}", f"softmax_sum_bs{i}", f"softmax_sum{i}", f"inverse{i}"]
        test_outputs += [f"exp{i}", f"exp_att_mask{i}", f"softmax_sum_bs{i}", f"softmax_sum{i}", f"inverse{i}"]
        # test_outputs += [f"isqrt_exp{i}"]
        # test_outputs += [f"isqrt_y_bs{i}" for i in [1,3,5]]
        # test_outputs += [f"isqrt_y_{i}" for i in [0,2,4]]
    # for output_name in test_outputs:
        # TestOutput(f"{prefix}_{output_name}", outputs[output_name])
    return (s0,s1)

__matmul_ct128x128_ct128x128_function__ = None
def matmul_ct128x128_ct128x128_function():
    function = CeriumFunction("matmul_ct128x128_ct128x128",1,0)
    with function:
        ct1 = [CiphertextArgument(f"A{i}",3*SCALE,9) for i in range(4)]
        ct2 = CiphertextArgument("B",3*SCALE,10)
        babySteps = [i*256 for i in range(8)]
        giantSteps = [i*256*8 for i in range(0,16)]
        pts = [PeriodicPlaintextInput(f"mask_128x128_rot_{i*128}",SCALE+SCALE//2,ct2.level(),128) for i in range(0,128)]
        diag = BsgsMultiplyAccumulate(ct2,pts,babySteps,giantSteps,rescaleLevels=1)

        diag_bs = HoistedRotate(diag, [(256*bs) for bs in range(8)])

        diag_bs_vec = MakeVector(diag_bs)
        diag_bs_vec = diag_bs_vec.doubleRescale()

        for n in range(4):
            ct1_gs = HoistedRotate(ct1[n],[(8*gs -2048*gs) for gs in range(16)])

            babysteps_accumulated = [None for _ in range(16)]

            for gs in range(0,128//8,1):
                l = MakeVector(HoistedRotate(ct1_gs[gs],[bs for bs in range(8)]))
                r = MakeVector(HoistedRotate(ct1_gs[gs],[(-128 + bs) for bs in range(8)]))
                diagRot = diag_bs_vec
                maskL = MakeVector([getPeriodicPlaintextInput(f"maskL_128x128_{8*gs + bs}",SCALE+SCALE//2,l.level(),128) for bs in range(8)])
                maskR = MakeVector([getPeriodicPlaintextInput(f"maskR_128x128_{8*gs + bs}",SCALE+SCALE//2,r.level(),128) for bs in range(8)])
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
            FunctionOutput(f"sumi{n}",sumi)
        
    global __matmul_ct128x128_ct128x128_function__
    __matmul_ct128x128_ct128x128_function__ = function

    
def matmul_ct128x128_ct128x128(A,B,prefix=""):
    assert __matmul_ct128x128_ct128x128_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__matmul_ct128x128_ct128x128_function__,{"A0":A[0],"A1":A[1],"A2": A[2], "A3": A[3], "B":B})
    sumi = [outputs[f"sumi{i}"] for i in range(4)]


    test_outputs = []
    test_outputs += ["diag"]
    test_outputs += [f"diag_bs_{i}" for i in range(8)]
    # test_outputs = [f"array_max_{i}" for i in range(7)]
    # test_outputs += [f"sign_{i}" for i in range(7)]
    # for i in range(2):
        # test_outputs += [f"max_values{i}", f"max_values_sub{i}", f"exp{i}", f"exp_att_mask{i}", f"softmax_sum_bs{i}", f"softmax_sum{i}", f"inverse{i}"]
        # test_outputs += [f"exp{i}", f"exp_att_mask{i}", f"softmax_sum_bs{i}", f"softmax_sum{i}", f"inverse{i}"]
        # test_outputs += [f"isqrt_exp{i}"]
        # test_outputs += [f"isqrt_y_bs{i}" for i in [1,3,5]]
        # test_outputs += [f"isqrt_y_{i}" for i in [0,2,4]]
    # for output_name in test_outputs:
        # TestOutput(f"{prefix}_{output_name}", outputs[output_name])
    return sumi
  

__attention_matmul_qkv_function__ = None
def attention_matmul_qkv_function():
    function = CeriumFunction(f"attention_matmul_qkv_1gpu",1,0)
    with function:
        prefix = "attention"
        x = [CiphertextArgument(f"X{i}",3*SCALE,13) for i in range(16)]
        level = x[0].level()

        masks = {i: getPlaintextInput(f"{prefix}_mask_{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}
        masks_freqs_cis = {i: getPlaintextInput(f"{prefix}_mask_freqs_cis_{i}",2*SCALE,level) for i in [-1,0,1,-129,-128,-127,127,128,129]}

        masks = [masks]*16
        masks_freqs_cis = [masks_freqs_cis]*16

        wq = get_plaintext_inputs_matmul(height=4096,width=4096,name=f"{prefix}_Wq",level=level,scale=2*SCALE,period=256,remapable=True)
        wk = get_plaintext_inputs_matmul(height=4096,width=1024,name=f"{prefix}_Wk",level=level,scale=2*SCALE,period=256,remapable=True)
        wv = get_plaintext_inputs_matmul(height=4096,width=1024,name=f"{prefix}_Wv",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

        Q, K, V = None, None, None

        Q = matmul_ct128x4096_pt4096x4096(x,wq,masks_freqs_cis,use_freqs_cis=True)
        K = matmul_ct128x4096_pt4096x1024(x,wk,masks_freqs_cis,use_freqs_cis=True)
        V = matmul_ct128x4096_pt4096x1024(x,wv,masks,use_freqs_cis=False)

        for i in range(16):
            Q[i] = Q[i].doubleRescale().doubleRescale()
            FunctionOutput(f"Q{i}",Q[i])

        for i in range(4):
            K[i] = K[i].doubleRescale().doubleRescale()
            FunctionOutput(f"K{i}",K[i])

        for i in range(4):
            V[i] = V[i].doubleRescale().rescale()
            FunctionOutput(f"V{i}",V[i])


    global __attention_matmul_qkv_function__
    __attention_matmul_qkv_function__ = function



def attention_matmul_qkv(x,block):
    assert __attention_matmul_qkv_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_matmul_qkv_function__,{f"X{i}":x[i] for i in range(16)},block)
    Q = [outputs[f"Q{i}"] for i in range(16)]
    K = [outputs[f"K{i}"] for i in range(4)]
    V = [outputs[f"V{i}"] for i in range(4)]
    return Q,K,V

__attention_matmul_o_function__ = None
def attention_matmul_o_function():
    function = CeriumFunction(f"attention_matmul_o_1gpu",1,0)
    with function:
        prefix = "attention"
        x = [CiphertextArgument(f"X{i}",3*SCALE,6) for i in range(16)]
        level = x[0].level()

        masks_re = {i: getPlaintextInput(f"{prefix}_maskO_re{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}
        masks_im = {i: getPlaintextInput(f"{prefix}_maskO_im{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

        masks = [masks_re,masks_im]*8

        wo = get_plaintext_inputs_matmul(height=4096,width=4096,name=f"{prefix}_Wo",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

        O = matmul_ct128x4096_pt4096x4096(x,wo,masks)

        for i in range(16):
            O[i] = O[i].doubleRescale().doubleRescale()
            FunctionOutput(f"O{i}",O[i])


    global __attention_matmul_o_function__
    __attention_matmul_o_function__ = function


def attention_matmul_o(x,block):
    assert __attention_matmul_o_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_matmul_o_function__,{f"X{i}":x[i] for i in range(16)},block)
    O = [outputs[f"O{i}"] for i in range(16)]
    return O



def attention(x,prefix,block,testOutputPredicate):

    Q,K,V = attention_matmul_qkv(x,block)
    for i in range(16):
        TestOutput(f"{prefix}_Q{i}",Q[i])
    for i in range(4):
        TestOutput(f"{prefix}_K{i}",K[i])
        TestOutput(f"{prefix}_V{i}",V[i])

    QK = [None] * 16
    for i in range(0,16,2):
        QK0, QK1 = matmul_ct128x128_ct128x128_transpose((Q[i],Q[i+1]),K[i//4],f"attention_{i}")
        QK[i] = QK0.modswitch().modswitch()
        QK[i+1] = QK1.modswitch().modswitch()
        TestOutput(f"{prefix}_QK_bs{i}",QK0,testOutputPredicate)
        TestOutput(f"{prefix}_QK_bs{i+1}",QK1,testOutputPredicate)

    softmax = [None] * 16
    for i in range(0,16,2):
        S0, S1 = softmax_128x128_vec(QK[i],QK[i+1],f"attention_{i}")
        softmax[i] = S0
        softmax[i+1] = S1
        TestOutput(f"{prefix}_softmax{i}",softmax[i],testOutputPredicate)
        TestOutput(f"{prefix}_softmax{i+1}",softmax[i+1],testOutputPredicate)

    attention = [None] * 16
    for i in range(0,16,4):
        ret = matmul_ct128x128_ct128x128(softmax[i:i+4],V[i//4],f"attention_{i}")
        attention[i] = ret[0]
        attention[i+1] = ret[1]
        attention[i+2] = ret[2]
        attention[i+3] = ret[3]
        TestOutput(f"{prefix}_attention{i}",attention[i],testOutputPredicate)
        TestOutput(f"{prefix}_attention{i+1}",attention[i+1],testOutputPredicate)
        TestOutput(f"{prefix}_attention{i+2}",attention[i+2],testOutputPredicate)
        TestOutput(f"{prefix}_attention{i+3}",attention[i+3],testOutputPredicate)

    O = attention_matmul_o(attention,block)
    for i in range(16):
        TestOutput(f"{prefix}_attention_out{i}",O[i])

    return (O)

__attention_matmul_qkv_parallel_function__ = None
def attention_matmul_qkv_parallel_function(gpus):
    assert gpus in [2,4,8]
    function = CeriumFunction(f"attention_matmul_qkv_{gpus}gpu",gpus,0)
    with function:
        prefix = "attention"
        x = [CiphertextArgument(f"X{i}",3*SCALE,13) for i in range(16)]
        level = x[0].level()

        def attention_q_streamfn(sid,X,Q,numStreams):
            if sid >= numStreams: return
            masks_freqs_cis = {i: getPlaintextInput(f"{prefix}_mask_freqs_cis_{i}",2*SCALE,level) for i in [-1,0,1,-129,-128,-127,127,128,129]}

            assert 16 % numStreams == 0
            count = 16 // numStreams

            masks_freqs_cis = [masks_freqs_cis]*count

            width = 4096 // numStreams

            wq = get_plaintext_inputs_matmul_offset(height=4096,heightOffset=0,width=width,widthOffset=sid * width,name=f"{prefix}_Wq",level=level,scale=2*SCALE,period=256,remapable=True)
            # wk = get_plaintext_inputs_matmul(height=4096,width=1024,name=f"{prefix}_Wk",level=level,scale=2*SCALE,period=256,remapable=True)
            # wv = get_plaintext_inputs_matmul(height=4096,width=1024,name=f"{prefix}_Wv",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

            # Q, K, V = None, None, None

            Q_ = matmul_ct128xHeight_ptHeightxWidth(height=4096,width=width,ct=X,pt=wq,masks=masks_freqs_cis,use_freqs_cis=True)
            Q_ = [q.doubleRescale().doubleRescale() for q in Q_]
            Q[sid*count : (sid+1)*count] = Q_

        streamSize = 1
        numStreams = gpus
        Q = [None]*16
        CeriumStream(streamSize,numStreams,attention_q_streamfn,x,Q,numStreams)

        def attention_kv_streamfn(sid,X,K,V,numStreams):
            if sid >= numStreams: return

            masks_freqs_cis = {i: getPlaintextInput(f"{prefix}_mask_freqs_cis_{i}",2*SCALE,level) for i in [-1,0,1,-129,-128,-127,127,128,129]}
            masks = {i: getPlaintextInput(f"{prefix}_mask_{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

            assert 4 % numStreams == 0
            count = 4 // numStreams

            masks = [masks]*count
            masks_freqs_cis = [masks_freqs_cis]*count

            width = 1024 // numStreams

            wk = get_plaintext_inputs_matmul_offset(height=4096,heightOffset=0,width=width,widthOffset=sid * width,name=f"{prefix}_Wk",level=level,scale=2*SCALE,period=256,remapable=True)
            wv = get_plaintext_inputs_matmul_offset(height=4096,heightOffset=0,width=width,widthOffset=sid * width,name=f"{prefix}_Wv",level=level,scale=SCALE + SCALE//2,period=256,remapable=True)

            K_ = matmul_ct128xHeight_ptHeightxWidth(height=4096,width=width,ct=X,pt=wk,masks=masks_freqs_cis,use_freqs_cis=True)
            K_ = [k.doubleRescale().doubleRescale() for k in K_]
            K[sid*count : (sid+1)*count] = K_


            V_ = matmul_ct128xHeight_ptHeightxWidth(height=4096,width=width,ct=X,pt=wv,masks=masks,use_freqs_cis=False)
            V_ = [v.doubleRescale().rescale() for v in V_]
            V[sid*count : (sid+1)*count] = V_

        streamSize = max(1,gpus//4)
        numStreams = min(4,gpus)
        K = [None]*4
        V = [None]*4
        CeriumStream(streamSize,numStreams,attention_kv_streamfn,x,K,V,numStreams)

        # raise Exception("")


        for i in range(16):
            FunctionOutput(f"Q{i}",Receive(Q[i]))

        for i in range(4):
            FunctionOutput(f"K{i}",Receive(K[i]))

        for i in range(4):
            FunctionOutput(f"V{i}",Receive(V[i]))


    global __attention_matmul_qkv_parallel_function__
    __attention_matmul_qkv_parallel_function__ = function



def attention_matmul_qkv_parallel(x,block):
    assert __attention_matmul_qkv_parallel_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_matmul_qkv_parallel_function__,{f"X{i}":x[i] for i in range(16)},block)
    Q = [outputs[f"Q{i}"] for i in range(16)]
    K = [outputs[f"K{i}"] for i in range(4)]
    V = [outputs[f"V{i}"] for i in range(4)]
    return Q,K,V

__attention_matmul_o_parallel_function__ = None
def attention_matmul_o_parallel_function(gpus):
    assert gpus  in [2,4,8]
    function = CeriumFunction(f"attention_matmul_o_{gpus}gpu",gpus,0)
    with function:
        prefix = "attention"
        x = [CiphertextArgument(f"X{i}",3*SCALE,6) for i in range(16)]
        level = x[0].level()

        def attention_o_streamfn(sid,X,O,numStreams):
            if sid >= numStreams: return
            masks_re = {i: getPlaintextInput(f"{prefix}_maskO_re{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}
            masks_im = {i: getPlaintextInput(f"{prefix}_maskO_im{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

            count = 16 // numStreams

            masks = [masks_re,masks_im]*(count//2)

            width = 4096 // numStreams

            wo = get_plaintext_inputs_matmul_offset(height=4096,heightOffset=0,width=width,widthOffset=sid * width,name=f"{prefix}_Wo",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

            O_ = matmul_ct128xHeight_ptHeightxWidth(height=4096,width=width,ct=X,pt=wo,masks=masks)
            O_ = [o.doubleRescale().doubleRescale() for o in O_]
            O[sid*(count) : (sid+1)*(count)] = O_

        streamSize = 1
        numStreams = gpus
        O = [None]*16
        CeriumStream(streamSize,numStreams,attention_o_streamfn,x,O,numStreams)

        for i in range(16):
            # O[i] = O[i].doubleRescale().doubleRescale()
            FunctionOutput(f"O{i}",Receive(O[i]))


    global __attention_matmul_o_parallel_function__
    __attention_matmul_o_parallel_function__ = function


def attention_matmul_o_parallel(x,block):
    assert __attention_matmul_o_parallel_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__attention_matmul_o_parallel_function__,{f"X{i}":x[i] for i in range(16)},block)
    O = [outputs[f"O{i}"] for i in range(16)]
    return O


__attention_score_softmax_parallel_function__ = None
def attention_score_softmax_parallel_function(gpus):
    assert gpus  in [2,4,8]
    function = CeriumFunction(f"attention_score_softmax_{gpus}gpu",gpus,0)
    with function:
        Q = [CiphertextArgument(f"Q{i}",3*SCALE,9) for i in range(16)]
        K = [CiphertextArgument(f"K{i}",3*SCALE,9) for i in range(4)]

        def softmax_streamfn(sid,Q,K,Softmax):
            if sid >= 8: return

            i = 2 * sid

            QK0, QK1 = matmul_ct128x128_ct128x128_transpose((Receive(Q[i]),Receive(Q[i+1])),Receive(K[i//4]),f"attention_{i}")
            QK0 = QK0.modswitch().modswitch()
            QK1 = QK1.modswitch().modswitch()

            S0, S1 = softmax_128x128_vec(QK0,QK1,f"attention_{i}")
            Softmax[i] = S0
            Softmax[i+1] = S1

        softmax = [None] * 16
        CeriumStream(1,8,softmax_streamfn,Q,K,softmax)

        for i in range(16):
            FunctionOutput(f"O{i}",Receive(softmax[i]))


    global __attention_score_softmax_parallel_function__
    __attention_score_softmax_parallel_function__ = function



def attention_score_softmax_parallel(Q,K):
    assert __attention_score_softmax_parallel_function__ is not None, "Function not defined"
    inputs = {f"Q{i}":Q[i] for i in range(16)}
    inputs.update({f"K{i}":K[i] for i in range(4)})
    outputs = CeriumFunctionCall(__attention_score_softmax_parallel_function__,inputs)
    O = [outputs[f"O{i}"] for i in range(16)]
    return O




def attention_parallel(x,prefix,block,testOutputPredicate):

    Q,K,V = attention_matmul_qkv_parallel(x,block)
    for i in range(16):
        TestOutput(f"{prefix}_Q{i}",Q[i],testOutputPredicate)
    for i in range(4):
        TestOutput(f"{prefix}_K{i}",K[i],testOutputPredicate)
        TestOutput(f"{prefix}_V{i}",V[i],testOutputPredicate)

    softmax = attention_score_softmax_parallel(Q,K)

    for i in range(16):
        TestOutput(f"{prefix}_softmax{i}",softmax[i],testOutputPredicate)

    def attention_streamfn(sid,softmax,V,attention):
        if sid >= 4: return
        i = 4 * sid
        softmax[i:i+4] = [Receive(s) for s in softmax[i:i+4]]
        # for i in range(0,16,4):
        ret = matmul_ct128x128_ct128x128(softmax[i:i+4],Receive(V[i//4]),f"attention_{i}")
        attention[i:i+4] = ret
        for j in range(4):
            TestOutput(f"{prefix}_attention{i+j}",attention[i+j],testOutputPredicate)

    attention = [None] * 16
    CeriumStream(1,4,attention_streamfn,softmax,V,attention)

    attention = [Receive(a) for a in attention]

    O = attention_matmul_o_parallel(attention,block)
    for i in range(16):
        TestOutput(f"{prefix}_attention_out{i}",O[i],testOutputPredicate)

    return (O)


def inverse_sqrt(x,outDepth,prefix):
    xInit = x
    minusZeroPtFour = PlaintextInput(f"{prefix}_isqrt_minus_zero_pt_4",x.scale(),x.level(),scalar=True)
    x = minusZeroPtFour - x 
    oneBy2N = PlaintextInput(f"{prefix}_isqrt_oneBy2N",2*SCALE,x.level(),scalar=True)
    x = x * oneBy2N
    x = x.doubleRescale()
    one = PlaintextInput(f"{prefix}_isqrt_one",x.scale(),x.level(),scalar=True)
    x = one + x

    for i in range(4):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()


    x = bootstrap(x,"b35",14)
    bootstrap_sf = PlaintextInput(f"{prefix}_isqrt_exp_bootstrap_sf",0,x.level(),scalar=True)
    x = x * bootstrap_sf
    for _ in range(3):
        x = x * x
        x = x.relinearize()
        x = x.doubleRescale()
    y = x + x
    zeroPt2 = PlaintextInput(f"{prefix}_isqrt_zeroPt2",y.scale(),y.level(),scalar=True)
    y = y + zeroPt2
    TestFunctionOutput("isqrt_y",y)

    x = xInit
    for i in range(10):

        
        y2 = (y * y).relinearize()
        y2 = y2.doubleRescale()

        x_ = x
        y_ = y
        while x_.level() > y_.level() + 2:
            x_ = x_.modswitch()
        xrescale = PlaintextInput(f"{prefix}_isqrt_xrescale_{i}",2*SCALE,x_.level(),scalar=True)
        x_ = (x_ * xrescale).doubleRescale()
        xy = (x_ * y_).relinearize()
        xy = xy.doubleRescale()
        xy3 = (xy * y2).relinearize()

        yrescale = PlaintextInput(f"{prefix}_isqrt_yrescale_{i}",2*SCALE,y.level(),scalar=True)
        y_ = y * yrescale
        y_ = y_.modswitch().modswitch()

        y = y_ - xy3

        y = y + y.conjugate()
        y = y.doubleRescale()
        if y.level() == 2: 
            TestFunctionOutput(f"isqrt_y_bs{i}",y)
            if i != 9: 
                y = bootstrap(y,"b35",8)
            else:
                y = bootstrap(y,"b35",outDepth)
        else:
            TestFunctionOutput(f"isqrt_y_{i}",y)

    bootstrap_sf = PlaintextInput(f"{prefix}_isqrt_y_bootstrap_sf",0,y.level(),scalar=True)
    y = y * bootstrap_sf
    return y



__rmsnorm_function__ = {"att": None, "ffn": None}
def rmsnorm_function(norm_type):
    function = CeriumFunction(f"rmsnorm_{norm_type}_1gpu",1,0)
    with function:
        x = [CiphertextArgument(f"X{i}",2*SCALE,2) for i in range(16)]

        for i in range(0,16,2):
            x[i:i+2] = BreakVector(bootstrap(MakeVector(x[i:i+2]),"b35_2",14))

        x_rescale_factor = PlaintextInput(f"rmsnorm_{norm_type}_x_rescale_factor",0,x[0].level(),scalar=True)
        x = [xx * x_rescale_factor for xx in x]
        x_sq = [xx*xx for xx in x]

        w_sum = x_sq[0]
        for i in range(1,16):
            w_sum += x_sq[i]
        w_sum = w_sum.relinearize()
        for i in range(8):
            w_sum += (w_sum >> 2**i)

        eps = PlaintextInput("rmsnorm_eps",w_sum.scale(),w_sum.level(),scalar=True)
        w_sum += eps

        TestFunctionOutput("w_sum",w_sum)

        oneByN = PlaintextInput("rmsnorm_oneByN",2*SCALE,w_sum.level(),remapable=True)
        w_mean = w_sum * oneByN

        for i in range(8):
            w_mean += (w_mean << 2**i)
        w_mean = w_mean.doubleRescale().doubleRescale()

        TestFunctionOutput("w_mean",w_mean)

        isqrt = inverse_sqrt(w_mean,14,"rmsnorm")

        TestFunctionOutput("inv_sqrt",isqrt)

        res = [None]*16
        for i in range(16):
            weight = PlaintextInput(f"rmsnorm_{norm_type}_weight{i}",2*SCALE,isqrt.level(),remapable=True)
            res[i] = ((isqrt * weight) * x[i]).relinearize().doubleRescale().rescale()
            FunctionOutput(f"res{i}",res[i])


    global __rmsnorm_function__
    __rmsnorm_function__[norm_type] = function


def rmsnorm(norm_type,x,prefix,block):
    assert norm_type in ["att","ffn"], "Invalid Type for rmsnorm"
    assert __rmsnorm_function__[norm_type] is not None, f"Function rmsnorm_{norm_type} not defined"
    outputs = CeriumFunctionCall(__rmsnorm_function__[norm_type],{f"X{i}":x[i] for i in range(16)},block)
    test_outputs = []
    test_outputs += ["w_sum","w_mean","inv_sqrt"]
    test_outputs += ["isqrt_y"]
    test_outputs += [f"isqrt_y_{i}" for i in range(0,10,2)]
    # test_outputs += ["inv_sqrt_div"]

    # for k in test_outputs:
        # TestOutput(f"{prefix}_{k}",outputs[k])
    
    res = [outputs[f"res{i}"] for i in range(16)]
    return res

__rmsnorm_function_parallel__ = {"att": None, "ffn": None}
def rmsnorm_function_parallel(norm_type,gpus):
    function = CeriumFunction(f"rmsnorm_{norm_type}_{gpus}gpu",gpus,0)
    with function:
        x = [CiphertextArgument(f"X{i}",2*SCALE,2) for i in range(16)]

        def bootstrap_stream_fn(sid,x,x_sq_sum):
            if sid >= 8:
                return
            pid = CurrentPartitionID()
            i = 2*sid
            x[i:i+2] = BreakVector(bootstrap(MakeVector(x[i:i+2]),"b35_2",14))
            x_rescale_factor = PlaintextInput(f"rmsnorm_{norm_type}_x_rescale_factor",0,x[0].level(),scalar=True)
            x[i] = x[i] * x_rescale_factor
            x[i+1] = x[i+1] * x_rescale_factor
            x_sq0 = (x[i]*x[i]).relinearize()
            x_sq1 = (x[i+1]*x[i+1]).relinearize()
            print(CurrentPartitionID(),i,x_sq0.level())
            if x_sq_sum[pid] is None:
                x_sq_sum[pid] = (x_sq0 + x_sq1)
            else:
                x_sq_sum[pid] += (x_sq0 + x_sq1)
            

        streamSize = 1
        numStreams = 8
        x_sq_sum = [None]*min(8,gpus)
        CeriumStream(streamSize,numStreams,bootstrap_stream_fn,x,x_sq_sum)

        x = [Receive(xi) for xi in x]

        w_sum = x_sq_sum[0]
        for i in range(1,min(8,gpus)):
            w_sum += x_sq_sum[i]

        def normalization_stream_fn(sid,w_sum_,res):
            if sid > 1:
                return
            for i in range(8):
                w_sum_ += (w_sum_ >> 2**i)

            eps = PlaintextInput("rmsnorm_eps",w_sum_.scale(),w_sum_.level(),scalar=True)
            w_sum_ += eps

            TestFunctionOutput("w_sum",w_sum)

            oneByN = PlaintextInput("rmsnorm_oneByN",2*SCALE,w_sum_.level(),remapable=True)
            w_mean = w_sum_ * oneByN

            for i in range(8):
                w_mean += (w_mean << 2**i)
            w_mean = w_mean.doubleRescale().doubleRescale()

            TestFunctionOutput("w_mean",w_mean)

            isqrt = inverse_sqrt(w_mean,14,"rmsnorm")

            TestFunctionOutput("inv_sqrt",isqrt)

            for i in range(16):
                weight = PlaintextInput(f"rmsnorm_{norm_type}_weight{i}",2*SCALE,isqrt.level(),remapable=True)
                res[i] = ((isqrt * weight) * x[i]).relinearize().doubleRescale().rescale()

        streamSize = 1
        numStreams = 1
        res = [None]*16
        CeriumStream(streamSize,numStreams,normalization_stream_fn,w_sum,res)

        for i in range(16):
            res[i] = Receive(res[i])
            FunctionOutput(f"res{i}",res[i])


    global __rmsnorm_function_parallel__
    __rmsnorm_function_parallel__[norm_type] = function


def rmsnorm_parallel(norm_type,x,prefix,block):
    assert norm_type in ["att","ffn"], "Invalid Type for rmsnorm"
    assert __rmsnorm_function_parallel__[norm_type] is not None, f"Function rmsnorm_parallel_{norm_type} not defined"
    outputs = CeriumFunctionCall(__rmsnorm_function_parallel__[norm_type],{f"X{i}":x[i] for i in range(16)},block)
    test_outputs = []
    test_outputs += ["w_sum","w_mean","inv_sqrt"]
    test_outputs += ["isqrt_y"]
    test_outputs += [f"isqrt_y_{i}" for i in range(0,10,2)]
    # test_outputs += ["inv_sqrt_div"]

    # for k in test_outputs:
        # TestOutput(f"{prefix}_{k}",outputs[k])
    
    res = [outputs[f"res{i}"] for i in range(16)]
    return res

__ffn_matmul_vw_function__ = None
def ffn_matmul_vw_function():
    function = CeriumFunction(f"ffn_matmul_vw_1gpu",1,0)
    with function:
        prefix = "ffn"
        x = [CiphertextArgument(f"X{i}",3*SCALE,13) for i in range(16)]
        for i in range(16):
            x[i] = x[i].modswitch()

        level = x[0].level()


        masks = {i: getPlaintextInput(f"{prefix}_maskW_{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

        masks = [masks]*56



        ww = get_plaintext_inputs_matmul(height=4096,width=14336,name=f"{prefix}_Ww",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

        V, W = None, None

        W = matmul_ct128x4096_pt4096x14336(x,ww,masks)

        for i in range(16):
            for _ in range(6):
                x[i] = x[i].modswitch()
        level = x[0].level()


        masks_re = {i: getPlaintextInput(f"{prefix}_maskV_re{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}
        masks_im = {i: getPlaintextInput(f"{prefix}_maskV_im{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

        masks_reim = [masks_re,masks_im]*28


        wv = get_plaintext_inputs_matmul(height=4096,width=14336,name=f"{prefix}_Wv",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)
        V = matmul_ct128x4096_pt4096x14336(x,wv,masks_reim)

        for i in range(56):
            W[i] = W[i].doubleRescale().doubleRescale()
            FunctionOutput(f"W{i}",W[i])

        for i in range(56):
            V[i] = V[i].doubleRescale().doubleRescale()
            FunctionOutput(f"V{i}",V[i])


    global __ffn_matmul_vw_function__
    __ffn_matmul_vw_function__ = function

def ffn_matmul_vw(x,block):
    assert __ffn_matmul_vw_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__ffn_matmul_vw_function__,{f"X{i}":x[i] for i in range(16)},block)
    V = [outputs[f"V{i}"] for i in range(56)]
    W = [outputs[f"W{i}"] for i in range(56)]
    return V,W

__ffn_matmul_o_function__ = None
def ffn_matmul_o_function():
    function = CeriumFunction(f"ffn_matmul_o_1gpu",1,0)
    with function:
        prefix = "ffn"
        x = [CiphertextArgument(f"X{i}",3*SCALE,6) for i in range(56)]
        level = x[0].level()

        masks_re = {i: getPlaintextInput(f"{prefix}_maskO_re{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}
        masks_im = {i: getPlaintextInput(f"{prefix}_maskO_im{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

        masks_reim = [masks_re,masks_im]*8


        wo = get_plaintext_inputs_matmul(height=14336,width=4096,name=f"{prefix}_Wo",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

        O = matmul_ct128x14336_pt14336x4096(x,wo,masks_reim)

        for i in range(16):
            O[i] = O[i].doubleRescale().rescale()
            FunctionOutput(f"O{i}",O[i])


    global __ffn_matmul_o_function__
    __ffn_matmul_o_function__ = function

def ffn_matmul_o(x,block):
    assert __ffn_matmul_o_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__ffn_matmul_o_function__,{f"X{i}":x[i] for i in range(56)},block)
    O = [outputs[f"O{i}"] for i in range(16)]
    return O

__ffn_matmul_vw_parallel_function__ = None
def ffn_matmul_vw_parallel_function(gpus):
    assert gpus in [2,4,8]
    function = CeriumFunction(f"ffn_matmul_vw_{gpus}gpu",gpus,0)
    with function:
        prefix = "ffn"
        x = [CiphertextArgument(f"X{i}",3*SCALE,13) for i in range(16)]
        for i in range(16):
            x[i] = x[i].modswitch()


        def ffn_vw_streamfn(sid,X,numStreams):
            if sid >= numStreams: return

            level = X[0].level()
            masks = {i: getPlaintextInput(f"{prefix}_maskW_{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

            assert 56 % numStreams == 0
            count = 56 // numStreams

            assert count % 2 == 0, "Count must be even for silu"

            masks = [masks]*count

            width = 14336 // numStreams

            print(sid * width, width)

            ww = get_plaintext_inputs_matmul_offset(height=4096,heightOffset=0,width=width,widthOffset=sid * width,name=f"{prefix}_Ww",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

            W = matmul_ct128xHeight_ptHeightxWidth(height=4096,width=width,ct=X,pt=ww,masks=masks)
            W = [w.doubleRescale().doubleRescale() for w in W]

            wv = get_plaintext_inputs_matmul_offset(height=4096,heightOffset=0,width=width,widthOffset=sid * width,name=f"{prefix}_Wv",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)
            Xms = X.copy()
            for i in range(16):
                for _ in range(6):
                    Xms[i] = Xms[i].modswitch()
            level = Xms[0].level()
            
            masks_re = {i: getPlaintextInput(f"{prefix}_maskV_re{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}
            masks_im = {i: getPlaintextInput(f"{prefix}_maskV_im{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

            masks_reim = [masks_re,masks_im]*(count//2)


            wv = get_plaintext_inputs_matmul_offset(height=4096,heightOffset=0,width=width,widthOffset=sid * width,name=f"{prefix}_Wv",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

            V = matmul_ct128xHeight_ptHeightxWidth(height=4096,width=width,ct=Xms,pt=wv,masks=masks_reim)
            V = [v.doubleRescale().doubleRescale() for v in V]

            for i in range(0,count):
                FunctionOutput(f"W{sid*count + i}",W[i])
                FunctionOutput(f"V{sid*count + i}",V[i])

            print(sid*count,sid*count + count, sid * width)


        streamSize = 1
        numStreams = min(gpus,7)
        CeriumStream(streamSize,numStreams,ffn_vw_streamfn,x,numStreams)


    global __ffn_matmul_vw_parallel_function__
    __ffn_matmul_vw_parallel_function__ = function

def ffn_matmul_vw_parallel(x,block):
    assert __ffn_matmul_vw_parallel_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__ffn_matmul_vw_parallel_function__,{f"X{i}":x[i] for i in range(16)},block)
    V = [outputs[f"V{i}"] for i in range(56)]
    W = [outputs[f"W{i}"] for i in range(56)]
    # W = [outputs[f"W{i}"] for i in range(56)]
    return (V,W)

__ffn_silu_parallel_function__ = None
def ffn_silu_parallel_function(gpus):
    assert gpus in [2,4,8]
    function = CeriumFunction(f"ffn_silu_{gpus}gpu",gpus,0)
    with function:
        prefix = "ffn"


        def ffn_silu_streamfn(sid,VWSilu,numStreams):
            if sid >= numStreams: return



            assert 56 % numStreams == 0
            count = 56 // numStreams

            assert count % 2 == 0, "Count must be even for silu"

            for i in range(0,count,2):
                idx = sid * count + i
                v0 = CiphertextArgument(f"V{idx}",2*SCALE,2)
                v1 = CiphertextArgument(f"V{idx + 1}",2*SCALE,2)
                w0 = CiphertextArgument(f"W{idx}",2*SCALE,8)
                w1 = CiphertextArgument(f"W{idx + 1}",2*SCALE,8)

                VWSilu[idx :idx + 2] = silu([v0,v1],[w0,w1],prefix)
                print("idx: ",sid,idx)


        streamSize = 1
        numStreams = min(gpus,7)
        VW = [None]*56
        CeriumStream(streamSize,numStreams,ffn_silu_streamfn,VW,numStreams)

        for i in range(56):
            FunctionOutput(f"VW{i}",Receive(VW[i]))


    global __ffn_silu_parallel_function__
    __ffn_silu_parallel_function__ = function

def ffn_silu_parallel(v,w):
    assert __ffn_silu_parallel_function__ is not None, "Function not defined"
    args = {}
    for i in range(56):
        args[f"V{i}"] = v[i]
        args[f"W{i}"] = w[i]
    outputs = CeriumFunctionCall(__ffn_silu_parallel_function__,args)
    VW = [outputs[f"VW{i}"] for i in range(56)]
    return VW

__ffn_matmul_o_parallel_function__ = None
def ffn_matmul_o_parallel_function(gpus):
    function = CeriumFunction(f"ffn_matmul_o_{gpus}gpu",gpus,0)
    with function:
        prefix = "ffn"
        x = [CiphertextArgument(f"X{i}",3*SCALE,6) for i in range(56)]
        level = x[0].level()
        

        def ffn_o_streamfn(sid,X,O,numStreams):
            if sid >= numStreams: return
            masks_re = {i: getPlaintextInput(f"{prefix}_maskO_re{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}
            masks_im = {i: getPlaintextInput(f"{prefix}_maskO_im{i}",SCALE+SCALE//2,level) for i in [-128,0,128]}

            count = 16 // numStreams

            masks = [masks_re,masks_im]*(count//2)

            width = 4096 // numStreams

            wo = get_plaintext_inputs_matmul_offset(height=14336,heightOffset=0,width=width,widthOffset=sid * width,name=f"{prefix}_Wo",level=level,scale=SCALE+SCALE//2,period=256,remapable=True)

            O_ = matmul_ct128xHeight_ptHeightxWidth(height=14336,width=width,ct=X,pt=wo,masks=masks)
            O_ = [o.doubleRescale().rescale() for o in O_]
            O[sid*(count) : (sid+1)*(count)] = O_

            print(sid*count,sid*count + count, sid * width)

        streamSize = 1
        numStreams = gpus
        O = [None]*16
        CeriumStream(streamSize,numStreams,ffn_o_streamfn,x,O,numStreams)
        for i in range(16):
            FunctionOutput(f"O{i}",Receive(O[i]))

    global __ffn_matmul_o_parallel_function__
    __ffn_matmul_o_parallel_function__ = function

def ffn_matmul_o_parallel(x,block):
    assert __ffn_matmul_o_parallel_function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__ffn_matmul_o_parallel_function__,{f"X{i}":x[i] for i in range(56)},block)
    O = [outputs[f"O{i}"] for i in range(16)]
    return O


def sign_new2(x,prefix,bootstrapAfter=False):
    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_g3_coeff1_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[3] = getPlaintextInput(f"{prefix}_g3_coeff3_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[5] = getPlaintextInput(f"{prefix}_g3_coeff5_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[7] = getPlaintextInput(f"{prefix}_g3_coeff7_{xLevel}",SCALE,x.level(),scalar=True)
    x = g3_f3(x,coeffs)
    x = x.rescale()

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_g3_coeff1_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[3] = getPlaintextInput(f"{prefix}_g3_coeff3_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[5] = getPlaintextInput(f"{prefix}_g3_coeff5_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[7] = getPlaintextInput(f"{prefix}_g3_coeff7_{xLevel}",SCALE,x.level(),scalar=True)
    x = g3_f3(x,coeffs)
    x = x.rescale()

    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    coeffs[1] = getPlaintextInput(f"{prefix}_f3_coeff1_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[3] = getPlaintextInput(f"{prefix}_f3_coeff3_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[5] = getPlaintextInput(f"{prefix}_f3_coeff5_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[7] = getPlaintextInput(f"{prefix}_f3_coeff7_{xLevel}",SCALE,x.level(),scalar=True)
    x = g3_f3(x,coeffs)

 
    xLevel = x.level()
    coeffs = [None for _ in range(8)]
    assert bootstrapAfter == False, "bootstrapAfter must be False for sign_new_vec"
    coeffs[1] = getPlaintextInput(f"{prefix}_f3_coeff1_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[3] = getPlaintextInput(f"{prefix}_f3_coeff3_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[5] = getPlaintextInput(f"{prefix}_f3_coeff5_{xLevel}",2*SCALE,x.level(),scalar=True)
    coeffs[7] = getPlaintextInput(f"{prefix}_f3_coeff7_{xLevel}",2*SCALE,x.level(),scalar=True)
    x = g3_f3_new(x,coeffs)
    x = x.doubleRescale()

    return x



def silu_internal(v,w):

    v = bootstrap(v,"b33_2",19)

    vRF0 = PlaintextInput("silu_vRF0",0,v.level(),scalar=True)
    vRF1 = PlaintextInput("silu_vRF1",0,v.level(),scalar=True)

    vr = v * vRF0
    v = v * vRF1
    
    C0 = PlaintextInput("silu_C0",vr.scale(),vr.level(),scalar=True)
    C1 = PlaintextInput("silu_C1",vr.scale(),vr.level(),scalar=True)
    C2 = PlaintextInput("silu_C2",vr.scale(),vr.level(),scalar=True)


    comp0 = (vr + C0)
    comp1 = (vr + C1)
    comp2 = (vr + C2)

    comp0 = BreakVector(comp0.rescale())
    comp1 = BreakVector(comp1.rescale())
    comp2 = BreakVector(comp2.rescale())

    s0 = MakeVector([sign_new2(comp0[0],"silu",bootstrapAfter=False),sign_new2(comp0[1],"silu",bootstrapAfter=False)])
    s1 = MakeVector([sign_new2(comp1[0],"silu",bootstrapAfter=False),sign_new2(comp1[1],"silu",bootstrapAfter=False)])
    s2 = MakeVector([sign_new2(comp2[0],"silu",bootstrapAfter=False),sign_new2(comp2[1],"silu",bootstrapAfter=False)])

    # TestFunctionOutput(f"comp0",comp0)
    # TestFunctionOutput(f"comp1",comp1)
    # TestFunctionOutput(f"comp2",comp2)
    TestFunctionOutputVec2(f"s0",s0)
    TestFunctionOutputVec2(f"s1",s1)
    TestFunctionOutputVec2(f"s2",s2)

    one = PlaintextInput("silu_one",s0.scale(),s0.level(),scalar=True)

    b1 = s0 - s1
    b2 = s1 - s2
    b3 = s2 + one

    TestFunctionOutputVec2(f"b1",b1)
    TestFunctionOutputVec2(f"b2",b2)
    # TestOutput(f"{prefix}_b3",b3)

    # print(b3.scale(),b3.level())

    for i in range(5):
        v = v.modswitch()

    print(v.level())

    v2 = (v * v).relinearize().doubleRescale()
    v4 = (v2 * v2).relinearize().doubleRescale()

    v6 = (v2.modswitch().modswitch() * v4).relinearize().doubleRescale()

    Q6 = PlaintextInput("silu_Q6",SCALE,v6.level(),scalar=True)
    Q = v6 * Q6

    Q4 = PlaintextInput("silu_Q4",SCALE,v4.level(),scalar=True)
    Qt4 = v4 * Q4
    Qt4 = Qt4.modswitch().modswitch()
    Q = Q + Qt4

    Q2 = PlaintextInput("silu_Q2",SCALE,v2.level(),scalar=True)
    Qt2 = v2 * Q2
    Qt2 = Qt2.modswitch().modswitch().modswitch().modswitch()
    Q = Q + Qt2

    Q1 = PlaintextInput("silu_Q1",SCALE,v.level(),scalar=True)
    Qt1 = v * Q1
    Qt1 = Qt1.modswitch().modswitch().modswitch().modswitch().modswitch().modswitch()
    Q = Q + Qt1

    Q = Q.doubleRescale()

    Q0 = PlaintextInput("silu_Q0",SCALE,Q.level(),scalar=True)
    Q = Q + Q0

    P2 = PlaintextInput("silu_P2",SCALE,v2.level(),scalar=True)
    Pt2 = v2 * P2
    Pt2 = Pt2.modswitch().modswitch()
    P = Pt2

    P1 = PlaintextInput("silu_P1",SCALE,v.level(),scalar=True)
    Pt1 = v * P1
    Pt1 = Pt1.modswitch().modswitch().modswitch().modswitch()
    P = P + Pt1

    P = P.modswitch().modswitch()

    P = P.doubleRescale()

    P0 = PlaintextInput("silu_P0",SCALE,P.level(),scalar=True)
    P = P + P0

    xRF2 = PlaintextInput("silu_xRF2",SCALE,v.level(),scalar=True)
    v = v * xRF2
    v = v.doubleRescale()

    for i in range(6):
        v = v.modswitch()

    QW = (w * Q).relinearize().doubleRescale()
    PW = (w * P).relinearize().doubleRescale()
    vw = (w * v).relinearize().doubleRescale()

    # TestFunctionOutput(f"res1",P)
    # TestFunctionOutput(f"res2",Q)
    # TestFunctionOutput(f"res3",x)

    res = (b1 * PW) + (b2 * QW) + (b3 * vw)
    res = res.relinearize()

    return res


__silu__function__ = None
def silu_function():
    function = CeriumFunction("silu",1,0)
    with function:
        v0 = CiphertextArgument("v0",2*SCALE,2)
        v1 = CiphertextArgument("v1",2*SCALE,2)
        v = MakeVector([v0,v1])
        w0 = CiphertextArgument("w0",2*SCALE,8)
        w1 = CiphertextArgument("w1",2*SCALE,8)
        w = MakeVector([w0,w1])
        res = silu_internal(v,w)
        res = BreakVector(res)
        FunctionOutput("res0",res[0])
        FunctionOutput("res1",res[1])
    global __silu__function__
    __silu__function__ = function

def silu(v,w,prefix):
    assert __silu__function__ is not None, "Function not defined"
    outputs = CeriumFunctionCall(__silu__function__,{"v0":v[0],"v1":v[1],"w0":w[0],"w1":w[1]})
    res0 = outputs["res0"]
    res1 = outputs["res1"]
    test_outputs = ["res1","res2"]
    test_outputs += ["comp0","comp1","comp2"]
    test_outputs += ["s0","s1","s2"]
    test_outputs += ["b1","b2"]
    # for output_name in test_outputs:
    #     TestOutput(f"{prefix}_{output_name}",outputs[output_name])
    return [res0,res1]

def ffn(X,blockID,testOutputPredicate):
    prefix = "ffn"
    V,W = ffn_matmul_vw(X,blockID)
    for i in range(56):
        TestOutput(f"{prefix}_V{i}",V[i],testOutputPredicate)
        TestOutput(f"{prefix}_W{i}",W[i],testOutputPredicate)
    VWSilu = [None]*56
    for i in range(0,56,2):
        VWSilu[i:i+2] = silu(V[i:i+2],W[i:i+2],prefix)
        TestOutput(f"{prefix}_VW{i}",VWSilu[i],testOutputPredicate)
        TestOutput(f"{prefix}_VW{i+1}",VWSilu[i+1],testOutputPredicate)
    O = ffn_matmul_o(VWSilu,blockID)
    for i in range(0,16):
        O[i] = O[i].rescale()
        TestOutput(f"{prefix}_O{i}",O[i],testOutputPredicate)
    return O

def ffn_parallel(X,blockID,testOutputPredicate):
    prefix = "ffn"
    V,W = ffn_matmul_vw_parallel(X,blockID)
    VWSilu = ffn_silu_parallel(V,W)
    for i in range(0,56,2):
        TestOutput(f"{prefix}_VW{i}",VWSilu[i],testOutputPredicate)
        TestOutput(f"{prefix}_VW{i+1}",VWSilu[i+1],testOutputPredicate)
    
    O = ffn_matmul_o_parallel(VWSilu,blockID)
    for i in range(0,16):
        O[i] = O[i].rescale()
        TestOutput(f"{prefix}_O{i}",O[i],testOutputPredicate)
    return O


def llama_blocks_1gpu(num_blocks=1):
        function = CeriumFunction(f"llama_1gpu_{num_blocks}blocks",1,0)
        with function:
            X = [CiphertextInput(f"rmsnorm_att_input{i}",2*SCALE,2) for i in range(16)]
            for bid in range(0,num_blocks):
                blockID = f"block{bid}"
                testOutputPredicate = (bid == num_blocks - 1)
                res = rmsnorm("att",X,"rmsnorm_att",blockID)
                for i in range(16):
                    TestOutput(f"rmsnorm_att_res{i}",res[i],testOutputPredicate)
                O = attention(res,"attention",blockID,testOutputPredicate)
                H = [None]*16

                for i in range(0,16,2):
                    H[i] = O[i] + X[i]
                    H[i+1] = O[i+1] + X[i+1]

                for i in range(0,16):
                    TestOutput(f"H{i}",H[i],testOutputPredicate)

                H_norm = rmsnorm("ffn",H,"rmsnorm_ffn",blockID)
                for i in range(16):
                    TestOutput(f"rmsnorm_ffn_res{i}",H_norm[i],testOutputPredicate)

                out = ffn(H_norm,blockID,testOutputPredicate)
                out = [out[i] + H[i] for i in range(16)]
                for i in range(16):
                    if bid != num_blocks - 1:
                        continue
                    Output(f"out{i}",out[i])
                X = out

def llama_blocks_multi_gpu(gpus,num_blocks=1):
        function = CeriumFunction(f"llama_{gpus}gpu_{num_blocks}blocks",gpus,0)
        with function:
            X = [CiphertextInput(f"rmsnorm_att_input{i}",2*SCALE,2) for i in range(16)]
            for bid in range(0,num_blocks):
                blockID = f"block{bid}"
                testOutputPredicate = (bid == num_blocks - 1)
                res = rmsnorm_parallel("att",X,"rmsnorm_att",blockID)
                for i in range(16):
                    TestOutput(f"rmsnorm_att_res{i}",res[i],predicate=testOutputPredicate)
                O = attention_parallel(res,"attention",blockID,testOutputPredicate)
                H = [None]*16

                for i in range(0,16,2):
                    H[i] = O[i] + X[i]
                    H[i+1] = O[i+1] + X[i+1]

                # for i in range(0,16):
                    # TestOutput(f"H{i}",H[i],predicate=testOutputPredicate)
                for i in range(16):
                    if bid != num_blocks - 1:
                        TestOutput(f"block{bid}_dbg_H{i}",H[i])
                    else:
                        TestOutput(f"H{i}",H[i],predicate=testOutputPredicate)

                H_norm = rmsnorm_parallel("ffn",H,"rmsnorm_ffn",blockID)
                for i in range(16):
                    TestOutput(f"rmsnorm_ffn_res{i}",H_norm[i],predicate=testOutputPredicate)

                out = ffn_parallel(H_norm,blockID,testOutputPredicate)
                out = [out[i] + H[i] for i in range(16)]
                for i in range(16):
                    if bid != num_blocks - 1:
                        TestOutput(f"block{bid}_dbg_out{i}",out[i])
                    else:
                        Output(f"out{i}",out[i])
                X = out



def llama_program_1gpu(num_blocks=1):
    num_gpus = 1
    testProgram = CeriumProgram('test', rns_bit_size=SCALE,num_gpus=num_gpus)
    with testProgram:
        bootstrap_32K_35lvl_t1_vec_function()
        bootstrap_32K_33lvl_t1_vec_function()

        rmsnorm_function("att")
        rmsnorm_function("ffn")

        softmax_128x128_vec_function()
        matmul_ct128x128_ct128x128_transpose_function("attention")
        matmul_ct128x128_ct128x128_function()
        attention_matmul_qkv_function()
        attention_matmul_o_function()

        ffn_matmul_vw_function()
        silu_function()
        ffn_matmul_o_function()

        llama_blocks_1gpu(num_blocks)


    return testProgram

def llama_program_multi_gpu(gpus,num_blocks=1):
    testProgram = CeriumProgram('test', rns_bit_size=SCALE,num_gpus=gpus)
    with testProgram:
        bootstrap_32K_35lvl_t1_vec_function()
        bootstrap_32K_33lvl_t1_vec_function()

        rmsnorm_function_parallel("att",gpus)
        rmsnorm_function_parallel("ffn",gpus)

        softmax_128x128_vec_function()
        matmul_ct128x128_ct128x128_transpose_function("attention")
        matmul_ct128x128_ct128x128_function()
        attention_matmul_qkv_parallel_function(gpus)
        attention_matmul_o_parallel_function(gpus)
        attention_score_softmax_parallel_function(gpus)

        silu_function()
        ffn_matmul_vw_parallel_function(gpus)
        ffn_silu_parallel_function(gpus)
        ffn_matmul_o_parallel_function(gpus)

        llama_blocks_multi_gpu(gpus,num_blocks)


    return testProgram




def main(gpus,num_blocks,prefix):

    global LVL
    # LVL = 51
    LVL = 52
    global SCALE
    SCALE = 28

    configureBootstrapLVL(LVL)
    # configureBootstrapGPU(True)
    
    def get_vregs(fname):
        if "attention_matmul_qkv" in fname:
            return 4096*3
        elif "attention_matmul_o" in fname:
            return 4096*2
        elif "bootstrap_32K_35lvl_t1_vec" in fname or "bootstrap_32K_33lvl_t1_vec" in fname:
            return 4096
        elif "ffn_matmul_vw" in fname or  "ffn_matmul_o" in fname:
            return 4096*3
        elif "matmul_ct128x128_ct128x128_transpose" in fname:
            return 4096
        elif "matmul_ct128x128_ct128x128" in fname:
            return 4096
        elif "test_block" in fname:
            return 1024
        elif "attention_score_softmax" in fname:
            return 1024
        else:
            return 4096

    def get_bcu(fname):
        if "attention_matmul_qkv" in fname:
            return 128
        elif "attention_matmul_o" in fname:
            return 128
        elif "bootstrap_32K_35lvl_t1_vec" in fname or "bootstrap_32K_33lvl_t1_vec" in fname:
            return 64
        elif "ffn_matmul_vw" in fname or  "ffn_matmul_o" in fname:
            return 128
        elif "matmul_ct128x128_ct128x128_transpose" in fname:
            return 128
        elif "matmul_ct128x128_ct128x128" in fname:
            return 128
        elif "test_block" in fname:
            return 32
        elif "attention_score_softmax" in fname:
            return 32
        else:
            return 128

    def get_block_size(fname):
        if "attention_matmul_qkv" in fname:
            return 64
        elif "attention_matmul_o" in fname:
            return 64
        elif "attention_score" in fname:
            return 1
        elif "bootstrap_32K_35lvl_t1_vec" in fname or "bootstrap_32K_33lvl_t1_vec" in fname:
            return 1024
        elif "ffn_matmul_vw" in fname or  "ffn_matmul_o" in fname:
            return 64
        elif "ffn_silu" in fname:
            return 1
        elif "matmul_ct128x128_ct128x128_transpose" in fname:
            return 1024
        elif "matmul_ct128x128_ct128x128" in fname:
            return 256
        else:
            return 1024
    
    def do_horizontal_fusion(fname):
        if "attention_score_softmax" in fname:
            return 0
        elif "ffn_silu" in fname:
            return 0
        elif "llama_" in fname:
            return 0
        return 1

    num_gpus = gpus
    if num_gpus == 1:
        program = llama_program_1gpu(num_blocks)
    else:
        program = llama_program_multi_gpu(gpus=num_gpus,num_blocks=num_blocks)
    functions = program.get_functions()
    skip = []
    
    if os.environ.get('LLAMA_SKIP_GLOBAL_FUNCTIONS','1') == '1':
        skip += ["bootstrap_32K_35lvl_t1_vec"]
        skip += ["bootstrap_32K_33lvl_t1_vec"]
        skip += ["softmax_128x128_vec"]
        skip += ["matmul_ct128x128_ct128x128_transpose"]
        skip += ["matmul_ct128x128_ct128x128"]
        skip += ["silu"]

    if os.environ.get('LLAMA_SKIP_GPU_LOCAL_FUNCTIONS','1') == '1':
        skip += [f"attention_matmul_qkv_{gpus}gpu"]
        skip += [f"attention_matmul_o_{gpus}gpu"]
        skip += [f"attention_score_softmax_{gpus}gpu"]

        skip += [f"rmsnorm_att_{gpus}gpu"]
        skip += [f"rmsnorm_ffn_{gpus}gpu"]

        skip += [f"ffn_matmul_vw_{gpus}gpu"]
        skip += [f"ffn_matmul_o_{gpus}gpu"]
        skip += [f"ffn_silu_{gpus}gpu"]

    print_graphs = os.environ.get('CERIUM_PRINT_GRAPHS','0')
    os.environ['CERIUM_PRINT_GRAPHS'] = print_graphs
    os.environ['CERIUM_REDUCTION_REWRITE']= '0'

    for name,f in functions.items():
        compile = True
        if name in skip:
            compile = False
        if not compile:
            continue
        print(f"Compiling: {name}")
        os.environ['CERIUM_BLOCK_SIZE'] = f"{get_block_size(name)}"
        os.environ['CERIUM_HORIZONTAL_FUSION'] = f"{do_horizontal_fusion(name)}"
        cerium_compile_function(f,gpus,get_vregs(name),get_bcu(name),prefix)
    

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str,default="outputs/")
    parser.add_argument('--gpus', type=int,default=1)
    parser.add_argument('--num_blocks', type=int,default=1)

    args = parser.parse_args()

    main(args.gpus,args.num_blocks,args.prefix)
