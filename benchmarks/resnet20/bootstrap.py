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

from numpy.polynomial import Chebyshev
from fftFactors import *
from primes import *
from cerium.dsl import *

bootstrapInputs = {}

def get_bootstrap_inputs(name,scale,level,scalar=False):
    partition_size = CurrentPartitionSize()
    partition_id = CurrentPartitionID()
    key = f"{name}_:p{partition_size}:i{partition_id}"
    if key not in bootstrapInputs.keys():
        bootstrapInputs[key] = PlaintextInput(name,scale,level,scalar)
    return bootstrapInputs[key]

def compute_chebyshev_new_ds_pmu(config,x,HK_deg,SCALE):
    if HK_deg != 30:
        raise Exception("")
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

def compute_homomorphic_mod_ds_vec(x,x_level,scale,initScale,HK_deg,HK_r,config,setScalar=False):
    x = x.rescale()
    level = x.level()
    doubleScale = 2*scale

    f0 = get_bootstrap_inputs(f'{config}_f0',4*scale - initScale,level)
    f1 = get_bootstrap_inputs(f'{config}_f1',4*scale - initScale,level)
    X = MakeVector([x,x]) * MakeVector([f0,f1])
    
    X = X.conjugate() + X
    X = X.doubleRescale()

    f1by4K = get_bootstrap_inputs(f'{config}_f1_4K',doubleScale,X.level(),scalar=setScalar)
    X = X + f1by4K

    val = compute_chebyshev_new_ds_pmu(config,X,HK_deg,scale)

    for i in range(HK_r):
        val = val * val
        val = val.relinearize().doubleRescale()
        m_one = get_bootstrap_inputs(f"{config}_m_one_r_"+str(i),val.scale(),val.level(),scalar=setScalar)
        val = (val + val) + m_one

    val_extract = BreakVector(val)
    val0 = val_extract[0]
    val1 = val_extract[1]
    piqx0 = get_bootstrap_inputs(f'{config}_piqx0',initScale - val0.scale() + scale,val0.level())
    piqx1 = get_bootstrap_inputs(f'{config}_piqx1',initScale - val1.scale() + scale,val1.level())

    z = (val0*piqx0) + (val1*piqx1)
    return z


def compute_chebyshev_new_ds_pmu_novec(config,x,HK_deg,SCALE):
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


def compute_homomorphic_mod_split_new_ds(x,x_level,scale,initScale,HK_deg,HK_r,config,partition_size):

    def compute_homomorphic_mod_streamfn(sid,x,z):
        if sid > 2:
            return

        x = x.rescale()
        doubleScale = 2*scale
        f = PlaintextInput(f'{config}_f{sid}',doubleScale,x.level())
        x = x*f

        x = x.conjugate() + x
        x = x.doubleRescale()

        f1by4K = PlaintextInput(f'{config}_f1_4K',doubleScale,x.level(),scalar=True)
        x = x  + f1by4K

        bootstrapInputs.clear()
        val = compute_chebyshev_new_ds_pmu_novec(config,x,HK_deg,scale)
        bootstrapInputs.clear()

        for i in range(HK_r):
            val = val * val
            val = val.relinearize().doubleRescale()
            m_one = PlaintextInput(f"{config}_m_one_r_{i}",val.scale(),val.level(),scalar=True)
            val = (val + val) + m_one

        piqx = PlaintextInput(f'{config}_piqx{sid}',initScale - val.scale() + scale,val.level())
        z[sid] = val*piqx 

    z = [None for _ in range(2)]

    PartitionSize = CurrentPartitionSize()
    streamSize = max(1,PartitionSize//2)
    numStreams = 2
    CeriumStream(streamSize,numStreams,compute_homomorphic_mod_streamfn,x,z)
    z = z[0] + z[1]
    return z

def compute_homomorphic_mod(x,x_level,scale,initScale,HK_deg,HK_r,config,SetScalar=True):
    if CurrentPartitionSize() == 1:
        return compute_homomorphic_mod_ds_vec(x,x_level,scale,initScale,HK_deg,HK_r,config,SetScalar)
    else:
        return compute_homomorphic_mod_split_new_ds(x,x_level,scale,initScale,HK_deg,HK_r,config,CurrentPartitionSize())


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

def bootstrap(x,config,availableLevels,gpus=1):
    return bootstrap_32K_33lvl_t1(x,availableLevels,gpus)

__bootstrap_32K_33lvl_t1_function__ = None
def bootstrap_32K_33lvl_function_init(gpus):
    bootstrapInputs.clear()
    function = CeriumFunction(f"bootstrap_32K_33_{gpus}gpus", gpus,0)
    with function:
        Partition(1,0)
        x = CiphertextArgument("x",56,2)
        availableLevels = 19
        scale = 28
        doubleScale = 2*scale
        initScale = x.scale()
        HK_deg = 30
        HK_r = 3

        Partition(gpus,0)
        x = x.toEphemeral()
        x = x.bootstrapModRaise(52) 

        prefix = f"b33_{availableLevels}"

        sf = get_bootstrap_inputs(f"{prefix}_sf",scale,x.level(),True)
        x = x*sf

        babySteps = [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168]
        giantSteps = [-16384, -8192, 0, 8192]

        pt = get_bsgs_plaintexts(f"{prefix}_au",babySteps,giantSteps,doubleScale,x.level())
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,2)

        babySteps = [0, 32, 64, 96, 128, 160, 192, 224]
        giantSteps = [-1024, -768, -512, -256, 0, 256, 512, 768]
        pt = get_bsgs_plaintexts(f"{prefix}_bu",babySteps,giantSteps,doubleScale,x.level(),period=1024)
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,2)

        babySteps = [0, 1, 2, 3, 4, 5, 6, 7]
        giantSteps = [-32, -24, -16, -8, 0, 8, 16, 24]
        pt = get_bsgs_plaintexts(f"{prefix}_cu",babySteps,giantSteps,doubleScale,x.level(),period=32)
        x = BsgsMultiplyAccumulate(x,pt,babySteps,giantSteps,2)


        z = compute_homomorphic_mod(x,x.level(),scale,initScale,HK_deg,HK_r,config=f"{prefix}")

        babySteps = [0, 1, 2, 3, 4, 5, 6, 7]
        giantSteps = [-32, -24, -16, -8, 0, 8, 16, 24]
        pt = get_bsgs_plaintexts(f"{prefix}_ap",babySteps,giantSteps,scale,z.level(),period=32)
        z = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,1)

        babySteps = [0, 32, 64, 96, 128, 160, 192, 224]
        giantSteps = [-1024, -768, -512, -256, 0, 256, 512, 768]
        pt = get_bsgs_plaintexts(f"{prefix}_bp",babySteps,giantSteps,scale,z.level(),period=1024)
        z = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,1)

        babySteps = [0, 1024, 2048, 3072]
        giantSteps = [-16384, -12288, -8192, -4096, 0, 4096, 8192, 12288]
        pt = get_bsgs_plaintexts(f"{prefix}_cp",babySteps,giantSteps,scale,z.level())
        z0 = BsgsMultiplyAccumulate(z,pt,babySteps,giantSteps,1)


        z0 = z0 + z0.conjugate()
        z0 = z0.rescale()
        
        Partition(1,0)
        FunctionOutput("z0",z0)

    global __bootstrap_32K_33lvl_t1_function__
    __bootstrap_32K_33lvl_t1_function__ = function

def bootstrap_32K_33lvl_t1(x,availableLevels,gpus=1):
    # assert availableLevels == 14
    if availableLevels > 19:
        raise ValueError("This bootstrap returns a max of 19 levels")
    assert __bootstrap_32K_33lvl_t1_function__ is not None, "bootstrap_32K_33lvl_t1_function is not defined"
    if x.level() < 2:
        raise IndexError("Invalid level for bootstrap")
    while x.level() != 2:
        x = x.modswitch()

    partitionSize = CurrentPartitionSize()
    partitionID = CurrentPartitionID()

    Partition(gpus,0)
    outputs = CeriumFunctionCall(__bootstrap_32K_33lvl_t1_function__,{"x":x})
    Partition(partitionSize,partitionID)
    z = outputs["z0"]
    while z.level() > availableLevels + 2:
        z = z.modswitch()
    return z


def hanki_approx(K,n,r,e):
    I = [(Ii - 1/4 -e,Ii -1/4 + e) for Ii in range(-K+1,K)]
    d = [1]*(2*K - 1)
    num_nodes = len(d)
    Nodes = [[] for _ in range(len(I))]
    def getNodes():
        i = 0
        for Ii in range(-K+1,K):
            di= d[i]
            ti = []
            for j in range(di):
                t = Ii - 1/4 + e*math.cos(((2*j + 1)/(2*di))*math.pi)
                ti.append(t)
            Nodes[i] = ti
            i = i+1
    def func(x):
        prod = 1
        for ti in Nodes:
            for t in ti:
                prod = prod*(x-t)
        return prod
    def getArgMax():
        max = -float("inf")
        imax = -float("inf")
        for i in range(len(I)):
            (a,b) = I[i]
            for x0 in np.linspace(a,b,100):
                eval = func(x0)
                if eval > max:
                    max = eval
                    imax = i
        return imax
    getNodes()
    while num_nodes <= n:
        imax = getArgMax()
        num_nodes = num_nodes + 1
        d[imax] = d[imax] + 1
        getNodes()
    Points = []
    NodesFlat = []
    for ti in Nodes:
        for t in ti:
            t_div = t/2**r
            NodesFlat.append(t_div/K)
            Points.append(math.cos(2*math.pi*t_div))
    return Chebyshev.fit(NodesFlat,Points,len(NodesFlat) - 1,[-1/2**r,1/2**r]) 

def nearest_power_of_2_geq(x):
    return 1<<(x-1).bit_length()

def get_hanki_raw_inputs(prefix,scale_factor,level,HK_K,HK_deg,HK_r,HK_e):

    hanki_cos2r = hanki_approx(HK_K,HK_deg,HK_r,HK_e).coef

    len2 = nearest_power_of_2_geq(len(hanki_cos2r))
    len2_half = len2 // 2

    hanki_cos2r = np.concatenate((hanki_cos2r, np.zeros(len2-len(hanki_cos2r))), axis=-1).flatten()
    hanki_cos2r[1::2] = 0

    raw_inputs = {}
    scales = [[] for _ in range(HK_deg+1)]
    scales_pr = [[] for _ in range(HK_deg+1)]
    levels = [[] for _ in range(HK_deg+1)]
    rescale_reqd = [True for i in range(HK_deg+1)]
    scales[1] = scale_factor
    levels[1] = level
    rescale_reqd[1] = False

    val_level = levels[1]
    val_scale = scale_factor

    if HK_deg & (HK_deg - 1) == 0:
        raise Exception("Unimplemented where HK_deg is power of 2")

    val_out_level = int(val_level - 2*np.floor(np.log2(15)))

    def adjust_for_patterson(coeffs):
        len2 = len(coeffs)
        assert len2 & (len2 - 1) == 0, "Length of coeffs must be a power of 2"
        len2_half = len2 // 2
        coeffs = [0 for _ in range(len2)]
        for i in range(len2_half):
            if i == 0:
                coeffs[i] = hanki_cos2r[i] 
                coeffs[len2_half + i] = hanki_cos2r[len2_half+i]
            else:
                coeffs[i] = hanki_cos2r[i] -1*hanki_cos2r[len2-i]
                coeffs[len2_half + i] = 2*hanki_cos2r[len2_half+i]
        return coeffs

    coeffs = adjust_for_patterson(hanki_cos2r)

    scale16 = scales[1]
    lv16 = levels[1]
    for i in range(4):
        scale16 = scale16 * scale16 / (Primes[lv16-1]*Primes[lv16-2])
        lv16 = lv16 - 2

    raw_inputs[f"{prefix}_c0_{1}"] = (coeffs[1],(Primes[val_out_level-1]*Primes[val_out_level-2]*Primes[val_out_level-5]*Primes[val_out_level-6]))
    raw_inputs[f"{prefix}_c1_{1}"] = (coeffs[len2_half + 1],(Primes[val_out_level-1]*Primes[val_out_level-2]*Primes[val_out_level-3]*Primes[val_out_level-4]*Primes[val_out_level-5]*Primes[val_out_level-6])/scale16)

    for i in range (2,17,2):
        if i == 2 or i % 4 == 0:
            if rescale_reqd[i//2]:
                lvl = levels[i//2]
                scales[i//2] = scales[i//2]/(Primes[lvl-1]*Primes[lvl-2])
                levels[i//2] = lvl - 2
                rescale_reqd[i//2] = False
            levels[i] = levels[i//2]
            scales[i] = scales[i//2]*scales[i//2]
            raw_inputs[f"{prefix}_m_one_{i}"] = (-1,scales[i])
        else:
            if rescale_reqd[i//2-1]:
                lvl = levels[i//2-1]
                scales[i//2-1] = scales[i//2-1]/(Primes[lvl-1]*Primes[lvl-2])
                levels[i//2-1] = lvl - 2
                rescale_reqd[i//2-1] = False
            if rescale_reqd[i//2 + 1]:
                lvl = levels[i//2 + 1]
                scales[i//2 + 1] = scales[i//2 +1]/(Primes[lvl-1]*Primes[lvl-2])
                levels[i//2 + 1] = lvl - 2
                rescale_reqd[i//2 + 1] = False
            lvl = levels[i//2 + 1]
            levels[i] = levels[i//2 + 1]
            scales[i] = scales[i//2-1]*scales[i//2 + 1]
            raw_inputs[f"{prefix}_rf{i}"] = (1,scales[i]/scales[2])
        scales_pr[i] = scales[i]

    for i in range (2,len2_half,2):

        raw_inputs[f"{prefix}_c0_{i}"] = (coeffs[i],val_scale/scales_pr[i]*(Primes[val_out_level-1]*Primes[val_out_level-2]*Primes[val_out_level-5]*Primes[val_out_level-6]))
        raw_inputs[f"{prefix}_c1_{i}"] = (coeffs[len2_half + i],val_scale/scales_pr[i]*(Primes[val_out_level-1]*Primes[val_out_level-2]*Primes[val_out_level-3]*Primes[val_out_level-4]*Primes[val_out_level-5]*Primes[val_out_level-6]/scale16))

    raw_inputs[f"{prefix}_c0_{0}"] = (coeffs[0],val_scale*(Primes[val_out_level-5]*Primes[val_out_level-6]))
    raw_inputs[f"{prefix}_c1_{0}"] = (coeffs[len2_half],val_scale*(Primes[val_out_level-3]*Primes[val_out_level-4]*Primes[val_out_level-5]*Primes[val_out_level-6])/scale16)
    val_level = val_out_level - 6

    return (raw_inputs,scale_factor,val_level)




def get_bootstrap_runner_inputs(config,availableLevels):

    Slots = 32768

    def generate_fft_inputs(prefix,matrices,scale,babySteps,giantSteps):
        for gs in giantSteps:
            for bs in babySteps:
                step = bs + gs
                name = prefix + str(step)
                if step in matrices.keys():
                    entries = matrices[step].entries()
                    entries = np.concatenate((entries[-gs:],entries[:-gs]))
                    raw_inputs[name] =(entries,scale)
                else:
                    raw_inputs[name] = ([0]*Slots,scale)
        return raw_inputs

    HK_K = 16
    HK_deg = 30
    HK_r = 3
    HK_e = 1/256

    if config == "b33":
        invFFTDepth = 5
        FFTDepth = 5

    fftMatrices = genFftMatrices(Slots,FFTDepth)
    fftInvMatrices = genInvFftMatrices(Slots,invFFTDepth)

    ground_base_indices = [0,1]
    Q_ground = 1
    for idx in ground_base_indices:
        Q_ground = Q_ground * Primes[idx]

    raw_inputs = {}

    xi = np.exp(2 * np.pi * 1j / (4*Slots))
    xic = np.conjugate(xi)
    w = xic**4

    if config == "b33":
        level = 33 + availableLevels

    if level > 52:
        raise Exception("Invalid number of available levels")

    prefix = f"{config}_{availableLevels}"
    raw_inputs[f"{prefix}_sf"] = (1,Primes[level-1])
    level = level - 1

    if config in ["b33"]:
        generate_fft_inputs(f'{prefix}_au',fftInvMatrices[0],Primes[level-1]*Primes[level-2],[0, 1024, 2048, 3072, 4096, 5120, 6144, 7168], [-16384, -8192, 0, 8192])
        generate_fft_inputs(f'{prefix}_bu',fftInvMatrices[1],Primes[level-3]*Primes[level-4],[0, 32, 64, 96, 128, 160, 192, 224],[-1024, -768, -512, -256, 0, 256, 512, 768])
        generate_fft_inputs(f'{prefix}_cu',fftInvMatrices[2],Primes[level-5]*Primes[level-6],[0, 1, 2, 3, 4, 5, 6, 7],[-32, -24, -16, -8, 0, 8, 16, 24])

    level = level - 6
    ixi = xic
    scale_factor = Primes[level-1]*Primes[level-2]*Primes[level-3]*Primes[level-4]*1.5

    Q_ground_times_K_inv = 1/(2*Q_ground*HK_K)

    f0 = [1]*Slots
    f0 = [xx * Q_ground_times_K_inv for xx in f0]
    raw_inputs[f"{prefix}_f0"] = (f0,scale_factor)

    f1 = [ixi**Slots]*Slots
    f1 = [xx * Q_ground_times_K_inv for xx in f1]
    raw_inputs[f"{prefix}_f1"] = (f1,scale_factor)

    scale_factor = scale_factor//(Primes[level-1]*Primes[level-2])
    level = level - 2

    f1_4K = -1/(4*HK_K)
    raw_inputs[f"{prefix}_f1_4K"] = (f1_4K,scale_factor)

    (hk_raw_inputs,scale_factor,val_level) = get_hanki_raw_inputs(prefix,scale_factor,level,HK_K,HK_deg,HK_r,HK_e)
    raw_inputs.update(hk_raw_inputs)

    for i in range(HK_r):
        scale_factor = scale_factor*scale_factor/(Primes[val_level-1]*Primes[val_level-2])
        raw_inputs[f"{prefix}_m_one_r_"+str(i)] = (-1,scale_factor)
        val_level = val_level - 2

    piq = (Q_ground)/((2*math.pi)*scale_factor)
    piqx0 = 1/2
    piqx0 *= piq
    piqx1 = [xi**Slots]*Slots
    piqx1 = [x*piq/2 for x in piqx1]
    raw_inputs[f"{prefix}_piqx0"] = (piqx0,Primes[val_level-1])
    raw_inputs[f"{prefix}_piqx1"] = (piqx1,Primes[val_level-1])

    level = val_level - 1

    if config in ["b33"]:
        generate_fft_inputs(f'{prefix}_ap',fftMatrices[0],Primes[level-1],[0, 1, 2, 3, 4, 5, 6, 7],[-32, -24, -16, -8, 0, 8, 16, 24])
        generate_fft_inputs(f'{prefix}_bp',fftMatrices[1],Primes[level-2],[0, 32, 64, 96, 128, 160, 192, 224],[-1024, -768, -512, -256, 0, 256, 512, 768])
        generate_fft_inputs(f'{prefix}_cp',fftMatrices[2],Primes[level-3],[0, 1024, 2048, 3072],[-16384, -12288, -8192, -4096, 0, 4096, 8192, 12288])

    return raw_inputs

