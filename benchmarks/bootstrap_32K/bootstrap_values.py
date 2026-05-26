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
import math

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
        # maxVals = []
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

def get_hanki_raw_inputs(prefix,scale_factor_in,scale_factor_out,level,HK_K,HK_deg,HK_r,HK_e):

    hanki_cos2r = hanki_approx(HK_K,HK_deg,HK_r,HK_e).coef

    len2 = nearest_power_of_2_geq(len(hanki_cos2r))
    len2_half = len2 // 2

    hanki_cos2r = np.concatenate((hanki_cos2r, np.zeros(len2-len(hanki_cos2r))), axis=-1).flatten()
    hanki_cos2r[1::2] = 0
    # raise Exception("")

    raw_inputs = {}
    scales = [[] for _ in range(HK_deg+1)]
    scales_pr = [[] for _ in range(HK_deg+1)]
    levels = [[] for _ in range(HK_deg+1)]
    rescale_reqd = [True for i in range(HK_deg+1)]
    scales[1] = scale_factor_in
    levels[1] = level
    rescale_reqd[1] = False

    val_level = levels[1]
    val_scale = scale_factor_out

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

    return (raw_inputs,scale_factor_out,val_level)




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

    if config == "b34":
        invFFTDepth = 4
        FFTDepth = 4
    if config == "b33" or config == "b33_multi":
        invFFTDepth = 4
        FFTDepth = 5
    elif config == "b31" or config == "b31_multi":
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

    if config == "b33" or config == "b33_multi":
        level = 33 + availableLevels
    else:
        raise Exception("Invalid Config")

    if level > 52:
        raise Exception("Invalid number of available levels")

    prefix = f"{config}_{availableLevels}"
    sfScale = Primes[level-1]
    raw_inputs[f"{prefix}_sf"] = (1,sfScale)
    # raw_inputs[f"{prefix}_sf_im"] = ([1j]*Slots,Primes[level-1])
    level = level - 1

    if config in ["b33"]: 
        generate_fft_inputs(f'{prefix}_au',fftInvMatrices[0],Primes[level-1]*Primes[level-2],[0, 2048, 4096, 6144],[-16384,-8192, 0, 8192])
        generate_fft_inputs(f'{prefix}_bu',fftInvMatrices[1],Primes[level-3]*Primes[level-4],[0, 128, 256, 384, 512, 640, 768, 896],[-1920, -896, 128, 1152 ])
        generate_fft_inputs(f'{prefix}_cu',fftInvMatrices[2],Primes[level-5]*Primes[level-6],[0, 8, 16, 24, 32, 40, 48, 56],[-120, -56, 8, 72])
        generate_fft_inputs(f'{prefix}_du',fftInvMatrices[3],Primes[level-7]*Primes[level-8],[0, 1, 2, 3, 4],[-7, -2, 3])
        level = level - 2
    elif config in ["b33_multi"]: 
        generate_fft_inputs(f'{prefix}_au',fftInvMatrices[0],Primes[level-1]*Primes[level-2],[0, 2048, 4096, 6144],[-16384,-8192, 0, 8192])
        generate_fft_inputs(f'{prefix}_bu',fftInvMatrices[1],Primes[level-3]*Primes[level-4],[0, 128, 256, 384],[-1920, -1408, -896, -384, 128, 640, 1152, 1664])
        generate_fft_inputs(f'{prefix}_cu',fftInvMatrices[2],Primes[level-5]*Primes[level-6],[0, 8, 16, 24],[-120, -88, -56, -24, 8, 40, 72, 104])
        generate_fft_inputs(f'{prefix}_du',fftInvMatrices[3],Primes[level-7]*Primes[level-8],[0, 1, 2],[-7, -4, -1, 2, 5])
        level = level - 2

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

    (hk_raw_inputs,scale_factor,val_level) = get_hanki_raw_inputs(prefix,scale_factor,scale_factor,level,HK_K,HK_deg,HK_r,HK_e)
    raw_inputs.update(hk_raw_inputs)

    for i in range(HK_r):
        scale_factor = scale_factor*scale_factor/(Primes[val_level-1]*Primes[val_level-2])
        raw_inputs[f"{prefix}_m_one_r_"+str(i)] = (-1,scale_factor)
        val_level = val_level - 2

    piq = (Q_ground)/((2*math.pi)*scale_factor)
    piqx0 = 1
    piqx0 *= piq
    piqx1 = [xi**Slots]*Slots
    piqx1 = [x*piq for x in piqx1]
    raw_inputs[f"{prefix}_piqx0"] = (piqx0,Primes[val_level-1])
    raw_inputs[f"{prefix}_piqx1"] = (piqx1,Primes[val_level-1])

    level = val_level - 1

    if config in ["b33", "b33_multi"]:
        generate_fft_inputs(f'{prefix}_ap',fftMatrices[0],Primes[level-1],[0, 1, 2, 3, 4, 5, 6, 7],[-32, -24, -16, -8, 0, 8, 16, 24])
        generate_fft_inputs(f'{prefix}_bp',fftMatrices[1],Primes[level-2],[0, 32, 64, 96, 128, 160, 192, 224],[-1024, -768, -512, -256, 0, 256, 512, 768])
        generate_fft_inputs(f'{prefix}_cp',fftMatrices[2],Primes[level-3],[0, 1024, 2048, 3072],[-16384, -12288, -8192, -4096, 0, 4096, 8192, 12288])

    return raw_inputs
