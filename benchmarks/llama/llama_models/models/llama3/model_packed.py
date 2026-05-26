# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from .args import ModelArgs

import numpy as np

import random
random.seed(1)

import pickle

# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.


rms_norm_values = []

ReferenceOutputs = {}

class RMSNorm3(torch.nn.Module):
    def __init__(self, layer_id, name, dim: int, eps: float = 1e-6):
        super().__init__()
        self.layer_id = layer_id
        self.name = name
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        w = x.pow(2).mean(-1,keepdim=True)
        w = w + self.eps #* x.shape[-1]
        DIV2 = (2)
        w = inv_sqrt(w / DIV2 ) / np.sqrt(DIV2)
        v = w
        return x * v

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def row_pack_div_tensor(DIV1,DIV2):
    tensor = torch.ones(SLOTS) * DIV1
    tensor[0:256] = DIV2
    return tensor

class RMSNorm2(torch.nn.Module):
    def __init__(self, layer_id, name, dim: int, eps: float = 1e-6):
        super().__init__()
        self.layer_id = layer_id
        self.name = name
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        run_refr = self.layer_id == 0

        w = x.pow(2).mean(-1,keepdim=True)
        w = w + self.eps #* x.shape[-1]
        """
        I've empirically observed that 
        For layers < 28, the values are small, i.e. close to eps = 1e-5
        For layers >= 2, slot 0 has a value ~ 60 and all the other slots have values ~ eps = 1e-5
        when the number of tokens is large

        For layer > 28, values are large enough 
        So this choice of factors should enable us to normalize the values for the sign function
        """
        div_tensor = torch.ones(w.shape)
        if w.shape[-2] != 1 and self.layer_id >= 2:
            div_tensor[:,0,:] = 60

        if self.layer_id > 28:
            pick = torch.ones(w.shape)
            DIV1 = (2)
            DIV2 = (2)
        else:
            pick = torch.ones(w.shape)
            if w.shape[-2] != 1 and self.layer_id >= 2:
                pick[:,0,:] = 0
                DIV1 = (0.01)
                DIV2 = (2)
            else:
                DIV1 = (0.01)
                DIV2 = (0.01)


        pick_inv = (1 - pick)

        div_tensor = pick * DIV1 + pick_inv * DIV2

        w = inv_sqrt(w / div_tensor) / torch.sqrt(div_tensor)

        v = w

        res = x * v

        res = res * self.weight

        if not run_refr:
            return res

        x_packed = row_pack_nx4096(x)

        for i in range(16): ReferenceOutputs[f"rmsnorm_{self.name}_input{i}"] = x_packed[i].tolist()
        w_packed = x_packed * x_packed
        w_sum = 0
        for i in range(16):
            w_sum += w_packed[i]
        
        for i in range(8):
            w_sum += rotate(w_sum,-2**i)

        w_sum += (self.eps * 4096)

        ReferenceOutputs[f"rmsnorm_{self.name}_w_sum"] = w_sum.tolist()

        oneByN = torch.zeros((SLOTS,))
        oneByN[255::256] = 1 / 4096
        div_tensor_packed = row_pack_div_tensor(DIV1,DIV2)

        oneByN = oneByN / div_tensor_packed
        w_mean = w_sum * oneByN

        for i in range(8):
            w_mean += rotate(w_mean,2**i)

        ReferenceOutputs[f"rmsnorm_{self.name}_w_mean"] = w_mean.tolist()

        w_inv_sqrt = inv_sqrt_packed(w_mean,f"rmsnorm_{self.name}") 
        
        ReferenceOutputs[f"rmsnorm_{self.name}_inv_sqrt"] = w_inv_sqrt.tolist()


        div_tensor_isqrt = 1/torch.sqrt(div_tensor_packed)

        res_packed = x_packed.clone()

        for i in range(16):
            weight = self.weight[256*i:256*i+256].broadcast_to(128,256).reshape(SLOTS) * div_tensor_isqrt
            res_packed[i] =  weight * w_inv_sqrt * x_packed[i]
            ReferenceOutputs[f"rmsnorm_{self.name}_res{i}"] = res_packed[i].tolist()


        res_packed = row_unpack_nx4096(res_packed,x.shape[1])
        diff = res_packed - res
        print(f"RMSNORM: {self.layer_id}",torch.mean(diff**2))
        return res_packed


    def _norm2(self, x):
        w = x.pow(2).mean(-1,keepdim=True)
        w = w + self.eps #* x.shape[-1]
        DIV1 = (0.01)
        DIV2 = (2)

        div_tensor = torch.ones(w.shape,dtype=float) * DIV1
        if w.shape[-2] != 1 and self.layer_id >= 2:
            div_tensor[:,0,:] = DIV2
        w = inv_sqrt(w / div_tensor) / torch.sqrt(div_tensor)

        v = w
        return x * v

    def forward(self, x):
        return self._norm(x.float()).type_as(x)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs,
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def prepare_rotary_embedding_masks():

    freqs_cis = precompute_freqs_cis(128, 128, 500000.0, True)

    freq_cis_packed = row_pack_nx64complex(freqs_cis.type(torch.complex128))

    maskL = torch.zeros((SLOTS),device=freqs_cis.device,dtype=torch.complex128)
    maskR = torch.zeros((SLOTS),device=freqs_cis.device,dtype=torch.complex128)
    maskL[0::2] = 1
    maskR[1::2] = 1j

    maskL *= freq_cis_packed
    maskR *= rotate(freq_cis_packed,-1)

    maskL_ = torch.zeros((SLOTS),device=freqs_cis.device,dtype=torch.complex128)
    maskR_ = torch.zeros((SLOTS),device=freqs_cis.device,dtype=torch.complex128)
    maskL_[0::2] = 0.5
    maskR_[0::2] = -0.5j

    mask0 = maskL * maskL_
    mask1 = maskR * rotate(maskL_,-1)
    mask1_ = maskL * maskR_
    mask0 += maskR * rotate(maskR_,-1)

    return mask0, mask1, mask1_


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis_reshape = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_reshape).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_reshape).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def next_power_of_2(n):
    if n & n - 1 == 0:
        return n
    else:
        return pow(2,int(np.log2(n)+1))

def prev_power_of_2(n):
    if n & n - 1 == 0:
        return n
    else:
        return pow(2,int(np.log2(n)))


def exp_approx(x):
    iters = 7
    result = 1 + x / 2**iters

    for _ in range(iters):
        result = result * result
    return result

def inv_sqrt(x):
    iters = 10
    y = exp_approx(-1 * (x * 0.5 + 0.2)) * 2 + 0.2 #- x * (1 / 1024)
    for _ in range(iters):
        y = y * (3 - x * y * y) * 0.5
    return y

def inv_sqrt_packed(x,prefix):
    iters = 10
    y = exp_approx(-1 * (x * 0.5 + 0.2)) * 2 + 0.2 #- x * (1 / 1024)
    ReferenceOutputs[f"{prefix}_isqrt_y"] = y.tolist()
    for i in range(iters):
        y = y * (3 - x * y * y) * 0.5
        ReferenceOutputs[f"{prefix}_isqrt_y_{i}"] = y.tolist()
        ReferenceOutputs[f"{prefix}_isqrt_y_bs{i}"] = y.tolist()
    return y

def inv_sqrt_softmax(x):
    iters = 6
    y = 0.2
    for _ in range(iters):
        y = y * (3 - x * y * y) * 0.5
    return y


def reciprocal(x):
    result = inv_sqrt(x) ** 2
    return result

def reciprocal_softmax(x):
    result = inv_sqrt_softmax(x) ** 2
    return result


def f(x):
    y = 35 / 128 * x**9 - 180/128 * x**7 + 378 / 128 * x**5 - 420 / 128 * x**3 + 315 / 128 * x
    return y

def f2(x):
    y = 3 / 8 * x**5 - 10 / 8 * x**3 + 15 / 8 * x
    return y

def f3(x):
    y = - 5/ 16 * x**7 + 21 / 16 * x**5 - 35 / 16 * x**3 + 35 / 16 * x
    return y

def f1(x):
    y = -1 / 2 * x**3 +  3 / 2 * x
    return y

def g(x):
    y = 46623 / 2**10 * x**9 - 113492 / 2**10 * x**7 + 97015 / 2**10 * x**5 - 34974 / 2**10 * x**3 + 5850 / 2**10 *x
    return y

def g3(x):
    y = - 12860/ 2**10 * x**7 + 25614 / 2**10 * x**5 - 16577 / 2**10 * x**3 + 4589/ 2**10 *x
    return y

def g2(x):
    y = 3796 / 2**10 * x**5 - 6108 / 2**10 * x**3 + 3334 / 2**10 *x
    return y

def g1(x):
    y =  -1359/ 2**10 * x**3 + 2126 / 2**10 *x
    return y

def shift(x, step):
    x = np.concatenate((x[step:], x[:step]))
    return x

def sgn(x):
    return sgn22(x)

def sgn22(x):
    PREC = 40
    x = x + torch.randint(-1, 2, x.shape, device=x.device) * (1 + 1j) / (1 << PREC)
    result = f3(f3((g3(g3(x)))))
    result = torch.real(result)
    return result

def sgn33(x):
    PREC = 40
    result = f3((g3(g3(x))))
    result += (torch.randint(-1,2,x.shape,device=x.device).double() / (1 << PREC) )
    return result
 

def sign_approx(x):
    return sgn22(x)

def sign_approx2(x):
    return sgn33(x)

def sign_approx_softmax(x):
    return sgn33(x)


def softmax_approx5(scores,attention_mask):

    scores = scores.float()
    n = scores.shape[-1]
    n2 = next_power_of_2(n)
    i = 1
    DIV = 80
    scores = scores / DIV
    max = scores
    while i < n2:
        sp = torch.tensor([(j^(i)) % n for j in range(n)], device=scores.device)
        shf = torch.index_select(max, -1, sp)
        sub = max - shf
        sign = sign_approx_softmax(sub)
        max = ((max + shf) +  sign * sub)/2
        i *= 2
    max = max[..., 0]
    scores_max = scores - max.unsqueeze(-1)
    scores_max = scores_max * DIV
    scores_max_exp = exp_approx(scores_max)

    if attention_mask is not None:
        scores_max_exp = scores_max_exp * attention_mask
    scores_max_sum = scores_max_exp.sum(dim=-1, keepdim=True)

    REC_DIV = 2
    if scores.shape[-1] > 1000:
        REC_DIV = 4

    scores_smax = scores_max_exp * reciprocal_softmax(scores_max_sum / REC_DIV) / REC_DIV 

    scores_smax /= 1.0001

    return scores_smax

def softmax_real(scores,mask):
    scores = scores.float()
    if mask is not None:
        scores = scores + mask  
    scores = F.softmax(scores.float(), dim=-1)
    return scores

SLOTS = 32*1024

def row_pack_nx64complex(x):
    n,w = x.shape
    assert w == 64
    assert n <= 128 

    res = torch.zeros((256,128),device=x.device,dtype=x.dtype)
    for i in range(n):
        res[2*i][0::2] = x[i]
        res[2*i+1][0::2] = x[i]
    return res.reshape(SLOTS)

def row_pack_nx32x128(x):
    bsz,n, p, w = x.shape
    assert bsz == 1
    assert p == 32
    assert w == 128
    assert n <= 128 

    res = torch.zeros((16,SLOTS),device=x.device,dtype=x.dtype)
    for j in range(16):
        for i in range(n):
            res[j][256*i:256*(i+1)] = x[0][i][2*j : 2*(j+1)].flatten(-2)
    return res

def row_unpack_nx32x128(x,n):
    p, w = x.shape
    assert p == 16
    assert w == SLOTS

    res = torch.zeros((1,n,32,128),device=x.device,dtype=x.dtype)
    for j in range(16):
        for i in range(n):
            res[0][i][2*j : 2*(j+1)] = x[j][256*i:256*(i+1)].reshape(-1,128)
    return res


def row_pack_nx8x128(x):
    bsz,n, p, w = x.shape
    assert bsz == 1
    assert p == 8
    assert w == 128
    assert n <= 128 

    res = torch.zeros((4,SLOTS),device=x.device,dtype=x.dtype)
    for j in range(4):
        for i in range(n):
            res[j][256*i:256*(i+1)] = x[0][i][2*j : 2*(j+1)].flatten(-2)
    return res

def row_unpack_nx8x128(x,n):
    p, w = x.shape
    assert p == 4
    assert w == SLOTS

    res = torch.zeros((1,n,8,128),device=x.device,dtype=x.dtype)
    for j in range(4):
        for i in range(n):
            res[0][i][2*j : 2*(j+1)] = x[j][256*i:256*(i+1)].reshape(-1,128)
    return res

def row_pack_nx4096(x):
    bsz,  n, w = x.shape
    assert bsz == 1
    assert w == 4096
    assert n <= 128 

    res = torch.zeros((16,SLOTS),device=x.device,dtype=x.dtype)
    for j in range(16):
        for i in range(n):
            res[j][256*i:256*(i+1)] = x[0][i][256*j: 256*(j+1)]
    return res

def row_pack_nx1024(x):
    bsz,  n, w = x.shape
    assert bsz == 1
    assert w == 1024 
    assert n <= 128 

    res = torch.zeros((4,SLOTS),device=x.device,dtype=x.dtype)
    for j in range(4):
        for i in range(n):
            res[j][256*i:256*(i+1)] = x[0][i][256*j: 256*(j+1)]
    return res

def row_pack_nx14336(x):
    bsz,  n, w = x.shape
    assert bsz == 1
    assert w == 14336
    assert n <= 128 

    res = torch.zeros((56,SLOTS),device=x.device,dtype=x.dtype)
    for j in range(56):
        for i in range(n):
            res[j][256*i:256*(i+1)] = x[0][i][256*j: 256*(j+1)]
    return res

def row_unpack_nx4096(x,n):
    w = x.shape
    assert w[0] == 16
    assert w[1] == 32768 
    assert n <= 128 

    res = torch.zeros((1,n,4096),device=x.device,dtype=x.dtype)
    for j in range(16):
        for i in range(n):
            res[0][i][256*j: 256*(j+1)] = x[j][256*i:256*(i+1)]
    return res

def row_unpack_nx1024(x,n):
    w = x.shape
    assert w[0] == 4
    assert w[1] == 32768 
    assert n <= 128 

    res = torch.zeros((1,n,1024),device=x.device,dtype=x.dtype)
    for j in range(4):
        for i in range(n):
            res[0][i][256*j: 256*(j+1)] = x[j][256*i:256*(i+1)]
    return res

def row_unpack_nxWidth(width,x,n):

    assert width % 256 == 0
    count = width // 256
    w = x.shape
    assert w[0] == count
    assert w[1] == 32768 
    assert n <= 128 

    res = torch.zeros((1,n,width),device=x.device,dtype=x.dtype)
    for j in range(count):
        for i in range(n):
            res[0][i][256*j: 256*(j+1)] = x[j][256*i:256*(i+1)]
    return res

def row_unpack_nx14336(x,n):
    return row_unpack_nxWidth(14336,x,n)

def row_pack_4096x1024(Mat):
    w, n = Mat.shape
    assert w == 4096
    assert n == 1024

    # res = np.zeros((3,6,SLOTS,))
    res = torch.zeros((4,32,SLOTS),device=Mat.device,dtype=Mat.dtype)
    for m in range(4):
        M = (Mat[:,m*256:m*256+128],Mat[:,m*256 + 128:m*256 + 256])
        for l in range(2):
            for i in range(16):
                for j in range(128):
                    res[m][i + 16*l][256*j : 128 + 256*j] = M[l % 2][i*256 + j]
                    res[m][i + 16*l][128 + 256*j : 256 + 256*j] = M[(l+1) % 2][i*256 + 128 + j]
    return res


def row_pack_4096x4096(Mat):
    n, w = Mat.shape
    assert w == 4096
    assert n == 4096

    # res = np.zeros((3,6,SLOTS,))
    res = torch.zeros((16,32,SLOTS),device=Mat.device,dtype=Mat.dtype)
    for m in range(16):
        M = (Mat[:,m*256:m*256+128],Mat[:,m*256 + 128:m*256 + 256])
        for l in range(2):
            for i in range(16):
                for j in range(128):
                    res[m][i + 16*l][256*j : 128 + 256*j] = M[l % 2][i*256 + j]
                    res[m][i + 16*l][128 + 256*j : 256 + 256*j] = M[(l+1) % 2][i*256 + 128 + j]
    return res

def row_pack_4096x14336(Mat):
    n, w = Mat.shape
    assert w == 14336
    assert n == 4096

    # res = np.zeros((3,6,SLOTS,))
    res = torch.zeros((56,32,SLOTS),device=Mat.device,dtype=Mat.dtype)
    for m in range(56):
        M = (Mat[:,m*256:m*256+128],Mat[:,m*256 + 128:m*256 + 256])
        for l in range(2):
            for i in range(16):
                for j in range(128):
                    res[m][i + 16*l][256*j : 128 + 256*j] = M[l % 2][i*256 + j]
                    res[m][i + 16*l][128 + 256*j : 256 + 256*j] = M[(l+1) % 2][i*256 + 128 + j]
    return res


def row_pack_14336x4096(Mat):
    n, w = Mat.shape
    assert w == 4096
    assert n == 14336

    res = torch.zeros((16,112,SLOTS),device=Mat.device,dtype=Mat.dtype)
    for m in range(16):
        M = (Mat[:,m*256:m*256+128],Mat[:,m*256 + 128:m*256 + 256])
        for l in range(2):
            for i in range(56):
                for j in range(128):
                    res[m][i + 56*l][256*j : 128 + 256*j] = M[l % 2][i*256 + j]
                    res[m][i + 56*l][128 + 256*j : 256 + 256*j] = M[(l+1) % 2][i*256 + 128 + j]
    return res



def row_pack_4096x1_new(Vec):
    res = torch.zeros((16,SLOTS),device=Vec.device,dtype=Vec.dtype)
    for m in range(16):
        for j in range(128):
            res[m][256*j: 256+ 256*j] = Vec[m*256 : m*256+256]
    return np.array(res)

def adjust_ct(x):
    maskL = torch.ones((SLOTS,),dtype=x.dtype).reshape(-1,128)
    maskR = torch.ones((SLOTS,),dtype=x.dtype).reshape(-1,128)
    maskL[1::2] = 0
    maskR[0::2] = 0
    maskL = maskL.flatten()
    maskR = maskR.flatten()
    L = rotate(x,128) * maskL
    R = rotate(x,-128) * maskR 
    return L + R

def prepare_adjust_ct_mask():
    mask0 = torch.ones((SLOTS))
    maskL = torch.ones((SLOTS,)).reshape(-1,128)
    maskR = torch.ones((SLOTS,)).reshape(-1,128)
    maskL[1::2] = 0
    maskR[0::2] = 0
    maskL = maskL.flatten()
    maskR = maskR.flatten()
    maskL = rotate(maskL,-128)
    maskR = rotate(maskL,128)
    return {0: mask0, 128:maskL, -128:maskR}

def prepare_adjust_and_apply_rotary_embeddings_masks():
    maskL = torch.ones((SLOTS,),dtype=torch.int32).reshape(-1,128)
    maskR = torch.ones((SLOTS,),dtype=torch.int32).reshape(-1,128)
    maskL[1::2] = 0
    maskR[0::2] = 0
    maskL = maskL.flatten()
    maskR = maskR.flatten()
    maskL = rotate(maskL,-128)
    maskR = rotate(maskR,128)

    mask0, mask1, mask1_ = prepare_rotary_embedding_masks()

    masks = {}
    masks[0] = mask0
    masks[128] = maskL * rotate(mask0,-128)
    masks[-128] = maskR * rotate(mask0,128)

    masks[1] = mask1
    masks[129] = maskL * rotate(mask1,-128)
    masks[-127] = maskR * rotate(mask1,128)

    masks[-1] = mask1_
    masks[-129] = maskR * rotate(mask1_,128)
    masks[127] = maskL * rotate(mask1_,-128)

    return masks

def adjust_ct_and_apply_rotary_embeddings(x,y,masks):
    x = x.float()
    y = y.float()
    rotation_indicesX = [-1,0,1]
    rotation_indicesY = [-129,-128,-127,127,128,129]
    products = [x * masks[i] for i in rotation_indicesX] + [y * masks[i] for i in rotation_indicesY]
    x_ = rotate_accumulate_many(products,rotation_indicesX + rotation_indicesY)
    x_ += torch.conj(x_)
    return x_.real

def rotate_accumulate_many(x,steps):
    assert len(x) == len(steps)
    result0 = torch.zeros((SLOTS,),device=x[0].device,dtype=x[0].dtype)
    for (i,s) in enumerate(steps):
        result0 += rotate(x[i],s)
    return result0

def hoisted_rotate(x,steps):
    result = {}
    for (i,s) in enumerate(steps):
        result[i] = rotate(x,s)
    return result

def multiply_ct128x4096_pt4096x128_new(ct,pt):

    babysteps = [i for i in range(16)]
    giantsteps = [i for i in range(-128,128,16)]
    acc0 = [None for i in range(len(giantsteps))]
    acc1 = [None for i in range(len(giantsteps))]

    babystep_rotate = [hoisted_rotate(ct[i],babysteps) for i in range(16)]

    for i in range(0,16):
        plaintexts0 = prepare_pts_matmul_ct128x128_pt128x128(pt[i])
        plaintexts1 = prepare_pts_matmul_ct128x128_pt128x128(pt[16+i])

        for (g,gs) in enumerate(giantsteps):
            for (b,bs) in enumerate(babysteps):
                temp0 = babystep_rotate[i][bs] * plaintexts0[bs+gs]
                temp1 = babystep_rotate[i][bs] * plaintexts1[bs+gs]
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
    
    result0 = rotate_accumulate_many(acc0,giantsteps)
    result1 = rotate_accumulate_many(acc1,giantsteps)

    result = result0 + adjust_ct(result1)
    return result

def multiply_ct128x4096_pt4096x4096_old(ct,pt):

    babysteps = [i for i in range(16)]
    giantsteps = [i for i in range(-128,128,16)]

    babystep_rotate = [hoisted_rotate(ct[i],babysteps) for i in range(16)]

    result = [None for j in range(16)]
    for j in range(16):
        acc0 = [None for _ in range(len(giantsteps))]
        acc1 = [None for _ in range(len(giantsteps))]
        for i in range(16):
            plaintexts0 = prepare_pts_matmul_ct128x128_pt128x128(pt[j][i])
            plaintexts1 = prepare_pts_matmul_ct128x128_pt128x128(pt[j][16+i])

            for (g,gs) in enumerate(giantsteps):
                for (b,bs) in enumerate(babysteps):
                    temp0 = babystep_rotate[i][bs] * plaintexts0[bs+gs]
                    temp1 = babystep_rotate[i][bs] * plaintexts1[bs+gs]
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
        
        result0 = rotate_accumulate_many(acc0,giantsteps)
        result1 = rotate_accumulate_many(acc1,giantsteps)

        result[j] = result0 + adjust_ct(result1)
    return torch.stack(result)

def multiply_ct128x4096_pt4096x4096_new(ct,pt,freqs_cis=None):
    return multiply_ct128x4096_pt4096xwidth_new(4096,ct,pt,freqs_cis)

def multiply_ct128x4096_pt4096x1024_new(ct,pt,freqs_cis=None):
    return multiply_ct128x4096_pt4096xwidth_new(1024,ct,pt,freqs_cis)

def multiply_ct128x4096_pt4096x14336_new(ct,pt,freqs_cis=None):
    return multiply_ct128x4096_pt4096xwidth_new(14336,ct,pt,freqs_cis)

def apply_rotary_embeddings_packed(x,masks):
    mask0, mask1, mask1_ = masks
    x = (x * mask0) + rotate(x * mask1,1) + rotate(x * mask1_,-1)
    x += torch.conj(x)
    x = torch.real(x)
    return x

def multiply_ct128x4096_pt4096xwidth_new(width,ct,pt,freqs_cis):

    # width = 4
    assert width % 256 == 0
    width = width // 256

    babysteps = [i for i in range(16)]
    giantsteps = [i for i in range(-128,128,16)]

    babystep_rotate = [hoisted_rotate(ct[i],babysteps) for i in range(16)]
    
    if freqs_cis is not None:
        masks = prepare_adjust_and_apply_rotary_embeddings_masks()
    else:
        masks = prepare_adjust_ct_mask()

    result = [None for j in range(width)]
    for j in range(width):
        acc0 = [None for _ in range(len(giantsteps))]
        acc1 = [None for _ in range(len(giantsteps))]
        for i in range(16):
            plaintexts0 = prepare_pts_matmul_ct128x128_pt128x128(pt[j][i])
            plaintexts1 = prepare_pts_matmul_ct128x128_pt128x128(pt[j][16+i])

            for (g,gs) in enumerate(giantsteps):
                for (b,bs) in enumerate(babysteps):
                    temp0 = babystep_rotate[i][bs] * plaintexts0[bs+gs]
                    temp1 = babystep_rotate[i][bs] * plaintexts1[bs+gs]
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
        
        result0 = rotate_accumulate_many(acc0,giantsteps)
        result1 = rotate_accumulate_many(acc1,giantsteps)

        if freqs_cis is None:
            # result[j] = result0 + adjust_ct(result1)
            products = [result0 * masks[0]] + [result1 * masks[i] for i in [-128,128]]
            result[j] = rotate_accumulate_many(products,[0,-128,128])
        else:
            result[j] = adjust_ct_and_apply_rotary_embeddings(result0, result1, masks).type_as(result0)

    return torch.stack(result).type(ct.dtype)

def multiply_ct128xHeight_ptHeightxWidth_new(height,width,ct,pt,freqs_cis=None):

    # width = 4
    assert width % 256 == 0
    width = width // 256

    assert height % 256 == 0
    height = height // 256

    babysteps = [i for i in range(16)]
    giantsteps = [i for i in range(-128,128,16)]

    babystep_rotate = [hoisted_rotate(ct[i],babysteps) for i in range(height)]
    
    if freqs_cis is not None:
        masks = prepare_adjust_and_apply_rotary_embeddings_masks()

    result = [None for j in range(width)]
    for j in range(width):
        acc0 = [None for _ in range(len(giantsteps))]
        acc1 = [None for _ in range(len(giantsteps))]
        for i in range(height):
            plaintexts0 = prepare_pts_matmul_ct128x128_pt128x128(pt[j][i])
            plaintexts1 = prepare_pts_matmul_ct128x128_pt128x128(pt[j][height+i])

            for (g,gs) in enumerate(giantsteps):
                for (b,bs) in enumerate(babysteps):
                    temp0 = babystep_rotate[i][bs] * plaintexts0[bs+gs]
                    temp1 = babystep_rotate[i][bs] * plaintexts1[bs+gs]
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
        
        result0 = rotate_accumulate_many(acc0,giantsteps)
        result1 = rotate_accumulate_many(acc1,giantsteps)

        if freqs_cis is None:
            result[j] = result0 + adjust_ct(result1)
        else:
            result[j] = adjust_ct_and_apply_rotary_embeddings(result0, result1, masks).type_as(result0)

    return torch.stack(result).type(ct.dtype)

def rotate(x,steps):
    # steps = steps % SLOTS
    return torch.cat((x[steps:],x[:steps]))

def prepare_pts_matmul_ct128x128_pt128x128(pt):
    plaintexts = {}
    pt_2 = pt.reshape(128,2,128).transpose(0,1)

    for i in range(0,128):
        maskL = torch.ones(256,device=pt.device,dtype=torch.int32)
        maskR = torch.ones(256,device=pt.device,dtype=torch.int32)
        
        maskL[128-i:128] = 0
        maskL[256-i:256] = 0
        maskR[0:128-i] = 0
        maskR[128:256-i] = 0

        d0 = torch.cat((torch.diag(pt_2[0],-i),torch.diag(pt_2[0],128-i)))
        d1 = torch.cat((torch.diag(pt_2[1],-i),torch.diag(pt_2[1],128-i)))
        d = torch.cat((d0,d1))

        plaintexts[i] = (d * maskL).broadcast_to((128,256)).flatten()
        plaintexts[-128 + i] = (d * maskR).broadcast_to((128,256)).flatten()

    for gs in range(-128,128,16):
        for bs in range(16):
            plaintexts[bs+gs] = rotate(plaintexts[bs+gs],-gs)#.broadcast_to((128,256)).flatten()

    return plaintexts


def multiply_ct128x128_pt128x128_new(ct,pt):

    START = 0
    STOP = START + 128

    plaintexts = prepare_pts_matmul_ct128x128_pt128x128(pt)

    sumi = torch.zeros((SLOTS,),device=ct.device,dtype=ct.dtype)

    babysteps = {}
    for bs in range(16):
        babysteps[bs] = rotate(ct,bs)

    for gs in range(-128,128,16):
        bsSum = torch.zeros((SLOTS,),device=ct.device,dtype=ct.dtype)
        for bs in range(16):
            temp = babysteps[bs] * plaintexts[bs+gs]
            bsSum += temp
        bsSum = rotate(bsSum,gs)
        sumi += bsSum
    return sumi

def pack_attention_mask(att_mask):
    if att_mask is None:
        return 1
    att_mask_packed = torch.zeros((SLOTS,),dtype=torch.int32)
    for i in range(att_mask.shape[0]):
        for j in range(128):
            idx = (i + j) % 128
            if idx < att_mask.shape[1]:
                att_mask_packed[256*i + j] = att_mask[i][idx]
                att_mask_packed[128 + 256*i + j] = att_mask[i][idx]
    return att_mask_packed

def multiply_ct128x64_ct128x64_trans_new(A,B):

    DIV = 80 

    mask = torch.zeros((SLOTS,),dtype=torch.float64)
    mask[::128] = 1 / (DIV * np.sqrt(128))
    Bs = hoisted_rotate(B,[256*i for i in range(0,8)])
    As = hoisted_rotate(A,[-2048*i for i in range(0,16)])
    for gs in range(0,16):
        Arot = As[gs]
        for bs in range(8):
            Brot = Bs[bs]
            sumi = Arot * Brot
            j = 1
            while j < 128:
                sumi += rotate(sumi,j)
                j *= 2
            temp = sumi * mask
            if bs == 0:
                sumbs = temp
            else:
                temp = rotate(temp,-bs)
                sumbs += temp

        if gs == 0:
            sum = sumbs
        else:
            sumbs = rotate(sumbs,(2048 - 8) * gs)
            sum += sumbs

    return sum

def sign2(x):
    return f3(f3(g3(g3(x))))

def max_packed(A,B):
    AB = A + B
    AB_ = A - B
    zeroPt5 = torch.ones((SLOTS,)) / 2
    s = sign2(AB_)
    result = zeroPt5 * (AB + AB_ * s)
    return result


def calculate_max(A,DIV):
    one100 = torch.zeros((SLOTS,))
    one100[127::128] = DIV
    x = A
    for i in range(7):
        s = rotate(x,-(2**i))
        s = max_packed(s,x)
        x = s
    x *= one100
    for i in range(7):
        s = rotate(x,2**i)
        s += x
        x = s
    return x

def calculate_sum(A, DIV):
    one = torch.zeros((SLOTS,),dtype=A.dtype)
    one[0::128] = 1 / DIV
    x = A
    for i in range(int(np.log2(128))):
        s = rotate(x,2**i)
        s += x
        x = s
    x *= one
    for i in range(int(np.log2(128))):
        s = rotate(x,-2**i)
        s += x
        x = s
    return x

def exp_approx_packed(A):
    iters = 7
    one = 1
    oneByTwoIters = 1 / (2**iters)
    result = (A * oneByTwoIters) + one

    for _ in range(iters):
        result = result * result
    return result

def inv_sqrt_packed_softmax(x):
    y = 0.2
    iters = 6
    for _ in range(iters):
        y = y * (3 - x * y * y) * 0.5
    return y

def reciprocal(x):
    result = inv_sqrt(x) ** 2
    return result

def reciprocal_packed(x):
    result = inv_sqrt_packed(x)
    result = mul_vec(result,result)
    return result

def calculate_max_exact(A: torch.Tensor,DIV):
    A_ = A.reshape(-1,128)
    max = torch.max(A_,dim=-1,keepdim=True).values.broadcast_to(256,128)
    max = max * DIV
    return max.reshape(SLOTS)



def softmax_128x128_new(A,att_mask,prefix=None,suffix=None):
    A = A.double()
    DIV = 80
    max_values = calculate_max(A,DIV)
    ReferenceOutputs[f"{prefix}_max_values{suffix}"] = max_values.tolist()
    A = A * DIV
    A = A - max_values
    ReferenceOutputs[f"{prefix}_max_values_sub{suffix}"] = A.tolist()
    exp_res = exp_approx_packed(A)
    ReferenceOutputs[f"{prefix}_exp{suffix}"] = exp_res.tolist()
    exp_res *= att_mask
    ReferenceOutputs[f"{prefix}_exp_att_mask{suffix}"] = exp_res.tolist()
    ISQRT_DIV = 2
    sum = calculate_sum(exp_res, ISQRT_DIV)
    ReferenceOutputs[f"{prefix}_softmax_sum{suffix}"] = sum.tolist()
    rec = inv_sqrt_packed_softmax(sum)
    rec = rec * rec
    ReferenceOutputs[f"{prefix}_inverse{suffix}"] = rec.tolist()
    rec /= ISQRT_DIV
    rec /= 1.0001
    return rec * exp_res

def multiply_ct128x128_ct128x128_new(ct1,ct2,prefix,suffix):
        
    pts = {}
    for gs in range(16):
        for bs in range(8):
            i = bs + 8*gs
            mask = torch.zeros((SLOTS,),dtype=torch.int32,device=ct1.device)
            mask[i::128] = 1
            # pts[i] = rotate(mask,-2048*gs)
            pts[i] = mask

    rotBs = [rotate(ct2,bs*256) for bs in range(8)]

    for gs in range(16):
        for bs in range(8):
            i = bs + 8*gs
            r = rotBs[bs]
            p = pts[i]
            if bs == 0:
                diagBs = r * p
            else:
                r = r * p
                diagBs = diagBs + r
        if gs == 0:
            diag = diagBs
        else:
            diagBs = rotate(diagBs,2048*gs)
            diag = diag + diagBs
    
    ReferenceOutputs[f"{prefix}_diag"] = diag.tolist()
    diag_bs = hoisted_rotate(diag, [256*bs for bs in range(8)])
    for bs in range(8):
        ReferenceOutputs[f"{prefix}_diag_bs_{bs}"] = diag_bs[bs].tolist()

    ct1_gs = [rotate(ct1,8*gs -2048*gs) for gs in range(16)]

    for gs in range(16):
        for bs in range(8):
            i = bs + 8*gs
            maskL = torch.ones((SLOTS,),dtype=torch.int32,device=ct1.device)
            maskR = torch.ones((SLOTS,),dtype=torch.int32,device=ct1.device)
            for y in range(SLOTS//128):
                maskL[128*y + 128 - i : 128*y + 128] = 0
                maskR[128*y: 128*y + 128 - i] = 0
            l = rotate(ct1_gs[gs],bs)
            r = rotate(ct1_gs[gs],-128+bs)
            v = (l * maskL) + (r*maskR)
            diagRot = diag_bs[bs]
            if bs == 0:
                sumbs = diagRot * v
            else:
                diagRot = diagRot * v
                sumbs = sumbs + diagRot
        if gs == 0:
            sumi = sumbs
        else:
            sumbs = rotate(sumbs,2048*gs)
            sumi = sumbs + sumi
    return sumi




class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        world_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):

        run_refr = self.layer_id == 0

        bsz, seqlen, _ = x.shape

        X = x
        Q = self.wq.weight.T
        K = self.wk.weight.T
        V = self.wv.weight.T
        O = self.wo.weight.T
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)


        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) 

        scores /= math.sqrt(self.head_dim)
        scores = softmax_approx5(scores, mask).type_as(xq)  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output_wo = self.wo(output)

        if not run_refr:
            return output_wo

        ###########################

        x_packed = row_pack_nx4096(X)

        prefix = "attention"

        for i in range(16):
            ReferenceOutputs[f"{prefix}_input{i}"] = x_packed[i].tolist()

        order = [0 , 4 , 1 , 5, 2 , 6 , 3 , 7] + [8 , 12 , 9 , 13, 10 , 14 , 11 , 15] + [16 , 20 , 17 , 21, 18 , 22 , 19 , 23] + [24 , 28 , 25 , 29, 26 , 30 , 27 , 31]

        Q = Q.view(4096,32,128)
        Q_ = Q.clone()
        for i in range(32):
                Q[:,i] = Q_[:,order[i]]
        Q = Q.view(4096,4096)
        
        wq_packed = row_pack_4096x4096(Q)
        wk_packed = row_pack_4096x1024(K)
        wv_packed = row_pack_4096x1024(V)
        mul_packedQ = multiply_ct128x4096_pt4096x4096_new(x_packed,wq_packed,freqs_cis)
        mul_packedK = multiply_ct128x4096_pt4096x1024_new(x_packed,wk_packed,freqs_cis)
        mul_packedV = multiply_ct128x4096_pt4096x1024_new(x_packed,wv_packed)

        plaintexts = prepare_pts_matmul_ct128x128_pt128x128(wq_packed[0][0])
        product000 = x_packed[0] * plaintexts[0]
        ReferenceOutputs[f"{prefix}_product000"] = product000.tolist()



        prod0 = multiply_ct128x128_pt128x128_new(x_packed[0],wq_packed[0][0])
        prod1 = multiply_ct128x128_pt128x128_new(x_packed[0],wq_packed[0][16])
        ReferenceOutputs[f"{prefix}_product00"] = prod0.tolist()
        ReferenceOutputs[f"{prefix}_product01"] = prod1.tolist()
        for i in range(1,16):
            p0 = multiply_ct128x128_pt128x128_new(x_packed[i],wq_packed[0][i])
            p1 = multiply_ct128x128_pt128x128_new(x_packed[i],wq_packed[0][16+i])
            ReferenceOutputs[f"{prefix}_product{i}0"] = p0.tolist()
            ReferenceOutputs[f"{prefix}_product{i}1"] = p1.tolist()
            prod0 += p0
            prod1 += p1

        ReferenceOutputs[f"{prefix}_product0"] = prod0.tolist()
        ReferenceOutputs[f"{prefix}_product1"] = prod1.tolist()

        ReferenceOutputs[f"{prefix}_product_adjust"] = (prod0 + adjust_ct(prod1)).tolist()
        ReferenceOutputs[f"{prefix}_product_adjust_freqs"] = (adjust_ct_and_apply_rotary_embeddings(prod0,prod1,prepare_adjust_and_apply_rotary_embeddings_masks())).tolist()




        for i in range(16):
            ReferenceOutputs[f"{prefix}_Q{i}"] = mul_packedQ[i].tolist()
        for i in range(4):
            ReferenceOutputs[f"{prefix}_K{i}"] = mul_packedK[i].tolist()
            ReferenceOutputs[f"{prefix}_V{i}"] = mul_packedV[i].tolist()



        scores_packed = torch.zeros(mul_packedQ.shape,device=x.device,dtype=mul_packedQ.dtype)

        att_mask = pack_attention_mask(mask)
        ReferenceOutputs[f"att_mask"] = att_mask.tolist()

        # raise Exception("")


        for i in range(16):
            scores_packed[i] = multiply_ct128x64_ct128x64_trans_new(mul_packedQ[i],mul_packedK[i//4])
            ReferenceOutputs[f"{prefix}_QK{i}"] = scores_packed[i].tolist()
            scores_packed[i] = softmax_128x128_new(scores_packed[i],att_mask,f"attention_{2*(i//2)}",f"{i%2}").type_as(mul_packedQ[i])
            ReferenceOutputs[f"{prefix}_softmax{i}"] = scores_packed[i].tolist()

        for i in range(16):
            scores_packed[i] = multiply_ct128x128_ct128x128_new(scores_packed[i],mul_packedV[i//4],f"attention_{4*(i//4)}",f"{i%4}")
            ReferenceOutputs[f"{prefix}_attention{i}"] = scores_packed[i].tolist()


        O = O.view(self.n_local_heads,self.head_dim,4096)
        O_ = O.clone()
        for i in range(32):
            O[i] = O_[order[i]]
        O = O.contiguous().view(4096,4096)

        wo_packed = row_pack_4096x4096(O)
        mul_packedO = multiply_ct128x4096_pt4096x4096_new(scores_packed,wo_packed)

        for i in range(16):
            ReferenceOutputs[f"{prefix}_attention_out{i}"] = mul_packedO[i].tolist()


        mul_packedO = row_unpack_nx4096(mul_packedO,seqlen)

        diff = output_wo - mul_packedO
        print(self.layer_id,torch.mean(diff**2))

        return mul_packedO

def approx_silu(x):
    
    x = x.float()

    res1 = -0.3067541139982155 -0.0819767021525476 * x -0.0055465625580307 * x**2
    res2 = 0.0085064025895951 + 0.5 * x + 0.2281430841728270 * x ** 2 -0.011113046708173 * x**4  + 0.0002743776353465 * x**6
    res3 = x

    DIV = 50

    comp_x = x 

    s0 = sign_approx((comp_x + 8)/DIV)
    s1 = sign_approx((comp_x + 4)/DIV)
    s2 = sign_approx((comp_x - 4)/DIV)

    b0 = 1 - s0
    b1 = s0 - s1
    b2 = s1 - s2
    b3 = s2 + 1

    # label0 = (x <= -8)
    # label1 = (-8 < x) * (x <= -4)
    # label2 = (-4 < x) * (x <= 4)
    # label3 = (x > 4)

    res = b1 * res1 + b2 * res2 + b3 * res3
    res = res / 2
    return res


silu_values = []
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        layer_id: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.layer_id = layer_id

    def forward(self, x):

        run_refr = self.layer_id == 0
        v = self.w1(x)
        w = self.w3(x)
        v = approx_silu(v).type_as(x)

        o = self.w2(v * w)

        if not run_refr: 
            return o

        x_packed = row_pack_nx4096(x)

        w1_packed = row_pack_4096x14336(self.w1.weight.T)
        w3_packed = row_pack_4096x14336(self.w3.weight.T)
        w2_packed = row_pack_14336x4096(self.w2.weight.T)

        mul_packedV = multiply_ct128x4096_pt4096x14336_new(x_packed,w1_packed)
        mul_packedW = multiply_ct128x4096_pt4096x14336_new(x_packed,w3_packed)

        silu_packed = approx_silu(mul_packedV).type_as(x_packed)

        vw_packed = silu_packed * mul_packedW

        for i in range(56):
            ReferenceOutputs[f"ffn_V{i}"] = mul_packedV[i].tolist()
            ReferenceOutputs[f"ffn_W{i}"] = mul_packedW[i].tolist()
            ReferenceOutputs[f"ffn_silu{i}"] = silu_packed[i].tolist()
            ReferenceOutputs[f"ffn_VW{i}"] = vw_packed[i].tolist()

        o_packed = multiply_ct128xHeight_ptHeightxWidth_new(14336,4096,vw_packed,w2_packed)

        for i in range(16):
            ReferenceOutputs[f"ffn_O{i}"] = o_packed[i].tolist()


        o_packed = row_unpack_nx4096(o_packed , x.shape[1])
        diffO = o_packed - o
        print(torch.mean(diffO.float()**2))

        print(f"x: {x.shape} v: {v.shape} w: {w.shape} o: {o.shape}, w2: {self.w2.weight.shape}")
        return o_packed


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, layer_id)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            layer_id=layer_id
        )
        self.layer_id = layer_id
        # self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention_norm = RMSNorm2(layer_id,f"att",args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm2(layer_id,f"ffn",args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        h_packed = row_pack_nx4096(h)
        for i in range(16): 
            ReferenceOutputs[f"H{i}"] = h_packed[i].tolist()
        out = h + self.feed_forward(self.ffn_norm(h))
        out_packed = row_pack_nx4096(out)
        for i in range(16): 
            ReferenceOutputs[f"out{i}"] = out_packed[i].tolist()

        raise Exception("Complete")
        return out


class TransformerPacked(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.norm = RMSNorm2(1000,"out",params.dim, eps=params.norm_eps)
        self.norm = RMSNorm3(1000,"out",params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        mask2 = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)


            mask2 = torch.full((seqlen, seqlen), float(1.0), device=tokens.device)
            mask2 = torch.tril(mask2, diagonal=0)

            # https://github.com/pytorch/pytorch/issues/100005
            # torch.triu is buggy when the device is mps: filled values are
            # nan instead of 0.
            if mask.device.type == torch.device("mps").type:
                mask = torch.nan_to_num(mask, nan=0.0)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)
            mask2 = torch.hstack([torch.ones((seqlen, start_pos), device=tokens.device), mask2]).int() #.type_as(h)
            mask = mask2

        refr_layer = 0
        data_sample = 0
        try:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
            h = self.norm(h)
            output = self.output(h).float()
            return output
        except Exception as e:
            print(f"Error Generating Packed Values for {layer} in {data_sample}: {e}")
            print("")
            ReferenceOutputs_ = {}
            with open(f'reference_outputs_{data_sample}/reference_outputs_layer{refr_layer}.pkl', 'rb') as f:
                ReferenceOutputs_ = pickle.load(f)
            ReferenceOutputs_.update(ReferenceOutputs)
            print(ReferenceOutputs.keys())
            with open(f'reference_outputs_{data_sample}/reference_outputs_layer{refr_layer}.pkl', 'wb') as f:
                pickle.dump(ReferenceOutputs_, f)



