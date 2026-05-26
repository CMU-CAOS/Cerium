# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from pathlib import Path
from typing import Optional

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

from .primes import Primes

import pickle

import os


# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.

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

    def _norm(self):

        if self.layer_id > 28:
            DIV1 = (2)
            DIV2 = (2)
        elif self.layer_id >= 2:
            DIV1 = (0.01)
            DIV2 = (2)
        else:
            DIV1 = (0.01)
            DIV2 = (0.01)

        Inputs = {}

        x_level = 16
        x_scale = Primes[x_level - 1]*Primes[x_level -2]

        xsq_scale = x_scale * x_scale

        Inputs["rmsnorm_eps"] = (self.eps * 4096, xsq_scale)

        oneByN = torch.zeros((SLOTS,))
        oneByN[255::256] = 1 / 4096

        div_tensor = row_pack_div_tensor(DIV1,DIV2)
        
        oneByN = oneByN / div_tensor

        oneByNscale = Primes[x_level-1] * Primes[x_level-2] * Primes[x_level-3] * Primes[x_level-4] *Primes[x_level-5] * Primes[x_level-6] / xsq_scale

        Inputs["rmsnorm_oneByN"] = (oneByN.tolist(),oneByNscale)

        rmsnorm_isqrt_out_level = x_level
        rmsnorm_isqrt_out_scale = Primes[rmsnorm_isqrt_out_level-3]*Primes[rmsnorm_isqrt_out_level-4]

        div_isqrt_inv_level = x_level
        div_isqrt_inv_scale = Primes[div_isqrt_inv_level-1]*Primes[div_isqrt_inv_level-2]*Primes[div_isqrt_inv_level-3]*Primes[div_isqrt_inv_level-4]*Primes[div_isqrt_inv_level-5]*Primes[div_isqrt_inv_level-6] / (x_scale * rmsnorm_isqrt_out_scale)

        div_tensor_isqrt = 1/torch.sqrt(div_tensor)
        Inputs[f"rmsnorm_div_isqrt"] = (div_tensor_isqrt.tolist(),div_isqrt_inv_scale)


        for i in range(16):
            weight = self.weight[256*i:256*i+256].broadcast_to(128,256).reshape(SLOTS) * div_tensor_isqrt
            Inputs[f"rmsnorm_{self.name}_weight{i}"] = (weight.tolist(),div_isqrt_inv_scale)


        return Inputs


    def forward(self):
        return self._norm()

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

def row_pack_4096x1024(Mat):
    w, n = Mat.shape
    assert w == 4096
    assert n == 1024

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

    torch.set_printoptions(threshold=torch.inf, linewidth=10000, sci_mode=False)

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

def rotate(x,steps):
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

        plaintexts[i] = (d * maskL)
        plaintexts[-128 + i] = (d * maskR)

    for gs in range(-128,128,16):
        for bs in range(16):
            plaintexts[bs+gs] = rotate(plaintexts[bs+gs],-gs)#[0:256]#.broadcast_to((128,256)).flatten()

    return plaintexts

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
    ):

        prefix = "attention"
        xLevel = 13
        Inputs = {}

        Q = self.wq.weight.T
        K = self.wk.weight.T
        V = self.wv.weight.T
        O = self.wo.weight.T

        order = [0 , 4 , 1 , 5, 2 , 6 , 3 , 7] + [8 , 12 , 9 , 13, 10 , 14 , 11 , 15] + [16 , 20 , 17 , 21, 18 , 22 , 19 , 23] + [24 , 28 , 25 , 29, 26 , 30 , 27 , 31]

        Q = Q.view(4096,32,128)
        Q_ = Q.clone()
        for i in range(32):
                Q[:,i] = Q_[:,order[i]]
        Q = Q.view(4096,4096)
        
        wq_packed = row_pack_4096x4096(Q)
        wk_packed = row_pack_4096x1024(K)
        wv_packed = row_pack_4096x1024(V)

        masks_freqs_cis_scale = Primes[xLevel-4] * Primes[xLevel - 5]
        masks_freqs_cis = prepare_adjust_and_apply_rotary_embeddings_masks()
        for i in [-1,0,1,-129,-128,-127,127,128,129]:
            Inputs[f"{prefix}_mask_freqs_cis_{i}"] = (masks_freqs_cis[i].tolist(),masks_freqs_cis_scale)


        masks = prepare_adjust_ct_mask()
        
        masks_V_scale= Primes[xLevel-4] * np.sqrt(Primes[xLevel - 5])
        for i in [-128,0,128]:
            Inputs[f"{prefix}_mask_{i}"] = (masks[i].tolist(),masks_V_scale)




        Wq_scale = Primes[xLevel-6]*Primes[xLevel-7]
        Wk_scale = Primes[xLevel-8]*Primes[xLevel-9]
        Wv_scale = np.sqrt(Primes[xLevel-5])*Primes[xLevel-6]

        for w in range(16):
            for h in range(32):
                pts = prepare_pts_matmul_ct128x128_pt128x128(wq_packed[w][h])
                for (k,val) in pts.items():
                    Inputs[f"{prefix}_Wq_{w}_{h}_{k}"] = (val.tolist(),Wq_scale)

        for w in range(4):
            for h in range(32):
                pts = prepare_pts_matmul_ct128x128_pt128x128(wk_packed[w][h])
                for (k,val) in pts.items():
                    Inputs[f"{prefix}_Wk_{w}_{h}_{k}"] = (val.tolist(),Wk_scale)
                pts = prepare_pts_matmul_ct128x128_pt128x128(wv_packed[w][h])
                for (k,val) in pts.items():
                    Inputs[f"{prefix}_Wv_{w}_{h}_{k}"] = (val.tolist(),Wv_scale)


        O = O.view(self.n_local_heads,self.head_dim,4096)
        O_ = O.clone()
        for i in range(32):
            O[i] = O_[order[i]]
        O = O.contiguous().view(4096,4096)

        wo_packed = row_pack_4096x4096(O)

        BootststrapScaleDiv = 1 << 6

        bootstrap_out_level = 16

        masks_O_scale= Primes[xLevel-4] * np.sqrt(Primes[bootstrap_out_level- 1]/BootststrapScaleDiv)

        Wo_scale = np.sqrt(Primes[bootstrap_out_level-1]/BootststrapScaleDiv)*Primes[bootstrap_out_level-2]

        for i in [-128,0,128]:
            Inputs[f"{prefix}_maskO_{i}"] = (masks[i].tolist(),masks_O_scale)
            Inputs[f"{prefix}_maskO_re{i}"] = (masks[i].tolist(),masks_O_scale)
            Inputs[f"{prefix}_maskO_im{i}"] = ((1j * masks[i]).tolist(),masks_O_scale)


        for w in range(16):
            for h in range(32):
                pts = prepare_pts_matmul_ct128x128_pt128x128(wo_packed[w][h])
                for (k,val) in pts.items():
                    Inputs[f"{prefix}_Wo_{w}_{h}_{k}"] = (val.tolist(),Wo_scale)
        
        Inputs[f"{prefix}_x_rescale_re"] = ([1],Primes[bootstrap_out_level-1]*Primes[bootstrap_out_level-2] / BootststrapScaleDiv)
        Inputs[f"{prefix}_x_rescale_im"] = ([1j],Primes[bootstrap_out_level-1]*Primes[bootstrap_out_level-2] / BootststrapScaleDiv)


        return Inputs

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
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

    def forward(self):

        prefix = "ffn"

        Inputs = {}


        vLevel = 6
        VBootstrapOutLevel = 21
        VBootststrapScaleDiv = 1000

        wLevel = 12

        xLevel = 13
        xScale = Primes[xLevel-1] * Primes[xLevel-2]*Primes[xLevel-3]

        Wv_scale = Primes[vLevel-1] * Primes[vLevel-2] * Primes[vLevel-3] * np.sqrt(Primes[VBootstrapOutLevel-1]/VBootststrapScaleDiv) * Primes[VBootstrapOutLevel-2] / xScale
        Ww_scale = Primes[wLevel-1] * Primes[wLevel-2] * Primes[wLevel-3] * np.sqrt(Primes[wLevel-5]) * Primes[wLevel-6] / xScale


        w1_packed = row_pack_4096x14336(self.w1.weight.T)
        w3_packed = row_pack_4096x14336(self.w3.weight.T)
        w2_packed = row_pack_14336x4096(self.w2.weight.T)

        masks_V_scale= Primes[vLevel-4] * np.sqrt(Primes[VBootstrapOutLevel- 1]/VBootststrapScaleDiv)
        masks_W_scale= Primes[wLevel-4] * np.sqrt(Primes[wLevel- 5])

        masks = prepare_adjust_ct_mask()
        
        for i in [-128,0,128]:
            Inputs[f"{prefix}_maskW_{i}"] = (masks[i].tolist(),masks_W_scale)
            Inputs[f"{prefix}_maskV_re{i}"] = (masks[i].tolist(),masks_V_scale)
            Inputs[f"{prefix}_maskV_im{i}"] = ((1j * masks[i]).tolist(),masks_V_scale)


        for w in range(56):
            for h in range(32):
                pts = prepare_pts_matmul_ct128x128_pt128x128(w1_packed[w][h])
                for (k,val) in pts.items():
                    Inputs[f"{prefix}_Wv_{w}_{h}_{k}"] = (val.tolist(),Wv_scale)
        for w in range(56):
            for h in range(32):
                pts = prepare_pts_matmul_ct128x128_pt128x128(w3_packed[w][h])
                for (k,val) in pts.items():
                    Inputs[f"{prefix}_Ww_{w}_{h}_{k}"] = (val.tolist(),Ww_scale)
        
        OBootststrapScaleDiv = 1 << 6
        OBootstrapOutLevel = 16
        vw_level = 6
        vw_scale = Primes[vw_level-1] * Primes[vw_level-2] * Primes[vw_level-3]
        masks_O_scale= Primes[vw_level - 4] * np.sqrt(Primes[OBootstrapOutLevel- 1]/OBootststrapScaleDiv)
        Wo_scale = np.sqrt(Primes[OBootstrapOutLevel-1]/OBootststrapScaleDiv)*Primes[OBootstrapOutLevel-2]

        for w in range(16):
            for h in range(112):
                pts = prepare_pts_matmul_ct128x128_pt128x128(w2_packed[w][h])
                for (k,val) in pts.items():
                    Inputs[f"{prefix}_Wo_{w}_{h}_{k}"] = (val.tolist(),Wo_scale)

        for i in [-128,0,128]:
            Inputs[f"{prefix}_maskO_re{i}"] = (masks[i].tolist(),masks_O_scale)
            Inputs[f"{prefix}_maskO_im{i}"] = ((1j * masks[i]).tolist(),masks_O_scale)

        
        return Inputs



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
        )
        self.layer_id = layer_id
        # self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention_norm = RMSNorm2(layer_id,f"att",args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm2(layer_id,f"ffn",args.dim, eps=args.norm_eps)

    def forward(
        self,
    ):
        Inputs = {}
        rmsnorm_attention_inputs = self.attention_norm()
        Inputs.update(rmsnorm_attention_inputs)

        attention_inputs = self.attention()
        Inputs.update(attention_inputs)

        rmsnorm_ffn_inputs = self.ffn_norm()
        Inputs.update(rmsnorm_ffn_inputs)

        ffn_inputs = self.feed_forward()
        Inputs.update(ffn_inputs)
        return Inputs


class TransformerParams(nn.Module):
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
    def forward(self):
        run_block = int(os.environ.get('LLAMA_PARAMS_BLOCK', '10000'))
        params_dir = (os.environ.get('LLAMA_PARAMS_DIR', 'params'))
        Path(params_dir).mkdir(parents=True, exist_ok=True)
        print("\nRunning Block:", run_block)
        Program_Inputs = {}

        Inputs = self.layers[run_block]()
        Program_Inputs.update(Inputs)
        with (Path(params_dir) / f"params_block{run_block}.pkl").open("wb") as f:
            pickle.dump(Program_Inputs, f)
