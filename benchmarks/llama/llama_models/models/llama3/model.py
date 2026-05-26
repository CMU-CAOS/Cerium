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
import os
from pathlib import Path

# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.


rms_norm_values = []

def get_data_sample() -> int:
    """Return the sample selected by ``run_unpacked.sh``."""
    value = os.environ.get("DATA_SAMPLE", "0")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"DATA_SAMPLE must be an integer sample identifier, got {value!r}"
        ) from error


def get_check_block() -> int:
    value = os.environ.get("BLOCK", "100")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"BLOCK must be an integer layer identifier, got {value!r}") from error


FHE_OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "fhe_outputs"

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


SLOTS = 32768
def row_unpack_nx1(x,n):
    w = x.shape
    assert w[0] == 32768 
    assert n <= 128 

    res = torch.zeros((1,n,1),device=x.device,dtype=x.dtype)
    for i in range(n):
        res[0][i][0] = x[256*i]
    return res

class RMSNorm2(torch.nn.Module):
    def __init__(self, layer_id, name, dim: int, eps: float = 1e-6):
        super().__init__()
        self.layer_id = layer_id
        self.name = name
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
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


        pick_inv = (1 - pick)

        div_tensor = pick * DIV1 + pick_inv * DIV2

        w = inv_sqrt(w / div_tensor)

        w /= torch.sqrt(div_tensor)


        v = w
        res = x * v

        res = res * self.weight
        return res

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
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

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


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
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

def exp_approx_softmax(x):
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
    return sgn22(x)


def softmax_approx(scores,attention_mask):

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
    scores_max_exp = exp_approx_softmax(scores_max )

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



SLOTS = 32768
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

def row_unpack_nx32x128(x,n):
    p, w = x.shape
    assert p == 16
    assert w == SLOTS

    res = torch.zeros((1,n,32,128),device=x.device,dtype=x.dtype)
    for j in range(16):
        for i in range(n):
            res[0][i][2*j : 2*(j+1)] = x[j][256*i:256*(i+1)].reshape(-1,128)
    return res

def row_unpack_32xnxn(x,n):
    res = row_unpack_nx32x128(x,n)
    res = res.transpose(1,2)[:,:,:,0:n]
    return res



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

        bsz, seqlen, _ = x.shape
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
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        scores = softmax_approx(scores, mask).type_as(xq)  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)


        return output

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
        layer_id: int
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
        v = self.w1(x)
        w = self.w3(x)
        v = approx_silu(v).type_as(x)
        o = self.w2(v * w)
        return o


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
            layer_id=layer_id,
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
        check_block = get_check_block()
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        if self.layer_id == check_block:
            data_sample = get_data_sample()
            ReferenceInputs = {}
            reference_file = FHE_OUTPUTS_DIR / f"sample{data_sample}_block{self.layer_id}.pkl"
            with reference_file.open("rb") as f:
                ReferenceInputs.update(pickle.load(f))
            out_ = torch.stack([(torch.tensor(ReferenceInputs[f"out{i}"],dtype=torch.complex128)*pow(-1j,i%2)).real for i in range(16)])
            out_ = row_unpack_nx4096(out_,out.shape[1])
            out = out_
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

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
            mask2 = torch.hstack([torch.ones((seqlen, start_pos), device=tokens.device), mask2]).type_as(h)
            mask = mask2

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
