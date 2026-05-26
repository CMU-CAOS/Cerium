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

"""Torch-native packed-parameter exporter for the saved BERT model."""
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import torch

from model import MODEL_PATH, load_model
from packing import (
    SLOTS,
    pack_2x1,
    rotate,
    row_pack_64x1,
    row_pack_768x1,
    row_pack_768x1_type2,
    row_pack_768x2,
    row_pack_768x64,
    row_pack_768x768,
    zeros,
)
from primes import Primes

_BSGS_PLANS = {}


def _bsgs_plan(device):
    """Return reusable integer index maps for BSGS packing on ``device``."""
    key = (device.type, device.index)
    if key not in _BSGS_PLANS:
        offsets = torch.arange(128, device=device).unsqueeze(1)
        rows = torch.arange(128, device=device).unsqueeze(0)
        source = ((offsets + rows) % 128) * 256 + rows
        local_columns = torch.arange(256, device=device).remainder(128).unsqueeze(0)
        keep_left = local_columns < (128 - offsets)
        keep_right = local_columns >= (128 - offsets)
        keys = torch.arange(-128, 128, device=device)
        giant_steps = torch.arange(-128, 128, 16, device=device).repeat_interleave(16)
        gather = (torch.arange(256, device=device).unsqueeze(0) - giant_steps.unsqueeze(1)).remainder(256)
        _BSGS_PLANS[key] = source, keep_left, keep_right, keys, gather
    return _BSGS_PLANS[key]


def bsgs_plaintexts(plaintext, prefix, type3=False):
    """Vectorized BSGS packing with the same values and key order as before."""
    source, keep_left, keep_right, keys, gather = _bsgs_plan(plaintext.device)
    pointer = torch.cat((plaintext[source], plaintext[source + 128]), dim=1)
    left = pointer * keep_left.to(dtype=plaintext.dtype)
    right = pointer * keep_right.to(dtype=plaintext.dtype)
    packed = torch.cat((right, left), dim=0).gather(1, gather)
    key_values = keys.tolist()
    if not type3:
        return {f"{prefix}_{key}": value for key, value in zip(key_values, packed.unbind())}

    left_mask = torch.cat((torch.ones(128, dtype=plaintext.dtype, device=plaintext.device), torch.zeros(128, dtype=plaintext.dtype, device=plaintext.device)))
    right_mask = 1 - left_mask
    rotated_left = left_mask.expand(256, -1).gather(1, gather)
    rotated_right = right_mask.expand(256, -1).gather(1, gather)
    result = {}
    for key, left_value, right_value in zip(key_values, (packed * rotated_left).unbind(), (packed * rotated_right).unbind()):
        result[f"{prefix}_{key + 128}"] = right_value
        result[f"{prefix}_{key - 128}"] = left_value
    return result


def add_plaintexts(destination, values, scale):
    destination.update({name: (value, scale) for name, value in values.items()})


def numpy_payload(values):
    """Convert packed Torch tensors to NumPy arrays for efficient pickling."""
    return {
        name: (value.detach().cpu().numpy() if torch.is_tensor(value) else value, scale)
        for name, (value, scale) in values.items()
    }


def linear1(parameters):
    HEAD_ORDER = (0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11)
    values = {}
    level = 13
    h_scale = Primes[level - 1] * Primes[level - 2] * Primes[level - 3]
    packed_weights = {
        name: [row_pack_768x64(parameters[name][HEAD_ORDER[2 * i]], parameters[name][HEAD_ORDER[2 * i + 1]]) for i in range(6)]
        for name in ("Wq", "Wk", "Wv")
    }
    packed_biases = {
        name: [row_pack_64x1(parameters[name][HEAD_ORDER[2 * i]], parameters[name][HEAD_ORDER[2 * i + 1]]) for i in range(6)]
        for name in ("Bq", "Bk", "Bv")
    }

    mask_scale = Primes[level - 4] * Primes[level - 5]
    mask_left = torch.ones(SLOTS, dtype=parameters["Wq"].dtype, device=parameters["Wq"].device).reshape(-1, 128)
    mask_right = mask_left.clone()
    mask_left[1::2], mask_right[0::2] = 0, 0
    values[f"attention_bsgs_Mask_{level}"] = (1, mask_scale)
    values[f"attention_bsgs_MaskL_{level}"] = (mask_left.flatten(), mask_scale)
    values[f"attention_bsgs_MaskR_{level}"] = (mask_right.flatten(), mask_scale)
    h_scale *= mask_scale

    weight_scales = {"Wq": Primes[level - 6] * Primes[level - 7], "Wk": Primes[level - 8] * Primes[level - 9], "Wv": Primes[level - 6] * Primes[level - 7]}
    for name in ("Wq", "Wk", "Wv"):
        for pair in range(6):
            for packed_row in range(6):
                add_plaintexts(values, bsgs_plaintexts(packed_weights[name][pair][packed_row], f"attention_{name}{pair}_{packed_row}"), weight_scales[name])

    denominator = Primes[level - 1] * Primes[level - 2] * Primes[level - 3] * Primes[level - 4]
    for pair in range(6):
        values[f"attention_Bq{pair}"] = (packed_biases["Bq"][pair], h_scale * weight_scales["Wq"] / denominator)
        values[f"attention_Bk{pair}"] = (packed_biases["Bk"][pair], h_scale * weight_scales["Wk"] / denominator)
        values[f"attention_Bv{pair}"] = (packed_biases["Bv"][pair], h_scale * weight_scales["Wv"] / denominator)

    mask = zeros(SLOTS, parameters["Wq"])
    mask[::128] = 1 / 40
    values["matmul_ct128x128_mask"] = (mask, Primes[15] * Primes[14] / (1 << 10))
    values["QK_t_bs_scale_factor"] = (1, 1 << 10)
    attention_level = level - 4
    for giant in range(16):
        for baby in range(8):
            shift = baby + 8 * giant
            rotation_mask = zeros(SLOTS, parameters["Wq"])
            rotation_mask[shift::128] = 1
            values[f"attention_matmul_mask_128x128_rot_{shift * 128}"] = (rotation_mask, Primes[attention_level - 5] * math.sqrt(Primes[attention_level - 6]))
            mask_left = torch.ones(SLOTS, dtype=parameters["Wq"].dtype, device=parameters["Wq"].device)
            mask_right = torch.ones_like(mask_left)
            for group in range(SLOTS // 128):
                mask_left[128 * group + 128 - shift:128 * group + 128] = 0
                mask_right[128 * group:128 * group + 128 - shift] = 0
            scale = math.sqrt(Primes[attention_level - 6]) * Primes[attention_level - 7]
            values[f"attention_matmul_maskL_128x128_{shift}"] = (mask_left, scale)
            values[f"attention_matmul_maskR_128x128_{shift}"] = (mask_right, scale)
    return values


def linear2(weight, bias):
    values = {}
    level, h1_level = 5, 13
    weight_scale = Primes[17] * Primes[16] / (1 << 10)
    h2_scale = Primes[level - 1] * Primes[level - 2] * Primes[level - 3]
    h2_scale = h2_scale * weight_scale / (Primes[level - 1] * Primes[level - 2])
    h1_scale = Primes[h1_level - 1] * Primes[h1_level - 2] * Primes[h1_level - 3]
    values["attention_h1_rescale_factor"] = (1, h1_scale * h2_scale / h1_scale)
    packed_weight, packed_bias = row_pack_768x768(weight), row_pack_768x1(bias)
    for chunk in range(3):
        for packed_row in range(3):
            add_plaintexts(values, bsgs_plaintexts(packed_weight[chunk][packed_row], f"attention_Wo_{chunk}_{packed_row}"), weight_scale)
        for packed_row in range(3, 6):
            add_plaintexts(values, bsgs_plaintexts(packed_weight[chunk][packed_row], f"attention_Wo_{chunk}_{packed_row}", type3=True), weight_scale)
        values[f"attention_Bo_{chunk}"] = (packed_bias[chunk], h2_scale)
    return values


def linear3(weight, bias):
    values = {}
    weight_scale = Primes[20] * Primes[19] / 1000
    bias_scale = Primes[4] * Primes[3] * Primes[2] * weight_scale / (Primes[4] * Primes[3])
    for part in range(4):
        packed_weight = row_pack_768x768(weight[:, 768 * part:768 * (part + 1)])
        packed_bias = row_pack_768x1(bias[768 * part:768 * (part + 1)])
        for chunk in range(3):
            for packed_row in range(3):
                add_plaintexts(values, bsgs_plaintexts(packed_weight[chunk][packed_row], f"ffn_Wi_{3 * part + chunk}_{packed_row}"), weight_scale)
            for packed_row in range(3, 6):
                add_plaintexts(values, bsgs_plaintexts(packed_weight[chunk][packed_row], f"ffn_Wi_{3 * part + chunk}_{packed_row}", type3=True), weight_scale)
            values[f"ffn_Bi_{3 * part + chunk}"] = (packed_bias[chunk], bias_scale)
    return values


def linear4(weight, bias):
    values = {}
    weight_scale = Primes[17] * Primes[16] / (1 << 10)
    for part in range(4):
        packed_weight = row_pack_768x768(weight[768 * part:768 * (part + 1), :])
        for chunk in range(3):
            for packed_row in range(3):
                add_plaintexts(values, bsgs_plaintexts(packed_weight[chunk][packed_row], f"ffn_Wo_{3 * part + chunk}_{packed_row}"), weight_scale)
            for packed_row in range(3, 6):
                add_plaintexts(values, bsgs_plaintexts(packed_weight[chunk][packed_row], f"ffn_Wo_{3 * part + chunk}_{packed_row}", type3=True), weight_scale)
    h_scale = Primes[4] * Primes[3] * Primes[2] * weight_scale / Primes[4]
    packed_bias = row_pack_768x1(bias)
    for chunk in range(3):
        values[f"ffn_Bo_{chunk}"] = (packed_bias[chunk], h_scale)
    h_scale /= Primes[3]
    h4_scale = Primes[4] * Primes[3] * Primes[2] / Primes[4]
    values["ffn_h_rescale_factor"] = (1, h_scale * Primes[3] / h4_scale)
    return values


def layer_norm(weight, bias, kind):
    values = {}
    level, bootstrap_out_level, divisor = 18, 13, 10
    h_scale = Primes[level - 1] * Primes[level - 2]
    var_scale = Primes[level - 5] * Primes[level - 6]
    one_sum_scale = math.sqrt(var_scale * Primes[level - 3] * Primes[level - 4]) * Primes[level - 1] * Primes[level - 2] / h_scale
    one_sq_sum_scale = var_scale * Primes[level - 1] * Primes[level - 2] * Primes[level - 3] * Primes[level - 4] / (h_scale * h_scale)
    one_sum, one_sq_sum = zeros(SLOTS, weight), zeros(SLOTS, weight)
    one_sum[::256] = (1 / 768) / divisor
    one_sq_sum[::256] = (1 / 768) / (divisor * divisor)
    values[f"layer_norm_{kind}_oneByN_sum"] = (one_sum, one_sum_scale)
    values[f"layer_norm_{kind}_oneByN_sq_sum"] = (one_sq_sum, one_sq_sum_scale)
    values[f"layer_norm_{kind}_epsilon"] = (1e-8, var_scale)
    isqrt_level = 8 if kind == "att" else 6
    isqrt_out_scale = Primes[isqrt_level - 1] * Primes[isqrt_level - 2]
    h_norm_scale = Primes[isqrt_level - 3] * Primes[isqrt_level - 4]
    h_mean_scale = h_scale * one_sum_scale / (Primes[level - 1] * Primes[level - 2])
    values[f"layer_norm_{kind}_H_rescale_factor"] = (1, h_norm_scale * Primes[level - 1] * Primes[level - 2] / h_scale)
    values[f"layer_norm_{kind}_H_mean_rescale_factor"] = (divisor, h_norm_scale * Primes[level - 3] * Primes[level - 4] / h_mean_scale)
    h_norm_scale = h_norm_scale * isqrt_out_scale / (Primes[isqrt_level - 1] * Primes[isqrt_level - 2])
    weight_scale = Primes[isqrt_level - 5] * Primes[isqrt_level - 6] if kind == "att" else Primes[bootstrap_out_level - 1] * Primes[bootstrap_out_level - 2] / (1 << 8)
    h_norm_scale = h_norm_scale * weight_scale / Primes[isqrt_level - 3]
    packed_weight, packed_bias = row_pack_768x1(weight) / divisor, row_pack_768x1(bias)
    for chunk in range(3):
        values[f"layernorm_{kind}_W_{chunk}"] = (packed_weight[chunk], weight_scale)
        values[f"layernorm_{kind}_B_{chunk}"] = (packed_bias[chunk], h_norm_scale)
    return values


def pool_classify(weight_pool, weight_classify, bias_pool, bias_classify):
    values = {}
    level = 13
    x_scale = Primes[level - 1] * Primes[level - 2] * Primes[level - 3]
    mask_scale = Primes[level - 4] * Primes[level - 5]
    mask_left = torch.ones(SLOTS, dtype=weight_pool.dtype, device=weight_pool.device).reshape(-1, 128)
    mask_right = mask_left.clone()
    mask_left[1::2], mask_right[0::2] = 0, 0
    mask_left, mask_right = mask_left.flatten(), mask_right.flatten()
    mask = zeros(SLOTS, weight_pool)
    mask[:256] = 1
    mask_left[256:] = 0
    mask_right[256:] = 0
    values[f"pool_classify_bsgs_Mask_{level}"] = (mask, mask_scale)
    values[f"pool_classify_bsgs_MaskL_{level}"] = (mask_left, mask_scale)
    values[f"pool_classify_bsgs_MaskR_{level}"] = (mask_right, mask_scale)
    x_scale *= mask_scale
    pool_scale = Primes[20] * Primes[19] / 1000
    packed_weight = row_pack_768x768(weight_pool)
    for chunk in range(3):
        for packed_row in range(6):
            add_plaintexts(values, bsgs_plaintexts(packed_weight[chunk][packed_row], f"pool_classify_Wp_{chunk}_{packed_row}"), pool_scale)
    values["pool_classify_Bp"] = (row_pack_768x1_type2(bias_pool), x_scale * pool_scale / (Primes[level - 1] * Primes[level - 2] * Primes[level - 3] * Primes[level - 4]))
    classifier_scale = Primes[3]
    values["pool_classify_Wc"] = (row_pack_768x2(weight_classify), classifier_scale)
    values["pool_classify_Bc"] = (pack_2x1(bias_classify), Primes[5] * Primes[4] * classifier_scale / Primes[5])
    return values


class AttentionPacker(torch.nn.Module):
    """Pack the Q/K/V and attention-output parameters of one BERT block."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self):
        parameters = {}
        for name, projections in (("q", self.attention.wq), ("k", self.attention.wk), ("v", self.attention.wv)):
            parameters[f"W{name}"] = torch.stack([projection.weight.detach().T for projection in projections])
            parameters[f"B{name}"] = torch.stack([projection.bias.detach() for projection in projections])
        parameters["W_attO"] = self.attention.output.weight.detach().T
        parameters["B_attO"] = self.attention.output.bias.detach()
        packed = linear1({
            "Wq": parameters["Wq"], "Wk": parameters["Wk"], "Wv": parameters["Wv"],
            "Bq": parameters["Bq"], "Bk": parameters["Bk"], "Bv": parameters["Bv"],
        })
        packed.update(linear2(parameters["W_attO"], parameters["B_attO"]))
        return packed


class LayerNormPacker(torch.nn.Module):
    """Pack one BERT layer-normalization module."""

    def __init__(self, layer_norm_module, kind):
        super().__init__()
        self.layer_norm_module = layer_norm_module
        self.kind = kind

    def forward(self):
        return layer_norm(
            self.layer_norm_module.w.detach(),
            self.layer_norm_module.b.detach(),
            self.kind,
        )


class FeedForwardPacker(torch.nn.Module):
    """Pack the intermediate and output linear modules of one BERT block."""

    def __init__(self, feed_forward):
        super().__init__()
        self.feed_forward = feed_forward

    def forward(self):
        packed = linear3(
            self.feed_forward.wi.weight.detach().T,
            self.feed_forward.wi.bias.detach(),
        )
        packed.update(linear4(
            self.feed_forward.wo.weight.detach().T,
            self.feed_forward.wo.bias.detach(),
        ))
        return packed


class TransformerBlockPacker(torch.nn.Module):
    """Mirror ``TransformerBlock`` while exporting packed parameters."""

    def __init__(self, block):
        super().__init__()
        self.attention = AttentionPacker(block.attention)
        self.attention_norm = LayerNormPacker(block.attention_norm, "att")
        self.ffn = FeedForwardPacker(block.ffn)
        self.ffn_norm = LayerNormPacker(block.ffn_norm, "ffn")

    def forward(self):
        packed = self.attention()
        packed.update(self.attention_norm())
        packed.update(self.ffn())
        packed.update(self.ffn_norm())
        return packed


class PoolClassifyPacker(torch.nn.Module):
    """Pack the pooler and classifier parameters of the final BERT block."""

    def __init__(self, pool_classify):
        super().__init__()
        self.pool_classify = pool_classify

    def forward(self):
        return pool_classify(
            self.pool_classify.pool_linear.weight.detach().T,
            self.pool_classify.pool_res.weight.detach().T,
            self.pool_classify.pool_linear.bias.detach(),
            self.pool_classify.pool_res.bias.detach(),
        )


class BERTPacker(torch.nn.Module):
    """A no-input forward pass that packs and serializes every model block."""

    def __init__(self, model, output_dir="params", task_name="rte"):
        super().__init__()
        self.blocks = torch.nn.ModuleList(TransformerBlockPacker(block) for block in model.layers)
        self.pool_classify = PoolClassifyPacker(model.pool_classify)
        self.output_dir = Path(output_dir)
        self.task_name = task_name

    @torch.inference_mode()
    def forward(self, block_num=None):
        if block_num is not None and not 0 <= block_num <= 12:
            raise ValueError(f"block_num must be in [0, 12], got {block_num}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        blocks = range(12) if block_num is None else (() if block_num == 12 else (block_num,))
        paths = []
        for block_id in blocks:
            packed = self.blocks[block_id]()
            path = self.output_dir / f"params_{self.task_name}_block{block_id}.pkl"
            with path.open("wb") as file:
                pickle.dump(numpy_payload(packed), file, protocol=pickle.HIGHEST_PROTOCOL)
            paths.append(path)
        if block_num is None or block_num == 12:
            packed = self.pool_classify()
            path = self.output_dir / f"params_{self.task_name}_block12.pkl"
            with path.open("wb") as file:
                pickle.dump(numpy_payload(packed), file, protocol=pickle.HIGHEST_PROTOCOL)
            paths.append(path)
        return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("packed_params"))
    parser.add_argument("--task-name", default="rte")
    parser.add_argument("--block-num", type=int, default=None)
    args = parser.parse_args()
    model = load_model(args.model_path)
    for path in BERTPacker(model, args.output_dir, args.task_name)(args.block_num):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
