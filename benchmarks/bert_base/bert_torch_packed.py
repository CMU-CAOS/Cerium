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

"""PyTorch implementation of the packed BERT inference path.

Activations are stored between transformer stages in the packed 128x768 layout.
Regular inference uses Torch dense operations; the selected reference block
uses packed BSGS dense operations and records packed intermediates.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

try:
    from .approximations import approx_gelu, approx_tanh, exp_approx, inv_sqrt, sign_approx, softmax_approx
    from .model import MODEL_PATH, load_model
    from .packing import (
        HIDDEN_SIZE, SEQUENCE_LENGTH, SLOTS, pack_128x768,
        row_pack_64x1, row_pack_768x1, row_pack_768x64,
        row_pack_768x768, unpack_128x768,
    )
except ImportError:  # Direct execution from the bert_torch directory.
    from approximations import approx_gelu, approx_tanh, exp_approx, inv_sqrt, sign_approx, softmax_approx
    from model import MODEL_PATH, load_model
    from packing import (
        HIDDEN_SIZE, SEQUENCE_LENGTH, SLOTS, pack_128x768,
        row_pack_64x1, row_pack_768x1, row_pack_768x64,
        row_pack_768x768, unpack_128x768,
    )

RUN_REFR: Optional[int] = None
REFERENCE_OUTPUTS = {}


def _capture(references, name, value):
    if references is not None:
        if value.numel() != SLOTS:
            raise ValueError(
                f"reference {name!r} must contain exactly {SLOTS} values, "
                f"got shape {tuple(value.shape)}"
            )
        references[name] = value.detach().reshape(SLOTS).clone()


def _capture_packed(references, name, values):
    for part, value in enumerate(values):
        _capture(references, f"{name}{part}", value)


def _rotate(values: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.roll(values, -steps, dims=-1)


def _bsgs_plaintexts_full(plaintext: torch.Tensor) -> torch.Tensor:
    """Build BSGS diagonals for one packed 128x128 plaintext."""
    offsets = torch.arange(128, device=plaintext.device).unsqueeze(1)
    columns = torch.arange(128, device=plaintext.device).unsqueeze(0)
    source = ((offsets + columns) % 128) * 256 + columns
    pointer = torch.cat((plaintext[source], plaintext[source + 128]), dim=1).repeat(1, 128)
    local = torch.arange(SLOTS, device=plaintext.device).remainder(128).unsqueeze(0)
    left = pointer * (local < (128 - offsets)).to(plaintext.dtype)
    right = pointer * (local >= (128 - offsets)).to(plaintext.dtype)
    return torch.cat((right, left), dim=0)


def bsgs_128x128(packed_input: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    diagonals = _bsgs_plaintexts_full(packed_weight)
    baby_steps = torch.stack([_rotate(packed_input, baby) for baby in range(16)])
    result = torch.zeros_like(packed_input)
    for giant in range(-128, 128, 16):
        rows = _rotate(diagonals[giant + 128:giant + 144], -giant)
        result = result + _rotate((baby_steps * rows).sum(dim=0), giant)
    return result


def _adjust_lanes(values: torch.Tensor) -> torch.Tensor:
    masks = torch.ones((2, SLOTS), dtype=values.dtype, device=values.device).reshape(2, -1, 128)
    masks[0, 1::2] = 0
    masks[1, 0::2] = 0
    return _rotate(values, 128) * masks[0].flatten() + _rotate(values, -128) * masks[1].flatten()


def bsgs_128x128_type3(packed_input: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    diagonals = _bsgs_plaintexts_full(packed_weight)
    lane_left = torch.ones(SLOTS, dtype=packed_input.dtype, device=packed_input.device).reshape(-1, 128)
    lane_right = lane_left.clone()
    lane_left[1::2] = 0
    lane_right[0::2] = 0
    lane_left, lane_right = lane_left.flatten(), lane_right.flatten()
    extended = {}
    for giant in range(-128, 128, 16):
        rotated = _rotate(diagonals[giant + 128:giant + 144], -giant)
        left_mask, right_mask = _rotate(lane_left, -giant), _rotate(lane_right, -giant)
        for baby in range(16):
            key = giant + baby
            extended[key + 128] = rotated[baby] * right_mask
            extended[key - 128] = rotated[baby] * left_mask
    baby_steps = torch.stack([_rotate(packed_input, baby) for baby in range(16)])
    result = torch.zeros_like(packed_input)
    for giant in range(-256, 256, 16):
        rows = torch.stack([extended[giant + baby] for baby in range(16)])
        result = result + _rotate((baby_steps * rows).sum(dim=0), giant)
    return result


def _packed_768x128(packed_input: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    first = sum((bsgs_128x128(packed_input[i], packed_weight[i]) for i in range(3)))
    second = sum((bsgs_128x128(packed_input[i], packed_weight[i + 3]) for i in range(3)))
    return first + _adjust_lanes(second)


def _packed_768x128_type3(packed_input: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    first = sum((bsgs_128x128(packed_input[i], packed_weight[i]) for i in range(3)))
    second = sum((bsgs_128x128_type3(packed_input[i], packed_weight[i + 3]) for i in range(3)))
    return first + second


def packed_linear_768x768(packed_input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    packed_weight = row_pack_768x768(weight.T)
    packed_bias = row_pack_768x1(bias) if bias is not None else torch.zeros((3, SLOTS), dtype=packed_input.dtype, device=packed_input.device)
    return torch.stack([_packed_768x128_type3(packed_input, packed_weight[i]) + packed_bias[i] for i in range(3)])


def _packed_head_pair(packed_input, first, second):
    weight = row_pack_768x64(first.weight.T, second.weight.T)
    bias = row_pack_64x1(first.bias, second.bias)
    return _packed_768x128(packed_input, weight) + bias


def packed_qk_128x64(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Packed 128x64 @ 64x128 used by ``linear1_new`` in bert_packed."""
    marker = torch.zeros_like(query)
    marker[::128] = 1 / 40
    key_baby_steps = [_rotate(key, 256 * baby) for baby in range(8)]
    result = torch.zeros_like(query)
    for giant in range(16):
        query_rotated = _rotate(query, -2048 * giant)
        baby_sum = torch.zeros_like(query)
        for baby, key_rotated in enumerate(key_baby_steps):
            partial = query_rotated * key_rotated
            for shift in (1, 2, 4, 8, 16, 32):
                partial = partial + _rotate(partial, shift)
            partial = partial * marker
            baby_sum = baby_sum + (_rotate(partial, -baby) if baby else partial)
        result = result + (_rotate(baby_sum, (2048 - 8) * giant) if giant else baby_sum)
    return result


def _packed_max(values: torch.Tensor) -> torch.Tensor:
    marker = torch.zeros_like(values)
    marker[127::128] = 40
    result = values
    for shift in (1, 2, 4, 8, 16, 32, 64):
        other = _rotate(result, -shift)
        difference = result - other
        result = ((result + other) + sign_approx(difference) * difference) * 0.5
    result = result * marker
    for shift in (1, 2, 4, 8, 16, 32, 64):
        result = result + _rotate(result, shift)
    return result


def _packed_sum(values: torch.Tensor) -> torch.Tensor:
    marker = torch.zeros_like(values)
    marker[::128] = 1
    result = values
    for shift in (1, 2, 4, 8, 16, 32, 64):
        result = result + _rotate(result, shift)
    result = result * marker
    for shift in (1, 2, 4, 8, 16, 32, 64):
        result = result + _rotate(result, -shift)
    return result


def packed_softmax_128x128(values: torch.Tensor, mask: torch.Tensor, references=None, prefix="softmax") -> torch.Tensor:
    """Packed softmax with the exact slot layout used by softmax_128x128_new."""
    valid = (mask.reshape(-1) != -1000).to(values.dtype)
    row = torch.arange(128, device=values.device).unsqueeze(1)
    column = torch.arange(128, device=values.device).unsqueeze(0)
    lane_mask = valid[(row + column).remainder(128)]
    attention_mask = torch.stack((lane_mask, lane_mask), dim=1).reshape(SLOTS)

    maximum = _packed_max(values)
    _capture(references, f"{prefix}_max_values", maximum)
    centered = values * 40 - maximum
    _capture(references, f"{prefix}_max_values_sub", centered)
    exponentials = exp_approx(centered)
    _capture(references, f"{prefix}_exp", exponentials)
    exponentials = exponentials * attention_mask
    _capture(references, f"{prefix}_exp_att_mask", exponentials)
    denominator = _packed_sum(exponentials)
    _capture(references, f"{prefix}_softmax_sum", denominator)
    reciprocal = inv_sqrt(denominator)
    return exponentials * reciprocal * reciprocal


def packed_attention_product(softmax: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Packed 128x128 @ 128x64 pair from multiply_ct128x128_ct128x128_new."""
    lane = torch.arange(SLOTS, device=value.device).remainder(128)
    value_baby_steps = [_rotate(value, 256 * baby) for baby in range(8)]
    diagonal = torch.zeros_like(value)
    for giant in range(16):
        diagonal_baby = torch.zeros_like(value)
        for baby, rotated in enumerate(value_baby_steps):
            index = baby + 8 * giant
            diagonal_baby = diagonal_baby + rotated * (lane == index).to(value.dtype)
        diagonal = diagonal + (_rotate(diagonal_baby, 2048 * giant) if giant else diagonal_baby)

    result = torch.zeros_like(value)
    for giant in range(16):
        softmax_giant = _rotate(softmax, 8 * giant - 2048 * giant)
        baby_sum = torch.zeros_like(value)
        for baby in range(8):
            index = baby + 8 * giant
            left_mask = (lane < 128 - index).to(value.dtype)
            right_mask = 1 - left_mask
            aligned = _rotate(softmax_giant, baby) * left_mask
            aligned = aligned + _rotate(softmax_giant, -128 + baby) * right_mask
            baby_sum = baby_sum + _rotate(diagonal, 256 * baby) * aligned
        result = result + (_rotate(baby_sum, 2048 * giant) if giant else baby_sum)
    return result


class PackedAttention(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.wq = attention.wq
        self.wk = attention.wk
        self.wv = attention.wv
        self.output = attention.output

    def forward(self, packed_hidden: torch.Tensor, mask: torch.Tensor, run_refr=False, references=None) -> torch.Tensor:
        if run_refr:
            attention_pairs = []
            order = (0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11)
            for pair in range(6):
                first, second = order[2 * pair], order[2 * pair + 1]
                q_packed = _packed_head_pair(packed_hidden, self.wq[first], self.wq[second])
                k_packed = _packed_head_pair(packed_hidden, self.wk[first], self.wk[second])
                v_packed = _packed_head_pair(packed_hidden, self.wv[first], self.wv[second])
                _capture(references, f"attention_Q{pair}", q_packed)
                _capture(references, f"attention_K{pair}", k_packed)
                _capture(references, f"attention_V{pair}", v_packed)
                scores = packed_qk_128x64(q_packed, k_packed)
                _capture(references, f"attention_scores{pair}", scores)
                softmax = packed_softmax_128x128(scores, mask, references, f"linear1_{pair}")
                _capture(references, f"attention_probs{pair}", softmax)
                attention = packed_attention_product(softmax, v_packed)
                _capture(references, f"attention_context_layer{pair}", attention)
                attention_pairs.append(attention)
            packed_attention = torch.stack([
                attention_pairs[2 * part] + _rotate(attention_pairs[2 * part + 1], -64)
                for part in range(3)
            ])
            for part in range(3):
                _capture(references, f"attention_context_layer_stacked{part}", packed_attention[part])
            projection = packed_linear_768x768(packed_attention, self.output.weight, self.output.bias)
            for part in range(3):
                _capture(references, f"attention_output{part}", projection[part])
            output = projection + packed_hidden
            for part in range(3):
                _capture(references, f"attention_residual_output{part}", output[part])
            return output

        hidden = unpack_128x768(packed_hidden)
        heads = []
        for query, key, value in zip(self.wq, self.wk, self.wv):
            q, k, v = query(hidden), key(hidden), value(hidden)
            heads.append(softmax_approx(q @ k.transpose(-2, -1), mask) @ v)
        return pack_128x768(self.output(torch.cat(heads, dim=-1)) + hidden)


class PackedLayerNorm(nn.Module):
    def __init__(self, layer_norm: nn.Module):
        super().__init__()
        self.w = layer_norm.w
        self.b = layer_norm.b

    def forward(self, packed_hidden: torch.Tensor, run_refr=False, references=None, prefix=None) -> torch.Tensor:
        if run_refr:
            squares = packed_hidden * packed_hidden
            hidden_sum = packed_hidden.sum(dim=0)
            square_sum = squares.sum(dim=0)
            for shift in (1, 2, 4, 8, 16, 32, 64, 128):
                hidden_sum = hidden_sum + _rotate(hidden_sum, shift)
                square_sum = square_sum + _rotate(square_sum, shift)
            _capture(references, f"{prefix}_H_sq_sum", square_sum)

            marker = torch.zeros_like(hidden_sum)
            marker[::256] = 1 / 768 / 10
            mean = hidden_sum * marker
            marker[::256] = 1 / 768 / 100
            variance = square_sum * marker
            for shift in (1, 2, 4, 8, 16, 32, 64, 128):
                mean = mean + _rotate(mean, -shift)
                variance = variance + _rotate(variance, -shift)
            variance = variance - mean * mean + 1e-8
            _capture(references, f"{prefix}_variance", variance)
            mean = mean * 10
            _capture(references, f"{prefix}_mean", mean)
            inverse_std = inv_sqrt(variance)
            _capture(references, f"{prefix}_isqrt", inverse_std)
            weight = row_pack_768x1(self.w / 10)
            bias = row_pack_768x1(self.b)
            output = (packed_hidden - mean) * inverse_std * weight + bias
            for part in range(3):
                _capture(references, f"{prefix}_output{part}", output[part])
            return output

        hidden = unpack_128x768(packed_hidden)
        mean = hidden.mean(dim=1, keepdim=True)
        variance = hidden.var(dim=1, keepdim=True)
        normalized = (hidden - mean) * inv_sqrt(variance / 100 + 1e-8) / 10
        output = pack_128x768(normalized * self.w + self.b)
        return output


class PackedFeedForward(nn.Module):
    def __init__(self, feed_forward: nn.Module):
        super().__init__()
        self.wi = feed_forward.wi
        self.wo = feed_forward.wo

    def forward(self, packed_hidden: torch.Tensor, run_refr=False, references=None) -> torch.Tensor:
        if run_refr:
            intermediate = []
            for part in range(4):
                start = 768 * part
                value = packed_linear_768x768(
                    packed_hidden,
                    self.wi.weight[start:start + 768],
                    self.wi.bias[start:start + 768],
                )
                for packed_part in range(3):
                    _capture(references, f"ffn_up{3 * part + packed_part}", value[packed_part])
                intermediate.append(approx_gelu(value))
                for packed_part in range(3):
                    _capture(references, f"ffn_gelu{3 * part + packed_part}", intermediate[-1][packed_part])
            output = torch.zeros_like(packed_hidden)
            for part, value in enumerate(intermediate):
                start = 768 * part
                output = output + packed_linear_768x768(
                    value,
                    self.wo.weight[:, start:start + 768],
                    None,
                )
            output = output + row_pack_768x1(self.wo.bias)
            for part in range(3):
                _capture(references, f"ffn_down{part}", output[part])
            output = output + packed_hidden
            _capture_packed(references, "ffn_residue", output)
            return output

        hidden = unpack_128x768(packed_hidden)
        return pack_128x768(self.wo(approx_gelu(self.wi(hidden))) + hidden)


class PackedTransformerBlock(nn.Module):
    def __init__(self, block: nn.Module, block_id: int):
        super().__init__()
        self.block_id = block_id
        self.attention = PackedAttention(block.attention)
        self.attention_norm = PackedLayerNorm(block.attention_norm)
        self.ffn = PackedFeedForward(block.ffn)
        self.ffn_norm = PackedLayerNorm(block.ffn_norm)

    def forward(self, packed_hidden: torch.Tensor, mask: torch.Tensor, run_refr=False, references=None) -> torch.Tensor:
        _capture_packed(references, "block_input", packed_hidden)
        hidden = self.attention(packed_hidden, mask, run_refr, references)
        hidden = self.attention_norm(hidden, run_refr, references, "layernorm_att")
        hidden = self.ffn(hidden, run_refr, references)
        hidden = self.ffn_norm(hidden, run_refr, references, "layernorm_ffn")
        _capture_packed(references, "block_output", hidden)
        return hidden


class PackedPoolClassify(nn.Module):
    def __init__(self, pool_classify: nn.Module):
        super().__init__()
        self.pool_linear = pool_classify.pool_linear
        self.pool_res = pool_classify.pool_res

    def forward(self, packed_hidden: torch.Tensor) -> torch.Tensor:
        first_token = unpack_128x768(packed_hidden)[0]
        return self.pool_res(approx_tanh(self.pool_linear(first_token))).float()


class PackedTransformer(nn.Module):
    """Packed-layout inference wrapper around a loaded ``Transformer`` model."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(PackedTransformerBlock(block, block_id) for block_id, block in enumerate(model.layers))
        self.pool_classify = PackedPoolClassify(model.pool_classify)
        self.reference_outputs = {}

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, reference_block=None) -> torch.Tensor:
        target_block = RUN_REFR if reference_block is None else reference_block
        if target_block is False:
            target_block = None
        elif target_block is True:
            raise ValueError("RUN_REFR must be False, None, or a block ID from 0 to 11")
        self.reference_outputs = {}
        hidden = pack_128x768(tokens)
        for block_id, layer in enumerate(self.layers):
            run_refr = target_block == block_id
            references = self.reference_outputs if run_refr else None
            hidden = layer(hidden, mask, run_refr, references)
        return self.pool_classify(hidden)

    def save_references(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for name, value in self.reference_outputs.items():
            if value.shape != (SLOTS,):
                raise ValueError(f"reference {name!r} has invalid shape {tuple(value.shape)}")
            payload[name] = value.cpu().numpy()
        print(f"Reference output keys ({len(payload)}):")
        for name in payload:
            print(name)
        with path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_packed_model(path: Path = MODEL_PATH, device: str = "cpu") -> PackedTransformer:
    model = load_model(path, device)
    packed_model = PackedTransformer(model)
    packed_model.eval()
    return packed_model


def load_sample(data_dir: Path, sample: int):
    tokens = torch.from_numpy(np.load(data_dir / "inputs" / f"inputs_{sample}_data.npy")).float()
    mask = torch.from_numpy((1 - np.load(data_dir / "inputs" / f"inputs_{sample}_mask.npy")) * -1000).float()
    return tokens, mask


def main():
    global RUN_REFR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[0] )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--run-refr", "--run-refr-block", dest="run_refr", type=int, choices=range(12), help="Block ID that uses packed BSGS and captures references.")
    parser.add_argument("--reference-output", type=Path, help="Optional reference pickle output path.")
    args = parser.parse_args()
    RUN_REFR = args.run_refr
    model = load_packed_model(args.model_path)
    tokens, mask = load_sample(args.data_dir, args.sample)
    print(model(tokens, mask))
    if RUN_REFR is not None:
        output = args.reference_output or Path(f"reference_outputs_{args.sample}") / f"block{RUN_REFR}.pkl"
        model.save_references(output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
