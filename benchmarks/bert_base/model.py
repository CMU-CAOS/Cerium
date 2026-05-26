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

import pickle
from pathlib import Path
from typing import Optional

import torch
from torch import nn

import numpy as np
try:
    from .approximations import approx_gelu, approx_tanh, inv_sqrt, softmax_approx
except ImportError:  # Direct execution from the bert_torch directory.
    from approximations import approx_gelu, approx_tanh, inv_sqrt, softmax_approx



class Attention(nn.Module):
    def __init__(self, block_id):
        super().__init__()

        self.n_heads = 12
        self.block_id = block_id

        self.wq = nn.ModuleList(torch.nn.Linear(in_features=768, out_features=64, bias=True) for _ in range(self.n_heads))
        self.wk = nn.ModuleList(torch.nn.Linear(in_features=768, out_features=64, bias=True) for _ in range(self.n_heads))
        self.wv = nn.ModuleList(torch.nn.Linear(in_features=768, out_features=64, bias=True) for _ in range(self.n_heads))
        self.output = torch.nn.Linear(in_features = 768, out_features = 768, bias=True)


    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        attention = [None] * self.n_heads
        for i in range(self.n_heads):
            xq = self.wq[i](x)
            xk = self.wk[i](x)
            xv = self.wv[i](x)
            qk = torch.matmul(xq, xk.transpose(-2, -1))
            qk = softmax_approx(qk,mask).type_as(xv)
            qkv = torch.matmul(qk, xv)
            attention[i] = qkv
        attention = torch.cat(attention, dim=-1)
        output = self.output(attention) + x
        return output


class LayerNorm(nn.Module):
    def __init__(self, block_id, type):
        super().__init__()
        self.n_heads = 12
        self.block_id = block_id

        if type == "att":
            self.w = torch.nn.Parameter(torch.ones(768))
            self.b = torch.nn.Parameter(torch.zeros(768))
        elif type == "out":
            self.w = torch.nn.Parameter(torch.ones(768))
            self.b = torch.nn.Parameter(torch.zeros(768))
        else:
            raise Exception("Unknown LayerNorm type")

    def forward(
        self,
        x: torch.Tensor,
    ):
        mean = torch.mean(x, axis=1, keepdims=True)
        variance = torch.var(x, axis=1, keepdims=True)

        # Calculate the layer normalization
        epsilon = 1e-8  # small value to avoid division by zero
        inv_sqrt_res = inv_sqrt(variance / 100 + epsilon)
        normalized_matrix = (x - mean) * inv_sqrt_res / 10

        # normalized_matrix = self.w(normalized_matrix)
        normalized_matrix = normalized_matrix * self.w + self.b
        return normalized_matrix

class FeedForward(nn.Module):
    def __init__(self, block_id):
        super().__init__()

        self.block_id = block_id

        self.wi = torch.nn.Linear(in_features = 768, out_features = 3072, bias=True)
        self.wo = torch.nn.Linear(in_features = 3072, out_features = 768, bias=True)


    def forward(
        self,
        x: torch.Tensor,
    ):
        l3 = self.wi(x)
        l3 = approx_gelu(l3)
        l4 = self.wo(l3) + x
        return l4 


class TransformerBlock(nn.Module):
    def __init__(self, block_id: int):
        super().__init__()
        self.attention = Attention(block_id)
        self.attention_norm = LayerNorm(block_id,"att")

        self.ffn = FeedForward(block_id)
        self.ffn_norm = LayerNorm(block_id,"out")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h =  self.attention(x,mask)
        h = self.attention_norm(h)
        h = self.ffn(h)
        h = self.ffn_norm(h)
        return h

class PoolClassify(nn.Module):
    def __init__(self):
        super().__init__()


        self.pool_linear = torch.nn.Linear(in_features = 768, out_features = 768, bias=True)
        self.pool_res = torch.nn.Linear(in_features = 768, out_features = 2, bias=True)


    def forward(
        self,
        x: torch.Tensor,
    ):
        pl = self.pool_linear(x) 
        pr = approx_tanh(pl)
        result = self.pool_res(pr)
        return result


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_blocks = 12

        self.blocks = torch.nn.ModuleList()
        for block_id in range(self.n_blocks):
            self.blocks.append(TransformerBlock(block_id))

        self.pool_classify = PoolClassify()

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, mask):
        h = tokens

        for block in self.layers:
            h = block(h,mask)
        output = self.pool_classify(h[0]).float()
        return output



data_dir = str(Path(__file__).resolve().parents[0] ) + "/"

def load_sample(data_dir,data_sample):
    H1 = np.load(data_dir + 'inputs/inputs_' + str(data_sample) + '_data.npy')
    H1 = torch.tensor(H1, dtype=torch.float32)
    input_mask = (1 - np.load(data_dir + 'inputs/inputs_' + str(data_sample) + '_mask.npy')) * -1000
    input_mask = torch.tensor(input_mask, dtype=torch.float32)
    return H1, input_mask



MODEL_PATH = Path(__file__).with_name("bert_torch_rte.pt")

_LEGACY_MODEL_CLASSES = {
    "Attention": Attention,
    "LayerNorm": LayerNorm,
    "FeedForward": FeedForward,
    "TransformerBlock": TransformerBlock,
    "PoolClassify": PoolClassify,
    "Transformer": Transformer,
}


class _LegacyModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in _LEGACY_MODEL_CLASSES:
            return _LEGACY_MODEL_CLASSES[name]
        return super().find_class(module, name)


class _LegacyModelPickle:
    Unpickler = _LegacyModelUnpickler
    load = staticmethod(pickle.load)
    dump = staticmethod(pickle.dump)
    loads = staticmethod(pickle.loads)
    dumps = staticmethod(pickle.dumps)


def load_model(path: Path = MODEL_PATH, device: str = "cpu") -> nn.Module:
    """Load a model previously saved with :func:`save_model`."""
    model = torch.load(path, map_location=device, weights_only=False, pickle_module=_LegacyModelPickle)
    for block in model.layers:
        attention = block.attention
        for name in ("wq", "wk", "wv"):
            modules = getattr(attention, name)
            if not isinstance(modules, nn.ModuleList):
                setattr(attention, name, nn.ModuleList(modules))
    model.eval()
    return model


if __name__ == '__main__':
    import argparse
    import tqdm
    from transformers import glue_compute_metrics as compute_metrics

    model = load_model()

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='rte')
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--sample_start', type=int, default=0)
    parser.add_argument('--sample_end', type=int, default=1)
    args = parser.parse_args()



    Preds = []
    label = []
    sample_num = args.sample_num
    pbar = tqdm.tqdm(range(sample_num))
    labels = np.load(data_dir + 'inputs/labels.npy')
    results = {}
    total = 0
    correct = 0
    Confs = []

    for data_sample in range(0,sample_num):

        input, mask = load_sample(data_dir, data_sample)


        scores = np.array(model.forward(input,mask).cpu())
        diff = np.abs(scores[0] - scores[1])
        pred = np.argmax(scores)

        pbar.update(1)
        total += 1 

        Preds.append(pred)
        Confs.append(scores)
        label.append(labels[data_sample])

        huggingface_eval = compute_metrics(args.task_name, np.array(Preds), np.array(label))
        results.update(huggingface_eval)
        pbar.set_postfix(results)

    huggingface_eval = compute_metrics(args.task_name, np.array(Preds), np.array(label))
    results.update(huggingface_eval)
