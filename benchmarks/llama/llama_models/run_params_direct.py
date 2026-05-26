import argparse
import json
import os
from pathlib import Path

import torch
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from models.checkpoint import maybe_reshard_state_dict
from models.llama3.args import ModelArgs
from models.llama3.model_params import TransformerParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=1)
    args = parser.parse_args()

    device = os.environ.get("DEVICE", "cpu")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(args.world_size)

    ckpt_dir = Path(args.ckpt_dir)
    with (ckpt_dir / "params.json").open() as params_file:
        model_args = ModelArgs(
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            **json.load(params_file),
        )
    state_dict = maybe_reshard_state_dict(
        sorted(ckpt_dir.glob("*.pth")), n_kv_heads=model_args.n_kv_heads
    )
    torch.set_default_device(device)
    if device == "cpu":
        torch.set_default_dtype(torch.float64)
    model = TransformerParams(model_args)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    model()


if __name__ == "__main__":
    main()
