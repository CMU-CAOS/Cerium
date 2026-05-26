import argparse
import os
import pickle
from pathlib import Path

import torch

from models.llama3.generation import Llama3
from models.llama3.model_packed import pack_attention_mask, row_pack_nx4096

from prompts import get_prompt



def get_device() -> str:
    return os.environ.get("DEVICE", "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--data_sample", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output_dir", default="llama_inputs/")
    args = parser.parse_args()

    device = get_device()
    generator = Llama3.build(
        ckpt_dir=args.ckpt_dir,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        world_size=args.world_size,
        device=device,
        packed_model=True,
    )

    prompt = args.prompt if args.prompt is not None else get_prompt(args.data_sample)
    token_ids = generator.formatter.encode_content(prompt).tokens
    print(f"Prompt: {prompt}")
    print(f"Prompt token count: {len(token_ids)}")
    if len(token_ids) > 127:
        raise ValueError("packed inputs support prompts of at most 128 tokens")
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        embeddings = generator.model.tok_embeddings(tokens)
        packed_embeddings = row_pack_nx4096(embeddings)
        attention_mask = torch.tril(
            torch.ones((len(token_ids), len(token_ids)), device=device, dtype=torch.int32)
        )
        packed_mask = pack_attention_mask(attention_mask.cpu())

    output_dir = Path(args.output_dir)
    data_sample = args.data_sample
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"sample{data_sample}_prompt.pkl").open("wb") as output_file:
        pickle.dump(
            {
                f"rmsnorm_att_input{i}": packed_embeddings[i].cpu().tolist()
                for i in range(16)
            },
            output_file,
        )
    with (output_dir / f"sample{data_sample}_attention_mask.pkl").open("wb") as output_file:
        pickle.dump({"att_mask": packed_mask.cpu().tolist()}, output_file)


if __name__ == "__main__":
    main()
