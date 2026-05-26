# Llama 3.1-8B model preparation

This submodule prepares the model-side inputs for Cerium's encrypted Llama 3-8B benchmark.

1. Download the licensed native Llama checkpoint.
2. Convert each transformer block's weights into packed parameter files.
3. Convert example prompts into packed prompt and attention-mask inputs.

## Prerequisites

Run the commands below from this `llama_models/` directory after building and activating the parent artifact's Cerium environment. The dependencies required by these workflows—PyTorch, FairScale, Fire, Hugging Face Hub, and their Python dependencies—are already pinned and installed from the top-level`requirements.txt`; no additional submodule-level `pip install` is needed.

You need a Hugging Face account that has accepted the license for
[`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B). Authenticate with `huggingface-cli login` or set `HF_TOKEN` before downloading.

## 1. Download and verify the model weights

```bash
python download_llama.py
```

The script downloads the native checkpoint files, verifies their MD5 digests,
and stores them in `model_weights/Llama3.1-8B/`:

- `consolidated.00.pth`
- `params.json`
- `tokenizer.model`

Model weights are not included in this submodule and remain subject to Meta's license.

## 2. Create packed model parameters

After the checkpoint has been downloaded, create parameters for all 32
transformer blocks:

```bash
MAX_JOBS=1 bash create_params.sh
```

The script writes `params/params_block<N>.pkl` for blocks `0` through `31` and skips already nonempty files. It runs up to `MAX_JOBS` block conversions in parallel (default: `8`); start with `MAX_JOBS=1` on a memory-constrained host. Set `LLAMA_PARAMS_DIR=/path/to/params` to use a different output location.

## 3. Create packed prompts and attention masks

Generate the five supplied example prompt/mask pairs:

```bash
for sample in 0 1 2 3 4; do
  bash run_packed_inputs.sh --data_sample "$sample"
done
```

The script reads the downloaded checkpoint and writes
`llama_inputs/sample<N>_prompt.pkl` and
`llama_inputs/sample<N>_attention_mask.pkl`. Its packed-input path supports prompts of at most 128 tokens.
