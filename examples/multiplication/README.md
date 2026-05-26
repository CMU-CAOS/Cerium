# Multiply ciphertext by plaintext

This folder is one step in the [Cerium examples](../README.md). Its [`program.py`](program.py) defines the encrypted computation, and its [`runner.py`](runner.py) supplies values and checks the result. Reusable setup is in [`common.py`](../common.py).

## Run

After the repository setup, from the `examples/` directory:

```bash
bash compile.sh multiplication 1
bash run_example.sh multiplication 1
```

Omit `1` to compile all supported GPU counts and run those visible in the container. Generated code and compiled runtime files are in `multiplication/outputs/`; the full run log is in `multiplication/logs/1gpus.log`.

## How it works

The DSL declares encrypted `x` and plaintext `factor`, then outputs their slotwise product. Each input uses scale 2^28, so the product is decoded at scale 2^56.

The runner encrypts `x` and has `IOGenerator` encode `factor` as a plaintext. It checks each slot against `x * factor`. This example shows how a plaintext parameter enters the runtime input map.

The compile step records which ciphertexts, plaintexts, and evaluation keys the function needs. The shared runner reads that description, prepares the inputs, executes the compiled CUDA function, decrypts output `z`, and checks it against the expected values. CKKS is approximate, so a small numerical error is normal.
