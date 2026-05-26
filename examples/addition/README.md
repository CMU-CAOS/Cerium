# Add two encrypted vectors

This folder is one step in the [Cerium examples](../README.md). Its [`program.py`](program.py) defines the encrypted computation, and its [`runner.py`](runner.py) supplies values and checks the result. Reusable setup is in [`common.py`](../common.py).

## Run

After the repository setup, from the `examples/` directory:

```bash
bash compile.sh addition 1
bash run_example.sh addition 1
```

Omit `1` to compile all supported GPU counts and run those visible in the container. Generated code and compiled runtime files are in `addition/outputs/`; the full run log is in `addition/logs/1gpus.log`.

## How it works

The DSL declares ciphertexts `x` and `y` at scale 2^56 and outputs `z = x + y`. Addition preserves the scale and does not need a multiplication key.

The runner encrypts two full vectors, decrypts `z`, and compares it with `x + y`. This is the smallest example of matching DSL input names to runtime values.

The compile step records which ciphertexts, plaintexts, and evaluation keys the function needs. The shared runner reads that description, prepares the inputs, executes the compiled CUDA function, decrypts output `z`, and checks it against the expected values. CKKS is approximate, so a small numerical error is normal.
