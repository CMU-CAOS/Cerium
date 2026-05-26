# Multiply and relinearize ciphertexts

This folder is one step in the [Cerium examples](../README.md). Its [`program.py`](program.py) defines the encrypted computation, and its [`runner.py`](runner.py) supplies values and checks the result. Reusable setup is in [`common.py`](../common.py).

## Run

After the repository setup, from the `examples/` directory:

```bash
bash compile.sh relinearization 1
bash run_example.sh relinearization 1
```

Omit `1` to compile all supported GPU counts and run those visible in the container. Generated code and compiled runtime files are in `relinearization/outputs/`; the full run log is in `relinearization/logs/1gpus.log`.

## How it works

The DSL multiplies encrypted `x` and `y`, then calls `.relinearize()` on the product. Relinearization uses an evaluation key to reduce the enlarged ciphertext representation after multiplication.

Both operands are encrypted at scale 2^56. The runner decodes the product at scale 2^112 and checks it against `x * y`.

The compile step records which ciphertexts, plaintexts, and evaluation keys the function needs. The shared runner reads that description, prepares the inputs, executes the compiled CUDA function, decrypts output `z`, and checks it against the expected values. CKKS is approximate, so a small numerical error is normal.
