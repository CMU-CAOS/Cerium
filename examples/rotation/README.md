# Rotate ciphertext slots

This folder is one step in the [Cerium examples](../README.md). Its [`program.py`](program.py) defines the encrypted computation, and its [`runner.py`](runner.py) supplies values and checks the result. Reusable setup is in [`common.py`](../common.py).

## Run

After the repository setup, from the `examples/` directory:

```bash
bash compile.sh rotation 1
bash run_example.sh rotation 1
```

Omit `1` to compile all supported GPU counts and run those visible in the container. Generated code and compiled runtime files are in `rotation/outputs/`; the full run log is in `rotation/logs/1gpus.log`.

## How it works

The DSL applies `x << 1`. This requests a one-slot left rotation and introduces a rotation evaluation key in the compiled input description.

The runner compares the decrypted result with `np.roll(x, -1)`, including wraparound from the last slot to the first.

The compile step records which ciphertexts, plaintexts, and evaluation keys the function needs. The shared runner reads that description, prepares the inputs, executes the compiled CUDA function, decrypts output `z`, and checks it against the expected values. CKKS is approximate, so a small numerical error is normal.
