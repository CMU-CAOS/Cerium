# Apply a periodic plaintext matrix

This folder is one step in the [Cerium examples](../README.md). Its [`program.py`](program.py) defines the encrypted computation, and its [`runner.py`](runner.py) supplies values and checks the result. Reusable setup is in [`common.py`](../common.py).

## Run

After the repository setup, from the `examples/` directory:

```bash
bash compile.sh periodic_plaintext_matrix 1
bash run_example.sh periodic_plaintext_matrix 1
```

Omit `1` to compile all supported GPU counts and run those visible in the container. Generated code and compiled runtime files are in `periodic_plaintext_matrix/outputs/`; the full run log is in `periodic_plaintext_matrix/logs/1gpus.log`.

## How it works

Adjacent ciphertext slots are treated as pairs `[a, b]`. The plaintext matrix `[[2, 1], [1, 3]]` gives `[2a+b, a+3b]` for every pair. Three `PeriodicPlaintextInput` values store the repeating diagonal and neighbor masks with period 2. Left and right one-slot rotations supply the two neighbors.

The runner provides two-element plaintext arrays `[2, 3]`, `[1, 0]`, and `[0, 1]`, encrypts the full vector, and checks every output pair against ordinary matrix arithmetic. This shows Cerium’s repeated-plaintext encoding.

The compile step records which ciphertexts, plaintexts, and evaluation keys the function needs. The shared runner reads that description, prepares the inputs, executes the compiled CUDA function, decrypts output `z`, and checks it against the expected values. CKKS is approximate, so a small numerical error is normal.
