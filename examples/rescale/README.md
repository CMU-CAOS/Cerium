# Rescale after multiplication

This folder follows the same [`program.py`](program.py), [`runner.py`](runner.py), and shared [`common.py`](../common.py) layout as the other [Cerium examples](../README.md).

## Run

After the repository setup, from the `examples/` directory:

```bash
bash compile.sh rescale 1
bash run_example.sh rescale 1
```

Omit `1` to compile all supported GPU counts and run those visible in the container. Generated code and runtime files go under `rescale/outputs/`; the one-GPU run log is `rescale/logs/1gpus.log`.

## How it works

The DSL declares encrypted `x` at scale `2^56` and plaintext `factor` at scale `2^28`, both at level 12. Their slotwise product has scale `2^84`. Calling `.rescale()` drops one RNS prime and reduces the level to 11, bringing the scale exponent back near 56. This prepares a product for further arithmetic without continually growing its scale.

The runner encrypts `x`, encodes `factor`, and compares the decrypted `z` with ordinary `x * factor`. Decoding uses the actual post-rescale scale, `(2^56 × 2^28) / Primes[11]`, because the dropped prime is close to, but not exactly, `2^28`. Small CKKS numerical error is expected.
