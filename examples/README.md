# Cerium examples

These small programs introduce the Cerium workflow one operation at a time. Each folder has a `program.py` that describes encrypted computation with the DSL, a `runner.py` that supplies values and checks the decrypted result, and a short guide. Shared compiler and CKKS runtime setup lives in [`common.py`](common.py). The build and run scripts live here so the examples use the same steps.

| Example | What it shows |
| --- | --- |
| [Addition](addition/README.md) | Add two encrypted vectors. |
| [Multiplication](multiplication/README.md) | Multiply an encrypted vector by a plaintext vector. |
| [Rotation](rotation/README.md) | Shift packed ciphertext slots. |
| [Relinearization](relinearization/README.md) | Multiply two ciphertexts and relinearize the result. |
| [Rescale](rescale/README.md) | Reduce the scale and level after multiplication. |
| [Periodic plaintext matrix](periodic_plaintext_matrix/README.md) | Apply a repeated 2×2 plaintext matrix to encrypted pairs. |
| [Function calling](function_calling/README.md) | Call a compiled encrypted helper from the entry function. |

Complete the Docker image, Cerium build, and Python environment setup in the repository root README first. Enter the `examples/` directory and start with one GPU:

```bash
cd examples
bash compile.sh addition 1
bash run_example.sh addition 1
```

Replace `addition` with another folder name. Omit the GPU count to compile the 1-, 2-, 4-, and 8-GPU variants and run those supported by the visible GPUs. Compilation lowers the DSL into CUDA and runtime metadata, then builds shared libraries under `<example>/outputs/<example>/compiled/`. During compilation, each `program.py` prints the resulting DSL expression’s `.scale()` and `.level()`; rescale and relinearization show values before and after the operation. The scale shown there is Cerium’s scale exponent in bits, while the runner uses the numeric CKKS scale to decode the ciphertext. The run script writes each full runtime log under `<example>/logs/` and prints sample values and the maximum absolute error.

The framework path is the same in each folder: `CeriumProgram` and `CeriumFunction` build a graph; `cerium_compile` emits CUDA and a description of required inputs; `IOGenerator` encodes plaintexts and generates evaluation keys; `FusedKernelProgram` loads the compiled functions; and the runner encrypts inputs, executes on GPUs, decrypts `z`, and compares it with ordinary NumPy arithmetic. CKKS arithmetic is approximate, so small differences are expected.

## What the runner does

Each `runner.py` chooses ordinary NumPy inputs, their CKKS encoding scales, an expected result, and the scale for decoding output `z`. It passes those four pieces to `run_program` in [`common.py`](common.py). The keys in its input map must match the names declared by `CiphertextInput`, `PlaintextInput`, or `PeriodicPlaintextInput` in the paired `program.py`.

`run_program` creates a `Context` from the 32,768-slot parameters in [`primes.py`](primes.py), then creates secret-key material and an encryptor. The scripts use fixed random seeds so their inputs and output checks are repeatable. Cerium's compiled `program_inputs` file describes the evaluation keys, plaintexts, and ciphertexts required by each function. `IOGenerator` creates the keys and plaintexts first, including those for called functions.

`FusedKernelProgram.make_program` loads the compiled shared libraries and builds the function call graph. The runner then asks `IOGenerator` to encrypt ciphertext inputs, copies them to the entry function's GPU(s), and calls `program.run()`. Finally, it retrieves encrypted `z`, decrypts it with the operation's output scale, and compares every slot with the NumPy result. For `rescale`, that decoding scale includes the exact dropped prime; it is not simply a power of two.
