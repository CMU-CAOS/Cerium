# Call a compiled Cerium function

This folder is one step in the [Cerium examples](../README.md). Its [`program.py`](program.py) defines the encrypted computation, and its [`runner.py`](runner.py) supplies values and checks the result. Reusable setup is in [`common.py`](../common.py).

## Run

After the repository setup, from the `examples/` directory:

```bash
bash compile.sh function_calling 1
bash run_example.sh function_calling 1
```

Omit `1` to compile all supported GPU counts and run those visible in the container. Generated code and compiled runtime files are in `function_calling/outputs/`; the full run log is in `function_calling/logs/1gpus.log`.

## How it works

`build_double_function()` defines a helper with `CiphertextArgument("value", ...)` and `FunctionOutput("doubled", value + value)`, then stores it in the module-level `double_function` variable. `build_main_function()` reads that variable, passes its encrypted input through `CeriumFunctionCall`, and names the returned value `z`. `build_program()` resets the global reference and registers the helper before its caller in the same `CeriumProgram`.

Both functions are compiled and linked. The runner invokes only `main_<N>gpus`; the runtime follows the compiled call graph and checks that `z = 2x`.

The compile step records which ciphertexts, plaintexts, and evaluation keys the function needs. The shared runner reads that description, prepares the inputs, executes the compiled CUDA function, decrypts output `z`, and checks it against the expected values. CKKS is approximate, so a small numerical error is normal.
