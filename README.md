# Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference

**Cerium is a multi-GPU framework for compiling and running terabyte-scale encrypted inference workloads.** It combines a Python-facing DSL, compiler optimizations, CUDA code generation, and a distributed runtime for CKKS homomorphic encryption.

> [!WARNING]
> Cerium is a research prototype built for benchmarking. It is not intended for production use.

Cerium is designed for researchers and systems developers exploring high-performance encrypted inference. Define a computation in the Python DSL, compile it, and run it with the CKKS runtime. The examples introduce this workflow and the benchmarks show it on larger workloads.

## What Cerium provides

- A Python DSL and bindings for describing encrypted workloads.
- A compiler with optimization passes, partitioning, and CUDA code generation.
- A multi-GPU runtime for CKKS evaluation, memory management, CUDA execution, and NCCL communication.
- Reproducible benchmark workloads for bootstrapping, ResNet-20, BERT-Base, and Llama 3.1-8B.

## How it fits together

```text
Python DSL / workload definition
            │
            ▼
 Cerium compiler and optimization passes
            │
            ▼
 CUDA code generation and multi-GPU partitioning
            │
            ▼
 CKKS runtime, CUDA execution, and NCCL communication
```

## Get started

### Prerequisites

Cerium is built in Docker. The host system needs Linux, Docker, NVIDIA Container Toolkit, and an NVIDIA driver compatible with CUDA 13.3.

The container environment uses Ubuntu 22.04, CUDA 13.3, GCC/G++ 12, CMake, Microsoft SEAL 4.1.1, NCCL 2.30.7, and Python 3.12. The default build targets GPU compute capabilities `80;86;89;90;100;120`; pass an alternative value as the first argument to `build_cerium.sh` when needed.

### Supported GPU architectures

| GPU architecture | Example GPU | CUDA compute capability |
| --- | --- | --- |
| NVIDIA Ampere | A100 | `80` |
| NVIDIA Hopper | H100 | `90` |
| NVIDIA Blackwell | B200 | `100` |

To build for other GPUs, pass a semicolon-separated CMake CUDA architecture list as the first argument to `build_cerium.sh`. For example, to build only for Ampere, Hopper, and Blackwell:

```bash
bash build_cerium.sh "80;90;100"
```

### Build

#### 1. Build the container

From the repository root, initialize submodules and build the Cerium development container:

```bash
git submodule update --init --recursive
docker build -t cerium/build-cu13.3:v1 docker/
```

#### 2. Build Cerium

Build the compiler and runtime inside the container, then create the Python environment used by the examples and benchmarks:

```bash
bash build_cerium.sh
bash activate_cerium_venv.sh
```

## Programming examples

For small DSL and runtime tutorials covering arithmetic, rotation, rescaling, periodic plaintexts, and function calls, see the [examples](examples/README.md).

## Benchmarks

Each benchmark has its own compile and run scripts. Start with bootstrapping after building Cerium:

```bash
cd benchmarks/bootstrap_32K
bash compile.sh
bash run_bootstrap_benchmarks.sh
```

The runner detects visible GPUs and evaluates supported 1-, 2-, 4-, and 8-GPU configurations. Other benchmarks are listed below.

| Workload | Location | Run |
| --- | --- | --- |
| Bootstrapping | `benchmarks/bootstrap_32K/` | `bash compile.sh && bash run_bootstrap_benchmarks.sh` |
| ResNet-20 | `benchmarks/resnet20/` | `bash compile.sh && bash run_resnet_benchmarks.sh` |
| BERT-Base | `benchmarks/bert_base/` | `bash compile.sh && bash run_bert_benchmarks.sh` |
| Llama 3.1-8B | `benchmarks/llama/` | See its [setup guide](benchmarks/llama/README.md). |

To obtain access to the Llama 3.1-8B model weights, visit [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B), then follow the Llama setup guide.

## Repository guide

| Path | Description |
| --- | --- |
| `cerium/compiler/` | Frontend, optimization passes, partitioning, and CUDA code generation. |
| `cerium/runtime/` | CKKS operations, evaluation, memory management, CUDA, and NCCL runtime support. |
| `cerium/python/` | Python DSL and bindings. |
| `examples/` | Small programs and runners that introduce the DSL and runtime. |
| `benchmarks/` | End-to-end encrypted-inference workloads. |
| `docker/` | Build image definition and SEAL installer. |

## Paper

Cerium is described in [*Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference*](https://doi.org/10.1145/3830418.3843864), presented at SOSP ’26.

> Siddharth Jayashankar, Joshua Kim, Michael B. Sullivan, Wenting Zheng, and Dimitrios Skarlatos. 2026. *Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference.* In ACM SIGOPS 32nd Symposium on Operating Systems Principles (SOSP ’26), September 29–October 02, 2026, Prague, Czech Republic. ACM, New York, NY, USA, 18 pages. [https://doi.org/10.1145/3830418.3843864](https://doi.org/10.1145/3830418.3843864)

If you use Cerium in your research, please cite:

```bibtex
@inproceedings{jayashankar2026cerium,
  author    = {Siddharth Jayashankar and Joshua Kim and Michael B. Sullivan and Wenting Zheng and Dimitrios Skarlatos},
  title     = {Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference},
  booktitle = {Proceedings of the ACM SIGOPS 32nd Symposium on Operating Systems Principles (SOSP '26)},
  year      = {2026},
  month     = sep,
  address   = {Prague, Czech Republic},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3830418.3843864},
  numpages  = {18}
}
```

## Contributing

Cerium is an active research codebase. If you extend the compiler, runtime, or workloads, please keep benchmark scripts reproducible and document hardware, software, and configuration assumptions.

## License

Cerium is licensed under Apache-2.0. See [LICENSE.txt](LICENSE.txt). External dependencies and the Llama model are subject to their respective licenses.
