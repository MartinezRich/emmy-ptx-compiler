# Emmy Kernel IR to PTX Compiler

A direct compiler from [Emmy](https://github.com/cloudrift-ai/emmy) Kernel IR to NVIDIA PTX, built for the Emmy Hackathon 2026.

The compiler reads Emmy's typed, scheduled Kernel IR and emits PTX directly to standard output. It does not use Emmy's CUDA C++ renderer or invoke `nvcc` for the core IR-to-PTX lowering.

## Results

The current implementation passes all **15/15 public problems** on the official hosted GPU runner.

| Category | Public problems | Result |
|---|---|---|
| Pointwise | p00–p03 | 4/4 PASS |
| Reduction | r01–r04 | 4/4 PASS |
| Matrix multiplication | m01–m02 | 2/2 PASS |
| Fused kernels | f01–f03 | 3/3 PASS |
| Attention-like | a01–a02 | 2/2 PASS |

Passing the public set does not guarantee every hidden IR variation; unsupported structures fail explicitly instead of silently emitting PTX with unverified semantics.

## Compiler pipeline

```text
Kernel IR JSON + meta.json
            |
            v
   Typed Emmy Graph / KernelOp / Tile
            |
            v
  Structure-based Tile classification
            |
            +--> Pointwise lowering
            +--> Cooperative reduction lowering
            +--> Structured loop lowering
            +--> Scheduled matmul lowering
            |
            v
       NVIDIA PTX on stdout
            |
            v
          ptxas
```

The IR describes the kernel computation and schedule. The adjacent `meta.json` supplies the launch contract: argument order, grid and block dimensions, dynamic shared-memory size, and zero-initialized outputs.

The compiler classifies a Tile from its statements rather than from a public problem ID or fixed kernel name.

## Implementation overview

### Shared PTX builder

The base builder manages:

- PTX virtual registers, labels, and predicates;
- mappings from IR SSA names and Tile axes to PTX operands;
- typed scalar operations and explicit conversions;
- multidimensional row-major address calculation;
- launch-index linearization for one-, two-, and three-dimensional grids and blocks.

The pointwise path uses grid-stride iteration and supports fundamental operations for `f16`, `f32`, `f64`, `i32`, and `i64`.

### Reduction lowering

Cooperative reductions preserve the structure present in the IR, including:

- strided loops and accumulator identities;
- warp-level shuffle combines;
- shared-memory loads and stores;
- CTA barriers and tree-style merging;
- supported sum, product, maximum, and minimum combines;
- guarded `f32` atomic-add output when the launch contract proves the destination is zero-initialized.

### Scheduled matrix multiplication

The matmul backend translates the scheduled memory pipeline rather than replacing it with a fixed triple loop. It lowers shared-memory declarations, `cp.async` copies, commit/wait operations, synchronization, inner loops, and accumulation in IR order.

Shared-memory addressing supports the IR's `NONE`, `B32`, `B64`, and `B128` XOR swizzle modes so producers and consumers use the same physical layout.

### Fused and attention-like kernels

The structured-loop backend translates nested `Loop`/`StridedLoop`, `Load`, `Let`, `Assign`, `Accum`, `Cond`, and `Write` statements in order. Accumulator initialization follows its loop scope instead of reconstructing a fixed RMSNorm, softmax, or attention formula.

## Requirements

- Python 3.12 or newer
- `emmy-ml==0.3.11`
- `numpy` and `requests` for the official test runner

Install the dependencies:

```bash
pip install emmy-ml==0.3.11 numpy requests
```

No local CUDA installation or GPU is required to generate PTX or use the hosted runner.

## Usage

The compiler follows the required command-line contract:

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

`stdout` contains PTX only. Unsupported IR is reported through a non-zero compiler exit rather than being mixed into the generated PTX.

## Validation

Run a public problem through the compiler and official hosted runner:

```bash
python submit.py p03_relu_mul --compiler compile.py
python submit.py r04_row_sum --compiler compile.py
python submit.py m02_matmul_rect --compiler compile.py
python submit.py f02_row_softmax --compiler compile.py
python submit.py a02_attention_128 --compiler compile.py
```

Each command compiles the problem's Kernel IR, sends the generated PTX to the hosted `ptxas` service, executes the kernel, and compares the output with `expected.npz` using the problem tolerance.

## Known boundaries

- The implementation targets the non-starred Kernel IR scope used by the competition; advanced TMA, tensor-core fragment, and WGMMA nodes are not implemented.
- Cross-thread `f64`/integer reductions and non-`f32` matmul or attention schedules are not treated as if they were equivalent to the validated paths.
- A packaged input is expected to contain the single `KernelOp` and materialized `Tile` produced by Emmy's competition generation pipeline.
- Unknown or unsafe combinations raise an explicit error.

## Repository contents

```text
compile.py   IR-to-PTX compiler submitted for evaluation
submit.py    official public-problem runner
problems/    15 public Kernel IR problems and expected outputs
examples/    hand-written PTX examples from the starter repository
TASK.md      official task and Kernel IR reference
MVP.md       official minimum acceptance target
docs/        hosted-runner API documentation
```

## Attribution

This repository is based on the official [cloudrift-ai/emmy-hackathon-2026](https://github.com/cloudrift-ai/emmy-hackathon-2026) starter repository. The submitted compiler implementation is in `compile.py`.
