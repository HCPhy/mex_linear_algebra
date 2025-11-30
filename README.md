# MEX Linear Algebra

This repository contains optimized MATLAB MEX functions for linear algebra operations over the Galois Field GF(2). These functions are designed for high performance, utilizing AVX2/AVX512 instructions and OpenMP multi-threading where available.

All functions support both **logical** (boolean) and **double** (integer) input matrices. For double inputs, values are treated modulo 2 (parity check).

## Features

- **`gf2_matmul_mex`**: Fast matrix multiplication over GF(2).
- **`null_gf2_mex`**: Efficient computation of the null space of a binary matrix.
- **`bitrank_mex`**: High-speed rank calculation for binary matrices.

## Prerequisites

- **MATLAB**: Required to run the scripts and use the MEX functions.
- **C Compiler**: A C compiler compatible with MATLAB (e.g., GCC, Clang, MSVC).
- **OpenMP** (Optional but recommended): For multi-threaded performance on Linux/macOS.
    - *macOS*: Install `libomp` via Homebrew: `brew install libomp`

## Installation & Compilation

To compile the MEX functions, run the `compile_mex.m` script in MATLAB. This script automatically detects your system architecture (Linux, macOS, Windows) and applies the appropriate optimization flags.

```matlab
compile_mex
```

The script will:
1. Compile source files from `src/` to `bin/`.
2. Add `bin/` to the MATLAB path.
3. Run a suite of verification tests to ensure correctness.

## Usage

### Matrix Multiplication (`gf2_matmul_mex`)

Computes $C = A \times B$ over GF(2). Accepts `logical` or `double` matrices.

```matlab
A = randi([0, 1], 100, 100); % double matrix
B = randi([0, 1], 100, 50);
C = gf2_matmul_mex(A, B);
```

### Null Space (`null_gf2_mex`)

Computes a basis for the null space of matrix $A$, such that $A \times Z = 0$ over GF(2).

```matlab
A = randi([0, 1], 50, 100);
Z = null_gf2_mex(A);
% Verify: gf2_matmul_mex(A, Z) should be all zeros.
```

### Rank (`bitrank_mex`)

Computes the rank of a binary matrix over GF(2).

```matlab
A = randi([0, 1], 100, 100);
r = bitrank_mex(A);
```

## Performance Notes

- **Linux (x86_64)**: Uses AVX512 if available, otherwise AVX2. OpenMP is enabled by default.
- **macOS (Apple Silicon)**: Uses NEON optimizations (via compiler auto-vectorization) and OpenMP (if `libomp` is installed).
- **Windows**: Uses AVX2 optimizations.

## License

[MIT License](LICENSE)
