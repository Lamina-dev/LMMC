# LMMC
Build for Lamina Project, powered by LMMP.

## Build

LMMP is built from the `LMMP` submodule as part of the LMMC build.

```pwsh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DLMMC_BUILD_TESTS=ON -DLMMC_LMMP_ASM=AUTO
cmake --build build
ctest --test-dir build --output-on-failure
```

`LMMC_LMMP_ASM` accepts `AUTO`, `GENERIC`, `X64`, or `ARM64`. `AUTO` selects the supported GAS/LLVM-compatible `.S` backend for the target architecture and falls back to the generic C implementation only on unsupported platforms.

## Implemented modules

- Dense vectors and matrices, elementwise operations, products, decompositions,
  direct solvers, and eigenvalue/SVD routines
- Sparse CSR/CSC/COO storage, arithmetic, direct and iterative solvers, and
  Jacobi/ILU preconditioners
- Nonlinear equations, optimization, quadrature, interpolation, and ODE
  integration
- Statistics, probability distributions, explicit and thread-local default
  random-number streams
- N-dimensional tensors, FFTs, complex arithmetic, and scalar special
  functions
- Standard-library adapters for math, constants, units, collections, linear
  algebra, statistics, and random-number operations

The standard-library C API is declared in `lmmc/stdlib.h` and uses the
`lmmc_std_*` function/type prefix and `LMMC_STD_*` constants. These LMMC
adapters implement the relevant LSR specification contracts; LSR is the
specification name, not the implementation API prefix.

Persistent LMMC objects use recoverable heap ownership and may be transferred
between initialized threads. Concurrent mutation requires external
synchronization. LMMP temporary algorithms retain their direct fail-fast
allocation contract.