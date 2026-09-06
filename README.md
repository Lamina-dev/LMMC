# LMMC
Build for Lamina Project, powered by LMMP.

## Build

LMMP is built from the `LMMP` submodule as part of the LMMC build.

```pwsh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DLMMC_BUILD_TESTS=ON -DLMMC_LMMP_ASM=AUTO
cmake --build build
ctest --test-dir build --output-on-failure
```

`LMMC_LMMP_ASM` accepts `AUTO`, `GENERIC`, `X64`, or `ARM64`. `AUTO` selects the supported GAS/LLVM-compatible `.S` backend for the target architecture and falls back to the generic C implementation only on unsupported platforms.

## CI

GitHub Actions runs on every push and pull request, matching the applicable
LMCAS checks:

- Linux: GCC with `AUTO` and `GENERIC` backends, and Clang with `AUTO`.
- Windows MinGW: Debug and Release, each with shared and static LMMC.
- Installed-package consumption through `find_package(LMMC CONFIG REQUIRED)`.
- AddressSanitizer/UndefinedBehaviorSanitizer, ThreadSanitizer, and clang-tidy.
- Coverage reports with minimum line coverage of 77% and branch coverage of 44%.
  JSON summary, XML, and HTML reports are uploaded as `coverage-report`.

All build jobs treat LMMC compiler warnings as errors. Analysis configuration
is local to this repository; a parent LMCAS checkout is not required. The workflow
validates packages but does not automatically publish GitHub Releases.

## Install and consume

A standalone build with bundled LMMP installs both libraries and their public
headers:

```pwsh
cmake --install build --prefix "$PWD/dist"
cmake -S test/package_consumer -B build-consumer -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/dist"
cmake --build build-consumer
```

Consumers link `LMMC::lmmc` after `find_package(LMMC CONFIG REQUIRED)`; the
LMMP dependency is transitive. To run `lmmc_package_consumer`, add the installed
`bin` directory to `PATH` on Windows, or the installed library directory
(`lib` or `lib64`) to `LD_LIBRARY_PATH` on Linux.

`LMMC_BUILD_SHARED=OFF` makes LMMC static; bundled LMMP remains shared.
External/prebuilt LMMP configurations remain build-only. When embedded in
LMCAS, installation and exports remain owned by LMCAS.

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