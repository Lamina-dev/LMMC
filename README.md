# LMMC
Build for Lamina Project, powered by LMMP.

## Build

LMMP is built from the `LMMP` submodule as part of the LMMC build.

LMMC supports Windows, Linux, and macOS; its platform scope follows upstream LMMP.

```pwsh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DLMMC_BUILD_TESTS=ON -DLMMC_LMMP_ASM=AUTO
cmake --build build
ctest --test-dir build --output-on-failure
```

`LMMC_LMMP_ASM` accepts `AUTO`, `GENERIC`, `X64`, or `ARM64`. On supported operating systems, `AUTO` selects the supported GAS/LLVM-compatible `.S` backend for the target architecture and falls back to the generic C implementation when that architecture has no assembly backend.
On macOS, `AUTO` selects the ARM64 backend on native Apple Silicon. Native
Intel builds use `GENERIC`; LMMP intentionally rejects the unsupported X64
assembly selection on Darwin.

### Development tests

Fresh product configurations leave testing disabled. Enable the C11 suite with
`LMMC_BUILD_TESTS=ON`, as above, or opt into CTest development defaults with
`BUILD_TESTING=ON`. Tests use cmocka assertions, per-case setup/teardown, and its
native group runner; CTest registers the executables under the `lmmc` label.

CMake first looks for a cmocka package of version 2.0.2 or newer, then fetches
the locked 2.0.2 tag `cmocka-2.0.2` when needed. For offline development, set
`FETCHCONTENT_SOURCE_DIR_CMOCKA` to that framework's local source directory.
The framework is a private test-executable dependency. Installed libraries,
headers, and CMake exports retain the LMMC/LMMP and standard-runtime boundary;
package checks reject development-framework files and exported link dependencies.

Use `ctest --test-dir build -L lmmc --output-on-failure` to run the suite.
CTest's `--output-junit` option provides a machine-readable report.

## CI

GitHub Actions runs on every push and pull request, matching the applicable
LMCAS checks:

- Linux: GCC with `AUTO` and `GENERIC` backends, and Clang with `AUTO`.
- Windows MinGW UCRT64: Debug and Release, each with shared and static LMMC.
- macOS: native Apple Silicon (`arm64`/`AUTO`) and Intel
  (`x86_64`/`GENERIC`), including Debug tests and relocated Release consumers.
- Installed-package consumption through `find_package(LMMC CONFIG REQUIRED)`.
- AddressSanitizer/UndefinedBehaviorSanitizer, ThreadSanitizer, and clang-tidy.
- Coverage reports with minimum line coverage of 77% and branch coverage of 44%.
  JSON summary, XML, and HTML reports are uploaded as `coverage-report`.

All build jobs treat LMMC compiler warnings as errors. Analysis configuration
is local to this repository; a parent LMCAS checkout is not required. The workflow
validates packages but does not automatically publish GitHub Releases.

Precision-sensitive kernels require a conforming fused multiply-add (`fma`).
Windows CI uses the [recommended MSYS2 UCRT64 environment](https://www.msys2.org/docs/environments/)
instead of legacy MINGW64/MSVCRT. Coverage builds use atomic counters so
multithreaded tests do not corrupt profiling data.

## Install and consume

A standalone build with bundled LMMP installs both libraries and their public
headers:

```pwsh
cmake --install build --prefix "$PWD/dist"
cmake -S test/package_consumer -B build-consumer -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/dist"
cmake --build build-consumer
```

Consumers link `LMMC::lmmc` after `find_package(LMMC CONFIG REQUIRED)`; C11
and the LMMP link dependency are transitive. To run `lmmc_package_consumer`,
add the installed `bin` directory to `PATH` on Windows or the installed library
directory (`lib` or `lib64`) to `LD_LIBRARY_PATH` on Linux. On macOS, the
installed dylibs use `@rpath` identities and the consumer's CMake-generated
RPATH; no `DYLD_LIBRARY_PATH` or `DYLD_FALLBACK_LIBRARY_PATH` injection is
required.

`LMMC_BUILD_SHARED=OFF` builds a position-independent LMMC static archive suitable
for linking into the shared LMCAS library on Linux. Bundled LMMP remains shared;
this option does not produce an entirely static dependency graph.
External/prebuilt LMMP configurations remain build-only. When embedded in
LMCAS, installation and exports remain owned by LMCAS.

### LMMC API migration

Rebuild callers and replace the previous API names with the current names below.

| Previous API | Current API |
| --- | --- |
| `lmmc_stats_nPr`, `lmmc_stats_nCr` | `lmmc_stats_npr`, `lmmc_stats_ncr` |
| `lmmc_std_math_I` | `lmmc_std_math_i` |
| `lmmc_tensor_t` | `lmmc_tensor3_t` |
| Fixed-rank `lmmc_tensor_*` operations | `lmmc_tensor3_*` |
| `lmmc_tensor_create`, `lmmc_tensor_get_nd`, `lmmc_tensor_set_nd` | `lmmc_tensor_nd_create`, `lmmc_tensor_nd_get`, `lmmc_tensor_nd_set` |
| `lmmc_tensor_permute`, `lmmc_tensor_contract`, `lmmc_tensor_mode_n_product` | `lmmc_tensor_nd_permute`, `lmmc_tensor_nd_contract`, `lmmc_tensor_nd_mode_n_product` |

The fixed-rank operations renamed above are `destroy`, `fill`, `set`, `get`,
`norm_fro`, `add`, `sub`, `mul`, `div`, `scale`, `sum`, `max`, `min`, `sum_axis`,
`reshape_view`, and `slice_view`. Existing `tensor3_create`/`tensor3_wrap` and
`tensor_nd_destroy`/`tensor_nd_reshape_view` names stay unchanged. Tensor layouts,
view ownership, and language-level `tensor.get`/`tensor.reshape`/`tensor.contract`
and `math.I` names are unchanged.

Use `lmmc/tensor3.h` or `lmmc/tensor_nd.h` for narrow declarations;
`lmmc/tensor.h` remains the official aggregate. The diagnostic allocation count
API returns `size_t` directly.

LMMC and LMCAS are currently unversioned: no version headers or CMake package
version metadata are installed, and package discovery does not select compatible
versions. LMMC uses the ELF SONAME `liblmmc.so` and an anonymous export node that
retains the public-symbol allowlist and hides all other symbols. On macOS its
identity is `@rpath/liblmmc.dylib`; bundled LMMP retains its upstream version and
uses `@rpath/liblmmp.1.dylib`. CMake generates the build-directory `Doxyfile`
without a project version; run `doxygen build/Doxyfile` after configuring to
generate documentation.

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