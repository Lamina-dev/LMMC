# LMMC
Build for Lamina Project, boosted by LAMMP technology.
## Build it
```pwsh
$root = (Get-Location).Path
$rt = Join-Path $root "build/lammp/dist/lammp/bin/Release"
$ar = Join-Path $root "build/lammp/dist/lammp/lib/Release"
cmake -S LAMMP -B build/lammp -G "Visual Studio 17 2022" -A x64 -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$rt" -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$ar" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$ar"
cmake --build build/lammp --config Release --target LammpCore
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## TODO List

- [x] dense matrix
- [x] vector
- [x] LU/QR decomposition
- [x] linear solve
- [x] CSR sparse matrix container
- [x] sparse calculation
- [x] CG
- [x] BiCGSTAB
- [x] GMRES(restart)
- [x] None
- [x] Jacobi
- [x] ILU0
- [x] ILUT
- [x] Non linear solve
- [x] ODE solve
- [x] statistics
- [x] 3d tensor

- [ ] sample prog
- [ ] benchmark
- [ ] solver logging
- [ ] preconditioner & iterative solver performance(assmbly?)
- [ ] large sparse and boundary condition testing
- [ ] parallelization(may not?)

due to lazy reason, the source code didnt conatin any comments, maybe i will use copilot to add some...