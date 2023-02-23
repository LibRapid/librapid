# CMake Integrations

## CMake Options

When using LibRapid in your CMake project, the following options are configurable:

- `LIBRAPID_BUILD_EXAMPLES => OFF` (Build examples?)
- `LIBRAPID_BUILD_TESTS => OFF` (Build tests?)
- `LIBRAPID_STRICT => OFF` (Force warnings into errors?)
- `LIBRAPID_QUIET => OFF` (Disable warnings)
- `LIBRAPID_GET_BLAS => OFF` (Clone a prebuilt version of OpenBLAS?)
- `LIBRAPID_USE_CUDA => ON` (Automatically search for CUDA?)
- `LIBRAPID_USE_OMP => ON` (Automatically search for OpenMP?)
- `LIBRAPID_USE_MULTIPREC => OFF` (Include multiprecision library -- more on this elsewhere in documentation)
- `LIBRAPID_OPTIMISE_SMALL_ARRAYS => OFF` (Optimise small arrays?)
- `LIBRAPID_FAST_MATH => OFF` (Use potentially less accurate operations to increase performance)
