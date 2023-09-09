# CMake Integration

## Installation

Link librapid like any other CMake library:

Clone the repository: ``git clone --recursive https://github.com/LibRapid/libRapid.git``

Add the following to your ``CMakeLists.txt``

```cmake
add_subdirectory(librapid)
target_link_libraries(yourTarget PUBLIC librapid)
```

:::{tip}
For a template `CMakelists.txt` file, see the `examples`
directory: [examples/CMakeLists.txt](https://github.com/LibRapid/librapid/blob/master/examples/templateCMakeLists.txt)
:::

## CMake Options

### ``LIBRAPID_BUILD_EXAMPLES``

```
DEFAULT: OFF
```

Build the suite of example programs in the ``examples`` directory.

### ``LIBRAPID_BUILD_TESTS``

```
DEFAULT: OFF
```

Build LibRapid's unit tests.

### ``LIBRAPID_CODE_COV``

```
DEFAULT: OFF
```

Enable code coverage for LibRapid's unit tests.

### ``LIBRAPID_STRICT``

``` 
DEFAULT: OFF
```

Enable strict compilation flags, turn on all warnings, and treat warnings as errors.

### ``LIBRAPID_QUIET``

```
DEFAULT: OFF
```

Disable all warnings from LibRapid. This is useful if you are using LibRapid as a dependency and want a cleaner
compilation output. Warnings should be minimal in the first place, but this option is provided just in case.

### ``LIBRAPID_USE_PRECOMPILED_HEADER``

```
DEFAULT: OFF
```

Enable the use of precompiled headers within LibRapid's compilation. This can be useful to accelerate compilation, but
can lead to some strange build errors, which is why it is disabled by default.

:::{warning}
One such build error occurs on some macOS systems with GCC. The resulting error is something along the lines of:

```
Unknown flag -Xarch_amd64
```

If you encounter this error, try disabling ``LIBRAPID_USE_PRECOMPILED_HEADER``.
:::

### ``LIBRAPID_GET_FFTW``

```
DEFAULT: OFF
```

Add FFTW as a dependency and link it with LibRapid. This is required for FFT support unless CUDA is enabled.

:::{danger}
FFTW is licensed under the GPL, which is not compatible with LibRapid's MIT license. If you are using LibRapid as a
dependency in an open source project, you may need to use LibRapid under a GPL license. If you forget, you'll
***probably*** be fine, but I can't guarantee anything. I'm not a lawyer, so don't take my word for it.
:::

### ``LIBRAPID_GET_BLAS``

```
DEFAULT: OFF
```

Download a precompiled OpenBLAS build for your platform, and link it with LibRapid. This is useful if you don't
(or can't) have BLAS installed on your system.

:::{warning}
Always prefer to use your system's BLAS installation if possible.
:::

### ``LIBRAPID_USE_OMP``

```
DEFAULT: ON
```

If OpenMP is found on the system, link LibRapid with it. This is required for multi-threading support and can
significantly improve performance.

:::{warning}
If this flag is enabled and OpenMP is not found installed on the system, the build will continue without OpenMP support.
:::

### ``LIBRAPID_USE_OPENCL``

```
DEFAULT: ON
```

Search for OpenCL and link LibRapid with it. This is required for OpenCL support.

:::warning
If this flag is enabled and OpenCL is not found installed on the system, the build will continue without OpenCL support.
:::

:::{danger}
If you are using OpenCL as a backend in your code, you must call ``librapid::configureOpenCL()`` before using any
OpenCL arrays. This function will initialise the OpenCL context and queue, compile the OpenCL kernels and configure the
OpenCL device for optimal performance. See the documentation for this function for more information.
:::

### ``LIBRAPID_USE_CUDA``

```
DEFAULT: ON
```

Search for CUDA and link LibRapid with it. This is required for GPU support.

:::warning
If this flag is enabled and CUDA is not found installed on the system, the build will continue without CUDA support.
:::

:::{danger}
LibRapid's CUDA support appears to only works on Windows, for some reason. I have no way of testing it on Linux or
MacOS, so I can't guarantee that it will work. If you have experience in this area, please feel free to contact me and
we can work together to get it working.
:::

### ``LIBRAPID_USE_MULTIPREC``

```
DEFAULT: OFF
```

If MPIR and MPFR are found on the system, LibRapid will automatically link with them. If not, LibRapid will build
custom, modified versions of these libraries. This is required for arbitrary precision support.

:::{warning}
This can lead to longer build times and larger binaries.
:::

### ``LIBRAPID_OPTIMISE_SMALL_ARRAYS``

```
DEFAULT: OFF
```

Enabling this flag removes multithreading support for trivial array operations. For relatively small arrays (on the
order of 1,000,000 elements), this can lead to a significant performance boost. For arrays larger than this,
multithreading can be more efficient.

### ``LIBRAPID_FAST_MATH``

```
DEFAULT: OFF
```

Enabling this flag enables fast math mode for all LibRapid functions. This can lead to a significant performance boost,
but may cause some functions to return slightly incorrect results due to lower precision operations being performed.

### ``LIBRAPID_NATIVE_ARCH``

```
DEFAULT: ON
```

Enabling this flag compiles librapid with the most advanced instruction set available on the system. This can lead to
significant performance boosts, but may cause the library to be incompatible with older systems.

:::warning
Compiling with this flag may also cause the binaries to be incompatible with other CPU architectures, so be careful
when distributing your programs.
:::

### ``LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH`` and ``LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH``

```
DEFAULT: 4
```

Set the default vector width for SIMD CUDA kernels. This must be in the range $[1, 4]$. Higher values will lead to
better performance in most cases, but can increase register pressure which may lead to lower performance than expected.
For optimal performance, you should try changing this value to suit your specific use case.

:::{warning}
This setting requires CUDA support to be enabled.
:::

### ``LIBRAPID_NO_WINDOWS_H``

```
DEFAULT: OFF
```

Prevent the inclusion of ``windows.h`` in LibRapid's headers. Sometimes the macros and functions defined in this header
can cause conflicts with other libraries, so this option is provided to prevent this.

:::{danger}
It is not possible to fully remove ``windows.h`` when compiling with CUDA support on Windows, but many of the modules
are still disabled. There is a possiblity that conflicts will still arise, but I am yet to encounter any.
:::

### ``LIBRAPID_MKL_CONFIG_PATH``

```
DEFAULT: ""
```

If you have Intel's OneAPI Math Kernel Library installed on your system, you can provide the path to the
``MKLConfig.cmake`` file here. This will force LibRapid to link with MKL and ignore any other BLAS libraries.
On systems with Intel CPUs, this can result in a significant performance boost.
