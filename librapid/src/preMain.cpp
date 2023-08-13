#include <librapid/librapid.hpp>

namespace librapid::detail {
    bool preMainRun = false;

    PreMain::PreMain() {
        if (!preMainRun) {
#if defined(LIBRAPID_WINDOWS) // && !defined(LIBRAPID_NO_WINDOWS_H)
            // Force the terminal to accept ANSI characters
            system(("chcp " + std::to_string(CP_UTF8)).c_str());
#endif // LIBRAPID_WINDOWS

            preMainRun            = true;
            global::cacheLineSize = cacheLineSize();

            // OpenCL compatible devices are detected after this function is called,
            // meaning nothing is found here. The user must call configureOpenCL()
            // manually.

            // #if defined(LIBRAPID_HAS_OPENCL)
            //			configureOpenCL();
            // #endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
            cudaSafeCall(cudaStreamCreate(&global::cudaStream));
            cublasSafeCall(cublasCreate(&global::cublasHandle));
            cublasSafeCall(cublasSetStream(global::cublasHandle, global::cudaStream));

            cudaSafeCall(cudaMallocAsync(
              &global::cublasLtWorkspace, global::cublasLtWorkspaceSize, global::cudaStream));

            cublasSafeCall(cublasLtCreate(&global::cublasLtHandle));
            cublasSafeCall(cublasSetWorkspace(
              global::cublasHandle, global::cublasLtWorkspace, global::cublasLtWorkspaceSize));
            // Stream is specified in the function calls
#endif // LIBRAPID_HAS_CUDA

            // Set the random seed to an initial value
            global::randomSeed = (size_t)now<time::nanosecond>();
        }
    }
} // namespace librapid::detail
