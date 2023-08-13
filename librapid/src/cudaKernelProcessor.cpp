#if defined(LIBRAPID_HAS_CUDA)

#    include <librapid/librapid.hpp>

namespace librapid::cuda {
    const std::string &loadKernel(const std::string &path, bool relative) {
        static std::map<std::string, std::string> mapping;

        if (mapping.find(path) != mapping.end()) { return mapping[path]; }

        auto basePath = fmt::format("{}/include/librapid/cuda/kernels/", LIBRAPID_SOURCE);

        std::string helperPath    = fmt::format("{}/kernelHelper.cuh", basePath);
        std::string vectorOpsPath = fmt::format("{}/vectorOps.cuh", basePath);
        std::string dualPath =
          fmt::format("{}/include/librapid/autodiff/dual.hpp", LIBRAPID_SOURCE);
        std::string kernelPath = fmt::format("{}{}.cu", relative ? (basePath + "/") : "", path);
        std::fstream helper(helperPath);
        std::fstream vectorOps(vectorOpsPath);
        std::fstream dual(dualPath);
        std::fstream kernel(kernelPath);
        LIBRAPID_ASSERT(helper.is_open(), "Failed to load CUDA helper functions");
        LIBRAPID_ASSERT(vectorOps.is_open(), "Failed to load CUDA vectorOps helper functions");
        LIBRAPID_ASSERT(dual.is_open(), "Failed to load dual number library");
        LIBRAPID_ASSERT(kernel.is_open(), "Failed to load CUDA kernel '{}.cu'", path);
        std::stringstream buffer;
        buffer << helper.rdbuf();
        buffer << "\n\n";
        buffer << vectorOps.rdbuf();
        buffer << "\n\n";
        buffer << dual.rdbuf();
        buffer << "\n\n";
        buffer << kernel.rdbuf();

        mapping[path] = path + "\n" + buffer.str();
        return mapping[path];
    }

    jitify::Program generateCudaProgram(const std::string &kernel) {
        return global::jitCache.program(kernel, {}, {fmt::format("-I{}", CUDA_INCLUDE_DIRS)});
    }
} // namespace librapid::cuda

#endif // LIBRAPID_HAS_CUDA
