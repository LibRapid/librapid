#ifndef LIBRAPID_OPENCL_CONFIGURE_HPP
#define LIBRAPID_OPENCL_CONFIGURE_HPP

namespace librapid {
#if defined(LIBRAPID_HAS_OPENCL)
    int64_t openclDeviceCompute(const cl::Device &device);
    void updateOpenCLDevices(bool verbose = false);
    cl::Device findFastestDevice(const std::vector<cl::Device> &devices);
    void addOpenCLKernelSource(const std::string &source);
    void addOpenCLKernelFile(const std::string &filename);
    void compileOpenCLKernels(bool verbose = false);
    void configureOpenCL(bool verbose = false, bool ask = false);
#endif // LIBRAPID_HAS_OPENCL
} // namespace librapid

#endif // LIBRAPID_OPENCL_CONFIGURE_HPP