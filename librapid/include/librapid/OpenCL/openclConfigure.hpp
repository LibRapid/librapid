#ifndef LIBRAPID_OPENCL_CONFIGURE_HPP
#define LIBRAPID_OPENCL_CONFIGURE_HPP

namespace librapid {
	int64_t openclDeviceCompute(const cl::Device &device);
	void updateOpenCLDevices(bool verbose = false);
	cl::Device findFastestDevice(const std::vector<cl::Device> &devices);
	void configureOpenCL(bool verbose = false);
	void addOpenCLKernelSource(const std::string &source);
	void addOpenCLKernelFile(const std::string &filename);
	void compileOpenCLKernels();
} // namespace librapid

#endif // LIBRAPID_OPENCL_CONFIGURE_HPP