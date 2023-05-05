#include <librapid/librapid.hpp>

namespace librapid {
#if defined(LIBRAPID_HAS_OPENCL)

	int64_t openclDeviceCompute(const cl::Device &device) {
		cl_uint aComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		cl_uint aClockFreq	  = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
		return static_cast<int64_t>(aComputeUnits * aClockFreq);
	}

	void updateOpenCLDevices(bool verbose) {
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		for (const auto & platform : platforms) {
			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			if (!devices.empty()) {
				if (verbose) fmt::print("Platform: {}\n", platform.getInfo<CL_PLATFORM_NAME>());

				for (auto &device : devices) {
					if (verbose) {
						fmt::print("\tDevice [id={}]: {}\n",
								   global::openclDevices.size(),
								   device.getInfo<CL_DEVICE_NAME>());

						fmt::print("\t\tCompute: {}\n",
								   device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
						fmt::print("\t\tClock: {}\n",
								   device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
						fmt::print("\t\tMemory: {}GB\n",
								   (device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() + (1 << 30)) /
									 (1 << 30));
					}

					global::openclDevices.push_back(device);
				}
			}
		}
	}

	cl::Device findFastestDevice(const std::vector<cl::Device> &devices) {
		LIBRAPID_ASSERT(!devices.empty(), "No OpenCL devices found");
		cl::Device fastest;
		int64_t fastestCompute = 0;
		for (auto &device : devices) {
			int64_t compute = openclDeviceCompute(device);
			if (compute > fastestCompute) {
				fastestCompute = compute;
				fastest		   = device;
			}
		}
		return fastest;
	}

	void configureOpenCL(bool verbose) {
		if (verbose) fmt::print("======= OpenCL Configuration =======\n");
		updateOpenCLDevices(verbose);

		if (!verbose) {
			// If not verbose, select the fastest device by default
			global::openCLDevice = findFastestDevice(global::openclDevices);
		} else {
			// Otherwise, prompt the user to select a device
			int64_t deviceIndex = -1;
			while (deviceIndex < 0 || deviceIndex >= global::openclDevices.size()) {
				std::string prompt =
				  fmt::format("Select OpenCL device [0-{}]: ", global::openclDevices.size() - 1);
				scn::prompt(prompt.c_str(), "{}", deviceIndex);
			}

			global::openCLDevice = global::openclDevices[deviceIndex];
		}

		global::openCLContext = cl::Context(global::openCLDevice);
		global::openCLQueue	  = cl::CommandQueue(global::openCLContext, global::openCLDevice);

		// Add kernel files
		auto basePath = fmt::format("{}/include/librapid/OpenCL/kernels/", LIBRAPID_SOURCE);
		addOpenCLKernelFile(basePath + "arithmetic.cl");

		// Compile kernels
		compileOpenCLKernels();
	}

	void addOpenCLKernelSource(const std::string &source) {
		global::openCLSources.emplace_back(source.c_str(), source.size());
	}

	void addOpenCLKernelFile(const std::string &filename) {
		std::ifstream file(filename);
		std::string source((std::istreambuf_iterator<char>(file)),
						   std::istreambuf_iterator<char>());
		char *cstr = new char[source.length() + 1];
		strcpy(cstr, source.c_str());
		global::openCLSources.emplace_back(cstr, source.size());
	}

	void compileOpenCLKernels() {
		global::openCLProgram = cl::Program(global::openCLContext, global::openCLSources);
		global::openCLProgram.build({global::openCLDevice});
		cl_build_status status =
		  global::openCLProgram.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(global::openCLDevice);

		if (status != CL_BUILD_SUCCESS) {
			std::string buildLog =
			  global::openCLProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(global::openCLDevice);
			std::string errorMsg = fmt::format("OpenCL kernel compilation failed:\n{}", buildLog);
			fmt::print(stderr, "{}\n", errorMsg);
			throw std::runtime_error(errorMsg);
		}
	}
#endif // LIBRAPID_HAS_OPENCL
} // namespace librapid
