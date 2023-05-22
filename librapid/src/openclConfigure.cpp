#include <librapid/librapid.hpp>

namespace librapid {
#if defined(LIBRAPID_HAS_OPENCL)

	bool testOpenCLDevice(const cl::Device &device) {
		try {
			cl::Context context(device);
			cl::CommandQueue queue(context, device);

			std::string source = R"V0G0N(
__kernel void testAddition(__global const float *a, __global const float *b, __global float *c) {
	const int i = get_global_id(0);
	c[i] = a[i] + b[i];
}
)V0G0N";
			cl::Program::Sources sources;
			sources.emplace_back(source.c_str(), source.length() + 1);

			cl_int err;
			cl::Program program(context, sources);
			err = program.build();

			// Check the build status
			cl_build_status buildStatus = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);

			if (buildStatus != CL_BUILD_SUCCESS) { return false; }

			std::vector<float> srcA = {1, 2, 3, 4, 5};
			std::vector<float> srcB = {5, 4, 3, 2, 1};
			std::vector<float> dst(5);
			size_t numElements = srcA.size();
			cl::Buffer bufA(context, CL_MEM_READ_ONLY, numElements * sizeof(float));
			cl::Buffer bufB(context, CL_MEM_READ_ONLY, numElements * sizeof(float));
			cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, numElements * sizeof(float));

			queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, numElements * sizeof(float), srcA.data());
			queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, numElements * sizeof(float), srcB.data());

			cl::Kernel kernel(program, "testAddition");
			kernel.setArg(0, bufA);
			kernel.setArg(1, bufB);
			kernel.setArg(2, bufC);

			cl::NDRange global_size(numElements);
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);
			queue.enqueueReadBuffer(bufC, CL_TRUE, 0, numElements * sizeof(float), dst.data());

			return dst == std::vector<float>({6, 6, 6, 6, 6});
		} catch (const std::exception &e) { return false; }
	}

	int64_t openclDeviceCompute(const cl::Device &device) {
		cl_uint aComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		cl_uint aClockFreq	  = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
		return static_cast<int64_t>(aComputeUnits * aClockFreq);
	}

	void updateOpenCLDevices(bool verbose) {
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		for (const auto &platform : platforms) {
			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			if (!devices.empty()) {
				if (verbose) fmt::print("Platform: {}\n", platform.getInfo<CL_PLATFORM_NAME>());

				for (auto &device : devices) {
					// Test the device to check it works
					if (!testOpenCLDevice(device) && verbose) {
						fmt::print(fmt::fg(fmt::color::red),
								   "\tDevice FAILED: {}\n",
								   device.getInfo<CL_DEVICE_NAME>());
						continue;
					}

					if (verbose) {
						auto format = fmt::emphasis::bold | fmt::fg(fmt::color::green);
						fmt::print(format,
								   "\tDevice [id={}]: {}\n",
								   global::openclDevices.size(),
								   device.getInfo<CL_DEVICE_NAME>());

						fmt::print(format,
								   "\t\tCompute Units: {}\n",
								   device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
						fmt::print(format,
								   "\t\tClock:         {}MHz\n",
								   device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
						fmt::print(format,
								   "\t\tMemory:        {}GB\n",
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

	void addOpenCLKernelSource(const std::string &source) {
		global::openCLSources.emplace_back(source.c_str(), source.size());
	}

	void addOpenCLKernelFile(const std::string &filename) {
		std::ifstream file(filename);
		std::string source((std::istreambuf_iterator<char>(file)),
						   std::istreambuf_iterator<char>());
		source += "\n\n\n";
		char *cstr = new char[source.length() + 1];
		strcpy(cstr, source.c_str());
		global::openCLSources.emplace_back(cstr, source.size());
	}

	void compileOpenCLKernels(bool verbose) {
		bool finished = false;
		std::thread printer;

		if (verbose) {
			printer = std::thread([&]() {
				auto format = fmt::fg(fmt::color::green);
				fmt::print(format, "Compiling OpenCL kernels...");
				while (!finished) {
					if (verbose) {
						fmt::print(format, ".");
						sleep(0.5);
					}
				}
				fmt::print("\n\n");
			});
		}

		global::openCLProgram = cl::Program(global::openCLContext, global::openCLSources);
		global::openCLProgram.build({global::openCLDevice});
		cl_build_status status =
		  global::openCLProgram.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(global::openCLDevice);

		finished = true;
		if (verbose) printer.join();

		if (status != CL_BUILD_SUCCESS) {
			std::string buildLog =
			  global::openCLProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(global::openCLDevice);
			std::string errorMsg = fmt::format("OpenCL kernel compilation failed:\n{}", buildLog);
			fmt::print(stderr, "{}\n", errorMsg);
			std::cout << std::endl;
			throw std::runtime_error(errorMsg);
		}
	}

	void configureOpenCL(bool verbose, bool ask) {
		LIBRAPID_ASSERT(!global::openCLConfigured, "OpenCL already configured");

		if (verbose) {
			auto format = fmt::emphasis::bold | fmt::fg(fmt::color::orange);
			fmt::print(format, "============== OpenCL Configuration ==============\n");
		}
		updateOpenCLDevices(verbose);

		if (!ask) {
			// Select the fastest device by default
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

		if (verbose) {
			auto format = fmt::emphasis::bold | fmt::fg(fmt::color::gold);

			std::string deviceDetails =
			  fmt::format("Selected Device: {}", global::openCLDevice.getInfo<CL_DEVICE_NAME>());
			fmt::print(format,
					   "\n{:=^{}}\n#  {}  #\n{:=^{}}\n\n",
					   "",
					   deviceDetails.length() + 6,
					   deviceDetails,
					   "",
					   deviceDetails.length() + 6);
		}

		global::openCLContext = cl::Context(global::openCLDevice);
		global::openCLQueue	  = cl::CommandQueue(global::openCLContext, global::openCLDevice);

		// Add kernel files
		auto basePath = fmt::format("{}/include/librapid/OpenCL/kernels/", LIBRAPID_SOURCE);
		addOpenCLKernelFile(basePath + "core.cl");
		addOpenCLKernelFile(basePath + "negate.cl");
		addOpenCLKernelFile(basePath + "arithmetic.cl");
		addOpenCLKernelFile(basePath + "abs.cl");
		addOpenCLKernelFile(basePath + "floorCeilRound.cl");
		addOpenCLKernelFile(basePath + "trigonometry.cl");
		addOpenCLKernelFile(basePath + "expLogPow.cl");
		addOpenCLKernelFile(basePath + "transpose.cl");
		addOpenCLKernelFile(basePath + "activations.cl");

		// Compile kernels
		compileOpenCLKernels(verbose);

		global::openCLConfigured = true;
	}
#endif // LIBRAPID_HAS_OPENCL
} // namespace librapid
