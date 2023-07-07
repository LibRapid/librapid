#include <librapid/librapid.hpp>

namespace librapid::opencl {
#if defined(LIBRAPID_HAS_OPENCL)
	std::string getOpenCLErrorString(int64_t error) {
		static const char *strings[] = {									  // Error Codes
										"CL_SUCCESS",						  //   0
										"CL_DEVICE_NOT_FOUND",				  //  -1
										"CL_DEVICE_NOT_AVAILABLE",			  //  -2
										"CL_COMPILER_NOT_AVAILABLE",		  //  -3
										"CL_MEM_OBJECT_ALLOCATION_FAILURE",	  //  -4
										"CL_OUT_OF_RESOURCES",				  //  -5
										"CL_OUT_OF_HOST_MEMORY",			  //  -6
										"CL_PROFILING_INFO_NOT_AVAILABLE",	  //  -7
										"CL_MEM_COPY_OVERLAP",				  //  -8
										"CL_IMAGE_FORMAT_MISMATCH",			  //  -9
										"CL_IMAGE_FORMAT_NOT_SUPPORTED",	  //  -10
										"CL_BUILD_PROGRAM_FAILURE",			  //  -11
										"CL_MAP_FAILURE",					  //  -12

										"",									  //  -13
										"",									  //  -14
										"",									  //  -15
										"",									  //  -16
										"",									  //  -17
										"",									  //  -18
										"",									  //  -19

										"",									  //  -20
										"",									  //  -21
										"",									  //  -22
										"",									  //  -23
										"",									  //  -24
										"",									  //  -25
										"",									  //  -26
										"",									  //  -27
										"",									  //  -28
										"",									  //  -29

										"CL_INVALID_VALUE",					  //  -30
										"CL_INVALID_DEVICE_TYPE",			  //  -31
										"CL_INVALID_PLATFORM",				  //  -32
										"CL_INVALID_DEVICE",				  //  -33
										"CL_INVALID_CONTEXT",				  //  -34
										"CL_INVALID_QUEUE_PROPERTIES",		  //  -35
										"CL_INVALID_COMMAND_QUEUE",			  //  -36
										"CL_INVALID_HOST_PTR",				  //  -37
										"CL_INVALID_MEM_OBJECT",			  //  -38
										"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", //  -39
										"CL_INVALID_IMAGE_SIZE",			  //  -40
										"CL_INVALID_SAMPLER",				  //  -41
										"CL_INVALID_BINARY",				  //  -42
										"CL_INVALID_BUILD_OPTIONS",			  //  -43
										"CL_INVALID_PROGRAM",				  //  -44
										"CL_INVALID_PROGRAM_EXECUTABLE",	  //  -45
										"CL_INVALID_KERNEL_NAME",			  //  -46
										"CL_INVALID_KERNEL_DEFINITION",		  //  -47
										"CL_INVALID_KERNEL",				  //  -48
										"CL_INVALID_ARG_INDEX",				  //  -49
										"CL_INVALID_ARG_VALUE",				  //  -50
										"CL_INVALID_ARG_SIZE",				  //  -51
										"CL_INVALID_KERNEL_ARGS",			  //  -52
										"CL_INVALID_WORK_DIMENSION",		  //  -53
										"CL_INVALID_WORK_GROUP_SIZE",		  //  -54
										"CL_INVALID_WORK_ITEM_SIZE",		  //  -55
										"CL_INVALID_GLOBAL_OFFSET",			  //  -56
										"CL_INVALID_EVENT_WAIT_LIST",		  //  -57
										"CL_INVALID_EVENT",					  //  -58
										"CL_INVALID_OPERATION",				  //  -59
										"CL_INVALID_GL_OBJECT",				  //  -60
										"CL_INVALID_BUFFER_SIZE",			  //  -61
										"CL_INVALID_MIP_LEVEL",				  //  -62
										"CL_INVALID_GLOBAL_WORK_SIZE",		  //  -63
										"CL_UNKNOWN_ERROR_CODE"};

		if (error >= -63 && error <= 0)
			return strings[-error];
		else
			return strings[64];
	}

	std::string getCLBlastErrorString(clblast::StatusCode status) {
		// clang-format off
		static const std::map<clblast::StatusCode, std::string> statusMap = {
		  {clblast::StatusCode::kSuccess, "CL_SUCCESS"},
		  {clblast::StatusCode::kOpenCLCompilerNotAvailable, "CL_COMPILER_NOT_AVAILABLE"},
		  {clblast::StatusCode::kTempBufferAllocFailure, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
		  {clblast::StatusCode::kOpenCLOutOfResources, "CL_OUT_OF_RESOURCES"},
		  {clblast::StatusCode::kOpenCLOutOfHostMemory, "CL_OUT_OF_HOST_MEMORY"},
		  {clblast::StatusCode::kOpenCLBuildProgramFailure, "CL_BUILD_PROGRAM_FAILURE"},
		  {clblast::StatusCode::kInvalidValue, "CL_INVALID_VALUE"},
		  {clblast::StatusCode::kInvalidCommandQueue, "CL_INVALID_COMMAND_QUEUE"},
		  {clblast::StatusCode::kInvalidMemObject, "CL_INVALID_MEM_OBJECT"},
		  {clblast::StatusCode::kInvalidBinary, "CL_INVALID_BINARY"},
		  {clblast::StatusCode::kInvalidBuildOptions, "CL_INVALID_BUILD_OPTIONS"},
		  {clblast::StatusCode::kInvalidProgram, "CL_INVALID_PROGRAM"},
		  {clblast::StatusCode::kInvalidProgramExecutable, "CL_INVALID_PROGRAM_EXECUTABLE"},
		  {clblast::StatusCode::kInvalidKernelName, "CL_INVALID_KERNEL_NAME"},
		  {clblast::StatusCode::kInvalidKernelDefinition, "CL_INVALID_KERNEL_DEFINITION"},
		  {clblast::StatusCode::kInvalidKernel, "CL_INVALID_KERNEL"},
		  {clblast::StatusCode::kInvalidArgIndex, "CL_INVALID_ARG_INDEX"},
		  {clblast::StatusCode::kInvalidArgValue, "CL_INVALID_ARG_VALUE"},
		  {clblast::StatusCode::kInvalidArgSize, "CL_INVALID_ARG_SIZE"},
		  {clblast::StatusCode::kInvalidKernelArgs, "CL_INVALID_KERNEL_ARGS"},
		  {clblast::StatusCode::kInvalidLocalNumDimensions, "CL_INVALID_WORK_DIMENSION"},
		  {clblast::StatusCode::kInvalidLocalThreadsTotal, "CL_INVALID_WORK_GROUP_SIZE"},
		  {clblast::StatusCode::kInvalidLocalThreadsDim, "CL_INVALID_WORK_ITEM_SIZE"},
		  {clblast::StatusCode::kInvalidGlobalOffset, "CL_INVALID_GLOBAL_OFFSET"},
		  {clblast::StatusCode::kInvalidEventWaitList, "CL_INVALID_EVENT_WAIT_LIST"},
		  {clblast::StatusCode::kInvalidEvent, "CL_INVALID_EVENT"},
		  {clblast::StatusCode::kInvalidOperation, "CL_INVALID_OPERATION"},
		  {clblast::StatusCode::kInvalidBufferSize, "CL_INVALID_BUFFER_SIZE"},
		  {clblast::StatusCode::kInvalidGlobalWorkSize, "CL_INVALID_GLOBAL_WORK_SIZE"},
		  {clblast::StatusCode::kNotImplemented, "Routine or functionality not implemented yet"},
		  {clblast::StatusCode::kInvalidMatrixA, "Matrix A is not a valid OpenCL buffer"},
		  {clblast::StatusCode::kInvalidMatrixB, "Matrix B is not a valid OpenCL buffer"},
		  {clblast::StatusCode::kInvalidMatrixC, "Matrix C is not a valid OpenCL buffer"},
		  {clblast::StatusCode::kInvalidVectorX, "Vector X is not a valid OpenCL buffer"},
		  {clblast::StatusCode::kInvalidVectorY, "Vector Y is not a valid OpenCL buffer"},
		  {clblast::StatusCode::kInvalidDimension, "Dimensions M, N, and K have to be larger than zero"},
		  {clblast::StatusCode::kInvalidLeadDimA, "LD of A is smaller than the matrix's first dimension"},
		  {clblast::StatusCode::kInvalidLeadDimB, "LD of B is smaller than the matrix's first dimension"},
		  {clblast::StatusCode::kInvalidLeadDimC, "LD of C is smaller than the matrix's first dimension"},
		  {clblast::StatusCode::kInvalidIncrementX, "Increment of vector X cannot be zero"},
		  {clblast::StatusCode::kInvalidIncrementY, "Increment of vector Y cannot be zero"},
		  {clblast::StatusCode::kInsufficientMemoryA, "Matrix A's OpenCL buffer is too small"},
		  {clblast::StatusCode::kInsufficientMemoryB, "Matrix B's OpenCL buffer is too small"},
		  {clblast::StatusCode::kInsufficientMemoryC, "Matrix C's OpenCL buffer is too small"},
		  {clblast::StatusCode::kInsufficientMemoryX, "Vector X's OpenCL buffer is too small"},
		  {clblast::StatusCode::kInsufficientMemoryY, "Vector Y's OpenCL buffer is too small"},
		  {clblast::StatusCode::kInsufficientMemoryTemp, "Temporary buffer provided to GEMM routine is too small"},
		  {clblast::StatusCode::kInvalidBatchCount, "Batch count needs to be positive"},
		  {clblast::StatusCode::kInsufficientMemoryTemp, "Temporary buffer provided to GEMM routine is too small"},
		  {clblast::StatusCode::kInvalidBatchCount, "The batch count needs to be positive"},
		  {clblast::StatusCode::kInvalidOverrideKernel, "Trying to override parameters for an invalid kernel"},
		  {clblast::StatusCode::kMissingOverrideParameter, "Missing override parameter(s) for the target kernel"},
		  {clblast::StatusCode::kInvalidLocalMemUsage, "Not enough local memory available on this device"},
		  {clblast::StatusCode::kNoHalfPrecision, "Half precision (16-bits) not supported by the device"},
		  {clblast::StatusCode::kNoDoublePrecision, "Double precision (64-bits) not supported by the device"},
		  {clblast::StatusCode::kInvalidVectorScalar, "The unit-sized vector is not a valid OpenCL buffer"},
		  {clblast::StatusCode::kInsufficientMemoryScalar, "The unit-sized vector's OpenCL buffer is too small"},
		  {clblast::StatusCode::kDatabaseError, "Entry for the device was not found in the database"},
		  {clblast::StatusCode::kUnknownError, "A catch-all error code representing an unspecified error"},
		  {clblast::StatusCode::kUnexpectedError, "A catch-all error code representing an unexpected exception"}};
		// clang-format on

		auto it = statusMap.find(status);
		if (it != statusMap.end())
			return it->second;
		else
			return "Unknown error";
	}
#endif // LIBRAPID_HAS_OPENCL
} // namespace librapid::opencl
