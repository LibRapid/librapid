#include <librapid/librapid.hpp>

namespace librapid::global {
	bool throwOnAssert				 = false;
	int64_t multithreadThreshold	 = 5000;
	int64_t gemmMultithreadThreshold = 100;
	int64_t numThreads				 = 1;

#if defined(LIBRAPID_HAS_CUDA)
	cudaStream_t cudaStream;
#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::global
