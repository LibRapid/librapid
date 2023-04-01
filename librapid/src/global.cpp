#include <librapid/librapid.hpp>

namespace librapid::global {
	bool throwOnAssert				 = false;
	int64_t multithreadThreshold	 = 5000;
	int64_t gemmMultithreadThreshold = 100;
	int64_t numThreads				 = 8;
	int64_t cacheLineSize			 = 64;
	int64_t memoryAlignment			 = LIBRAPID_DEFAULT_MEM_ALIGN;

#if defined(LIBRAPID_HAS_CUDA)
	cudaStream_t cudaStream;
	jitify::JitCache jitCache;
#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::global
