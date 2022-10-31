#include <librapid/librapid.hpp>

namespace librapid::global {
	bool throwOnAssert				 = false;
	int64_t multithreadThreshold	 = 5000;
	int64_t gemmMultithreadThreshold = 100;
	int64_t numThreads				 = 1;
} // namespace librapid::global
