#include <librapid/test/librapid_test.hpp>
#include <librapid/array/extent.hpp>
#include <librapid/array/multiarray.hpp>

namespace librapid::test {
	int testLibrapid(int x) { return x * x; }

	void streamTest() {
#ifdef LIBRAPID_HAS_CUDA
		cudaStream_t stream;

		cudaSafeCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

		cudaSafeCall(cudaStreamDestroy(stream));
#endif // LIBRAPID_HAS_CUDA
	}
} // namespace librapid::test