#include <librapid/test/librapid_test.hpp>

namespace librapid {
	namespace test {
		int testLibrapid(int x) {
			return x * x;
		}

		void streamTest() {
#ifdef LIBRAPID_HAS_CUDA
			cudaStream_t stream;

			cudaSafeCall(cudaStreamCreateWithFlags(&stream,
												   cudaStreamNonBlocking));

			cudaSafeCall(cudaStreamDestroy(stream));
#endif // LIBRAPID_HAS_CUDA
		}
	}
}