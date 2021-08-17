#ifndef LIBRAPID_TEST
#define LIBRAPID_TEST

namespace librapid
{
	namespace test
	{
		int testLibrapid(int x)
		{
			return x * x;
		}

		void streamTest()
		{
		#ifdef LIBRAPID_HAS_CUDA
			cudaStream_t stream;

			cudaSafeCall(cudaStreamCreateWithFlags(&stream,
							cudaStreamNonBlocking));

			cudaSafeCall(cudaStreamDestroy(stream));
		#endif // LIBRAPID_HAS_CUDA
		}
	}
}

#endif // LIBRAPID_TEST