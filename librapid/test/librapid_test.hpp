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
			cudaStream_t stream;

			cudaSafeCall(cudaStreamCreateWithFlags(&stream,
							cudaStreamNonBlocking));

			cudaSafeCall(cudaStreamDestroy(stream));
		}
	}
}

#endif // LIBRAPID_TEST