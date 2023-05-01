#include <librapid/librapid.hpp>

namespace librapid::fft {
	namespace detail {
#if defined(LIBRAPID_HAS_CUDA)
		// TODO: Move this to "global.hpp"
		static bool cuFFTInitialized = false;
#endif // LIBRAPID_HAS_CUDA

		namespace cpu {
			void rfft(Complex<double> *output, double *input, size_t n) {
#if defined(LIBRAPID_HAS_FFTW) || defined(LIBRAPID_HAS_CUDA)
				unsigned int mode = FFTW_ESTIMATE;

#	if !defined(LIBRAPID_HAS_CUDA)
				fftw_plan_with_nthreads((int)global::numThreads);
#	endif // LIBRAPID_HAS_CUDA

				fftw_plan plan = fftw_plan_dft_r2c_1d(
				  (int)n, input, reinterpret_cast<fftw_complex *>(output), mode);
				fftw_execute(plan);
				fftw_destroy_plan(plan);
#else
				LIBRAPID_ASSERT(false,
								"FFTW or cuFFT is required for Fourier transform. See "
								"documentation for details.");
#endif	  // LIBRAPID_HAS_FFTW || LIBRAPID_HAS_CUDA
			}
		} // namespace cpu

#if defined(LIBRAPID_HAS_CUDA)
		namespace gpu {
			void rfft(Complex<double> *output, double *input, size_t n) {
				cufftHandle plan;
				cufftPlan1d(&plan, (int)n, CUFFT_D2Z, 1);
				cufftSetStream(plan, global::cudaStream);
				cufftExecD2Z(plan, input, reinterpret_cast<cufftDoubleComplex *>(output));
				cufftDestroy(plan);
			}

			void rfft(Complex<float> *output, float *input, size_t n) {
				cufftHandle plan;
				cufftPlan1d(&plan, (int)n, CUFFT_R2C, 1);
				cufftSetStream(plan, global::cudaStream);
				cudaStreamSynchronize(global::cudaStream);
				cufftExecR2C(plan, input, reinterpret_cast<cufftComplex *>(output));
				cufftDestroy(plan);
			}
		} // namespace gpu
#endif	  // LIBRAPID_HAS_CUDA
	}	  // namespace detail

	LIBRAPID_NODISCARD Array<Complex<double>, device::CPU> fft(Array<double, device::CPU> &array) {
		LIBRAPID_ASSERT(array.ndim() == 1, "FFT only implemented for 1D arrays");
		int64_t outSize = array.shape()[0] / 2 + 1;
		Array<Complex<double>, device::CPU> res(Shape({outSize}));
		double *input			= array.storage().begin();
		Complex<double> *output = res.storage().begin();
		detail::cpu::rfft(output, input, array.shape()[0]);
		return res;
	}

#if defined(LIBRAPID_HAS_CUDA)
	LIBRAPID_NODISCARD Array<Complex<double>, device::GPU> fft(Array<double, device::GPU> &array) {
		LIBRAPID_ASSERT(array.ndim() == 1, "FFT only implemented for 1D arrays");
		int64_t outSize = array.shape()[0] / 2 + 1;
		Array<Complex<double>, device::GPU> res(Shape({outSize}));
		double *input			= array.storage().begin().get();
		Complex<double> *output = res.storage().begin().get();
		detail::gpu::rfft(output, input, array.shape()[0]);
		return res;
	}

	LIBRAPID_NODISCARD Array<Complex<float>, device::GPU> fft(Array<float, device::GPU> &array) {
		LIBRAPID_ASSERT(array.ndim() == 1, "FFT only implemented for 1D arrays");
		int64_t outSize = array.shape()[0] / 2 + 1;
		Array<Complex<float>, device::GPU> res(Shape({outSize}));
		float *input		   = array.storage().begin().get();
		Complex<float> *output = res.storage().begin().get();
		detail::gpu::rfft(output, input, array.shape()[0]);
		return res;
	}
#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::fft
