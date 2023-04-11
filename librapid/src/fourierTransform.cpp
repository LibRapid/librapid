#include <librapid/librapid.hpp>

namespace librapid::fft {
	namespace detail {
		namespace cpu {
			void rfft(Complex<double> *output, double *input, size_t n) {
				unsigned int mode = FFTW_ESTIMATE;
				fftw_plan_with_nthreads((int)global::numThreads);
				fftw_plan plan = fftw_plan_dft_r2c_1d(
				  (int)n, input, reinterpret_cast<fftw_complex *>(output), mode);
				fftw_execute(plan);
				fftw_destroy_plan(plan);
			}
		} // namespace cpu

		namespace gpu {

		}
	} // namespace detail

	LIBRAPID_NODISCARD Array<Complex<double>, device::CPU> fft(Array<double, device::CPU> &array) {
		LIBRAPID_ASSERT(array.ndim() == 1, "FFT only implemented for 1D arrays");
		int64_t outSize = array.shape()[0] / 2 + 1;
		Array<Complex<double>, device::CPU> res(Shape({outSize}));
		double *input			= array.storage().begin();
		Complex<double> *output = res.storage().begin();
		detail::cpu::rfft(output, input, array.shape()[0]);
		return res;
	}
} // namespace librapid::fft
