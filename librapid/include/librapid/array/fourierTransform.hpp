#ifndef LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP
#define LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP

namespace librapid::fft {
	namespace detail {
		namespace cpu {
			void rfft(Complex<double> *output, double *input, size_t n);
			void rfft(Complex<float> *output, float *input, size_t n);
		} // namespace cpu

		namespace gpu {

		}
	} // namespace detail

	LIBRAPID_NODISCARD Array<Complex<double>, device::CPU>
	fft(Array<double, device::CPU> &array);
} // namespace librapid::fft

#endif // LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP