#ifndef LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP
#define LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP

namespace librapid::fft {
	namespace detail {
		namespace cpu {
			void rfft(Complex<double> *output, double *input, size_t n);
		} // namespace cpu

#if defined(LIBRAPID_HAS_CUDA)
		namespace gpu {
			void rfft(Complex<double> *output, double *input, size_t n);
		} // namespace gpu
#endif	  // LIBRAPID_HAS_CUDA
	}	  // namespace detail

	LIBRAPID_NODISCARD Array<Complex<double>, device::CPU> fft(Array<double, device::CPU> &array);

#if defined(LIBRAPID_HAS_CUDA)
	LIBRAPID_NODISCARD Array<Complex<double>, device::GPU> fft(Array<double, device::GPU> &array);
	LIBRAPID_NODISCARD Array<Complex<float>, device::GPU> fft(Array<float, device::GPU> &array);
#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::fft

#endif // LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP