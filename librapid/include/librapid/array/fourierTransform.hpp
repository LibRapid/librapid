#ifndef LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP
#define LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP

namespace librapid::fft {
	namespace detail {
		namespace cpu {
			template<typename T>
			void rfft(Complex<T> *output, T *input, size_t n) {
				pocketfft::shape_t shape	 = {n};
				pocketfft::stride_t strideIn = {sizeof(T)};
				pocketfft::stride_t strideOut = {sizeof(Complex<T>)};
				size_t axis	 = 0;
				bool forward = true;
				T fct		 = 1.0;
				pocketfft::r2c(shape,
							   strideIn,
							   strideOut,
							   axis,
							   forward,
							   input,
							   reinterpret_cast<std::complex<T> *>(output),
							   fct,
							   global::numThreads);
			}

#if defined(LIBRAPID_HAS_CUDA)
			LIBRAPID_INLINE void rfft(Complex<double> *output, double *input, size_t n) {
				unsigned int mode = FFTW_ESTIMATE;
				fftw_plan plan	  = fftw_plan_dft_r2c_1d(
					 (int)n, input, reinterpret_cast<fftw_complex *>(output), mode);
				fftw_execute(plan);
				fftw_destroy_plan(plan);
			}

			LIBRAPID_INLINE void rfft(Complex<float> *output, float *input, size_t n) {
				unsigned int mode = FFTW_ESTIMATE;
				fftwf_plan plan	  = fftwf_plan_dft_r2c_1d(
					(int)n, input, reinterpret_cast<fftwf_complex *>(output), mode);
				fftwf_execute(plan);
				fftwf_destroy_plan(plan);
			}
#elif defined(LIBRAPID_HAS_FFTW)
			LIBRAPID_INLINE void rfft(Complex<double> *output, double *input, size_t n) {
				unsigned int mode = FFTW_ESTIMATE;
				fftw_plan_with_nthreads((int)global::numThreads);
				fftw_plan plan = fftw_plan_dft_r2c_1d(
				  (int)n, input, reinterpret_cast<fftw_complex *>(output), mode);
				fftw_execute(plan);
				fftw_destroy_plan(plan);
			}

			LIBRAPID_INLINE void rfft(Complex<float> *output, float *input, size_t n) {
				unsigned int mode = FFTW_ESTIMATE;
				fftwf_plan_with_nthreads((int)global::numThreads);
				fftwf_plan plan = fftwf_plan_dft_r2c_1d(
				  (int)n, input, reinterpret_cast<fftwf_complex *>(output), mode);
				fftwf_execute(plan);
				fftwf_destroy_plan(plan);
			}
#endif
		} // namespace cpu

#if defined(LIBRAPID_HAS_CUDA)
		namespace gpu {
			LIBRAPID_INLINE void rfft(Complex<double> *output, double *input, size_t n) {
				cufftHandle plan;
				cufftPlan1d(&plan, (int)n, CUFFT_D2Z, 1);
				cufftSetStream(plan, global::cudaStream);
				cufftExecD2Z(plan, input, reinterpret_cast<cufftDoubleComplex *>(output));
				cufftDestroy(plan);
			}

			LIBRAPID_INLINE void rfft(Complex<float> *output, float *input, size_t n) {
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


	/// \brief Compute the real-valued discrete Fourier transform of a 1D array
	///
	/// Given a 1D array of real numbers, compute the discrete Fourier transform of the array. This
	/// returns an array of length \f$\frac{n}{2} + 1\f$ where \f$n\f$ is the length of the input
	/// array. The returned array contains the non-redundant half of the resulting transform, since
	/// the other half can be obtained by taking the complex conjugate of the first half.
	///
	/// \tparam ShapeType The shape type of the input array
	/// \tparam StorageScalar The scalar type of the input array
	/// \tparam StorageAllocator The allocator type of the input array
	/// \param array The input array
	/// \return The discrete Fourier transform of the input array
	template<typename ShapeType, typename StorageScalar, typename StorageAllocator>
	LIBRAPID_NODISCARD Array<Complex<StorageScalar>, backend::CPU>
	rfft(array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>> &array) {
		LIBRAPID_ASSERT(array.ndim() == 1, "RFFT only implemented for 1D arrays");
		int64_t outSize = array.shape()[0] / 2 + 1;
		Array<Complex<StorageScalar>, backend::CPU> res(Shape({outSize}));
		StorageScalar *input		   = array.storage().begin();
		Complex<StorageScalar> *output = res.storage().begin();
		detail::cpu::rfft(output, input, array.shape()[0]);
		return res;
	}

#if defined(LIBRAPID_HAS_CUDA)
	template<typename Scalar>
	LIBRAPID_NODISCARD Array<Complex<Scalar>, backend::CUDA> rfft(Array<Scalar, backend::CUDA> &array) {
		LIBRAPID_ASSERT(array.ndim() == 1, "RFFT only implemented for 1D arrays");
		int64_t outSize = array.shape()[0] / 2 + 1;
		Array<Complex<Scalar>, backend::CUDA> res(Shape({outSize}));
		Scalar *input			= array.storage().begin().get();
		Complex<Scalar> *output = res.storage().begin().get();
		detail::gpu::rfft(output, input, array.shape()[0]);
		return res;
	}
#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::fft

#endif // LIBRAPID_ARRAY_FOURIER_TRANFORM_HPP