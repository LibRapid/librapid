#ifndef LIBRAPID_ARRAY_UTILS
#define LIBRAPID_ARRAY_UTILS

#include <librapid/config.hpp>
#include <librapid/autocast/autocast.hpp>

namespace librapid {
	namespace imp {
		/**
		 * \rst
		 *
		 * Calculate the number of digits before and after the decimal point
		 * of a number. If the number is an integer, there are zero digits after
		 * the decimal point.
		 *
		 * For example, the number :math:`3.141` has one digit before the decimal
		 * point, and three digits after it.
		 *
		 * \endrst
		 */
		inline void autocastBeforeAfterDecimal(const RawArray &src,
											   std::pair<int64_t, int64_t> &res) {
			std::stringstream stream;
			stream.precision(10);

			stream << std::boolalpha;

			if (src.location == Accelerator::CPU) {
				std::visit([&](auto *value) {
					stream << *value;
				}, src.data);
			}
#ifdef LIBRAPID_HAS_CUDA
			else {
				std::visit([&](auto *value) {
					using A = std::remove_pointer<decltype(value)>::type;

					A tmp;

#ifdef LIBRAPID_CUDA_STREAM
					cudaSafeCall(cudaMemcpyAsync(&tmp, value, sizeof(A),
												 cudaMemcpyDeviceToHost, cudaStream));
#else
					cudaSafeCall(cudaMemcpy(tmp, value, sizeof(A),
								 cudaMemcpyDeviceToHost));
#endif // LIBRAPID_CUDA_STREAM

					stream << tmp;
				}, src.data);
			}
#else
			else
			{
				throw std::invalid_argument("CUDA support was not enabled, so an"
											" Array on the GPU cannot be printed");
			}
#endif

			std::string str = stream.str();

			int64_t index;

			// Align the +/- for complex datatypes
			if (src.dtype == Datatype::CFLOAT64) {
				index = str.find('+', 1);
				if (index == std::string::npos)
					index = str.find('-', 1);

				res = {index, str.length() - index - 1};
				return;
			}

			if (isFloating(src.dtype) && str.find_last_of('.') == std::string::npos) {
				res = {str.length(), 0};
				return;
			}

			index = str.find_last_of('.');
			if (index == std::string::npos) {
				res = {str.length(), 0};
				return;
			}

			res = {index, str.length() - index - 1};
		}

		inline void autocastFormatValue(const RawArray &src, std::string &res) {
			std::stringstream stream;
			stream.precision(10);

			stream << std::boolalpha;

			if (src.location == Accelerator::CPU) {
				std::visit([&](auto *value) {
					// if (src.dtype == Datatype::INT8 ||
					// 	src.dtype == Datatype::UINT8)
					// {
					// 	stream << (int) *value;
					// }
					// else
					// {
					stream << *value;
					// }
				}, src.data);
			}
#ifdef LIBRAPID_HAS_CUDA
			else {
				std::visit([&](auto *value) {
					using A = std::remove_pointer<decltype(value)>::type;

					auto tmp = (A *) malloc(sizeof(A));

					if (tmp == nullptr)
						throw std::bad_alloc();

#ifdef LIBRAPID_CUDA_STREAM
					cudaSafeCall(cudaMemcpyAsync(tmp, value, sizeof(A),
												 cudaMemcpyDeviceToHost, cudaStream));
#else
					cudaSafeCall(cudaMemcpy(tmp, value, sizeof(A), cudaMemcpyDeviceToHost));
#endif // LIBRAPID_CUDA_STREAM

					if (std::is_same<A, int8_t>::value ||
						std::is_same<A, uint8_t>::value)
						stream << (int) *tmp;
					else
						stream << *tmp;

					free(tmp);
				}, src.data);
			}
#endif

			res = stream.str();
			if (isFloating(src.dtype) && res.find_last_of('.') == std::string::npos)
				res += ".";
		}

		template<typename _Ty>
		void arrayOpEq(void *dataStart, Accelerator location, const _Ty &val) {
			if (location == Accelerator::CPU) {
				*((_Ty *) dataStart) = val;
			} else {
#ifdef LIBRAPID_HAS_CUDA
#ifdef LIBRAPID_CUDA_STREAM
				cudaSafeCall(cudaMemcpyAsync((_Ty *) dataStart, &val, sizeof(char),
											 cudaMemcpyHostToDevice, cudaStream));
#else
				cudaSafeCall(cudaDeviceSynchronize());
				cudaSafeCall(cudaMemcpy((_Ty *) dataStart, &val, sizeof(char),
							 cudaMemcpyHostToDevice));
#endif // LIBRAPID_CUDA_STREAM
#endif // LIBRAPID_HAS_CUDA
			}
		}
	}
}

#endif // LIBRAPID_ARRAY_UTILS