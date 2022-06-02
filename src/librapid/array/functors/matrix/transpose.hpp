#pragma once

#include "../../../internal/config.hpp"
#include "../../traits.hpp"
#include "../../helpers/extent.hpp"

namespace librapid::functors::matrix {
	template<typename Type_>
	class Transpose {
	public:
		using Type	  = Type_;
		using Scalar  = typename internal::traits<Type_>::Scalar;
		using RetType = Scalar;
		using Packet  = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags =
		  internal::flags::Matrix | internal::flags::Unary | internal::flags::HasCustomEval |
		  internal::flags::SupportsScalar | internal::flags::RequireEval;

		Transpose() = default;

		template<typename T, int64_t d>
		explicit Transpose(const Extent<T, d> &order) : m_order(order) {};

		Transpose(const Transpose<Type> &other) = default;

		Transpose<Type> &operator=(const Transpose<Type> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return val; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return val;
		}

		template<typename Input, typename Output>
		void customEval(const Input &input_, Output &output) const {
			auto input = input_.eval();

			Extent<int64_t, 32> extent = input.extent();
			int64_t dims			   = extent.dims();

			Extent<int64_t, 32> inputStride	 = extent.stride();
			Extent<int64_t, 32> outputStride = output.extent().stride().swivel(m_order);

			auto coord = Extent<int64_t, 32>::zero(dims);
			int64_t idim;
			int64_t ndim = dims;

			// Reverse everything???
			auto tmpExtent		 = extent;
			auto tmpInputStride	 = inputStride;
			auto tmpOutputStride = outputStride;

			for (int64_t i = 0; i < dims; ++i) {
				tmpExtent[dims - i - 1]		  = extent[i];
				tmpInputStride[dims - i - 1]  = inputStride[i];
				tmpOutputStride[dims - i - 1] = outputStride[i];
			}

			const auto *__restrict inputData = input.storage().heap();
			auto *__restrict outputData		 = output.storage().heap();

			if (dims == 2 && m_order[0] == 1 && m_order[1] == 0) {
				// Faster for matrix transposition
				// This is disgusting, but it's faster than using OMPs "if"
				if (extent.size() < 500) {
					for (int64_t i = 0; i < extent.size(); ++i) {
						int64_t a	  = i / extent[0];
						int64_t b	  = i % extent[0];
						outputData[i] = inputData[b * extent[1] + a];
					}
				} else {
#pragma omp parallel for shared(inputData, outputData, extent) num_threads(matrixThreads)
					for (int64_t i = 0; i < extent.size(); ++i) {
						int64_t a	  = i / extent[0];
						int64_t b	  = i % extent[0];
						outputData[i] = inputData[b * extent[1] + a];
					}
				}
				return;
			}

			int64_t inputIndex	= 0;
			int64_t outputIndex = 0;

			do {
				outputData[outputIndex] = inputData[inputIndex];
				for (idim = 0; idim < ndim; ++idim) {
					if (++coord[idim] == tmpExtent[idim]) {
						coord[idim] = 0;
						inputIndex	= inputIndex - (tmpExtent[idim] - 1) * tmpInputStride[idim];
						outputIndex = outputIndex - (tmpExtent[idim] - 1) * tmpOutputStride[idim];
					} else {
						inputIndex	= inputIndex + tmpInputStride[idim];
						outputIndex = outputIndex + tmpOutputStride[idim];
						break;
					}
				}
			} while (idim < ndim);
		}

		template<typename T, int64_t d>
		LR_NODISCARD("")
		Extent<T, d> genExtent(const Extent<T, d> &extent) const {
			return extent.swivel(m_order);
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const {
			LR_ASSERT(false, "Array transpose has no dedicated GPU kernel");
			return "ERROR";
		}

	private:
		Extent<int64_t, 32> m_order;
	};
} // namespace librapid::functors::matrix