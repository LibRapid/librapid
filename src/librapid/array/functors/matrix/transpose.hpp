#pragma once

#include "../../../internal/config.hpp"
#include "../../traits.hpp"
#include "../../helpers/extent.hpp"
#include "../../../modified/modified.hpp"

namespace librapid::functors::matrix {
	template<typename Type_>
	class Transpose {
	public:
		using Type					   = Type_;
		using Scalar				   = typename internal::traits<Type_>::Scalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = internal::flags::Matrix | internal::flags::Unary |
										 internal::flags::HasCustomEval |
										 internal::flags::RequireInput;
		// | internal::flags::RequireEval;

		Transpose() = default;

		template<typename T, int64_t d>
		explicit Transpose(const ExtentType<T, d> &order) : m_order(order) {};

		Transpose(const Transpose<Type> &other) = default;

		Transpose<Type> &operator=(const Transpose<Type> &other) {
			m_order = other.m_order;
			return *this;
		}

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return 0; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return 1;
		}

		template<typename Derived>
		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOpInput(const Derived &other, int64_t index) const {
			return 123;

			auto extent = other.extent();
			auto size = extent.size();
			auto swivelled = extent.reverseIndex(index).swivel(m_order);
			auto first = extent.index(swivelled);
			return other.scalar(first);
		}

		template<typename Derived>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOpInput(const Derived &other, int64_t index) const {
			Scalar buffer[internal::traits<Scalar>::PacketWidth];
			auto extent = other.extent();
			auto size = extent.size();
			auto swivelled = extent.reverseIndex(index).swivel(m_order);
			auto first = extent.index(swivelled);

			for (int64_t i = 0; i < internal::traits<Scalar>::PacketWidth; ++i) {
				buffer[i] = other.scalar(first);
				first += extent[1];
			}

			Packet res(&(buffer[0]));
			return res;
		}

		template<typename Input, typename Output>
		void customEval(const Input &input_, Output &output) const {
			auto input = input_.eval();

			using InputDevice  = typename internal::traits<Input>::Device;
			using OutputDevice = typename internal::traits<Output>::Device;
			using Device	   = typename memory::PromoteDevice<InputDevice, OutputDevice>;

			ExtentType<int64_t, 32> extent = input.extent();
			int64_t dims				   = extent.dims();

			ExtentType<int64_t, 32> inputStride	 = extent.stride();
			ExtentType<int64_t, 32> outputStride = output.extent().stride().swivel(m_order);

			auto coord = ExtentType<int64_t, 32>::zero(dims);
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

				// Only works in-place

				// Scalar *buffer = memory::malloc<Scalar, Device>(extent.size());
				// detail::transpose(true, inputData, extent[0], extent[1], outputData);
				// return;
				// memory::free<Scalar, Device>(buffer);
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
		ExtentType<T, d> genExtent(const ExtentType<T, d> &extent) const {
			return extent.swivel(m_order);
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const {
			LR_ASSERT(false, "Array transpose has no dedicated GPU kernel");
			return "ERROR";
		}

	private:
		ExtentType<int64_t, 32> m_order;
	};
} // namespace librapid::functors::matrix
