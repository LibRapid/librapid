#pragma once

#include "../../../internal/config.hpp"
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
										 internal::flags::RequireInput |
										 internal::flags::NoPacketOp;

		Transpose() = default;

		template<typename T, int64_t d>
		explicit Transpose(const ExtentType<T, d> &order) : m_order(order) {};

		Transpose(const Transpose<Type> &other) = default;

		Transpose<Type> &operator=(const Transpose<Type> &other) = default;

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
			auto extent	   = other.extent();
			auto swivelled = extent.reverseIndex(index).swivelled(m_order);
			auto first	   = extent.index(swivelled);
			return other.scalar(first);
		}

		template<typename Derived>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOpInput(const Derived &other, int64_t index) const {
			/*
			 * This works great if all Array dimensions are a multiple of the packet size, otherwise
			 * it's entirely useless and doesn't work at all :)
			using BaseScalar = typename internal::traits<Scalar>::BaseScalar;
			BaseScalar buffer[internal::traits<BaseScalar>::PacketWidth];
			auto extent	   = other.extent();
			auto size	   = extent.size();
			auto swivelled = extent.reverseIndexAdjusted(index).swivelled(m_order);
			auto first	   = extent.indexAdjusted(swivelled);
			auto stride	   = extent.strideAdjusted();
			auto inc	   = stride[m_order[extent.dims() - 1]];

			if constexpr (std::is_same_v<Scalar, bool>) {
				LR_ASSERT(false, "Boolean Arrays do not currently support Matrix operations");
			} else {
				for (int64_t i = 0; i < internal::traits<BaseScalar>::PacketWidth; ++i) {
					buffer[i] = other.scalar(first);
					first += inc;
				}
			}

			Packet res(&(buffer[0]));
			return res;
			 */

			return -1;
		}

		template<typename T, int64_t d, int64_t a>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d, a> &extent) const {
			return extent.swivelled(m_order);
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const {
			LR_ASSERT(false, "Array transpose has no dedicated GPU kernel");
			return "ERROR";
		}

	private:
		ExtentType<int64_t, 32> m_order;
	};
} // namespace librapid::functors::matrix
