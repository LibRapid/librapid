#pragma once

#include "../../../internal/config.hpp"

namespace librapid::functors::misc {
	template<typename Type_>
	class FillArray {
	public:
		using Type = Type_;
		// Use base scalar specially for boolean arrays
		using Scalar				   = typename internal::traits<Type_>::BaseScalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = 0;

		FillArray() = default;

		explicit FillArray(const Type &val) : m_val(val) {}

		FillArray(const FillArray<Type> &other) = default;

		FillArray &operator=(const FillArray &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return m_val; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return m_val;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const {
			// This should be compiled down to just "m_val"
			return fmt::format("{0}({1}) + {0}(0) *", internal::traits<Type>::Name, m_val);
		}

		template<typename T, int64_t d, int64_t a>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d, a> &extent) const {
			return extent;
		}

	private:
		const Type m_val;
	};
} // namespace librapid::functors::misc
