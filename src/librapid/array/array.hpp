#pragma once

#include "../internal/config.hpp"
#include "traits.hpp"
#include "helpers/extent.hpp"
#include "arrayBase.hpp"
#include "cwisebinop.hpp"
#include "denseStorage.hpp"

namespace librapid {
	namespace internal {
		template<typename Scalar_, typename Device_>
		struct traits<Array<Scalar_, Device_>> {
			using Scalar = Scalar_;
			using Device = Device_;
			using Packet = typename traits<Scalar>::Packet;
		};
	} // namespace internal

	template<typename Scalar_, typename Device_>
	class Array : public ArrayBase<Array<Scalar_, Device_>, Device_> {
	public:
		using Scalar	 = Scalar_;
		using Device	 = Device_;
		using Packet	 = typename internal::traits<Scalar>::Packet;
		using Type		 = Array<Scalar, Device>;
		using Base		 = ArrayBase<Type, Device>;
		using ExtentType = Extent<int64_t, 32>;

		Array() = default;

		template<typename T, int64_t d>
		explicit Array(const Extent<T, d> &extent) : Base(extent) {}

		template<typename OtherDerived>
		explicit Array(const OtherDerived &other) : Base(other.extent()) {
			Base::assign(other);
		}

		template<typename OtherDerived>
		Array &operator=(const OtherDerived &other) {
			return Base::assign(other);
		}

		LR_FORCE_INLINE void writePacket(int64_t index, const Packet &p) {
			LR_ASSERT(index >= 0 && index < Base::extent().size(), "Index {} is out of range", index);
			p.store(Base::storage().heap() + index);
		}

		LR_FORCE_INLINE void writeScalar(int64_t index, const Scalar &s) {
			Base::storage()[index] = s;
		}

		LR_NODISCARD("") std::string str() const {
			std::string res = "[";
			for (int64_t i = 0; i < Base::extent().size(); ++i) {
				res += fmt::format("{}, ", Base::storage().heap()[i]);
			}
			return res;
		}

	private:
	};
} // namespace librapid
