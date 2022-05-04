#pragma once

#include "../internal/config.hpp"
#include "traits.hpp"
#include "helpers/extent.hpp"
#include "helpers/internalUtils.hpp"
#include "arrayBase.hpp"
#include "cwisebinop.hpp"
#include "denseStorage.hpp"
#include "commaInitializer.hpp"

namespace librapid {
	namespace internal {
		template<typename Scalar_, typename Device_>
		struct traits<Array<Scalar_, Device_>> {
			using Scalar				   = Scalar_;
			using Device				   = Device_;
			using Packet				   = typename traits<Scalar>::Packet;
			using StorageType			   = memory::DenseStorage<Scalar, Device>;
			static constexpr int64_t Flags = 0;
		};
	} // namespace internal

	template<typename Scalar_, typename Device_ = device::CPU>
	class Array : public ArrayBase<Array<Scalar_, Device_>, Device_> {
	public:
#if !defined(LIBRAPID_HAS_CUDA)
		static_assert(std::is_same_v<Device_, device::CPU>, "CUDA support was not enabled");
#endif

		using Scalar	  = Scalar_;
		using Device	  = Device_;
		using Packet	  = typename internal::traits<Scalar>::Packet;
		using Type		  = Array<Scalar, Device>;
		using Base		  = ArrayBase<Type, Device>;
		using ExtentType  = Extent<int64_t, 32>;
		using StorageType = typename internal::traits<Type>::StorageType;

		Array() = default;

		template<typename T, int64_t d>
		explicit Array(const Extent<T, d> &extent) : Base(extent) {}

		template<typename OtherDerived>
		explicit Array(const OtherDerived &other) : Base(other.extent()) {
			Base::assign(other);
		}

		Array &operator=(const Array<Scalar, Device> &other) { return Base::assign(other); }

		template<typename OtherDerived>
		Array &operator=(const OtherDerived &other) {
			using ScalarOther = typename internal::traits<OtherDerived>::Scalar;
			static_assert(std::is_same_v<Scalar, ScalarOther>,
						  "Cannot assign Arrays with different types. Please use Array::cast<T>()");

			return Base::assign(other);
		}

		internal::CommaInitializer<Type> operator<<(const Scalar &value) {
			return internal::CommaInitializer<Type>(*this, value);
		}

		LR_NODISCARD("") Scalar &operator[](int64_t index) const {
			return Base::storage().get(index);
		}

		template<typename... T>
		LR_NODISCARD("")
		const Scalar &operator()(T... indices) const {
			LR_ASSERT(sizeof...(T) == Base::extent().dims(),
					  "Array with {0} dimensions requires {0} access indices. Received {1}",
					  Base::extent().dims(),
					  sizeof...(indices));

			int64_t index = internal::extentIndexProd(Base::extent(), 0, indices...);
			return Base::storage().get(index);
		}

		template<typename... T>
		LR_NODISCARD("")
		Scalar &operator()(T... indices) {
			LR_ASSERT(sizeof...(T) == Base::extent().dims(),
					  "Array with {0} dimensions requires {0} access indices. Received {1}",
					  Base::extent().dims(),
					  sizeof...(indices));

			int64_t index = internal::extentIndexProd(Base::extent(), 0, indices...);
			return Base::storage().get(index);
		}

		LR_FORCE_INLINE void writePacket(int64_t index, const Packet &p) {
			LR_ASSERT(
			  index >= 0 && index < Base::extent().size(), "Index {} is out of range", index);
			p.store(Base::storage().heap() + index);
		}

		LR_FORCE_INLINE void writeScalar(int64_t index, const Scalar &s) {
			Base::storage()[index] = s;
		}

		LR_NODISCARD("") std::string str() const {
			std::string res = "[";
			for (int64_t i = 0; i < Base::extent().size(); ++i) {
				res += fmt::format("{}, ", Base::storage().get(i));
			}
			return res;
		}

	private:
	};
} // namespace librapid
