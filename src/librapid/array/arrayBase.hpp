#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "helpers/extent.hpp"
#include "traits.hpp"
#include "functors/functors.hpp"
#include "cast.hpp"

namespace librapid {
	namespace internal {
		template<typename Derived>
		struct traits<ArrayBase<Derived, device::CPU>> {
			using Scalar				= typename traits<Derived>::Scalar;
			using Device				= device::CPU;
			using StorageType			= memory::DenseStorage<Scalar, device::CPU>;
			static const uint64_t Flags = 0;
		};

		template<typename Derived>
		struct traits<ArrayBase<Derived, device::GPU>> {
			using Scalar				= typename traits<Derived>::Scalar;
			using Device				= device::GPU;
			using StorageType			= memory::DenseStorage<Scalar, device::CPU>;
			static const uint64_t Flags = 0;
		};
	} // namespace internal

	template<typename Derived, typename Device>
	class ArrayBase {
	public:
		using Scalar				= typename internal::traits<Derived>::Scalar;
		using This					= ArrayBase<Derived, Device>;
		using Packet				= typename internal::traits<Derived>::Packet;
		using StorageType			= typename internal::traits<Derived>::StorageType;
		static const uint64_t Flags = internal::traits<This>::Flags;

		ArrayBase() = default;

		template<typename T_, int64_t d_>
		explicit ArrayBase(const Extent<T_, d_> &extent) :
				m_isScalar(extent.size() == 0), m_extent(extent), m_storage(extent.size()) {}

		template<typename OtherDerived>
		LR_INLINE Derived &operator=(const OtherDerived &other) {
			return assign(other);
		}

		template<typename T>
		LR_NODISCARD("")
		LR_FORCE_INLINE auto cast() const {
			using ScalarType = typename internal::traits<T>::Scalar;
			using RetType	 = unary::Cast<ScalarType, Derived>;
			return RetType(derived());
		}

		template<typename OtherDerived, typename OtherDevice>
		LR_NODISCARD("Do not ignore the result of an arithmetic operation")
		auto operator+(const ArrayBase<OtherDerived, OtherDevice> &other) const {
			using ScalarOther = typename internal::traits<OtherDerived>::Scalar;
			using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;
			using RetType	  = binop::
			  CWiseBinop<functors::binary::ScalarSum<Scalar, ScalarOther>, Derived, OtherDerived>;

			static_assert(
			  std::is_same_v<Scalar, ScalarOther>,
			  "Cannot add Arrays with different data types. Please use Array::cast<T>()");

			return RetType(derived(), other.derived());
		}

		template<typename OtherDerived, typename OtherDevice>
		LR_NODISCARD("Do not ignore the result of an arithmetic operation")
		auto operator-(const ArrayBase<OtherDerived, OtherDevice> &other) const {
			using ScalarOther = typename internal::traits<OtherDerived>::Scalar;
			using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;
			using RetType	  = binop::
			  CWiseBinop<functors::binary::ScalarDiff<Scalar, ScalarOther>, Derived, OtherDerived>;

			static_assert(
			  std::is_same_v<Scalar, ScalarOther>,
			  "Cannot add Arrays with different data types. Please use Array::cast<T>()");

			return RetType(derived(), other.derived());
		}

		template<typename OtherDerived, typename OtherDevice>
		LR_NODISCARD("Do not ignore the result of an arithmetic operation")
		auto operator*(const ArrayBase<OtherDerived, OtherDevice> &other) const {
			using ScalarOther = typename internal::traits<OtherDerived>::Scalar;
			using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;
			using RetType	  = binop::
			  CWiseBinop<functors::binary::ScalarProd<Scalar, ScalarOther>, Derived, OtherDerived>;

			static_assert(
			  std::is_same_v<Scalar, ScalarOther>,
			  "Cannot add Arrays with different data types. Please use Array::cast<T>()");

			return RetType(derived(), other.derived());
		}

		template<typename OtherDerived, typename OtherDevice>
		LR_NODISCARD("Do not ignore the result of an arithmetic operation")
		auto operator/(const ArrayBase<OtherDerived, OtherDevice> &other) const {
			using ScalarOther = typename internal::traits<OtherDerived>::Scalar;
			using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;
			using RetType	  = binop::
			  CWiseBinop<functors::binary::ScalarDiv<Scalar, ScalarOther>, Derived, OtherDerived>;

			static_assert(
			  std::is_same_v<Scalar, ScalarOther>,
			  "Cannot add Arrays with different data types. Please use Array::cast<T>()");

			return RetType(derived(), other.derived());
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE void loadFrom(int64_t index, const OtherDerived &other) {
			LR_ASSERT(index >= 0 && index < m_extent.size(),
					  "Index {} is out of range for Array with extent {}",
					  index, m_extent.str());
			derived().writePacket(index, other.packet(index));
		}

		template<typename ScalarType>
		LR_FORCE_INLINE void loadFromScalar(int64_t index, const ScalarType &other) {
			LR_ASSERT(index >= 0 && index < m_extent.size(),
					  "Index {} is out of range for Array with extent {}",
					  index, m_extent.str());
			derived().writeScalar(index, other.scalar(index));
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE Derived &assign(const OtherDerived &other) {
			LR_ASSERT(m_extent == other.extent(),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			using Selector = functors::AssignSelector<Derived, OtherDerived, false>;
			return Selector::run(derived(), other.derived());
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE Derived &assignLazy(const OtherDerived &other) {
			LR_ASSERT(m_extent == other.extent(),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			using Selector = functors::AssignOp<Derived, OtherDerived>;
			Selector::run(derived(), other.derived());
			return derived();
		}

		LR_NODISCARD("") LR_FORCE_INLINE const Derived &derived() const {
			return *static_cast<const Derived *>(this);
		}

		LR_FORCE_INLINE virtual Packet packet(int64_t index) const {
			Packet p;
			p.load(m_storage.heap() + index);
			return p;
		}

		template<typename T>
		std::string genKernel(std::vector<T> &vec, int64_t &index) const {
			vec.emplace_back(m_storage.heap());
			return fmt::format("arg{}", index++);
		}

		LR_FORCE_INLINE virtual Scalar scalar(int64_t index) const { return m_storage[index]; }

		LR_NODISCARD("") LR_FORCE_INLINE Derived &derived() {
			return *static_cast<Derived *>(this);
		}

		LR_NODISCARD("") bool isScalar() const { return m_isScalar; }
		LR_NODISCARD("") const StorageType &storage() const { return m_storage; }
		LR_NODISCARD("") Extent<int64_t, 32> extent() const { return m_extent; }

	private:
		bool m_isScalar = false;
		Extent<int64_t, 32> m_extent;
		StorageType m_storage;
	};
} // namespace librapid