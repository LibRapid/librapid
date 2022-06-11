#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "helpers/extent.hpp"
#include "traits.hpp"
#include "functors/functors.hpp"
#include "cast.hpp"

#define IMPL_BINOP(NAME, TYPE)                                                                     \
	template<typename OtherDerived, typename OtherDevice>                                          \
	LR_NODISCARD("")                                                                               \
	auto NAME(const ArrayBase<OtherDerived, OtherDevice> &other) const {                           \
		using ScalarOther = typename internal::traits<OtherDerived>::Scalar;                       \
		using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;             \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, ScalarOther>, Derived, OtherDerived>;   \
		static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;                      \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
                                                                                                   \
		static_assert(                                                                             \
		  is_same_v<Scalar, ScalarOther>,                                                     \
		  "Cannot operate on Arrays with different types. Please use Array::cast<T>()");           \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		LR_ASSERT(extent() == other.extent(),                                                      \
				  "Arrays must have equal extents. Cannot operate on Arrays with {} and {}",       \
				  extent().str(),                                                                  \
				  other.extent().str());                                                           \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(derived(), other.derived()).eval();                                     \
		else                                                                                       \
			return RetType(derived(), other.derived());                                            \
	}

#define IMPL_BINOP_SCALAR(NAME, TYPE)                                                              \
	template<typename OtherScalar,                                                                 \
			 typename enable_if_t<internal::traits<OtherScalar>::IsScalar, int> = 0>          \
	LR_NODISCARD("")                                                                               \
	auto NAME(const OtherScalar &other) const {                                                    \
		using ResDevice = Device;                                                                  \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, OtherScalar>, Derived, OtherScalar>;    \
		static constexpr uint64_t Flags	   = internal::traits<OtherScalar>::Flags;                 \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(derived(), other).eval();                                               \
		else                                                                                       \
			return RetType(derived(), other);                                                      \
	}

#define IMPL_BINOP_SCALAR_EXTERNAL(NAME, TYPE)                                                     \
	template<typename OtherScalar,                                                                 \
			 typename Derived,                                                                     \
			 typename Device,                                                                      \
			 typename enable_if_t<internal::traits<OtherScalar>::IsScalar, int> = 0>          \
	LR_NODISCARD("")                                                                               \
	auto NAME(const OtherScalar &other, const ArrayBase<Derived, Device> &arr) {                   \
		using Scalar	= typename internal::traits<Derived>::Scalar;                              \
		using ResDevice = Device;                                                                  \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, OtherScalar>, OtherScalar, Derived>;    \
		static constexpr uint64_t Flags	   = internal::traits<OtherScalar>::Flags;                 \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(other, arr.derived()).eval();                                           \
		else                                                                                       \
			return RetType(other, arr.derived());                                                  \
	}

#define IMPL_UNOP(NAME, TYPE)                                                                        \
	LR_NODISCARD("")                                                                                 \
	auto NAME() const {                                                                              \
		using RetType					   = unop::CWiseUnop<functors::unop::TYPE<Scalar>, Derived>; \
		static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;                        \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;        \
                                                                                                     \
		static_assert(!(Required & ~(Flags & Required)),                                             \
					  "Scalar type is incompatible with Functor");                                   \
                                                                                                     \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))               \
			return RetType(derived()).eval();                                                        \
		else                                                                                         \
			return RetType(derived());                                                               \
	}

namespace librapid {
	namespace internal {
		template<typename Derived>
		struct traits<ArrayBase<Derived, device::CPU>> {
			static constexpr bool IsScalar	= false;
			using Valid						= std::true_type;
			using Scalar					= typename traits<Derived>::Scalar;
			using BaseScalar				= typename traits<Scalar>::BaseScalar;
			using Device					= device::CPU;
			using StorageType				= memory::DenseStorage<Scalar, device::CPU>;
			static constexpr uint64_t Flags = traits<Derived>::Flags;
		};

		template<typename Derived>
		struct traits<ArrayBase<Derived, device::GPU>> {
			using Scalar					= typename traits<Derived>::Scalar;
			using Device					= device::GPU;
			using StorageType				= memory::DenseStorage<Scalar, device::CPU>;
			static constexpr uint64_t Flags = traits<Derived>::Flags;
		};
	} // namespace internal

	template<typename Derived, typename Device>
	class ArrayBase {
	public:
		using Scalar					= typename internal::traits<Derived>::Scalar;
		using This						= ArrayBase<Derived, Device>;
		using Packet					= typename internal::traits<Derived>::Packet;
		using StorageType				= typename internal::traits<Derived>::StorageType;
		static constexpr uint64_t Flags = internal::traits<This>::Flags;

		friend Derived;

		ArrayBase() = default;

		template<typename T_, int64_t d_>
		explicit ArrayBase(const ExtentType<T_, d_> &extent) :
				m_isScalar(extent.size() == 0), m_extent(extent), m_storage(extent.size()) {}

		template<typename T_, int64_t d_>
		explicit ArrayBase(const ExtentType<T_, d_> &extent, int) :
				m_isScalar(extent.size() == 0), m_extent(extent) {}

		template<typename OtherDerived>
		ArrayBase(const OtherDerived &other) {
			assign(other);
		}

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

		IMPL_BINOP(operator+, ScalarSum)
		IMPL_BINOP(operator-, ScalarDiff)
		IMPL_BINOP(operator*, ScalarProd)
		IMPL_BINOP(operator/, ScalarDiv)

		IMPL_BINOP_SCALAR(operator+, ScalarSum)
		IMPL_BINOP_SCALAR(operator-, ScalarDiff)
		IMPL_BINOP_SCALAR(operator*, ScalarProd)
		IMPL_BINOP_SCALAR(operator/, ScalarDiv)

		IMPL_BINOP(operator|, BitwiseOr)
		IMPL_BINOP(operator&, BitwiseAnd)
		IMPL_BINOP(operator^, BitwiseXor)

		IMPL_UNOP(operator-, UnaryMinus)
		IMPL_UNOP(operator~, BitwiseNot)
		IMPL_UNOP(operator!, UnaryNot)

		template<typename T = int64_t, int64_t d = 32>
		auto transposed(const ExtentType<T, d> &order_ = {}) const {
			using RetType = unop::CWiseUnop<functors::matrix::Transpose<Derived>, Derived>;
			static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;
			static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;

			static_assert(!(Required & ~(Flags & Required)),
						  "Scalar type is incompatible with Functor");

			ExtentType<int64_t, 32> order;
			if (order_.dims() == -1) {
				// Default order is to reverse all indices
				order = ExtentType<int64_t, 32>::zero(m_extent.dims());
				for (int64_t i = 0; i < m_extent.dims(); ++i) {
					order[m_extent.dims() - i - 1] = i;
				}
			} else {
				order = order_;
			}

			auto res = (1 * *this).eval();
			res.transpose(order);
			return res;
		}

		LR_NODISCARD("Do not ignore the result of an evaluated calculation")
		auto eval() const { return derived(); }

		template<typename OtherDerived>
		LR_FORCE_INLINE void loadFrom(int64_t index, const OtherDerived &other) {
			LR_ASSERT(index >= 0 && index < m_extent.size(),
					  "Index {} is out of range for Array with extent {}",
					  index,
					  m_extent.str());
			derived().writePacket(index, other.packet(index));
		}

		template<typename ScalarType>
		LR_FORCE_INLINE void loadFromScalar(int64_t index, const ScalarType &other) {
			LR_ASSERT(index >= 0 && index < m_extent.size(),
					  "Index {} is out of range for Array with extent {}",
					  index,
					  m_extent.str());
			derived().writeScalar(index, other.scalar(index));
		}

		LR_FORCE_INLINE Derived &assign(const Scalar &other) {
			// Construct if necessary
			if (!m_storage) {
				m_extent   = Extent(1);
				m_storage  = StorageType(m_extent.size());
				m_isScalar = true;
			}

			LR_ASSERT(m_isScalar, "Cannot assign Scalar to non-scalar Array");
			m_storage[0] = other;
			return derived();
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE Derived &assign(const OtherDerived &other) {
			// Construct if necessary
			if (!m_storage) {
				m_extent  = other.extent();
				m_storage = StorageType(m_extent.size());
			}

			LR_ASSERT(m_extent == other.extent(),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			m_isScalar	   = other.isScalar();
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

		LR_FORCE_INLINE Packet packet(int64_t index) const {
			Packet p;
			if constexpr (is_same_v<Scalar, bool>)
				p.load(m_storage.heap() + (index / 64));
			else
				p.load(m_storage.heap() + index);
			return p;
		}

		LR_FORCE_INLINE Scalar scalar(int64_t index) const { return m_storage[index]; }

		template<typename T>
		std::string genKernel(std::vector<T> &vec, int64_t &index) const {
			vec.emplace_back(m_storage.heap());
			return fmt::format("arg{}", index++);
		}

		LR_NODISCARD("") LR_FORCE_INLINE Derived &derived() {
			return *static_cast<Derived *>(this);
		}

		LR_NODISCARD("") bool isScalar() const { return m_isScalar; }
		LR_NODISCARD("") const StorageType &storage() const { return m_storage; }
		LR_NODISCARD("") StorageType &storage() { return m_storage; }
		LR_NODISCARD("") ExtentType<int64_t, 32> extent() const { return m_extent; }
		LR_NODISCARD("") ExtentType<int64_t, 32> &extent() { return m_extent; }

	private:
		bool m_isScalar = false;
		ExtentType<int64_t, 32> m_extent;
		StorageType m_storage;
	};

	IMPL_BINOP_SCALAR_EXTERNAL(operator+, ScalarSum)
	IMPL_BINOP_SCALAR_EXTERNAL(operator-, ScalarDiff)
	IMPL_BINOP_SCALAR_EXTERNAL(operator*, ScalarProd)
	IMPL_BINOP_SCALAR_EXTERNAL(operator/, ScalarDiv)
} // namespace librapid

#undef IMPL_BINOP
#undef IMPL_BINOP_SCALAR
#undef IMPL_BINOP_SCALAR_EXTERNAL
#undef IMPL_UNOP