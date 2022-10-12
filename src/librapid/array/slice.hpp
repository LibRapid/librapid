#pragma once

namespace librapid {
	namespace internal {
		template<typename Derived>
		struct traits<ArraySlice<Derived>> {
			static constexpr bool IsScalar	  = false;
			static constexpr bool IsEvaluated = false;
			using Valid						  = std::true_type;
			using Type						  = ArraySlice<Derived>;
			using Scalar					  = typename traits<Derived>::Scalar;
			using BaseScalar				  = typename traits<Scalar>::BaseScalar;
			using Packet					  = typename traits<Scalar>::Packet;
			using Device					  = typename traits<Derived>::Device;
			using StorageType				  = memory::DenseStorage<Scalar, Device>;
			static constexpr ui64 Flags		  = (traits<Derived>::Flags) & ~flags::Evaluated;
		};
	} // namespace internal

	// Slice of an Array object
	template<typename Derived>
	class ArraySlice
			: public ArrayBase<ArraySlice<Derived>, typename internal::traits<Derived>::Device> {
	public:
		using Scalar				= typename internal::traits<Derived>::Scalar;
		using Packet				= typename internal::traits<Derived>::Packet;
		using Device				= typename internal::traits<Derived>::Device;
		using Type					= ArraySlice<Derived>;
		using Base					= ArrayBase<Type, Device>;
		static constexpr ui64 Flags = internal::traits<Derived>::Flags;

		ArraySlice() = delete;

		ArraySlice(const Derived &derived, const Extent &offset, const Extent &stride) :
				Base(derived.extent()), m_derived(derived), m_offset(offset), m_stride(stride) {}

		ArraySlice(const ArraySlice &other) : Base(other.extent()) { m_derived = other.m_derived; }

		ArraySlice(ArraySlice &&other) : Base(other.extent()) {
			m_derived = std::move(other.m_derived);
		}

		ArraySlice &operator=(const ArraySlice &other) {
			m_derived = other.m_derived;
			return *this;
		}

		ArraySlice &operator=(ArraySlice &&other) {
			m_derived = std::move(other.m_derived);
			return *this;
		}

		LR_NODISCARD("") ArraySlice operator[](i64 index) const {
			LR_ASSERT(false, "NOT YET IMPLEMENTED");
			return ArraySlice(m_derived);
		}

		template<typename... T>
		LR_NODISCARD("")
		Scalar operator()(T... indices) const {
			Extent tmpIndex(indices...);
			for (i64 i = 0; i < tmpIndex.ndim(); ++i) {
				tmpIndex[i] = tmpIndex[i] * m_stride[i] + m_offset[i];
			}
			return m_derived(tmpIndex);
		}

		LR_NODISCARD("")
		Scalar operator()(const Extent &index) const {
			Extent tmpIndex(index);
			for (i64 i = 0; i < tmpIndex.ndim(); ++i) {
				tmpIndex[i] = tmpIndex[i] * m_stride[i] + m_offset[i];
			}
			return m_derived(tmpIndex);
		}

		// Evaluate the slice and return a new array
		LR_NODISCARD("") Array<Scalar, Device> eval() const {
			Array<Scalar, Device> result(m_extent);
			for (i64 i = 0; i < m_extent.size(); ++i) {
				Extent currentIndex = m_extent.reverseIndex(i);
				result(i)			= (*this)(currentIndex);
			}
			return result;
		}

	protected:
		Derived m_derived;
		Extent m_extent;
		Extent m_stride;
		Extent m_offset;
	};
} // namespace librapid