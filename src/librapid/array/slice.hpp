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

	class Slice {
	public:
		Slice()							= default;
		Slice(const Slice &)			= default;
		Slice(Slice &&)					= default;
		Slice &operator=(const Slice &) = default;
		Slice &operator=(Slice &&)		= default;

		Slice &start(const Extent &start) {
			m_start = start;
			return *this;
		}

		Slice &stop(const Extent &stop) {
			m_stop = stop;
			return *this;
		}

		Slice &stride(const Extent &step) {
			m_stride = step;
			return *this;
		}

		LR_NODISCARD("") const Extent &start() const { return m_start; }
		LR_NODISCARD("") Extent &start() { return m_start; }
		LR_NODISCARD("") const Extent &stop() const { return m_stop; }
		LR_NODISCARD("") Extent &stop() { return m_stop; }
		LR_NODISCARD("") const Extent &stride() const { return m_stride; }
		LR_NODISCARD("") Extent &stride() { return m_stride; }

	private:
		Extent m_start;
		Extent m_stop;
		Extent m_stride;
	};

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

		ArraySlice(const Derived &derived, const Slice &slice) :
				Base(derived.extent()), m_derived(derived), m_slice(slice) {
			// Calculate the extent of the resulting array
			std::vector<i64> tmpExtent(derived.ndim());
			for (i64 i = 0; i < tmpExtent.size(); ++i) {
				tmpExtent[i] = (m_slice.stop()[i] - m_slice.start()[i]) / m_slice.stride()[i];
			}
			m_extent = Extent(tmpExtent);
		}

		ArraySlice(const ArraySlice &other) : Base(other.extent()) { m_derived = other.m_derived; }

		ArraySlice(ArraySlice &&other) : Base(other.extent()) {
			m_derived = std::move(other.m_derived);
		}

		ArraySlice &operator=(const ArraySlice &other) {
			m_derived = other.m_derived;
			return *this;
		}

		ArraySlice &operator=(ArraySlice &&other) noexcept {
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
			return m_derived(tmpIndex);
		}

		LR_NODISCARD("")
		Scalar operator()(const Extent &index) const {
			Extent tmpIndex(index);
			for (i64 i = 0; i < tmpIndex.ndim(); ++i) {
				tmpIndex[i] = tmpIndex[i] * m_slice.stride()[i] + m_slice.start()[i];
			}
			return m_derived(tmpIndex);
		}

		// Evaluate the slice and return a new array
		LR_NODISCARD("") Array<Scalar, Device> eval() const {
			Array<Scalar, Device> result(m_extent);
			auto resStorage = result.storage();
			for (i64 i = 0; i < m_extent.size(); ++i) {
				resStorage[i]		= at(i);
			}
			return result;
		}

	private:
		LR_NODISCARD("")
		Scalar at(i64 index) const {
			Extent tmpIndex = m_extent.reverseIndex(index);
			for (i64 i = 0; i < tmpIndex.ndim(); ++i) {
				tmpIndex[i] = tmpIndex[i] * m_slice.stride()[i] + m_slice.start()[i];
			}
			return m_derived.scalar(m_derived.extent().index(tmpIndex));
		}

	protected:
		Derived m_derived;
		Slice m_slice;
		Extent m_extent;
	};
} // namespace librapid