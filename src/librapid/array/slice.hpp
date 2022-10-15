#include <utility>

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
			static constexpr ui64 Flags =
			  (traits<Derived>::Flags | flags::NoPacketOp) & ~flags::Evaluated;
		};
	} // namespace internal

	class Slice {
	public:
		Slice() = default;
		Slice(const Extent &start, const Extent &stop, const Extent &stride) :
				m_start(start), m_stop(stop), m_stride(stride) {}
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

		ArraySlice() = default;

		ArraySlice(const Derived &derived, Slice slice) :
				Base(derived.extent()), m_derived(derived), m_slice(std::move(slice)) {
			// Check the slice is valid
			if (m_slice.start().ndim() == -1) {
				m_slice.start() = Extent(std::vector<i64>(derived.ndim(), AUTO));
			}

			if (m_slice.stop().ndim() == -1) {
				m_slice.stop() = Extent(std::vector<i64>(derived.ndim(), AUTO));
			}

			if (m_slice.stride().ndim() == -1) {
				m_slice.stride() = Extent(std::vector<i64>(derived.ndim(), AUTO));
			}

			// Calculate the extent of the resulting array
			std::vector<i64> tmpExtent(derived.ndim());
			for (i64 i = 0; i < tmpExtent.size(); ++i) {
				// Adjust the start and stop values to account for negative values
				if (m_slice.start()[i] < 0) { m_slice.start()[i] += derived.extent()[i]; }
				if (m_slice.stop()[i] < 0) { m_slice.stop()[i] += derived.extent()[i]; }

				// Automatic slice settings -- clip to the start/end of the array
				if (m_slice.start()[i] == AUTO) { m_slice.start()[i] = 0; }
				if (m_slice.stop()[i] == AUTO) {
					if (m_slice.stride()[i] < 0) {
						m_slice.stop()[i] = -1;
					} else {
						m_slice.stop()[i] = derived.extent()[i];
					}
				}
				i64 delta = m_slice.stop()[i] - m_slice.start()[i];
				if (m_slice.stride()[i] == AUTO) {
					m_slice.stride()[i] = internal::copySign(i64(1), delta);
				}

				tmpExtent[i] = roundUpTo(delta, m_slice.stride()[i]) / m_slice.stride()[i];

				LR_ASSERT(tmpExtent[i] > 0, "Invalid slice");
			}
			m_extent = Extent(tmpExtent);
		}

		ArraySlice(const ArraySlice &other) : Base(other.extent()) {
			m_derived = other.m_derived;
			m_slice	  = other.m_slice;
			m_extent  = other.m_extent;
		}

		ArraySlice(ArraySlice &&other) noexcept : Base(other.extent()) {
			m_derived = std::move(other.m_derived);
			m_slice	  = std::move(other.m_slice);
			m_extent  = std::move(other.m_extent);
		}

		// ArraySlice &operator=(const ArraySlice &other) {
		// 	m_derived = other.m_derived;
		// 	m_slice	  = other.m_slice;
		// 	m_extent  = other.m_extent;
		// 	return *this;
		// }

		// ArraySlice &operator=(ArraySlice &&other) noexcept {
		// 	m_derived = std::move(other.m_derived);
		// 	m_slice	  = std::move(other.m_slice);
		// 	m_extent  = std::move(other.m_extent);
		// 	return *this;
		// }

		template<typename OtherDerived>
		ArraySlice &operator=(const OtherDerived &other) {
			LR_ASSERT(other.extent() == m_extent, "Array extents must match");

			for (i64 i = 0; i < m_extent.size(); ++i) {
				Extent tmpIndex	  = m_extent.reverseIndex(i);
				(*this)(tmpIndex) = other.scalar(i);
			}
			return *this;
		}

		auto operator[](i64 index) const { return eval()[index]; }

		template<typename... T>
		LR_NODISCARD("")
		auto operator()(T... indices) const {
			Extent index(indices...);
			for (i64 i = 0; i < index.ndim(); ++i) {
				index[i] = index[i] * m_slice.stride()[i] + m_slice.start()[i];
			}
			return m_derived(index);
		}

		LR_NODISCARD("")
		auto operator()(Extent index) const {
			for (i64 i = 0; i < index.ndim(); ++i) {
				index[i] = index[i] * m_slice.stride()[i] + m_slice.start()[i];
			}
			return m_derived(index);
		}

		template<typename... T>
		LR_NODISCARD("")
		auto operator()(T... indices) {
			return const_cast<const Type *>(this)->operator()(indices...);
		}

		LR_NODISCARD("")
		auto operator()(const Extent &index) {
			return const_cast<const Type *>(this)->operator()(index);
		}

		// Evaluate the slice and return a new array
		LR_NODISCARD("") Array<Scalar, Device> eval() const {
			Array<Scalar, Device> result(m_extent);
			auto resStorage = result.storage();
			for (i64 i = 0; i < m_extent.size(); ++i) { resStorage[i] = at(i); }
			return result;
		}

		LR_NODISCARD("") LR_FORCE_INLINE Scalar scalar(i64 index) const { return at(index); }

		LR_NODISCARD("") LR_INLINE const Extent &extent() const { return m_extent; }
		LR_NODISCARD("") LR_INLINE Extent &extent() { return m_extent; }

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

template<typename Derived>
struct fmt::formatter<librapid::ArraySlice<Derived>> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::ArraySlice<Derived> &slice, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), slice.eval().str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};