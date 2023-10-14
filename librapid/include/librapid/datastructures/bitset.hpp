#ifndef LIBRAPID_BITSET_HPP
#define LIBRAPID_BITSET_HPP

namespace librapid {
	template<uint64_t numBits_ = 64, bool stackAlloc_ = true>
	class BitSet {
	public:
		template<uint64_t otherBits, bool otherStackAlloc>
		using BitSetMerger =
		  BitSet<(numBits_ > otherBits ? numBits_ : otherBits), stackAlloc_ && otherStackAlloc>;

		friend BitSet<numBits_, !stackAlloc_>;

		using ElementType						 = uint64_t;
		static constexpr bool stackAlloc		 = stackAlloc_;
		static constexpr uint64_t bitsPerElement = sizeof(ElementType) * 8;
		static constexpr uint64_t numBits		 = numBits_;
		static constexpr uint64_t numElements	 = (numBits + bitsPerElement - 1) / bitsPerElement;
		using StorageType =
		  std::conditional_t<stackAlloc, std::array<ElementType, numElements>, ElementType *>;

		BitSet() { emptyInit(); }

		BitSet(const BitSet &other) {
			init();
			if constexpr (stackAlloc) {
				m_data = other.data();
			} else {
				for (uint64_t i = 0; i < numElements; ++i) { m_data[i] = other.data()[i]; }
			}
		}

		BitSet(BitSet &&other) = default;

		constexpr BitSet(uint64_t value) {
			static_assert(numElements > 0, "Not enough bits in BitSet");
			emptyInit();
			m_data[0] = value;
		}

		constexpr BitSet(const std::string &str, char zero = '0', char one = '1') {
			emptyInit();
			uint64_t stringLength = str.length();
			for (uint64_t i = 0; i < stringLength; ++i) {
				if (str[i] == zero)
					continue;
				else if (str[i] == one)
					set(stringLength - i - 1, true);
				else
					LIBRAPID_ERROR("Invalid character in BitSet string: {}", str[i]);
			}
		}

		BitSet &operator=(const BitSet &other) = default;
		BitSet &operator=(BitSet &&other)	   = default;

		~BitSet() {
			if constexpr (!stackAlloc) delete[] m_data;
		}

		BitSet &set(uint64_t index, bool value) {
			uint64_t element = index / bitsPerElement;
			uint64_t bit	 = index % bitsPerElement;
			if (value)
				m_data[element] |= (1ULL << bit);
			else
				m_data[element] &= ~(1ULL << bit);
			return *this;
		}

		BitSet &set(uint64_t start, uint64_t end, bool value) {
			if (start > end) std::swap(start, end);
			uint64_t blockStart = start / bitsPerElement + 1;
			uint64_t blockEnd	= end / bitsPerElement;

			// 1. Set the bits before the blocked elements
			// 2. Set the blocked elements
			// 3. Set the bits after the blocked elements

			for (uint64_t i = start; i < end && i < blockStart * bitsPerElement; ++i) set(i, value);
			for (uint64_t i = blockStart; i < blockEnd; ++i) m_data[i] = value ? ~0ULL : 0ULL;
			for (uint64_t i = blockEnd * bitsPerElement; i < end; ++i) set(i, value);

			return *this;
		}

		LIBRAPID_NODISCARD bool get(uint64_t index) const {
			uint64_t element = index / bitsPerElement;
			uint64_t bit	 = index % bitsPerElement;
			return m_data[element] & (1ULL << bit);
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool any() const {
			for (uint64_t i = 0; i < numElements; ++i)
				if (m_data[i]) return true;
			return false;
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool all() const {
			for (uint64_t i = 0; i < numElements; ++i)
				if (m_data[i] != ~(ElementType(0))) return false;
			return true;
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool none() const {
			for (uint64_t i = 0; i < numElements; ++i)
				if (m_data[i]) return false;
			return true;
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE uint64_t first() const {
			for (uint64_t i = 0; i < numElements; ++i) {
				if (m_data[i]) {
#if defined(LIBRAPID_CLANG) || defined(LIBRAPID_GCC)
					if constexpr (sizeof(ElementType) == 8) {
						return __builtin_ctzll(m_data[i]) + i * bitsPerElement;
					} else if constexpr (sizeof(ElementType) == 4) {
						return __builtin_ctz(m_data[i]) + i * bitsPerElement;
					} else if constexpr (sizeof(ElementType) == 2) {
						return __builtin_ctzs(m_data[i]) + i * bitsPerElement;
					} else if constexpr (sizeof(ElementType) == 1) {
						return __builtin_ctz(m_data[i]) + i * bitsPerElement;
					}
#elif defined(LIBRAPID_MSVC)
					unsigned long index;
					if constexpr (sizeof(ElementType) == 8) {
						_BitScanForward64(&index, m_data[i]);
					} else {
						_BitScanForward(&index, m_data[i]);
					}
					return index + i * bitsPerElement;
#else
					for (uint64_t j = 0; j < bitsPerElement; ++j) {
						if (m_data[i] & (1ULL << j)) return i * bitsPerElement + j;
					}
#endif
				}
			}
			return numBits;
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE uint64_t last() const {
			for (uint64_t i = numElements - 1; i >= 0; --i) {
				if (m_data[i]) {
#if defined(LIBRAPID_CLANG) || defined(LIBRAPID_GCC)
					if constexpr (sizeof(ElementType) == 8) {
						return 63 - __builtin_clzll(m_data[i]) + i * bitsPerElement;
					} else if constexpr (sizeof(ElementType) == 4) {
						return 31 - __builtin_clz(m_data[i]) + i * bitsPerElement;
					} else if constexpr (sizeof(ElementType) == 2) {
						return 15 - __builtin_clzs(m_data[i]) + i * bitsPerElement;
					} else if constexpr (sizeof(ElementType) == 1) {
						return 7 - __builtin_clz(m_data[i]) + i * bitsPerElement;
					}
#elif defined(LIBRAPID_MSVC)
					unsigned long index;
					if constexpr (sizeof(ElementType) == 8) {
						_BitScanReverse64(&index, m_data[i]);
					} else {
						_BitScanReverse(&index, m_data[i]);
					}
					return index + i * bitsPerElement;
#else
					for (uint64_t j = bitsPerElement - 1; j >= 0; --j) {
						if (m_data[i] & (1ULL << j)) return i * bitsPerElement + j;
					}
#endif
				}
			}
			return numBits;
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE uint64_t popCount() const {
			uint64_t res = 0;
			for (uint64_t i = 0; i < numElements; ++i) { res += std::popcount(m_data[i]); }
			return res;
		}

		template<uint64_t otherBits = numBits, bool otherStackAlloc = stackAlloc>
		BitSet &operator|=(const BitSet<otherBits, otherStackAlloc> &other) {
			constexpr uint64_t otherElements = std::decay_t<decltype(other)>::numElements;

			uint64_t index = 0;
			while (index < min(numElements, otherElements)) {
				m_data[index] |= other.data()[index];
				++index;
			}

			return *this;
		}

		template<uint64_t otherBits = numBits, bool otherStackAlloc = stackAlloc>
		BitSet &operator&=(const BitSet<otherBits, otherStackAlloc> &other) {
			// If otherBits < numBits, zero out the top of this
			// Otherwise, make sure we don't read out of bounds

			constexpr uint64_t otherElements = std::decay_t<decltype(other)>::numElements;

			uint64_t index = 0;
			while (index < min(numElements, otherElements)) {
				m_data[index] &= other.data()[index];
				++index;
			}

			while (index < numElements) {
				m_data[index] = 0;
				++index;
			}

			return *this;
		}

		template<uint64_t otherBits = numBits, bool otherStackAlloc = stackAlloc>
		BitSet &operator^=(const BitSet<otherBits, otherStackAlloc> &other) {
			constexpr uint64_t otherElements = std::decay_t<decltype(other)>::numElements;

			uint64_t index = 0;
			while (index < min(numElements, otherElements)) {
				m_data[index] ^= other.data()[index];
				++index;
			}

			return *this;
		}

		BitSet &operator<<=(int64_t shift) {
			if (shift < 0) return *this >>= -shift;
			if (shift == 0) return *this;

			int64_t elementShift = shift / bitsPerElement;
			int64_t digitShift	 = shift % bitsPerElement;

			// Clear high bits
			for (int64_t i = numElements - elementShift; i < numElements; ++i) { m_data[i] = 0; }

			// Shift elements
			for (int64_t i = numElements - 1; i >= elementShift; --i) {
				ElementType tmp = 0;
				tmp |= m_data[i - elementShift] << digitShift;

				if (i - elementShift - 1 >= 0) {
					tmp |= m_data[i - elementShift - 1] >> (bitsPerElement - digitShift);
				}

				m_data[i] = tmp;
			}

			// Clear low bits
			for (int64_t i = 0; i < elementShift; ++i) { m_data[i] = 0; }

			return *this;
		}

		BitSet &operator>>=(int64_t shift) {
			if (shift < 0) return *this <<= -shift; // Handle negative shift
			if (shift == 0) return *this;			// Handle zero shift

			int64_t elementShift = shift / bitsPerElement;
			int64_t digitShift	 = shift % bitsPerElement;

			// Clear low bits
			for (int64_t i = 0; i < elementShift; ++i) { m_data[i] = 0; }

			// Shift elements
			for (int64_t i = 0; i < numElements - elementShift; ++i) {
				ElementType tmp = 0;
				tmp |= m_data[i + elementShift] >> digitShift;

				if (i + elementShift + 1 < numElements) {
					tmp |= m_data[i + elementShift + 1] << (bitsPerElement - digitShift);
				}

				m_data[i] = tmp;
			}

			// Clear high bits
			for (int64_t i = numElements - elementShift; i < numElements; ++i) { m_data[i] = 0; }

			return *this;
		}

		template<uint64_t otherBits = numBits, bool otherStackAlloc = stackAlloc>
		auto operator|(const BitSet<otherBits, otherStackAlloc> &other) const
		  -> BitSetMerger<otherBits, otherStackAlloc> {
			BitSetMerger<otherBits, otherStackAlloc> res = *this;
			res |= other;
			return res;
		}

		template<uint64_t otherBits = numBits, bool otherStackAlloc = stackAlloc>
		auto operator&(const BitSet<otherBits, otherStackAlloc> &other) const
		  -> BitSetMerger<otherBits, otherStackAlloc> {
			BitSetMerger<otherBits, otherStackAlloc> res = *this;
			res &= other;
			return res;
		}

		template<uint64_t otherBits = numBits, bool otherStackAlloc = stackAlloc>
		auto operator^(const BitSet<otherBits, otherStackAlloc> &other) const
		  -> BitSetMerger<otherBits, otherStackAlloc> {
			BitSetMerger<otherBits, otherStackAlloc> res = *this;
			res ^= other;
			return res;
		}

		BitSet operator<<(int64_t shift) const {
			BitSet res = *this;
			res <<= shift;
			return res;
		}

		BitSet operator>>(int64_t shift) const {
			BitSet res = *this;
			res >>= shift;
			return res;
		}

		BitSet operator~() const {
			BitSet res = *this;

			for (uint64_t i = 0; i < numElements; ++i) { res.m_data[i] = ~res.m_data[i]; }
			res.m_data[numElements - 1] &= highMask();

			return res;
		}

		bool operator==(const BitSet &other) const {
			for (uint64_t i = 0; i < numElements; ++i) {
				if (m_data[i] != other.data()[i]) return false;
			}
			return true;
		}

		const auto &data() const { return m_data; }
		auto &data() { return m_data; }

		template<typename Integer											 = ElementType,
				 typename std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE int toInt() const {
#if defined(LIBRAPID_DEBUG)
			static bool warned = false;
			if (!warned && last() >= sizeof(Integer) * 8) {
				warned = true;
				LIBRAPID_WARN(
				  "BitSet::toInt() called with BitSet that is too large for the "
				  "requested integer type");
			}
#endif

			return m_data[0];
		}

		template<typename T, typename Char, typename Ctx>
		void str(const fmt::formatter<T, Char> &format, Ctx &ctx) const {
			for (int64_t i = numBits - 1; i >= 0; --i) { format.format(get(i), ctx); }
		}

	protected:
		constexpr uint64_t highMask() const {
			ElementType res = 0;
			for (uint64_t i = 0; i < numBits % bitsPerElement; ++i) { res |= 1ULL << i; }
			return res;
		}

		void zero() {
			for (uint64_t i = 0; i < numElements; ++i) { m_data[i] = 0; }
		}

		void init() {
			if constexpr (!stackAlloc) { m_data = new ElementType[numElements]; }
		}

		void emptyInit() {
			init();
			zero();
		}

	private:
		StorageType m_data;
	};

	template<typename T>
	uint64_t popCount(const T &value) {
		return std::popcount(value);
	}

	template<uint64_t numBits, bool stackAlloc>
	uint64_t popCount(const BitSet<numBits, stackAlloc> &bitset) {
		uint64_t res = 0;
		for (uint64_t i = 0; i < BitSet<numBits, stackAlloc>::numElements; ++i) {
			res += popCount(bitset.data()[i]);
		}
		return res;
	}
} // namespace librapid

template<size_t numBits, typename Char>
struct fmt::formatter<librapid::BitSet<numBits>, Char> {
private:
	using Type = librapid::BitSet<numBits>;
	using Base = fmt::formatter<int, Char>;
	Base m_base;

public:
	template<typename ParseContext>
	FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const Char * {
		return m_base.parse(ctx);
	}

	template<typename FormatContext>
	FMT_CONSTEXPR auto format(const Type &val, FormatContext &ctx) const -> decltype(ctx.out()) {
		val.str(m_base, ctx);
		return ctx.out();
	}
};

LIBRAPID_SIMPLE_IO_NORANGE(uint64_t numBits, librapid::BitSet<numBits>)

// std ostream support
template<uint64_t numElements>
std::ostream &operator<<(std::ostream &os, const librapid::BitSet<numElements> &bitset) {
	return os << fmt::format("{}", bitset);
}

#endif // LIBRAPID_BITSET_HPP