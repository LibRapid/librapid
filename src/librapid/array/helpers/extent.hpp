#pragma once

namespace librapid {
	template<typename T, i64 maxDims, i64 align_ = 1>
	class ExtentType {
	public:
		using Type					 = T;
		static constexpr i64 MaxDims = maxDims;
		static constexpr i64 Align	 = align_;

		ExtentType() = default;

		template<typename... T_>
		explicit ExtentType(T_... args) :
				m_dims((i64)sizeof...(T_)), m_data {static_cast<T>(args)...} {}

		template<typename T_>
		ExtentType(const std::initializer_list<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			i64 i = 0;
			for (const auto &val : args) m_data[i++] = val;
		}

		template<typename T_>
		ExtentType(const std::vector<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			for (i64 i = 0; i < m_dims; ++i) m_data[i] = args[i];
		}

		template<typename T_, i64 d_, i64 a_>
		ExtentType(const ExtentType<T_, d_, a_> &e) {
			LR_ASSERT(e.ndim() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = e.ndim();
			for (i64 i = 0; i < m_dims; ++i) { m_data[i] = e[i]; }
		}

		ExtentType &operator=(const ExtentType &other) {
			if (this == &other) return *this;
			m_dims = other.ndim();
			for (i64 i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
			return *this;
		}

		template<typename T_, i64 d_, i64 a_>
		ExtentType &operator=(const ExtentType<T_, d_, a_> &other) {
			LR_ASSERT(other.ndim() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = other.ndim();
			for (i64 i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
			return *this;
		}

		static ExtentType zero(i64 dims) {
			// Data is already zeroed
			ExtentType res;
			res.m_dims = dims;
			return res;
		}

		static ExtentType one(i64 dims) {
			ExtentType res;
			res.m_dims = dims;
			for (i64 i = 0; i < dims; ++i) res.m_data[i] = 1;
			return res;
		}

		ExtentType stride() const {
			ExtentType res = zero(m_dims);
			i64 prod	   = 1;
			for (i64 i = m_dims - 1; i >= 0; --i) {
				res[i] = prod;
				prod *= m_data[i];
			}
			return res;
		}

		ExtentType strideAdjusted() const {
			ExtentType res = zero(m_dims);
			i64 prod	   = 1;
			for (i64 i = m_dims - 1; i >= 0; --i) {
				res[i] = prod;
				prod *= adjusted(i);
			}
			return res;
		}

		template<typename First, typename... Other>
		T index(First index, Other... others) const {
			return indexImpl(0, index, others...);
		}

		T index(const ExtentType &index) const {
			LR_ASSERT(
			  index.ndim() == m_dims,
			  "Cannot get index of Extent with {} dimensions using Extent with {} dimensions",
			  m_dims,
			  index.ndim());

			T res			   = 0;
			ExtentType strides = stride();
			for (i64 i = 0; i < index.ndim(); ++i) { res += strides[i] * index[i]; }
			return res;
		}

		template<typename First, typename... Other>
		T indexAdjusted(First index, Other... others) const {
			return indexImplAdjusted(0, index, others...);
		}

		T indexAdjusted(const ExtentType &index) const {
			LR_ASSERT(
			  index.ndim() == m_dims,
			  "Cannot get index of Extent with {} dimensions using Extent with {} dimensions",
			  m_dims,
			  index.ndim());

			T res			   = 0;
			ExtentType strides = strideAdjusted();
			for (i64 i = 0; i < index.ndim(); ++i) {
				LR_ASSERT(index.m_data[i] >= 0 && index[i] <= adjusted(i),
						  "Index {} is out of range for Extent with adjusted dimension {}",
						  index[i],
						  adjusted(i));
				res += strides[i] * index[i];
			}
			return res;
		}

		ExtentType reverseIndex(i64 index) const {
			ExtentType res	   = zero(m_dims);
			ExtentType strides = stride();
			for (i64 i = 0; i < m_dims; ++i) {
				res[i] = index / strides[i];
				index -= strides[i] * res[i];
			}
			return res;
		}

		ExtentType partial(i64 start = 0, i64 end = -1) const {
			if (end == -1) end = m_dims - 1;
			ExtentType res;
			res.m_dims = m_dims - 1;
			for (i64 i = start; i < end + 1; ++i) { res[i - start] = m_data[i]; }
			return res;
		}

		template<typename T_ = T, i64 d = maxDims, i64 a = align_>
		LR_NODISCARD("")
		ExtentType swivelled(const ExtentType<T_, d, a> &order) const {
			LR_ASSERT(
			  order.ndim() == m_dims,
			  "Swivel order must contain the same number of dimensions as the Extent to swivelled");

#if defined(LIBRAPID_DEBUG)
			// Check the order contains only valid numbers
			for (i64 i = 0; i < order.ndim(); ++i) {
				bool found = false;
				for (i64 j = 0; j < order.ndim(); ++j) {
					if (order[j] == i) {
						found = true;
						break;
					}
				}
				LR_ASSERT(found, "Swivel missing index {}", i);
			}
#endif

			ExtentType res = zero(m_dims);
			for (i64 i = 0; i < order.ndim(); ++i) { res[order[i]] = m_data[i]; }
			return res;
		}

		template<typename T_ = T, i64 d = maxDims, i64 a = align_>
		void swivel(const ExtentType<T_, d, a> &order) {
			*this = swivelled(order);
		}

		LR_NODISCARD("") LR_FORCE_INLINE i64 size() const {
			i64 res = 1;
			for (i64 i = 0; i < m_dims; ++i) res *= m_data[i];
			return res;
		}

		LR_NODISCARD("") LR_FORCE_INLINE i64 sizeAdjusted() const {
			i64 res = 1;
			for (i64 i = 0; i < m_dims; ++i) res *= adjusted(i);
			return res;
		}

		LR_NODISCARD("") LR_FORCE_INLINE i64 ndim() const { return m_dims; }

		const T &operator[](i64 index) const {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return m_data[index];
		}

		T &operator[](i64 index) {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return m_data[index];
		}

		T adjusted(i64 index) const {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return roundUpTo(m_data[index], Align);
		}

		template<typename T_, i64 d_, i64 a_>
		LR_NODISCARD("")
		bool operator==(const ExtentType<T_, d_, a_> &other) const {
			if (m_dims != other.m_dims) return false;
			for (i64 i = 0; i < m_dims; ++i)
				if (m_data[i] != other.m_data[i]) return false;
			return true;
		}

		template<typename T_, i64 d_, i64 a_>
		LR_NODISCARD("")
		bool operator!=(const ExtentType<T_, d_, a_> &other) const {
			return !(*this == other);
		}

		LR_NODISCARD("") std::string str() const {
			std::string res = "Extent(";
			for (i64 i = 0; i < m_dims - 1; ++i) res += fmt::format("{}, ", m_data[i]);
			return res + fmt::format("{})", m_data[m_dims - 1]);
		}

		LR_NODISCARD("") std::vector<T> toVector() const {
			std::vector<T> res;
			res.reserve(m_dims);
			for (i64 i = 0; i < m_dims; ++i) res.push_back(m_data[i]);
			return res;
		}

	private:
		template<typename First>
		T indexImpl(T index, First first) const {
			LR_ASSERT(first >= 0 && first < m_data[index],
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			i64 extentProd = 1;
			for (i64 i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first;
		}

		template<typename First, typename... Other>
		T indexImpl(T index, First first, Other... others) const {
			LR_ASSERT(first >= 0 && first < m_data[index],
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			i64 extentProd = 1;
			for (i64 i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first + indexImpl(index + 1, others...);
		}

		template<typename First>
		T indexImplAdjusted(T index, First first) const {
			LR_ASSERT(first >= 0 && first < adjusted(index),
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			i64 extentProd = 1;
			for (i64 i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first;
		}

		template<typename First, typename... Other>
		T indexImplAdjusted(T index, First first, Other... others) const {
			LR_ASSERT(first >= 0 && first < adjusted(index),
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			i64 extentProd = 1;
			for (i64 i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first + indexImpl(index + 1, others...);
		}

	private:
		i64 m_dims = -1;
		T m_data[maxDims] {};
	};

	template<typename T, i64 d, i64 a>
	inline std::string str(const ExtentType<T, d, a> &val,
						   const StrOpt &options = DEFAULT_STR_OPT) {
		return val.str();
	}

	using Extent = ExtentType<i64, 32, 1>;
} // namespace librapid