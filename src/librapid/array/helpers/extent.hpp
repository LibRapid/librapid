#pragma once

#include "../../internal/config.hpp"

namespace librapid {
	template<typename T, int64_t maxDims, int64_t align_ = 1>
	class ExtentType {
	public:
		using Type = T;
		static constexpr int64_t MaxDims = maxDims;
		static constexpr int64_t Align = align_;

		ExtentType() = default;

		template<typename... T_>
		explicit ExtentType(T_... args) : m_dims(sizeof...(T_)), m_data {args...} {}

		template<typename T_>
		ExtentType(const std::initializer_list<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			int64_t i = 0;
			for (const auto &val : args) m_data[i++] = val;
		}

		template<typename T_>
		ExtentType(const std::vector<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			for (int64_t i = 0; i < m_dims; ++i) m_data[i] = args[i];
		}

		template<typename T_, int64_t d_, int64_t a_>
		ExtentType(const ExtentType<T_, d_, a_> &e) {
			LR_ASSERT(e.dims() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = e.dims();
			for (int64_t i = 0; i < m_dims; ++i) { m_data[i] = e[i]; }
		}

		ExtentType &operator=(const ExtentType &other) {
			if (this == &other) return *this;
			m_dims = other.dims();
			for (int64_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
			return *this;
		}

		template<typename T_, int64_t d_, int64_t a_>
		ExtentType &operator=(const ExtentType<T_, d_, a_> &other) {
			LR_ASSERT(other.dims() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = other.dims();
			for (int64_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
			return *this;
		}

		static ExtentType zero(int64_t dims) {
			// Data is already zeroed
			ExtentType res;
			res.m_dims = dims;
			return res;
		}

		ExtentType stride() const {
			ExtentType res = zero(m_dims);
			int64_t prod   = 1;
			for (int64_t i = m_dims - 1; i >= 0; --i) {
				res[i] = prod;
				prod *= m_data[i];
			}
			return res;
		}

		ExtentType strideAdjusted() const {
			ExtentType res = zero(m_dims);
			int64_t prod   = 1;
			for (int64_t i = m_dims - 1; i >= 0; --i) {
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
			  index.dims() == m_dims,
			  "Cannot get index of Extent with {} dimensions using Extent with {} dimensions",
			  m_dims,
			  index.dims());

			T res			   = 0;
			ExtentType strides = stride();
			for (int64_t i = 0; i < index.dims(); ++i) {
				LR_ASSERT(index.m_data[i] >= 0 && index.m_data[i] <= m_data[i],
						  "Index {} is out of range for Extent with dimension {}",
						  index.m_data[i],
						  m_data[i]);
				res += strides[i] * index[i];
			}
			return res;
		}

		template<typename First, typename... Other>
		T indexAdjusted(First index, Other... others) const {
			return indexImplAdjusted(0, index, others...);
		}

		T indexAdjusted(const ExtentType &index) const {
			LR_ASSERT(
			  index.dims() == m_dims,
			  "Cannot get index of Extent with {} dimensions using Extent with {} dimensions",
			  m_dims,
			  index.dims());

			T res			   = 0;
			ExtentType strides = strideAdjusted();
			for (int64_t i = 0; i < index.dims(); ++i) {
				LR_ASSERT(index.m_data[i] >= 0 && index[i] <= adjusted(i),
						  "Index {} is out of range for Extent with adjusted dimension {}",
						  index[i],
						  adjusted(i));
				res += strides[i] * index[i];
			}
			return res;
		}

		ExtentType reverseIndex(int64_t index) const {
			ExtentType res	   = zero(m_dims);
			ExtentType strides = stride();
			for (int64_t i = 0; i < m_dims; ++i) {
				res[i] = index / strides[i];
				index -= strides[i] * res[i];
			}
			return res;
		}

		ExtentType partial(int64_t start = 0, int64_t end = -1) const {
			if (end == -1) end = m_dims - 1;
			ExtentType res;
			res.m_dims = m_dims - 1;
			for (int64_t i = start; i < end + 1; ++i) { res[i - start] = m_data[i]; }
			return res;
		}

		template<typename T_ = T, int64_t d = maxDims, int64_t a = align_>
		LR_NODISCARD("")
		ExtentType swivelled(const ExtentType<T_, d, a> &order) const {
			LR_ASSERT(
			  order.dims() == m_dims,
			  "Swivel order must contain the same number of dimensions as the Extent to swivelled");

#if defined(LIBRAPID_DEBUG)
			// Check the order contains only valid numbers
			for (int64_t i = 0; i < order.dims(); ++i) {
				bool found = false;
				for (int64_t j = 0; j < order.dims(); ++j) {
					if (order[j] == i) {
						found = true;
						break;
					}
				}
				LR_ASSERT(found, "Swivel missing index {}", i);
			}
#endif

			ExtentType res = zero(m_dims);
			for (int64_t i = 0; i < order.dims(); ++i) { res[order[i]] = m_data[i]; }
			return res;
		}
#
		template<typename T_ = T, int64_t d = maxDims, int64_t a = align_>
		void swivel(const ExtentType<T_, d, a> &order) {
			*this = swivelled(order);
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t size() const {
			int64_t res = 1;
			for (int64_t i = 0; i < m_dims; ++i) res *= m_data[i];
			return res;
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t sizeAdjusted() const {
			int64_t res = 1;
			for (int64_t i = 0; i < m_dims; ++i) res *= adjusted(i);
			return res;
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t dims() const { return m_dims; }

		const T &operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return m_data[index];
		}

		T &operator[](int64_t index) {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return m_data[index];
		}

		T adjusted(int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return roundUpTo(m_data[index], Align);
		}

		template<typename T_, int64_t d_, int64_t a_>
		LR_NODISCARD("")
		bool operator==(const ExtentType<T_, d_, a_> &other) const {
			if (m_dims != other.m_dims) return false;
			for (int64_t i = 0; i < m_dims; ++i)
				if (m_data[i] != other.m_data[i]) return false;
			return true;
		}

		LR_NODISCARD("") std::string str() const {
			std::string res = "Extent(";
			for (int64_t i = 0; i < m_dims - 1; ++i) res += fmt::format("{}, ", m_data[i]);
			return res + fmt::format("{})", m_data[m_dims - 1]);
		}

	private:
		template<typename First>
		T indexImpl(T index, First first) const {
			LR_ASSERT(first >= 0 && first < m_data[index],
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first;
		}

		template<typename First, typename... Other>
		T indexImpl(T index, First first, Other... others) const {
			LR_ASSERT(first >= 0 && first < m_data[index],
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first + indexImpl(index + 1, others...);
		}

		template<typename First>
		T indexImplAdjusted(T index, First first) const {
			LR_ASSERT(first >= 0 && first < adjusted(index),
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first;
		}

		template<typename First, typename... Other>
		T indexImplAdjusted(T index, First first, Other... others) const {
			LR_ASSERT(first >= 0 && first < adjusted(index),
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first + indexImpl(index + 1, others...);
		}

	private:
		T m_dims = -1;
		T m_data[maxDims] {};
	};

	template<typename T, int64_t d, int64_t a>
	inline std::string str(const ExtentType<T, d, a> &val,
						   const StrOpt &options = DEFAULT_STR_OPT) {
		return val.str();
	}

	using Extent = ExtentType<int64_t, 32, 1>;
} // namespace librapid