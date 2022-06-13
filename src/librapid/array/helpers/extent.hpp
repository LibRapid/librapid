#pragma once

#include "../../internal/config.hpp"

namespace librapid {
	template<typename T, int64_t maxDims, int64_t align = internal::traits<T>::PacketWidth>
	class ExtentType {
	public:
		ExtentType() = default;

		template<typename... T_>
		explicit ExtentType(T_... args) : m_dims(sizeof...(T_)), m_data {args...} {
			for (int64_t i = 0; i < m_dims; ++i) m_dataAdjusted[i++] = roundUpTo(m_dims[i], align);
		}

		template<typename T_>
		ExtentType(const std::initializer_list<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			int64_t i = 0;
			for (const auto &val : args) {
				m_data[i]			= val;
				m_dataAdjusted[i++] = roundUpTo(val, align);
			}
		}

		template<typename T_>
		ExtentType(const std::vector<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			for (int64_t i = 0; i < m_dims; ++i) {
				m_data[i]		  = args[i];
				m_dataAdjusted[i] = roundUpTo(args[i], align);
			}
		}

		template<typename T_, int64_t d_, int64_t a_>
		ExtentType(const ExtentType<T_, d_, a_> &e) {
			LR_ASSERT(e.dims() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = e.dims();
			for (int64_t i = 0; i < m_dims; ++i) {
				m_data[i] = e[i];
				m_data[i] = e.adjusted(i);
			}
		}

		template<typename T_, int64_t d_, int64_t a_>
		ExtentType &operator=(const ExtentType<T_, d_, a_> &other) {
			if (this == &other) return *this;
			LR_ASSERT(other.dims() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = other.dims();
			for (int64_t i = 0; i < m_dims; ++i) {
				m_data[i] = other[i];
				m_data[i] = other.adjusted(i);
			}
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
				prod *= m_dataAdjusted[i];
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
				LR_ASSERT(index.m_data[i] >= 0 && index.m_data[i] <= m_data[i],
						  "Index {} is out of range for Extent with dimension {}",
						  index.m_data[i],
						  m_data[i]);
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

		ExtentType reverseIndexAdjusted(int64_t index) const {
			ExtentType res	   = zero(m_dims);
			ExtentType strides = strideAdjusted();
			for (int64_t i = 0; i < m_dims; ++i) {
				res[i] = index / strides[i];
				index -= strides[i] * res[i];
			}
			return res;
		}

		ExtentType partial(int64_t start = 0, int64_t end = -1) const {
			if (end == -1) end = m_dims;
			ExtentType<T, maxDims> res;
			res.m_dims = m_dims - 1;
			for (int64_t i = start; i < end; ++i) {
				res[i - start]			= m_data[i];
				res.adjusted(i - start) = m_dataAdjusted[i];
			}
			return res;
		}

		template<typename T_ = T, int64_t d = maxDims>
		LR_NODISCARD("")
		ExtentType swivel(const ExtentType<T_, d> &order) const {
			LR_ASSERT(
			  order.dims() == m_dims,
			  "Swivel order must contain the same number of dimensions as the Extent to swivel");

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
			for (int64_t i = 0; i < order.dims(); ++i) {
				res[order[i]]		   = m_data[i];
				res.adjusted(order[i]) = m_dataAdjusted[i];
			}
			return res;
		}

		template<typename T_ = T, int64_t d = maxDims>
		void swivelInplace(const ExtentType<T_, d> &order) {
			*this = swivel(order);
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t size() const {
			int64_t res = 1;
			for (int64_t i = 0; i < m_dims; ++i) res *= m_data[i];
			return res;
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t sizeAdjusted() const {
			int64_t res = 1;
			for (int64_t i = 0; i < m_dims; ++i) res *= m_dataAdjusted[i];
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

		const T &adjusted(int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return m_dataAdjusted[index];
		}

		T &adjusted(int64_t index) {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return m_dataAdjusted[index];
		}

		template<typename T_, int64_t d_>
		LR_NODISCARD("")
		bool operator==(const ExtentType<T_, d_> &other) const {
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
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= m_data[i];
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
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= m_data[i];
			return extentProd * first + indexImpl(index + 1, others...);
		}

		template<typename First>
		T indexImplAdjusted(T index, First first) const {
			LR_ASSERT(first >= 0 && first < m_dataAdjusted[index],
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= m_dataAdjusted[i];
			return extentProd * first;
		}

		template<typename First, typename... Other>
		T indexImplAdjusted(T index, First first, Other... others) const {
			LR_ASSERT(first >= 0 && first < m_dataAdjusted[index],
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= m_dataAdjusted[i];
			return extentProd * first + indexImpl(index + 1, others...);
		}

	private:
		T m_dims = -1;
		T m_data[maxDims] {};
		T m_dataAdjusted[maxDims] {align};
	};

	using Extent = ExtentType<int64_t, 32>;
} // namespace librapid