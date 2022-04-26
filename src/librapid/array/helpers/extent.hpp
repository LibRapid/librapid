#pragma once

#include "../../internal/config.hpp"

namespace librapid {
	template<typename T, int64_t maxDims>
	class Extent {
	public:
		Extent() = default;

		template<typename... T_>
		explicit Extent(T_... args) : m_dims(sizeof...(T_)), m_data {args...} {}

		template<typename T_>
		Extent(const std::initializer_list<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			int64_t i = 0;
			for (const auto &val : args) m_data[i++] = val;
		}

		template<typename T_>
		Extent(const std::vector<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			for (int64_t i = 0; i < m_dims; ++i) m_data[i] = args[i];
		}

		template<typename T_, int64_t d_>
		Extent(const Extent<T_, d_> &e) {
			LR_ASSERT(e.dims() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = e.dims();
			for (int64_t i = 0; i < m_dims; ++i) { m_data[i] = e[i]; }
		}

		template<typename T_, int64_t d_>
		Extent<T, maxDims> &operator=(const Extent<T_, d_> &other) {
			if (this == &other) return *this;
			LR_ASSERT(other.dims() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = other.dims();
			for (int64_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
			return *this;
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t size() const {
			int64_t res = 1;
			for (int64_t i = 0; i < m_dims; ++i) res *= m_data[i];
			return res;
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t dims() const { return m_dims; }

		const T &operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < maxDims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index, maxDims);
			return m_data[index];
		}

		T &operator[](int64_t index) {
			LR_ASSERT(index >= 0 && index < maxDims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index);
			return m_data[index];
		}

		template<typename T_, int64_t d_>
		LR_NODISCARD("") bool operator==(const Extent<T_, d_> &other) const {
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
		T m_dims = -1;
		T m_data[maxDims] {};
	};
} // namespace librapid