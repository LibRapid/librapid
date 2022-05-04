#pragma once

#include "../internal/config.hpp"

namespace librapid::internal {
	template<typename T, int64_t dims, typename First>
	T extentIndexProd(const Extent<T, dims> &extent, int64_t index, First first) {
		return first;
	}

	template<typename T, int64_t dims, typename First, typename... Other>
	T extentIndexProd(const Extent<T, dims> &extent, int64_t index, First first, Other... others) {
		int64_t extentProd = 1;
		for (int64_t i = index; i < extent.dims() - 1; ++i) extentProd *= extent[i];
		return extentProd * first + extentIndexProd(extent, index + 1, others...);
	}
} // namespace librapid::internal