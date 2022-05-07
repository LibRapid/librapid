#pragma once

#include "../internal/config.hpp"

namespace librapid::internal {
	template<typename T, int64_t dims, typename First>
	T extentIndexProd(const Extent<T, dims> &extent, int64_t index, First first) {
		LR_ASSERT(first >= 0 && first < extent[index],
				  "Index {} is out of range for dimension {} of an Array with {}",
				  first,
				  index,
				  extent.str());

		return first;
	}

	template<typename T, int64_t dims, typename First, typename... Other>
	T extentIndexProd(const Extent<T, dims> &extent, int64_t index, First first, Other... others) {
		LR_ASSERT(first >= 0 && first < extent[index],
				  "Index {} is out of range for dimension {} of an Array with {}",
				  first,
				  index,
				  extent.str());

		int64_t extentProd = 1;
		for (int64_t i = index + 1; i < extent.dims(); ++i) extentProd *= extent[i];
		return extentProd * first + extentIndexProd(extent, index + 1, others...);
	}
} // namespace librapid::internal