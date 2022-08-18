#pragma once

#include "../internal/config.hpp"
#include "coreMath.hpp"

namespace librapid {
	template<typename T>
	T mean(const std::vector<T> &vals) {
		T sum = T();
		for (const auto &val : vals) { sum += val; }
		return sum / vals.size();
	}

	template<typename T>
	T standardDeviation(const std::vector<T> &vals) {
		T x = mean(vals);
		std::vector<T> variance;
		for (const auto &val : vals) variance.emplace_back((x - val) * (x - val));
		return ::librapid::sqrt(mean(variance));
	}
} // namespace librapid