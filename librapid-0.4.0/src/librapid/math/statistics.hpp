#pragma once

#include "../internal/config.hpp"

namespace librapid {
	template<typename T>
	T mean(const std::vector<T> &vals) {
		T sum = T();
		for (const auto &val : vals) sum += val;
		return sum / vals.size();
	}

	template<typename T>
	T standardDeviation(const std::vector<T> &vals) {
		T x = mean(vals);
		std::vector<T> variance2;
		for (const auto &val : vals) variance2.emplace_back((x - val) * (x - val));
		return sqrt(mean(variance2));
	}
} // namespace librapid