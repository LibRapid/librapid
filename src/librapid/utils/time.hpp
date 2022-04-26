#pragma once

#include "../internal/config.hpp"

namespace librapid {
	namespace time {
		constexpr int64_t day		  = 86400e9;
		constexpr int64_t hour		  = 3600e9;
		constexpr int64_t minute	  = 60e9;
		constexpr int64_t second	  = 1e9;
		constexpr int64_t millisecond = 1e6;
		constexpr int64_t microsecond = 1e3;
		constexpr int64_t nanosecond  = 1;
	} // namespace time

	template<int64_t scale = time::second>
	LR_NODISCARD("") double now() {
		using namespace std::chrono;
		return (double)high_resolution_clock::now().time_since_epoch().count() / (double)scale;
	}

	template<int64_t scale = time::second>
	void sleep(double time) {
		using namespace std::chrono;
		time *= scale;
		auto start = high_resolution_clock::now();
		while ((double)(high_resolution_clock::now() - start).count() < time - 75) {}
	}
} // namespace librapid