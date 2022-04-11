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
	void sleep(double time) {
		using namespace std::chrono;
		using namespace std::this_thread;

		static double estimate = 1e6; // Assume millisecond precision to begin with
		static double mean	   = 1e6;
		static double m2	   = 0;
		static int64_t count   = 1;

		time *= scale; // Convert to nanoseconds

		// int64_t maxDelay = (int64_t) time >> 2;

//		while (time > estimate + 2e6) {
//			auto start = high_resolution_clock::now();
//			sleep_for(std::chrono::nanoseconds(1000));
//			auto end = high_resolution_clock::now();
//
//			double observed = (double)(end - start).count();
//			time -= observed;
//
//			++count;
//			double delta = observed - mean;
//			mean += delta / (double)count;
//			m2 += delta * (observed - mean);
//			double stddev = sqrt(m2 / (double)(count - 1));
//			estimate	  = mean + stddev;
//		}

		// spin lock
		auto start = high_resolution_clock::now();
		while ((double)(high_resolution_clock::now() - start).count() < time - 75) {}
	}
} // namespace librapid