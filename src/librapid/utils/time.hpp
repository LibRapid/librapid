#pragma once

#include <utility>

#include "../internal/config.hpp"
#include "../math/statistics.hpp"

namespace librapid {
	// Forward declare this function
	template<typename T>
	T map(T val, T start1, T stop1, T start2, T stop2);

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
	LR_NODISCARD("")
	LR_FORCE_INLINE double now() {
		using namespace std::chrono;
		return (double)high_resolution_clock::now().time_since_epoch().count() / (double)scale;
	}

	static inline double sleepOffset = 0;

	template<int64_t scale = time::second>
	LR_FORCE_INLINE void sleep(double time) {
		using namespace std::chrono;
		time *= scale;
		auto start = now<time::nanosecond>();
		while (now<time::nanosecond>() - start < time - sleepOffset) {}
	}

	template<int64_t scale = time::second>
	std::string formatTime(double time, const std::string &format = "{:6f}") {
		double ns					= time * scale;
		int numUnits				= 8;
		static std::string prefix[] = {"ns", "µs", "ms", "s", "m", "h", "d", "y"};
		static double divisor[]		= {1000, 1000, 1000, 60, 60, 24, 365, 1e300};
		for (int i = 0; i < numUnits; ++i) {
			if (ns < divisor[i]) return fmt::format(format + "{}", ns, prefix[i]);
			ns /= divisor[i];
		}
		return fmt::format("{}ns", time * ns);
	}

	class Timer {
	public:
		explicit Timer(std::string name = "Timer") :
				m_name(std::move(name)), m_start(now<time::nanosecond>()) {}

		~Timer() {
			double end = now<time::nanosecond>();
			fmt::print("[ TIMER ] {} : {}\n", m_name, formatTime<time::nanosecond>(end - m_start));
		}

	private:
		std::string m_name;
		double m_start;
	};

	template<typename LAMBDA, typename... Args>
	double timeFunction(const LAMBDA &op, int64_t iters = -1, int64_t samples = -1, Args... vals) {
		if (samples < 1) samples = 10;

		// Call the function to compile any kernels
		op(vals...);

		if (iters < 1) {
			// Run the function once and time it to see how many iterations make sense
			int64_t tmpIters = 1;
			double start	 = now<time::nanosecond>();
			op(vals...);
			double end = now<time::nanosecond>();

			if (end - start < 1000) {
				tmpIters = 1000;
				start	 = now<time::nanosecond>();
				for (int64_t i = 0; i < tmpIters; ++i) op(vals...);
				end = now<time::nanosecond>();
			}

			// Make each sample take around 1 second
			iters = 5.0e8 / ((end - start) / (double)tmpIters);
			if (iters < 1) iters = 1;
		}
		std::vector<double> times;

		double globalStart = now();
		for (int64_t sample = 0; sample < samples; ++sample) {
			double start = now<time::nanosecond>();
			for (int64_t iter = 0; iter < iters; ++iter) { op(vals...); }
			double end = now<time::nanosecond>();
			times.emplace_back((end - start) / (double)iters);

			if (now() - globalStart > 15) {
				samples = sample + 1;
				break;
			}
		}

		// Calculate average (mean) time and standard deviation
		double avg = mean(times);
		double std = standardDeviation(times);
		fmt::print("Mean {} ± {} after {} samples, each with {} iterations\n",
				   formatTime<time::nanosecond>(avg),
				   formatTime<time::nanosecond>(std),
				   samples,
				   iters);
		return avg;
	}
} // namespace librapid