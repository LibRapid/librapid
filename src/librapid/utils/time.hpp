#pragma once

#include <utility>

#include "../internal/config.hpp"

namespace librapid {
	// Forward declare mean and standardDeviation
	template<typename T>
	T mean(const std::vector<T> &);
	template<typename T>
	T standardDeviation(const std::vector<T> &);

	// Forward declare this function
	template<typename T>
	T map(T val, T start1, T stop1, T start2, T stop2);

	namespace time {
		constexpr int64_t day		  = (int64_t) 86400e9;
		constexpr int64_t hour		  = (int64_t) 3600e9;
		constexpr int64_t minute	  = (int64_t) 60e9;
		constexpr int64_t second	  = (int64_t) 1e9;
		constexpr int64_t millisecond = (int64_t) 1e6;
		constexpr int64_t microsecond = (int64_t) 1e3;
		constexpr int64_t nanosecond  = (int64_t) 1;
	} // namespace time

	template<int64_t scale = time::second>
	LR_NODISCARD("")
	LR_FORCE_INLINE double now() {
		using namespace std::chrono;
		return (double)high_resolution_clock::now().time_since_epoch().count() / (double)scale;
	}

	static double sleepOffset = 0;

	template<int64_t scale = time::second>
	LR_FORCE_INLINE void sleep(double time) {
		using namespace std::chrono;
		time *= scale;
		auto start = now<time::nanosecond>();
		while (now<time::nanosecond>() - start < time - sleepOffset) {}
	}

	template<int64_t scale = time::second>
	std::string formatTime(double time, const std::string &format = "{:.3f}") {
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
	double timeFunction(const LAMBDA &op, int64_t samples = -1, int64_t iters = -1,
						double time = -1, Args... vals) {
		if (samples < 1) samples = 10;

		// Call the function to compile any kernels
		op(vals...);

		double loopTime		   = 1e300;
		int64_t itersCompleted = 0;
		if (iters < 1) {
			loopTime	   = 5e9 / (double)samples;
			iters		   = 1e10;
			itersCompleted = 0;
		}

		if (time > 0) {
			loopTime = time * time::second;
			loopTime /= (double)samples;
		}

		std::vector<double> times;

		for (int64_t sample = 0; sample < samples; ++sample) {
			itersCompleted = 0;
			double start   = now<time::nanosecond>();
			while (itersCompleted++ < iters && now<time::nanosecond>() - start < loopTime) {
				op(vals...);
			}
			double end = now<time::nanosecond>();
			times.emplace_back((end - start) / (double)itersCompleted);
		}

		// Calculate average (mean) time and standard deviation
		double avg = mean(times);
		double std = standardDeviation(times);
		fmt::print("Mean {:>9} ± {:>9} after {} samples, each with {} iterations\n",
				   formatTime<time::nanosecond>(avg, "{:.3f}"),
				   formatTime<time::nanosecond>(std, "{:.3f}"),
				   samples,
				   itersCompleted - 1);
		return avg;
	}
} // namespace librapid