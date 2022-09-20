#pragma once

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
		constexpr i64 day		  = i64(86400e9);
		constexpr i64 hour		  = i64(3600e9);
		constexpr i64 minute	  = i64(60e9);
		constexpr i64 second	  = i64(1e9);
		constexpr i64 millisecond = i64(1e6);
		constexpr i64 microsecond = i64(1e3);
		constexpr i64 nanosecond  = i64(1);
	} // namespace time

	template<i64 scale = time::second>
	LR_NODISCARD("")
	LR_FORCE_INLINE double now() {
		using namespace std::chrono;
#if defined(LIBRAPID_OS_WINDOWS)
		using rep	   = i64;
		using period   = std::nano;
		using duration = std::chrono::duration<rep, period>;

		static const i64 clockFreq = []() -> i64 {
			LARGE_INTEGER frequency;
			QueryPerformanceFrequency(&frequency);
			return frequency.QuadPart;
		}();

		LARGE_INTEGER count;
		QueryPerformanceCounter(&count);
		return duration(count.QuadPart * static_cast<i64>(std::nano::den) / clockFreq).count() /
			   (double)scale;
#else
		return (double)high_resolution_clock::now().time_since_epoch().count() / (double)scale;
#endif
	}

	static double sleepOffset = 0;

	template<i64 scale = time::second>
	LR_FORCE_INLINE void sleep(double time) {
		using namespace std::chrono;
		time *= scale;
		auto start = now<time::nanosecond>();
		while (now<time::nanosecond>() - start < time - sleepOffset) {}
	}

	template<i64 scale = time::second>
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

	struct Bench {
		i64 samples;
		i64 iters;
		double avg;
		double stddev;
	};

	template<typename LAMBDA, typename... Args>
	LR_INLINE Bench timeFunction(const LAMBDA &op, i64 samples = -1, i64 iters = -1,
								 double time = -1, Args... vals) {
		if (samples < 1) samples = 10;

		// Call the function to compile any kernels
		op(vals...);

		double loopTime		   = 1e300;
		i64 itersCompleted = 0;
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

		for (i64 sample = 0; sample < samples; ++sample) {
			itersCompleted = 0;
			double start   = now<time::nanosecond>();
			while (itersCompleted++ < iters && now<time::nanosecond>() - start < loopTime) {
				op(vals...);
			}
			double end = now<time::nanosecond>();
			times.emplace_back((end - start) / (double)itersCompleted);
		}

		// Calculate average (mean) time and standard deviation
		double avg	  = mean(times);
		double stddev = standardDeviation(times);
		return {samples, itersCompleted - 1, avg, stddev};
	}

	LR_INLINE std::string formatBench(const Bench &bench, bool includeIters = true) {
		std::string res = fmt::format("Mean {:>9} ± {:>9}",
									  formatTime<time::nanosecond>(bench.avg, "{:.3f}"),
									  formatTime<time::nanosecond>(bench.stddev, "{:.3f}"));

		if (includeIters) {
			res +=
			  fmt::format(" after {} samples, each with {} iterations", bench.samples, bench.iters);
		}

		return res;
	}

	template<typename LAMBDA, typename... Args>
	LR_INLINE void timeVerbose(const LAMBDA &op, i64 samples = -1, i64 iters = -1,
							   double time = -1, Args... vals) {
		Bench b = timeFunction(op, samples, iters, time, vals...);
		fmt::print("{}\n", formatBench(b));
	}
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<>
struct fmt::formatter<librapid::Bench> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::Bench &b, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), formatBench(b));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API