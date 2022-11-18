#ifndef LIBRAPID_UTILS_TIME_HPP
#define LIBRAPID_UTILS_TIME_HPP

namespace librapid {
	namespace time {
		constexpr int64_t day		  = int64_t(86400e9);
		constexpr int64_t hour		  = int64_t(3600e9);
		constexpr int64_t minute	  = int64_t(60e9);
		constexpr int64_t second	  = int64_t(1e9);
		constexpr int64_t millisecond = int64_t(1e6);
		constexpr int64_t microsecond = int64_t(1e3);
		constexpr int64_t nanosecond  = int64_t(1);
	} // namespace time

	template<int64_t scale = time::second>
	LIBRAPID_NODISCARD double now() {
		using namespace std::chrono;
#if defined(LIBRAPID_OS_WINDOWS)
		using rep	   = int64_t;
		using period   = std::nano;
		using duration = std::chrono::duration<rep, period>;

		static const int64_t clockFreq = []() -> int64_t {
			LARGE_INTEGER frequency;
			QueryPerformanceFrequency(&frequency);
			return frequency.QuadPart;
		}();

		LARGE_INTEGER count;
		QueryPerformanceCounter(&count);
		return duration(count.QuadPart * static_cast<int64_t>(std::nano::den) / clockFreq).count() /
			   (double)scale;
#else
		return (double)high_resolution_clock::now().time_since_epoch().count() / (double)scale;
#endif
	}

	constexpr static double sleepOffset = 0;

	template<int64_t scale = time::second>
	LIBRAPID_ALWAYS_INLINE void sleep(double time) {
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

	/// A timer class that can be used to measure a multitude of things.
	/// The timer can be started, stopped and reset, and can, optionally, output
	/// the time between construction and destruction to the console.
	class Timer {
	public:
		/// Create a new timer with a given name
		/// \param name The name of the timer
		/// \param printOnDestruct Whether to print the time between construction and destruction
		explicit Timer(std::string name = "Timer", bool printOnDestruct = false);

		Timer(const Timer &)			= default;
		Timer(Timer &&)					= default;
		Timer &operator=(const Timer &) = default;
		Timer &operator=(Timer &&)		= default;

		/// Timer destructor
		~Timer();

		/// Start the timer
		void start();

		/// Stop the timer
		void stop();

		/// Reset the timer
		void reset();

		/// Get the elapsed time in a given unit
		/// \tparam scale The unit to return the time in
		/// \return The elapsed time in the given unit
		template<int64_t scale = time::second>
		LIBRAPID_NODISCARD double elapsed() const {
			if (m_end == -1) return (now<time::nanosecond>() - m_start) / (double)scale;
			return (m_end - m_start) / (double)scale;
		}

		/// Print the current elapsed time of the timer
		void print() const;

	private:
		std::string m_name;
		bool m_printOnDestruct;
		double m_start;
		double m_end;
	};
} // namespace librapid

#endif // LIBRAPID_UTILS_TIME_HPP