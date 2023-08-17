#ifndef LIBRAPID_UTILS_TIME_HPP
#define LIBRAPID_UTILS_TIME_HPP

namespace librapid {
	namespace time {
		constexpr int64_t nanosecond  = int64_t(1);
		constexpr int64_t microsecond = nanosecond * 1000;
		constexpr int64_t millisecond = microsecond * 1000;
		constexpr int64_t second      = millisecond * 1000;
		constexpr int64_t minute      = second * 60;
		constexpr int64_t hour        = minute * 60;
		constexpr int64_t day         = hour * 24;
	} // namespace time

	template<int64_t scale = time::second>
	LIBRAPID_NODISCARD double now() {
		using namespace std::chrono;
#if defined(LIBRAPID_OS_WINDOWS)
		using rep      = int64_t;
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

	namespace detail {
		struct FormattedTime {
			double time;
			std::string unit;
		};
	} // namespace detail

	template<int64_t scale = time::second>
	detail::FormattedTime formatTime(double time, const std::string &format = "{:.3f}") {
		double ns    = time * scale;
		int numUnits = 8;

		static std::string prefix[] = {
			"ns",
#if defined(LIBRAPID_OS_WINDOWS) && defined(LIBRAPID_NO_WINDOWS_H)
			"µs",
#else
			"us",
#endif
			"ms",
			"s",
			"m",
			"h",
			"d",
			"y"
		};

		static double divisor[] = {1000, 1000, 1000, 60, 60, 24, 365, 1e300};
		for (int i = 0; i < numUnits; ++i) {
			// if (ns < divisor[i]) return std::operator+(fmt::format(format, ns), prefix[i]);
			if (ns < divisor[i]) return {ns, prefix[i]};
			ns /= divisor[i];
		}
		return {ns, prefix[numUnits - 1]};
	}

	/// A timer class that can be used to measure a multitude of things.
	/// The timer can be started, stopped and reset, and can, optionally, output
	/// the time between construction and destruction to the console.
	class Timer {
	public:
		/// Create a new timer with a given name
		/// \param name The name of the timer
		/// \param printOnDestruct Whether to print the time between construction and destruction
		explicit Timer(std::string name = "") :
				m_name(std::move(name)), m_start(now<time::nanosecond>()), m_end(-1) {}

		Timer(const Timer &)            = default;
		Timer(Timer &&)                 = default;
		Timer &operator=(const Timer &) = default;
		Timer &operator=(Timer &&)      = default;

		template<size_t scale = time::second>
		Timer &setTargetTime(double time) {
			m_iters      = 0;
			m_targetTime = time * (double)scale;
			m_start      = now<time::nanosecond>();
			return *this;
		}

		/// Start the timer
		void start() {
			m_start = now<time::nanosecond>();
			m_end   = -1;
		}

		/// Stop the timer
		void stop() { m_end = now<time::nanosecond>(); }

		/// Reset the timer
		void reset() {
			m_start = now<time::nanosecond>();
			m_end   = -1;
		}

		/// Get the elapsed time in a given unit
		/// \tparam scale The unit to return the time in
		/// \return The elapsed time in the given unit
		template<int64_t scale = time::second>
		LIBRAPID_NODISCARD double elapsed() const {
			if (m_end == -1) return (now<time::nanosecond>() - m_start) / (double)scale;
			return (m_end - m_start) / (double)scale;
		}

		/// Get the average time in a given unit
		/// \tparam scale The unit to return the time in
		/// \return The average time in the given unit
		template<int64_t scale = time::second>
		LIBRAPID_NODISCARD double average() const {
			return elapsed<scale>() / (double)m_iters;
		}

		bool isRunning() {
			++m_iters;
			return now<time::nanosecond>() - m_start < m_targetTime;
		}

		/// Print the current elapsed time of the timer
		template<typename T, typename Char, typename Ctx>
		void str(const fmt::formatter<T, Char> &formatter, Ctx &ctx) const {
			double tmpEnd = m_end;
			if (tmpEnd < 0) tmpEnd = now<time::nanosecond>();
			// return fmt::format(
			//   "{}Elapsed: {} | Average: {}",
			//   (m_name.empty() ? "" : m_name + ": "),
			//   formatTime<time::nanosecond>(tmpEnd - m_start, format),
			//   formatTime<time::nanosecond>((tmpEnd - m_start) / (double)m_iters, format));

			auto [elapsed, elapsedUnit] = formatTime<time::nanosecond>(tmpEnd - m_start);
			auto [average, averageUnit] =
			  formatTime<time::nanosecond>((tmpEnd - m_start) / (double)m_iters);
			fmt::format_to(ctx.out(), "{}Elapsed: ", m_name);
			formatter.format(elapsed, ctx);
			fmt::format_to(ctx.out(), "{} | Average: ", elapsedUnit);
			formatter.format(average, ctx);
			fmt::format_to(ctx.out(), "{}", averageUnit);
		}

	private:
		std::string m_name = "Timer";
		double m_start     = 0;
		double m_end       = 0;

		size_t m_iters      = 0;
		double m_targetTime = 0;
	};
} // namespace librapid

template<typename Char>
struct fmt::formatter<librapid::Timer, Char> {
public:
	using Base = fmt::formatter<double, Char>;
	Base m_base;

	template<typename ParseContext>
	FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const char * {
		return m_base.parse(ctx);
	}

	template<typename FormatContext>
	FMT_CONSTEXPR auto format(const librapid::Timer &val, FormatContext &ctx) const
	  -> decltype(ctx.out()) {
		val.str(m_base, ctx);
		return ctx.out();
	}
};

template<typename Char>
struct fmt::formatter<librapid::detail::FormattedTime, Char> {
public:
	using Base = fmt::formatter<double, Char>;
	Base m_base;

	template<typename ParseContext>
	FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const char * {
		return m_base.parse(ctx);
	}

	template<typename FormatContext>
	FMT_CONSTEXPR auto format(const librapid::detail::FormattedTime &val,
							  FormatContext &ctx) const -> decltype(ctx.out()) {
		m_base.format(val.time, ctx);
		fmt::format_to(ctx.out(), "{}", val.unit);
		return ctx.out();
	}
};

#endif // LIBRAPID_UTILS_TIME_HPP