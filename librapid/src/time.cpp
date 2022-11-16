#include <librapid/librapid.hpp>

namespace librapid {
	Timer::Timer(std::string name, bool printOnDestruct) :
			m_name(std::move(name)), m_start(now<time::nanosecond>()), m_end(-1),
			m_printOnDestruct(printOnDestruct) {}

	Timer::~Timer() {
		m_end = now<time::nanosecond>();
		if (m_printOnDestruct) print();
	}

	void Timer::start() {
		m_start = now<time::nanosecond>();
		m_end	= -1;
	}

	void Timer::stop() { m_end = now<time::nanosecond>(); }

	void Timer::reset() {
		m_start = now<time::nanosecond>();
		m_end	= -1;
	}

	void Timer::print() const {
		double tmpEnd = m_end;
		if (tmpEnd < 0) tmpEnd = now<time::nanosecond>();

		fmt::print(
		  "[ TIMER ] {} : {}\n", m_name, formatTime<time::nanosecond>(tmpEnd - m_start));
	}
}
