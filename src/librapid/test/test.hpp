#pragma once

#include "../internal/config.hpp"

namespace librapid::test {
	namespace detail {
		static inline auto noOp = []() { return true; };

		template<typename T>
		bool isClose(T a, T b) {
			if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, half> ||
						  std::is_same_v<T, mpf> || std::is_same_v<T, mpq> ||
						  std::is_same_v<T, mpfr>)
				return abs(a - b) < internal::traits<T>::epsilon();
			return a == b;
		}

		template<typename T>
		bool isClose(const std::vector<T> &a, const std::vector<T> &b) {
			if (a.size() != b.size()) { return false; }
			for (size_t i = 0; i < a.size(); i++)
				if (!isClose(a[i], b[i])) return false;
			return true;
		}
	} // namespace detail

	template<typename LAMBDA_>
	class Test {
	public:
		using Lambda = LAMBDA_;
		using Expect = std::invoke_result_t<Lambda>;

		enum PassState { NOT_RUN, PASSED, FAILED };

	public:
		explicit Test(const LAMBDA_ &func) :
				m_name("Unnamed Test"), m_description("None"), m_test(func), m_expect() {}

		Test(const Test &other) :
				m_name(other.getName()), m_description(other.getDescription()),
				m_test(other.getTest()), m_expect(other.getExpect()), m_bench(other.getBench()),
				m_allowClose(other.getAllowClose()) {}

		Test &operator=(const Test &other) {
			m_name		  = other.m_name;
			m_description = other.m_description;
			m_test		  = other.m_test;
			m_expect	  = other.m_expect;
			m_bench		  = other.m_bench;
			m_allowClose  = other.m_allowClose;
			return *this;
		}

		Test &name(const std::string &name) {
			m_name = name;
			return *this;
		}

		Test &description(const std::string &desc) {
			m_description = desc;
			return *this;
		}

		Test &expect(const Expect &expect) {
			m_expect = expect;
			return *this;
		}

		Test &bench(const bool &b) {
			m_bench = b;
			return *this;
		}

		Test &allowClose(const bool &b) {
			m_allowClose = b;
			return *this;
		}

		template<typename... Args>
		void run(Args... args) {
			Expect result;
			bool threw		= false;
			double execTime = 0;
			std::exception error;

			const std::string pass = "PASSED";
			const std::string fail = "FAILED";

			double tryStart = now();
			try {
				double start = now();
				result		 = m_test(args...);
				double end	 = now();
				execTime	 = end - start;
			} catch (std::exception &e) {
				result = Expect();
				error  = e;
				threw  = true;
			}
			double tryEnd = now();
			if (threw) execTime = tryEnd - tryStart;

			if (!threw && (m_allowClose ? detail::isClose(result, m_expect) : result == m_expect)) {
				m_pass = PASSED;
				std::string benchRes;
				// If we are benchmarking properly, time it
				if (m_bench) {
					benchRes = formatBench(timeFunction(m_test, -1, -1, 1, args...), false);
				} else {
					benchRes = formatTime(execTime);
				}

				// If the test passed, just say that it passed.
				fmt::print(
				  fmt::fg(fmt::color::green),
				  fmt::format("[ TEST ] {:<50}   {:<10}   {:>8}\n", m_name, pass, benchRes));
			} else {
				m_pass = FAILED;

				// If the test fails, print some more detailed information
				fmt::print(fmt::fg(fmt::color::red), fmt::format("\n{:#<81}\n", ""));

				fmt::print(
				  fmt::fg(fmt::color::red),
				  fmt::format("{:<50}   {:<10}   {:>8}\n", m_name, fail, formatTime(execTime)));

				fmt::print(fmt::fg(fmt::color::red), fmt::format("\n{}\n\n", m_description));

				fmt::print(fmt::fg(fmt::color::red), fmt::format("Expected:\n{}\n\n", m_expect));

				if (!threw) {
					// If no exception was thrown, print the resulting value
					fmt::print(fmt::fg(fmt::color::red), fmt::format("Received:\n{}\n", result));
				} else {
					// An exception was thrown, so the resulting value is *almost* certainly
					// invalid
					fmt::print(fmt::fg(fmt::color::red),
							   fmt::format("Received:\nERROR: {}\n", error.what()));
				}

				fmt::print(fmt::fg(fmt::color::red), fmt::format("{:#<81}\n\n", ""));
			}
		}

		LR_NODISCARD("") operator bool() const { return passed(); }

		auto getName() const { return m_name; }
		auto getDescription() const { return m_description; }
		auto getTest() const { return m_test; }
		auto getExpect() const { return m_expect; }
		auto getBench() const { return m_bench; }
		auto getAllowClose() const { return m_allowClose; }
		auto passed() const { return m_pass == PASSED; }

	private:
		std::string m_name;
		std::string m_description;
		Lambda m_test;
		Expect m_expect;
		bool m_bench	  = true;
		bool m_allowClose = false;

		PassState m_pass = NOT_RUN;
	};
} // namespace librapid::test