#pragma once

#include "../internal/config.hpp"

namespace librapid::test {
	namespace detail {
		auto noOp = []() { return true; };
	} // namespace detail

	template<typename LAMBDA_>
	class Test {
	public:
		using Lambda = LAMBDA_;
		using Expect = std::invoke_result_t<Lambda>;

		enum PassState { NOT_RUN, PASSED, FAILED };

	public:
		explicit Test(const LAMBDA_ &func) :
				m_name("Unnamed Test"), m_description("None"), m_test(func), m_expect(0) {}

		Test(const Test &other) :
				m_name(other.getName()), m_description(other.getDescription()),
				m_test(other.getTest()), m_expect(other.getExpect()) {}

		Test &operator=(const Test &other) {
			m_name		  = other.m_name;
			m_description = other.m_description;
			m_test		  = other.m_test;
			m_expect	  = other.m_expect;
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

		template<typename... Args>
		void run(Args... args) {
			auto result = m_test(args...);
			if (result == m_expect) {
				fmt::print("[ PASSED ]\n");
			} else {
				fmt::print("[ ERROR ]\n");
			}
		}

		auto getName() const { return m_name; }
		auto getDescription() const { return m_description; }
		auto getTest() const { return m_test; }
		auto getExpect() const { return m_expect; }

	private:
		std::string m_name;
		std::string m_description;
		Lambda m_test;
		Expect m_expect;

		PassState m_pass = NOT_RUN;
	};
} // namespace librapid::test