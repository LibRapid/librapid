#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 0.001;

// #define SCALAR float
// #define BACKEND lrc::backend::CPU

TEST_CASE("Test Array Generation Methods", "[array-lib]") {
	SECTION("Test zeros()") {
		auto a = lrc::zeros<double, lrc::backend::CPU>({3, 4, 5});
		REQUIRE(a.shape() == lrc::Shape({3, 4, 5}));
		REQUIRE(a.storage().size() == 60);
		for (size_t i = 0; i < a.storage().size(); i++) { REQUIRE(a.storage()[i] - 0 < tolerance); }
	}

	SECTION("Test ones()") {
		auto a = lrc::ones<double, lrc::backend::CPU>({3, 4, 5});
		REQUIRE(a.shape() == lrc::Shape({3, 4, 5}));
		REQUIRE(a.storage().size() == 60);
		for (size_t i = 0; i < a.storage().size(); i++) { REQUIRE(a.storage()[i] - 1 < tolerance); }
	}

	SECTION("Test ordered()") {
		auto a = lrc::ordered<double, lrc::backend::CPU>({3, 4, 5});
		REQUIRE(a.shape() == lrc::Shape({3, 4, 5}));
		REQUIRE(a.storage().size() == 60);
		for (size_t i = 0; i < a.storage().size(); i++) { REQUIRE(a.storage()[i] - i < tolerance); }
	}

	SECTION("Test arange()") {
		auto a = lrc::arange<double, lrc::backend::CPU>(0, 10, 1);
		REQUIRE(a.shape() == lrc::Shape({10}));
		REQUIRE(a.storage().size() == 10);
		for (size_t i = 0; i < a.storage().size(); i++) { REQUIRE(a.storage()[i] - i < tolerance); }

		auto b = lrc::arange<double, lrc::backend::CPU>(0, 10, 2);
		REQUIRE(b.shape() == lrc::Shape({5}));
		REQUIRE(b.storage().size() == 5);
		for (size_t i = 0; i < b.storage().size(); i++) {
			REQUIRE(b.storage()[i] - i * 2 < tolerance);
		}
	}

	SECTION("Test linspace()") {
		auto a = lrc::linspace<double, lrc::backend::CPU>(0, 10, 10, false);
		REQUIRE(a.shape() == lrc::Shape({10}));
		REQUIRE(a.storage().size() == 10);
		for (size_t i = 0; i < a.storage().size(); i++) { REQUIRE(a.storage()[i] - i < tolerance); }

		auto b = lrc::linspace<double, lrc::backend::CPU>(0, 10, 100, false);
		REQUIRE(b.shape() == lrc::Shape({100}));
		REQUIRE(b.storage().size() == 100);
		for (size_t i = 0; i < b.storage().size(); i++) {
			REQUIRE(b.storage()[i] - static_cast<double>(i) / 10 < tolerance);
		}

		auto c = lrc::linspace<double, lrc::backend::CPU>(0, 10, 10, true);
		REQUIRE(c.shape() == lrc::Shape({10}));
		REQUIRE(c.storage().size() == 10);
		for (size_t i = 0; i < c.storage().size(); i++) {
			REQUIRE(c.storage()[i] - static_cast<double>(i) * (10.0 / 9.0) < tolerance);
		}
	}
}
