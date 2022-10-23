#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

TEST_CASE("Test Storage<T>", "[storage]") {
	lrc::Shape shape1({1, 2, 3, 4});
	REQUIRE(shape1.str() == "(1, 2, 3, 4)");

	lrc::Shape zero = lrc::Shape<size_t, 32>::zeros(3);
	REQUIRE(zero.str() == "(0, 0, 0)");

	lrc::Shape ones = lrc::Shape<size_t, 32>::ones(3);
	REQUIRE(ones.str() == "(1, 1, 1)");

	REQUIRE(shape1.ndim() == 4);
	REQUIRE(shape1[0] == 1);
	REQUIRE(shape1[1] == 2);
	REQUIRE(shape1[2] == 3);
	REQUIRE(shape1[3] == 4);

	SECTION("Benchmarks") {
		BENCHMARK("Shape::zeros(5)") {
			auto shape = lrc::Shape<size_t, 32>::zeros(5);
			return shape.size();
		};

		BENCHMARK("Shape::ones(5)") {
			auto shape = lrc::Shape<size_t, 32>::ones(5);
			return shape.size();
		};
	}
}
