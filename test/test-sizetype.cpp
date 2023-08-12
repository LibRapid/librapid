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

    REQUIRE(shape1.size() == 24);
    REQUIRE(zero.size() == 0);
    REQUIRE(ones.size() == 1);

    REQUIRE(shape1 == shape1);
    REQUIRE_FALSE(shape1 != shape1);
    REQUIRE_FALSE(shape1 == zero);
    REQUIRE(shape1 != zero);

    REQUIRE(ones == lrc::Shape({1, 1, 1}));
    REQUIRE(zero == lrc::Shape({0, 0, 0}));
    REQUIRE(lrc::Shape({1, 2, 3, 4}) == lrc::Shape({1, 2, 3, 4}));
    REQUIRE(lrc::Shape({1, 2, 3, 4}) != lrc::Shape({1, 2, 3, 5}));

    REQUIRE(lrc::shapesMatch(lrc::Shape({1, 2, 3, 4}), lrc::Shape({1, 2, 3, 4})));
    REQUIRE_FALSE(lrc::shapesMatch(lrc::Shape({1, 2, 3, 4}), lrc::Shape({1, 2, 3, 5})));
    REQUIRE(lrc::shapesMatch(
      lrc::Shape({1, 2, 3, 4}), lrc::Shape({1, 2, 3, 4}), lrc::Shape({1, 2, 3, 4})));

    SECTION("Benchmarks") {
        BENCHMARK("Shape::zeros(5)") {
            auto shape = lrc::Shape<size_t, 32>::zeros(5);
            return shape.size();
        };

        BENCHMARK("Shape::ones(5)") {
            auto shape = lrc::Shape<size_t, 32>::ones(5);
            return shape.size();
        };

        auto lhs = lrc::Shape<size_t, 128>::ones(128);
        auto rhs = lrc::Shape<size_t, 128>::ones(128);
        BENCHMARK("Equality") { return lhs == rhs; };
    }
}
