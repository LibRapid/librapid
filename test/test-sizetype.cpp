#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

TEST_CASE("Test Storage<T>", "[storage]") {
    lrc::Shape shape1({1, 2, 3, 4});
    REQUIRE(fmt::format("{}", shape1) == "Shape(1, 2, 3, 4)");

    lrc::Shape zero = lrc::Shape::zeros(3);
    REQUIRE(fmt::format("{}", zero) == "Shape(0, 0, 0)");

    lrc::Shape ones = lrc::Shape::ones(3);
    REQUIRE(fmt::format("{}", ones) == "Shape(1, 1, 1)");

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
}
