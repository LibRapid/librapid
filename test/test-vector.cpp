#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

// Use a high tolerance to ensure no floating point rounding errors cause tests to fail.
// If the results are within this tolerance, they are likely correct.
constexpr double tolerance = 1e-3;
#define VEC_TYPE lrc::GenericVector
#define SCALAR   double

TEST_CASE("Temporary") { REQUIRE(1 == 1); }
