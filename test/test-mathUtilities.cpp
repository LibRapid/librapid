#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

TEST_CASE("Test Math Utilities", "[math]") {
    REQUIRE(lrc::clamp(5.f, 0.f, 10.f) == 5.f);
    REQUIRE(lrc::clamp(0.f, 0.f, 10.f) == 0.f);
    REQUIRE(lrc::clamp(10.f, 0.f, 10.f) == 10.f);
    REQUIRE(lrc::clamp(-10.f, 0.f, 10.f) == 0.f);
    REQUIRE(lrc::clamp(20.f, 0.f, 10.f) == 10.f);

    REQUIRE(lrc::lerp(0.f, 0.f, 1.f) == 0.f);
    REQUIRE(lrc::lerp(0.5f, 0.f, 1.f) == 0.5f);
    REQUIRE(lrc::lerp(1.f, 0.f, 1.f) == 1.f);
    REQUIRE(lrc::lerp(0.f, 0.f, 10.f) == 0.f);
    REQUIRE(lrc::lerp(0.5f, 0.f, 10.f) == 5.f);
    REQUIRE(lrc::lerp(1.f, 0.f, 10.f) == 10.f);
    REQUIRE(lrc::lerp(2.f, 0.f, 10.f) == 20.f);

    REQUIRE(lrc::smoothStep(0.f) == 0.f);
    REQUIRE(lrc::smoothStep(0.5f) == 0.5f);
    REQUIRE(lrc::smoothStep(1.f) == 1.f);
    REQUIRE(lrc::smoothStep(0.f, 0.f, 10.f) == 0.f);
    REQUIRE(lrc::smoothStep(5.f, 0.f, 10.f) == 0.5f);
    REQUIRE(lrc::smoothStep(10.f, 0.f, 10.f) == 1.f);
    REQUIRE(lrc::smoothStep(20.f, 0.f, 10.f) == 1.f);
    REQUIRE(lrc::smoothStep(0.25f, 0.f, 1.f) - 0.1035f < 0.001f);
    REQUIRE(lrc::smoothStep(0.75f, 0.f, 1.f) - 0.8965f < 0.001f);
}
