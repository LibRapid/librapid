#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;
constexpr double tolerance = 0.001;

#define SCALAR float
#define DEVICE lrc::device::CPU
