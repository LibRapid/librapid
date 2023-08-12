#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc              = librapid;
constexpr double tolerance = 0.001;
using CPU                  = lrc::backend::CPU;
using OPENCL               = lrc::backend::OpenCL;
using CUDA                 = lrc::backend::CUDA;

#define TEST_SIGMOID(SCALAR, BACKEND)                                                              \
    TEST_CASE(fmt::format("Test Sigmoid -- [ {} | {} ]", STRINGIFY(SCALAR), STRINGIFY(BACKEND)),   \
              "[sigmoid]") {                                                                       \
        SECTION("Forward Sigmoid") {                                                               \
            int64_t n    = 100;                                                                    \
            auto sigmoid = lrc::ml::Sigmoid();                                                     \
            auto data    = lrc::linspace<SCALAR, BACKEND>(-10, 10, n);                             \
            auto f       = [](SCALAR x) { return 1 / (1 + lrc::exp(-x)); };                        \
                                                                                                   \
            auto result = lrc::zeros<SCALAR, BACKEND>(lrc::Shape({n}));                            \
            sigmoid.forward(result, data);                                                         \
            auto result2 = sigmoid(data);                                                          \
            auto result3 = sigmoid.forward(data);                                                  \
                                                                                                   \
            for (int64_t i = 0; i < n; ++i) {                                                      \
                REQUIRE(lrc::isClose((SCALAR)result(i), (SCALAR)f(data(i)), tolerance));           \
                REQUIRE(lrc::isClose((SCALAR)result2(i), (SCALAR)f(data(i)), tolerance));          \
                REQUIRE(lrc::isClose((SCALAR)result3(i), (SCALAR)f(data(i)), tolerance));          \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        SECTION("Backward Sigmoid") {                                                              \
            int64_t n    = 100;                                                                    \
            auto sigmoid = lrc::ml::Sigmoid();                                                     \
            auto data    = lrc::linspace<SCALAR, BACKEND>(-10, 10, n);                             \
            auto f       = [](SCALAR x) { return 1 / (1 + lrc::exp(-x)); };                        \
            auto fPrime  = [](SCALAR x) { return x * (1 - x); };                                   \
                                                                                                   \
            auto result = lrc::zeros<SCALAR, BACKEND>(lrc::Shape({n}));                            \
            sigmoid.forward(result, data);                                                         \
            sigmoid.backward(result, result);                                                      \
            auto result2 = sigmoid.backward(sigmoid(data));                                        \
            auto result3 = sigmoid.backward(sigmoid.forward(data));                                \
                                                                                                   \
            for (int64_t i = 0; i < n; ++i) {                                                      \
                REQUIRE(lrc::isClose((SCALAR)result(i), (SCALAR)fPrime(f(data(i))), tolerance));   \
                REQUIRE(lrc::isClose((SCALAR)result2(i), (SCALAR)fPrime(f(data(i))), tolerance));  \
                REQUIRE(lrc::isClose((SCALAR)result3(i), (SCALAR)fPrime(f(data(i))), tolerance));  \
            }                                                                                      \
        }                                                                                          \
    }

TEST_SIGMOID(float, CPU)
TEST_SIGMOID(double, CPU)

#if defined(LIBRAPID_HAS_OEPNCL)
TEST_SIGMOID(float, OPENCL)
TEST_SIGMOID(double, OPENCL)
#endif

#if defined(LIBRAPID_HAS_CUDA)
TEST_SIGMOID(float, CUDA)
TEST_SIGMOID(double, CUDA)
#endif
