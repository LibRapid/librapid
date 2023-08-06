#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 1e-5;
using CPU				   = lrc::backend::CPU;
using OPENCL			   = lrc::backend::OpenCL;
using CUDA				   = lrc::backend::CUDA;

// #define SCALAR	float
// #define BACKEND CPU

#define TEST_OP(NAME, SCALAR)                                                                      \
	auto NAME##X = lrc::NAME(x).eval();                                                            \
	for (int i = 0; i < NAME##X.shape().size(); ++i) {                                             \
		REQUIRE(lrc::isClose((SCALAR)NAME##X(i), (SCALAR)lrc::NAME((SCALAR)x(i)), tolerance));     \
	}

#define TRIG_TEST_IMPL(SCALAR, BACKEND)                                                            \
	TEST_CASE(fmt::format("Test Trigonometry -- {} {}", STRINGIFY(SCALAR), STRINGIFY(BACKEND)),    \
			  "[array-lib]") {                                                                     \
		/* Valid range for all functions */                                                        \
		auto x = lrc::linspace<SCALAR, BACKEND>(0.1, 0.5, 100, false);                             \
                                                                                                   \
		TEST_OP(sin, SCALAR);                                                                      \
		TEST_OP(cos, SCALAR);                                                                      \
		TEST_OP(tan, SCALAR);                                                                      \
		TEST_OP(asin, SCALAR);                                                                     \
		TEST_OP(acos, SCALAR);                                                                     \
		TEST_OP(atan, SCALAR);                                                                     \
		TEST_OP(sinh, SCALAR);                                                                     \
		TEST_OP(cosh, SCALAR);                                                                     \
		TEST_OP(tanh, SCALAR);                                                                     \
                                                                                                   \
		TEST_OP(exp, SCALAR);                                                                      \
		TEST_OP(log, SCALAR);                                                                      \
		TEST_OP(log2, SCALAR);                                                                     \
		TEST_OP(log10, SCALAR);                                                                    \
		TEST_OP(sqrt, SCALAR);                                                                     \
		TEST_OP(cbrt, SCALAR);                                                                     \
		TEST_OP(abs, SCALAR);                                                                      \
		TEST_OP(floor, SCALAR);                                                                    \
		TEST_OP(ceil, SCALAR);                                                                     \
	}

TRIG_TEST_IMPL(float, CPU)
TRIG_TEST_IMPL(double, CPU)

#if defined(LIBRAPID_HAS_OPENCL)
TEST_CASE("Configure OpenCL") { lrc::configureOpenCL(true); }
TRIG_TEST_IMPL(float, OPENCL)
TRIG_TEST_IMPL(double, OPENCL)
#endif

#if defined(LIBRAPID_HAS_CUDA)
TRIG_TEST_IMPL(float, CUDA)
TRIG_TEST_IMPL(double, CUDA)
#endif
