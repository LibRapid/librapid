#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 1e-5;
using CPU				   = lrc::backend::CPU;
using OPENCL			   = lrc::backend::OpenCL;
using CUDA				   = lrc::backend::CUDA;

#define SCALAR	float
#define BACKEND CPU

#define TEST_OP(NAME)                                                                              \
	auto NAME##X = lrc::NAME(x).eval();                                                            \
	for (int i = 0; i < NAME##X.shape().size(); ++i) {                                             \
		REQUIRE(lrc::isClose(NAME##X(i), lrc::NAME((SCALAR)x(i)), tolerance));                     \
	}

#define TRIG_TEST_IMPL(SCALAR, BACKEND)                                                            \
	TEST_CASE(fmt::format("Test Trigonometry -- {} {}", STRINGIFY(SCALAR), STRINGIFY(BACKEND)),    \
			  "[array-lib]") {                                                                     \
		auto x = lrc::linspace<SCALAR, BACKEND>(0, 1, 100, false);                                 \
                                                                                                   \
		TEST_OP(sin);                                                                              \
		TEST_OP(cos);                                                                              \
		TEST_OP(tan);                                                                              \
		TEST_OP(asin);                                                                             \
		TEST_OP(acos);                                                                             \
		TEST_OP(atan);                                                                             \
		TEST_OP(sinh);                                                                             \
		TEST_OP(cosh);                                                                             \
		TEST_OP(tanh);                                                                             \
                                                                                                   \
		TEST_OP(exp);                                                                              \
		TEST_OP(log);                                                                              \
		TEST_OP(log2);                                                                             \
		TEST_OP(log10);                                                                            \
		TEST_OP(sqrt);                                                                             \
		TEST_OP(cbrt);                                                                             \
		TEST_OP(abs);                                                                              \
		TEST_OP(floor);                                                                            \
		TEST_OP(ceil);                                                                             \
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
