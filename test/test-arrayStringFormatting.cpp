#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 0.001;
using CPU				   = lrc::backend::CPU;
using OPENCL			   = lrc::backend::OpenCL;
using CUDA				   = lrc::backend::CUDA;

#define TEST_STRING_FORMATTING(SCALAR, BACKEND)                                                    \
	SECTION(                                                                                       \
	  fmt::format("Test String Formatting [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {    \
		lrc::Array<SCALAR, BACKEND> testA(lrc::Array<SCALAR, BACKEND>::ShapeType({2, 3}));         \
		testA << 1, 2, 3, 4, 5, 6;                                                                 \
                                                                                                   \
		REQUIRE(fmt::format("{}", testA) == fmt::format("[[{} {} {}]\n [{} {} {}]]",               \
														SCALAR(1),                                 \
														SCALAR(2),                                 \
														SCALAR(3),                                 \
														SCALAR(4),                                 \
														SCALAR(5),                                 \
														SCALAR(6)));                               \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testB(lrc::Array<SCALAR, BACKEND>::ShapeType({2, 3}));         \
		testB << 10, 2, 3, 4, 5, 6;                                                                \
                                                                                                   \
		REQUIRE(fmt::format("{}", testB) == fmt::format("[[{} {} {}]\n [{} {} {}]]",               \
														SCALAR(10),                                \
														SCALAR(2),                                 \
														SCALAR(3),                                 \
														SCALAR(4),                                 \
														SCALAR(5),                                 \
														SCALAR(6)));                               \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testC(lrc::Array<SCALAR, BACKEND>::ShapeType({2, 2, 2}));      \
		testC << 100, 2, 3, 4, 5, 6, 7, 8;                                                         \
		REQUIRE(fmt::format("{}", testC) ==                                                        \
				fmt::format("[[[{} {}]\n  [{} {}]]\n\n [[{} {}]\n  [{} {}]]]",                     \
							SCALAR(100),                                                           \
							SCALAR(2),                                                             \
							SCALAR(3),                                                             \
							SCALAR(4),                                                             \
							SCALAR(5),                                                             \
							SCALAR(6),                                                             \
							SCALAR(7),                                                             \
							SCALAR(8)));                                                           \
	}

TEST_CASE("Test Array -- int8_t CPU", "[array-lib]") { TEST_STRING_FORMATTING(int8_t, CPU); }
TEST_CASE("Test Array -- uint8_t CPU", "[array-lib]") { TEST_STRING_FORMATTING(uint8_t, CPU); }
TEST_CASE("Test Array -- int16_t CPU", "[array-lib]") { TEST_STRING_FORMATTING(int16_t, CPU); }
TEST_CASE("Test Array -- uint16_t CPU", "[array-lib]") { TEST_STRING_FORMATTING(uint16_t, CPU); }
TEST_CASE("Test Array -- int32_t CPU", "[array-lib]") { TEST_STRING_FORMATTING(int32_t, CPU); }
TEST_CASE("Test Array -- uint32_t CPU", "[array-lib]") { TEST_STRING_FORMATTING(uint32_t, CPU); }
TEST_CASE("Test Array -- int64_t CPU", "[array-lib]") { TEST_STRING_FORMATTING(int64_t, CPU); }
TEST_CASE("Test Array -- uint64_t CPU", "[array-lib]") { TEST_STRING_FORMATTING(uint64_t, CPU); }
TEST_CASE("Test Array -- float CPU", "[array-lib]") { TEST_STRING_FORMATTING(float, CPU); }
TEST_CASE("Test Array -- double CPU", "[array-lib]") { TEST_STRING_FORMATTING(double, CPU); }

#if defined(LIBRAPID_USE_MULTIPREC)
TEST_CASE("Test Array -- lrc::mpfr CPU", "[array-lib]") { TEST_STRING_FORMATTING(lrc::mpfr, CPU); }
#endif // LIBRAPID_USE_MULTIPREC

#if defined(LIBRAPID_HAS_OPENCL)

TEST_CASE("Configure OpenCL") { lrc::configureOpenCL(true); }

TEST_CASE("Test Array -- int32_t OpenCL", "[array-lib]") {
	TEST_STRING_FORMATTING(int32_t, OPENCL);
}
TEST_CASE("Test Array -- uint32_t OpenCL", "[array-lib]") {
	TEST_STRING_FORMATTING(uint32_t, OPENCL);
}
TEST_CASE("Test Array -- int64_t OpenCL", "[array-lib]") {
	TEST_STRING_FORMATTING(int64_t, OPENCL);
}
TEST_CASE("Test Array -- uint64_t OpenCL", "[array-lib]") {
	TEST_STRING_FORMATTING(uint64_t, OPENCL);
}
TEST_CASE("Test Array -- float OpenCL", "[array-lib]") { TEST_STRING_FORMATTING(float, OPENCL); }
TEST_CASE("Test Array -- double OpenCL", "[array-lib]") { TEST_STRING_FORMATTING(double, OPENCL); }

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

TEST_CASE("Test Array -- int8_t CUDA", "[array-lib]") { TEST_STRING_FORMATTING(int8_t, CUDA); }
TEST_CASE("Test Array -- uint8_t CUDA", "[array-lib]") { TEST_STRING_FORMATTING(uint8_t, CUDA); }
TEST_CASE("Test Array -- int16_t CUDA", "[array-lib]") { TEST_STRING_FORMATTING(int16_t, CUDA); }
TEST_CASE("Test Array -- uint16_t CUDA", "[array-lib]") { TEST_STRING_FORMATTING(uint16_t, CUDA); }
// TEST_CASE("Test Array -- int8_t CUDA", "[array-lib]") { TEST_ALL(int8_t, CUDA); }
// TEST_CASE("Test Array -- uint8_t CUDA", "[array-lib]") { TEST_ALL(uint8_t, CUDA); }
// TEST_CASE("Test Array -- int16_t CUDA", "[array-lib]") { TEST_ALL(int16_t, CUDA); }
// TEST_CASE("Test Array -- uint16_t CUDA", "[array-lib]") { TEST_ALL(uint16_t, CUDA); }
TEST_CASE("Test Array -- int32_t CUDA", "[array-lib]") { TEST_STRING_FORMATTING(int32_t, CUDA); }
TEST_CASE("Test Array -- uint32_t CUDA", "[array-lib]") { TEST_STRING_FORMATTING(uint32_t, CUDA); }
TEST_CASE("Test Array -- int64_t CUDA", "[array-lib]") { TEST_STRING_FORMATTING(int64_t, CUDA); }
TEST_CASE("Test Array -- uint64_t CUDA", "[array-lib]") { TEST_STRING_FORMATTING(uint64_t, CUDA); }
TEST_CASE("Test Array -- float CUDA", "[array-lib]") { TEST_STRING_FORMATTING(float, CUDA); }
TEST_CASE("Test Array -- double CUDA", "[array-lib]") { TEST_STRING_FORMATTING(double, CUDA); }

#endif // LIBRAPID_HAS_CUDA
