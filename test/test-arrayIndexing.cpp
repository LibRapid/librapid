#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 0.001;
using CPU				   = lrc::backend::CPU;
using OPENCL			   = lrc::backend::OpenCL;
using CUDA				   = lrc::backend::CUDA;

#define TEST_INDEXING(SCALAR, BACKEND)                                                             \
	SECTION(fmt::format("Test Indexing [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {       \
		lrc::Array<SCALAR, BACKEND> testA(lrc::Array<SCALAR, BACKEND>::ShapeType({5, 3}));         \
		testA << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;                                \
		std::string index0 = fmt::format("[{} {} {}]", SCALAR(1), SCALAR(2), SCALAR(3));           \
		std::string index1 = fmt::format("[{} {} {}]", SCALAR(4), SCALAR(5), SCALAR(6));           \
		std::string index2 = fmt::format("[{} {} {}]", SCALAR(7), SCALAR(8), SCALAR(9));           \
		std::string index3 = fmt::format("[{} {} {}]", SCALAR(10), SCALAR(11), SCALAR(12));        \
		std::string index4 = fmt::format("[{} {} {}]", SCALAR(13), SCALAR(14), SCALAR(15));        \
		REQUIRE(fmt::format("{}", testA[0]) == index0);                                            \
		REQUIRE(fmt::format("{}", testA[1]) == index1);                                            \
		REQUIRE(fmt::format("{}", testA[2]) == index2);                                            \
		REQUIRE(fmt::format("{}", testA[3]) == index3);                                            \
		REQUIRE(fmt::format("{}", testA[4]) == index4);                                            \
		REQUIRE(fmt::format("{}", testA[0][0]) == fmt::format("{}", SCALAR(1)));                   \
		REQUIRE(fmt::format("{}", testA[1][1]) == fmt::format("{}", SCALAR(5)));                   \
		REQUIRE(fmt::format("{}", testA[2][2]) == fmt::format("{}", SCALAR(9)));                   \
                                                                                                   \
		testA[1][2] = 123;                                                                         \
                                                                                                   \
		REQUIRE(testA[0][0].get() == SCALAR(1));                                                   \
		REQUIRE(testA[1][1].get() == SCALAR(5));                                                   \
		REQUIRE(testA[2][2].get() == SCALAR(9));                                                   \
		REQUIRE(testA[1][2].get() == SCALAR(123));                                                 \
                                                                                                   \
		testA[0][0] = 123;                                                                         \
		testA[1][1] = 456;                                                                         \
		testA[2][2] = 789;                                                                         \
		REQUIRE((SCALAR)testA.storage()[0] == SCALAR(123));                                        \
		REQUIRE((SCALAR)testA.storage()[4] == SCALAR(456));                                        \
		REQUIRE((SCALAR)testA.storage()[8] == SCALAR(789));                                        \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testB(lrc::Array<SCALAR, BACKEND>::ShapeType({10}));           \
		testB << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;                                                    \
		REQUIRE(testB[0].get() == SCALAR(1));                                                      \
		REQUIRE(testB[9].get() == SCALAR(10));                                                     \
	}

TEST_CASE("Test Array -- int8_t CPU", "[array-lib]") { TEST_INDEXING(int8_t, CPU); }
TEST_CASE("Test Array -- uint8_t CPU", "[array-lib]") { TEST_INDEXING(uint8_t, CPU); }
TEST_CASE("Test Array -- int16_t CPU", "[array-lib]") { TEST_INDEXING(int16_t, CPU); }
TEST_CASE("Test Array -- uint16_t CPU", "[array-lib]") { TEST_INDEXING(uint16_t, CPU); }
TEST_CASE("Test Array -- int32_t CPU", "[array-lib]") { TEST_INDEXING(int32_t, CPU); }
TEST_CASE("Test Array -- uint32_t CPU", "[array-lib]") { TEST_INDEXING(uint32_t, CPU); }
TEST_CASE("Test Array -- int64_t CPU", "[array-lib]") { TEST_INDEXING(int64_t, CPU); }
TEST_CASE("Test Array -- uint64_t CPU", "[array-lib]") { TEST_INDEXING(uint64_t, CPU); }
TEST_CASE("Test Array -- float CPU", "[array-lib]") { TEST_INDEXING(float, CPU); }
TEST_CASE("Test Array -- double CPU", "[array-lib]") { TEST_INDEXING(double, CPU); }

#if defined(LIBRAPID_USE_MULTIPREC)
TEST_CASE("Test Array -- lrc::mpfr CPU", "[array-lib]") { TEST_INDEXING(lrc::mpfr, CPU); }
#endif // LIBRAPID_USE_MULTIPREC

#if defined(LIBRAPID_HAS_OPENCL)

TEST_CASE("Configure OpenCL") { lrc::configureOpenCL(true); }

TEST_CASE("Test Array -- int32_t OpenCL", "[array-lib]") { TEST_INDEXING(int32_t, OPENCL); }
TEST_CASE("Test Array -- uint32_t OpenCL", "[array-lib]") { TEST_INDEXING(uint32_t, OPENCL); }
TEST_CASE("Test Array -- int64_t OpenCL", "[array-lib]") { TEST_INDEXING(int64_t, OPENCL); }
TEST_CASE("Test Array -- uint64_t OpenCL", "[array-lib]") { TEST_INDEXING(uint64_t, OPENCL); }
TEST_CASE("Test Array -- float OpenCL", "[array-lib]") { TEST_INDEXING(float, OPENCL); }
TEST_CASE("Test Array -- double OpenCL", "[array-lib]") { TEST_INDEXING(double, OPENCL); }

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

TEST_CASE("Test Array -- int8_t CUDA", "[array-lib]") { TEST_INDEXING(int8_t, CUDA); }
TEST_CASE("Test Array -- uint8_t CUDA", "[array-lib]") { TEST_INDEXING(uint8_t, CUDA); }
TEST_CASE("Test Array -- int16_t CUDA", "[array-lib]") { TEST_INDEXING(int16_t, CUDA); }
TEST_CASE("Test Array -- uint16_t CUDA", "[array-lib]") { TEST_INDEXING(uint16_t, CUDA); }

// TEST_CASE("Test Array -- int8_t CUDA", "[array-lib]") { TEST_ALL(int8_t, CUDA); }
// TEST_CASE("Test Array -- uint8_t CUDA", "[array-lib]") { TEST_ALL(uint8_t, CUDA); }
// TEST_CASE("Test Array -- int16_t CUDA", "[array-lib]") { TEST_ALL(int16_t, CUDA); }
// TEST_CASE("Test Array -- uint16_t CUDA", "[array-lib]") { TEST_ALL(uint16_t, CUDA); }
TEST_CASE("Test Array -- int32_t CUDA", "[array-lib]") { TEST_INDEXING(int32_t, CUDA); }
TEST_CASE("Test Array -- uint32_t CUDA", "[array-lib]") { TEST_INDEXING(uint32_t, CUDA); }
TEST_CASE("Test Array -- int64_t CUDA", "[array-lib]") { TEST_INDEXING(int64_t, CUDA); }
TEST_CASE("Test Array -- uint64_t CUDA", "[array-lib]") { TEST_INDEXING(uint64_t, CUDA); }
TEST_CASE("Test Array -- float CUDA", "[array-lib]") { TEST_INDEXING(float, CUDA); }
TEST_CASE("Test Array -- double CUDA", "[array-lib]") { TEST_INDEXING(double, CUDA); }

#endif // LIBRAPID_HAS_CUDA
