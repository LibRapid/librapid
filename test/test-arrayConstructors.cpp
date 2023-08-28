#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 0.001;
using CPU				   = lrc::backend::CPU;
using OPENCL			   = lrc::backend::OpenCL;
using CUDA				   = lrc::backend::CUDA;

#define TEST_CONSTRUCTORS(SCALAR, BACKEND)                                                         \
	SECTION(fmt::format("Test Constructors [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {   \
		lrc::Array<SCALAR, BACKEND> testA;                                                         \
		REQUIRE(testA.shape() == lrc::Array<SCALAR, BACKEND>::ShapeType {0});                      \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testB(lrc::Array<SCALAR, BACKEND>::ShapeType {3, 4});          \
		REQUIRE(testB.shape() == lrc::Array<SCALAR, BACKEND>::ShapeType {3, 4});                   \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testC(lrc::Array<SCALAR, BACKEND>::ShapeType {3, 4}, 5);       \
		REQUIRE(testC.shape() == lrc::Array<SCALAR, BACKEND>::ShapeType {3, 4});                   \
		REQUIRE(testC.storage()[0] == 5);                                                          \
		REQUIRE(testC.storage()[1] == 5);                                                          \
		REQUIRE(testC.storage()[2] == 5);                                                          \
		REQUIRE(testC.storage()[9] == 5);                                                          \
		REQUIRE(testC.storage()[10] == 5);                                                         \
		REQUIRE(testC.storage()[11] == 5);                                                         \
                                                                                                   \
		lrc::ArrayF<SCALAR, 2, 2> testD(3);                                                        \
		REQUIRE(testD.storage()[0] == 3);                                                          \
		REQUIRE(testD.storage()[1] == 3);                                                          \
		REQUIRE(testD.storage()[2] == 3);                                                          \
		REQUIRE(testD.storage()[3] == 3);                                                          \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND>::ShapeType tmpShape({2, 3});                                   \
		lrc::Array<SCALAR, BACKEND> testE(std::move(tmpShape));                                    \
		REQUIRE(testE.shape() == lrc::Array<SCALAR, BACKEND>::ShapeType {2, 3});                   \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testF(testC);                                                  \
		REQUIRE(testF.shape() == lrc::Array<SCALAR, BACKEND>::ShapeType {3, 4});                   \
		REQUIRE(testF.storage()[0] == 5);                                                          \
		REQUIRE(testF.storage()[1] == 5);                                                          \
		REQUIRE(testF.storage()[2] == 5);                                                          \
		REQUIRE(testF.storage()[9] == 5);                                                          \
		REQUIRE(testF.storage()[10] == 5);                                                         \
		REQUIRE(testF.storage()[11] == 5);                                                         \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testG(lrc::Array<SCALAR, BACKEND>::ShapeType {3, 4}, 10);      \
		testC = testG;                                                                             \
		REQUIRE(testC.storage()[0] == 10);                                                         \
		REQUIRE(testC.storage()[1] == 10);                                                         \
		REQUIRE(testC.storage()[2] == 10);                                                         \
		REQUIRE(testC.storage()[9] == 10);                                                         \
		REQUIRE(testC.storage()[10] == 10);                                                        \
		REQUIRE(testC.storage()[11] == 10);                                                        \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testH(lrc::Array<SCALAR, BACKEND>::ShapeType {3, 3});          \
		testH << 1, 2, 3, 4, 5, 6, 7, 8, 9;                                                        \
		REQUIRE(testH.storage()[0] == 1);                                                          \
		REQUIRE(testH.storage()[1] == 2);                                                          \
		REQUIRE(testH.storage()[2] == 3);                                                          \
		REQUIRE(testH.storage()[3] == 4);                                                          \
		REQUIRE(testH.storage()[4] == 5);                                                          \
		REQUIRE(testH.storage()[5] == 6);                                                          \
		REQUIRE(testH.storage()[6] == 7);                                                          \
		REQUIRE(testH.storage()[7] == 8);                                                          \
		REQUIRE(testH.storage()[8] == 9);                                                          \
                                                                                                   \
		/* It is necessary to define the type of the data, otherwise bad things happen for the     \
		 * MPFR type */                                                                            \
		using InitList =                                                                           \
		  std::initializer_list<std::initializer_list<std::initializer_list<SCALAR>>>;             \
		using Vec = std::vector<std::vector<std::vector<SCALAR>>>;                                 \
                                                                                                   \
		/* Due to the way the code works, if this passes for a 3D array, it *must* pass for all    \
		 * other dimensions */                                                                     \
		auto testI =                                                                               \
		  lrc::Array<SCALAR, BACKEND>::fromData(InitList({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));   \
		REQUIRE(fmt::format("{}", testI) ==                                                        \
				fmt::format("[[[{} {}]\n  [{} {}]]\n\n [[{} {}]\n  [{} {}]]]",                     \
							SCALAR(1),                                                             \
							SCALAR(2),                                                             \
							SCALAR(3),                                                             \
							SCALAR(4),                                                             \
							SCALAR(5),                                                             \
							SCALAR(6),                                                             \
							SCALAR(7),                                                             \
							SCALAR(8)));                                                           \
                                                                                                   \
		auto testJ =                                                                               \
		  lrc::Array<SCALAR, BACKEND>::fromData(Vec({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));        \
		REQUIRE(fmt::format("{}", testJ) ==                                                        \
				fmt::format("[[[{} {}]\n  [{} {}]]\n\n [[{} {}]\n  [{} {}]]]",                     \
							SCALAR(1),                                                             \
							SCALAR(2),                                                             \
							SCALAR(3),                                                             \
							SCALAR(4),                                                             \
							SCALAR(5),                                                             \
							SCALAR(6),                                                             \
							SCALAR(7),                                                             \
							SCALAR(8)));                                                           \
	}

TEST_CASE("Test Array -- int8_t CPU", "[array-lib]") { TEST_CONSTRUCTORS(int8_t, CPU); }
TEST_CASE("Test Array -- uint8_t CPU", "[array-lib]") { TEST_CONSTRUCTORS(uint8_t, CPU); }
TEST_CASE("Test Array -- int16_t CPU", "[array-lib]") { TEST_CONSTRUCTORS(int16_t, CPU); }
TEST_CASE("Test Array -- uint16_t CPU", "[array-lib]") { TEST_CONSTRUCTORS(uint16_t, CPU); }
TEST_CASE("Test Array -- int32_t CPU", "[array-lib]") { TEST_CONSTRUCTORS(int32_t, CPU); }
TEST_CASE("Test Array -- uint32_t CPU", "[array-lib]") { TEST_CONSTRUCTORS(uint32_t, CPU); }
TEST_CASE("Test Array -- int64_t CPU", "[array-lib]") { TEST_CONSTRUCTORS(int64_t, CPU); }
TEST_CASE("Test Array -- uint64_t CPU", "[array-lib]") { TEST_CONSTRUCTORS(uint64_t, CPU); }
TEST_CASE("Test Array -- float CPU", "[array-lib]") { TEST_CONSTRUCTORS(float, CPU); }
TEST_CASE("Test Array -- double CPU", "[array-lib]") { TEST_CONSTRUCTORS(double, CPU); }

#if defined(LIBRAPID_USE_MULTIPREC)
TEST_CASE("Test Array -- lrc::mpfr CPU", "[array-lib]") { TEST_CONSTRUCTORS(lrc::mpfr, CPU); }
#endif // LIBRAPID_USE_MULTIPREC

#if defined(LIBRAPID_HAS_OPENCL)

TEST_CASE("Configure OpenCL") { lrc::configureOpenCL(true); }

TEST_CASE("Test Array -- int32_t OpenCL", "[array-lib]") { TEST_CONSTRUCTORS(int32_t, OPENCL); }
TEST_CASE("Test Array -- uint32_t OpenCL", "[array-lib]") { TEST_CONSTRUCTORS(uint32_t, OPENCL); }
TEST_CASE("Test Array -- int64_t OpenCL", "[array-lib]") { TEST_CONSTRUCTORS(int64_t, OPENCL); }
TEST_CASE("Test Array -- uint64_t OpenCL", "[array-lib]") { TEST_CONSTRUCTORS(uint64_t, OPENCL); }
TEST_CASE("Test Array -- float OpenCL", "[array-lib]") { TEST_CONSTRUCTORS(float, OPENCL); }
TEST_CASE("Test Array -- double OpenCL", "[array-lib]") { TEST_CONSTRUCTORS(double, OPENCL); }

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

TEST_CASE("Test Array -- int8_t CUDA", "[array-lib]") { TEST_CONSTRUCTORS(int8_t, CUDA); }
TEST_CASE("Test Array -- uint8_t CUDA", "[array-lib]") { TEST_CONSTRUCTORS(uint8_t, CUDA); }
TEST_CASE("Test Array -- int16_t CUDA", "[array-lib]") { TEST_CONSTRUCTORS(int16_t, CUDA); }
TEST_CASE("Test Array -- uint16_t CUDA", "[array-lib]") { TEST_CONSTRUCTORS(uint16_t, CUDA); }

// TEST_CASE("Test Array -- int8_t CUDA", "[array-lib]") { TEST_ALL(int8_t, CUDA); }
// TEST_CASE("Test Array -- uint8_t CUDA", "[array-lib]") { TEST_ALL(uint8_t, CUDA); }
// TEST_CASE("Test Array -- int16_t CUDA", "[array-lib]") { TEST_ALL(int16_t, CUDA); }
// TEST_CASE("Test Array -- uint16_t CUDA", "[array-lib]") { TEST_ALL(uint16_t, CUDA); }
TEST_CASE("Test Array -- int32_t CUDA", "[array-lib]") { TEST_CONSTRUCTORS(int32_t, CUDA); }
TEST_CASE("Test Array -- uint32_t CUDA", "[array-lib]") { TEST_CONSTRUCTORS(uint32_t, CUDA); }
TEST_CASE("Test Array -- int64_t CUDA", "[array-lib]") { TEST_CONSTRUCTORS(int64_t, CUDA); }
TEST_CASE("Test Array -- uint64_t CUDA", "[array-lib]") { TEST_CONSTRUCTORS(uint64_t, CUDA); }
TEST_CASE("Test Array -- float CUDA", "[array-lib]") { TEST_CONSTRUCTORS(float, CUDA); }
TEST_CASE("Test Array -- double CUDA", "[array-lib]") { TEST_CONSTRUCTORS(double, CUDA); }

#endif // LIBRAPID_HAS_CUDA
