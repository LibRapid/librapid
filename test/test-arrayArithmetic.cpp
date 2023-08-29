#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 0.001;
using CPU				   = lrc::backend::CPU;
using OPENCL			   = lrc::backend::OpenCL;
using CUDA				   = lrc::backend::CUDA;

#define TEST_ARITHMETIC(SCALAR, BACKEND)                                                           \
	SECTION(                                                                                       \
	  fmt::format("Test Array Operations [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {     \
		lrc::Array<SCALAR, BACKEND>::ShapeType shape({37, 41});                                    \
		lrc::Array<SCALAR, BACKEND> testA(shape); /* Prime-dimensioned to force wrapping */        \
		lrc::Array<SCALAR, BACKEND> testB(shape);                                                  \
                                                                                                   \
		for (int64_t i = 0; i < shape[0]; ++i) {                                                   \
			for (int64_t j = 0; j < shape[1]; ++j) {                                               \
				SCALAR a = j + i * shape[1] + 1;                                                   \
				SCALAR b = i + j * shape[0] + 1;                                                   \
                                                                                                   \
				testA[i][j] = a;                                                                   \
				testB[i][j] = b != 0 ? b : 1;                                                      \
			}                                                                                      \
		}                                                                                          \
                                                                                                   \
		auto negResult = (-testA).eval();                                                          \
		bool negValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(negResult.scalar(i) == -(testA.scalar(i)))) {                                    \
				REQUIRE(lrc::isClose(negResult.scalar(i), -(testA.scalar(i)), tolerance));         \
				negValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(negValid);                                                                         \
                                                                                                   \
		auto sumResult = (testA + testB).eval();                                                   \
		bool sumValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(sumResult.scalar(i) == testA.scalar(i) + testB.scalar(i))) {                     \
				REQUIRE(lrc::isClose(                                                              \
				  sumResult.scalar(i), testA.scalar(i) + testB.scalar(i), tolerance));             \
				sumValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(sumValid);                                                                         \
                                                                                                   \
		auto diffResult = (testA - testB).eval();                                                  \
		bool diffValid	= true;                                                                    \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(diffResult.scalar(i) == testA.scalar(i) - testB.scalar(i))) {                    \
				REQUIRE(lrc::isClose(                                                              \
				  diffResult.scalar(i), testA.scalar(i) - testB.scalar(i), tolerance));            \
				diffValid = false;                                                                 \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(diffValid);                                                                        \
                                                                                                   \
		auto prodResult = (testA * testB).eval();                                                  \
		bool prodValid	= true;                                                                    \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(prodResult.scalar(i) == testA.scalar(i) * testB.scalar(i))) {                    \
				REQUIRE(lrc::isClose(                                                              \
				  prodResult.scalar(i), testA.scalar(i) * testB.scalar(i), tolerance));            \
				prodValid = false;                                                                 \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(prodValid);                                                                        \
                                                                                                   \
		auto divResult = (testA / testB).eval();                                                   \
		bool divValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(divResult.scalar(i) == testA.scalar(i) / testB.scalar(i))) {                     \
				REQUIRE(lrc::isClose(                                                              \
				  divResult.scalar(i), testA.scalar(i) / testB.scalar(i), tolerance));             \
				divValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(diffValid);                                                                        \
	}                                                                                              \
	do {                                                                                           \
	} while (false)

#define TEST_ARITHMETIC_ARRAY_SCALAR(SCALAR, BACKEND)                                              \
	SECTION(fmt::format(                                                                           \
	  "Test Array-Scalar Operations [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {          \
		lrc::Array<SCALAR, BACKEND>::ShapeType shape({37, 41});                                    \
		lrc::Array<SCALAR, BACKEND> testA(shape); /* Prime-dimensioned to force wrapping */        \
                                                                                                   \
		for (int64_t i = 0; i < shape[0]; ++i) {                                                   \
			for (int64_t j = 0; j < shape[1]; ++j) {                                               \
				SCALAR a	= j + i * shape[1] + SCALAR(1);                                        \
				testA[i][j] = a;                                                                   \
			}                                                                                      \
		}                                                                                          \
                                                                                                   \
		auto sumResult = (testA + SCALAR(1)).eval();                                               \
		bool sumValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(sumResult.scalar(i) == testA.scalar(i) + SCALAR(1))) {                           \
				REQUIRE(                                                                           \
				  lrc::isClose(sumResult.scalar(i), testA.scalar(i) + SCALAR(1), tolerance));      \
				sumValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(sumValid);                                                                         \
                                                                                                   \
		auto diffResult = (testA - SCALAR(1)).eval();                                              \
		bool diffValid	= true;                                                                    \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(diffResult.scalar(i) == testA.scalar(i) - SCALAR(1))) {                          \
				REQUIRE(                                                                           \
				  lrc::isClose(diffResult.scalar(i), testA.scalar(i) - SCALAR(1), tolerance));     \
				diffValid = false;                                                                 \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(diffValid);                                                                        \
                                                                                                   \
		auto prodResult = (testA * SCALAR(2)).eval();                                              \
		bool prodValid	= true;                                                                    \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(prodResult.scalar(i) == testA.scalar(i) * SCALAR(2))) {                          \
				REQUIRE(                                                                           \
				  lrc::isClose(prodResult.scalar(i), testA.scalar(i) * SCALAR(2), tolerance));     \
				prodValid = false;                                                                 \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(prodValid);                                                                        \
                                                                                                   \
		auto divResult = (testA / SCALAR(2)).eval();                                               \
		bool divValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(divResult.scalar(i) == testA.scalar(i) / SCALAR(2))) {                           \
				REQUIRE(                                                                           \
				  lrc::isClose(divResult.scalar(i), testA.scalar(i) / SCALAR(2), tolerance));      \
				divValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(diffValid);                                                                        \
	}                                                                                              \
	do {                                                                                           \
	} while (false)

#define TEST_ARITHMETIC_SCALAR_ARRAY(SCALAR, BACKEND)                                              \
	SECTION(fmt::format(                                                                           \
	  "Test Scalar-Array Operations [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {          \
		lrc::Array<SCALAR, BACKEND>::ShapeType shape({37, 41});                                    \
		lrc::Array<SCALAR, BACKEND> testB(shape);                                                  \
                                                                                                   \
		for (int64_t i = 0; i < shape[0]; ++i) {                                                   \
			for (int64_t j = 0; j < shape[1]; ++j) {                                               \
				SCALAR b	= i + j * shape[0] + 1;                                                \
				testB[i][j] = b != 0 ? b : 1;                                                      \
			}                                                                                      \
		}                                                                                          \
                                                                                                   \
		auto sumResult = (SCALAR(1) + testB).eval();                                               \
		bool sumValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(sumResult.scalar(i) == SCALAR(1) + testB.scalar(i))) {                           \
				REQUIRE(                                                                           \
				  lrc::isClose(sumResult.scalar(i), SCALAR(1) + testB.scalar(i), tolerance));      \
				sumValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(sumValid);                                                                         \
                                                                                                   \
		auto diffResult = (1 - testB).eval();                                                      \
		bool diffValid	= true;                                                                    \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(diffResult.scalar(i) == SCALAR(1) - testB.scalar(i))) {                          \
				REQUIRE(                                                                           \
				  lrc::isClose(diffResult.scalar(i), SCALAR(1) - testB.scalar(i), tolerance));     \
				diffValid = false;                                                                 \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(diffValid);                                                                        \
                                                                                                   \
		auto prodResult = (SCALAR(2) * testB).eval();                                              \
		bool prodValid	= true;                                                                    \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(prodResult.scalar(i) == SCALAR(2) * testB.scalar(i))) {                          \
				REQUIRE(                                                                           \
				  lrc::isClose(prodResult.scalar(i), SCALAR(2) * testB.scalar(i), tolerance));     \
				prodValid = false;                                                                 \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(prodValid);                                                                        \
                                                                                                   \
		auto divResult = (SCALAR(2) / testB).eval();                                               \
		bool divValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(divResult.scalar(i) == SCALAR(2) / testB.scalar(i))) {                           \
				REQUIRE(                                                                           \
				  lrc::isClose(divResult.scalar(i), SCALAR(2) / testB.scalar(i), tolerance));      \
				divValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(diffValid);                                                                        \
	}                                                                                              \
	do {                                                                                           \
	} while (false)

#define TEST_ALL(SCALAR, BACKEND)                                                                  \
	TEST_ARITHMETIC(SCALAR, BACKEND);                                                              \
	TEST_ARITHMETIC_ARRAY_SCALAR(SCALAR, BACKEND);                                                 \
	TEST_ARITHMETIC_SCALAR_ARRAY(SCALAR, BACKEND);

TEST_CASE("Test Array -- int32_t CPU", "[array-lib]") { TEST_ALL(int32_t, CPU); }
TEST_CASE("Test Array -- uint32_t CPU", "[array-lib]") { TEST_ALL(uint32_t, CPU); }
TEST_CASE("Test Array -- int64_t CPU", "[array-lib]") { TEST_ALL(int64_t, CPU); }
TEST_CASE("Test Array -- uint64_t CPU", "[array-lib]") { TEST_ALL(uint64_t, CPU); }
TEST_CASE("Test Array -- float CPU", "[array-lib]") { TEST_ALL(float, CPU); }
TEST_CASE("Test Array -- double CPU", "[array-lib]") { TEST_ALL(double, CPU); }

#if defined(LIBRAPID_USE_MULTIPREC)
TEST_CASE("Test Array -- lrc::mpfr CPU", "[array-lib]") { TEST_ALL(lrc::mpfr, CPU); }
#endif // LIBRAPID_USE_MULTIPREC

#if defined(LIBRAPID_HAS_OPENCL)

TEST_CASE("Configure OpenCL") { lrc::configureOpenCL(true); }

TEST_CASE("Test Array -- int32_t OpenCL", "[array-lib]") { TEST_ALL(int32_t, OPENCL); }
TEST_CASE("Test Array -- uint32_t OpenCL", "[array-lib]") { TEST_ALL(uint32_t, OPENCL); }
TEST_CASE("Test Array -- int64_t OpenCL", "[array-lib]") { TEST_ALL(int64_t, OPENCL); }
TEST_CASE("Test Array -- uint64_t OpenCL", "[array-lib]") { TEST_ALL(uint64_t, OPENCL); }
TEST_CASE("Test Array -- float OpenCL", "[array-lib]") { TEST_ALL(float, OPENCL); }
TEST_CASE("Test Array -- double OpenCL", "[array-lib]") { TEST_ALL(double, OPENCL); }

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

// TEST_CASE("Test Array -- int8_t CUDA", "[array-lib]") { TEST_ALL(int8_t, CUDA); }
// TEST_CASE("Test Array -- uint8_t CUDA", "[array-lib]") { TEST_ALL(uint8_t, CUDA); }
// TEST_CASE("Test Array -- int16_t CUDA", "[array-lib]") { TEST_ALL(int16_t, CUDA); }
// TEST_CASE("Test Array -- uint16_t CUDA", "[array-lib]") { TEST_ALL(uint16_t, CUDA); }
TEST_CASE("Test Array -- int32_t CUDA", "[array-lib]") { TEST_ALL(int32_t, CUDA); }
TEST_CASE("Test Array -- uint32_t CUDA", "[array-lib]") { TEST_ALL(uint32_t, CUDA); }
TEST_CASE("Test Array -- int64_t CUDA", "[array-lib]") { TEST_ALL(int64_t, CUDA); }
TEST_CASE("Test Array -- uint64_t CUDA", "[array-lib]") { TEST_ALL(uint64_t, CUDA); }
TEST_CASE("Test Array -- float CUDA", "[array-lib]") { TEST_ALL(float, CUDA); }
TEST_CASE("Test Array -- double CUDA", "[array-lib]") { TEST_ALL(double, CUDA); }

#endif // LIBRAPID_HAS_CUDA
