#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc			   = librapid;
constexpr double tolerance = 0.001;
using CPU				   = lrc::backend::CPU;
using OPENCL			   = lrc::backend::OpenCL;
using CUDA				   = lrc::backend::CUDA;

#define TEST_COMPARISONS(SCALAR, BACKEND)                                                          \
	SECTION(                                                                                       \
	  fmt::format("Test Array Comparisons [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {    \
		lrc::Array<SCALAR, BACKEND>::ShapeType shape({53, 79});                                    \
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
		auto gtResult = (testA > testB).eval();                                                    \
		bool gtValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(gtResult.scalar(i) == testA.scalar(i) > testB.scalar(i))) {                      \
				REQUIRE(gtResult.scalar(i) == testA.scalar(i) > testB.scalar(i));                  \
				gtValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(gtValid);                                                                          \
                                                                                                   \
		auto geResult = (testA >= testB).eval();                                                   \
		bool geValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(geResult.scalar(i) == testA.scalar(i) >= testB.scalar(i))) {                     \
				REQUIRE(geResult.scalar(i) == testA.scalar(i) >= testB.scalar(i));                 \
				geValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(geValid);                                                                          \
                                                                                                   \
		auto ltResult = (testA < testB).eval();                                                    \
		bool ltValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(ltResult.scalar(i) == testA.scalar(i) < testB.scalar(i))) {                      \
				REQUIRE(ltResult.scalar(i) == testA.scalar(i) < testB.scalar(i));                  \
				ltValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(ltValid);                                                                          \
                                                                                                   \
		auto leResult = (testA <= testB).eval();                                                   \
		bool leValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(leResult.scalar(i) == testA.scalar(i) <= testB.scalar(i))) {                     \
				REQUIRE(leResult.scalar(i) == testA.scalar(i) <= testB.scalar(i));                 \
				leValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(leValid);                                                                          \
                                                                                                   \
		auto eqResult = (testA == testB).eval();                                                   \
		bool eqValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(eqResult.scalar(i) == (testA.scalar(i) == testB.scalar(i)))) {                   \
				REQUIRE(eqResult.scalar(i) == (testA.scalar(i) == testB.scalar(i)));               \
				eqValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(eqValid);                                                                          \
                                                                                                   \
		auto neResult = (testA != testB).eval();                                                   \
		bool neValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(neResult.scalar(i) == (testA.scalar(i) != testB.scalar(i)))) {                   \
				REQUIRE(neResult.scalar(i) == (testA.scalar(i) != testB.scalar(i)));               \
				neValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(neValid);                                                                          \
	}                                                                                              \
	do {                                                                                           \
	} while (false)

#define TEST_COMPARISONS_ARRAY_SCALAR(SCALAR, BACKEND)                                             \
	SECTION(fmt::format(                                                                           \
	  "Test Array-Scalar Comparisons [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {         \
		lrc::Array<SCALAR, BACKEND>::ShapeType shape({53, 79});                                    \
		lrc::Array<SCALAR, BACKEND> testA(shape); /* Prime-dimensioned to force wrapping */        \
                                                                                                   \
		for (int64_t i = 0; i < shape[0]; ++i) {                                                   \
			for (int64_t j = 0; j < shape[1]; ++j) {                                               \
				SCALAR a	= j + i * shape[1] + 1;                                                \
				testA[i][j] = a;                                                                   \
			}                                                                                      \
		}                                                                                          \
                                                                                                   \
		auto gtResult = (testA > SCALAR(64)).eval();                                               \
		bool gtValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(gtResult.scalar(i) == testA.scalar(i) > SCALAR(64))) {                           \
				REQUIRE(gtResult.scalar(i) == testA.scalar(i) > SCALAR(64));                       \
				gtValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(gtValid);                                                                          \
                                                                                                   \
		auto geResult = (testA >= SCALAR(64)).eval();                                              \
		bool geValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(geResult.scalar(i) == testA.scalar(i) >= SCALAR(64))) {                          \
				REQUIRE(geResult.scalar(i) == testA.scalar(i) >= SCALAR(64));                      \
				geValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(geValid);                                                                          \
                                                                                                   \
		auto ltResult = (testA < SCALAR(64)).eval();                                               \
		bool ltValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(ltResult.scalar(i) == testA.scalar(i) < SCALAR(64))) {                           \
				REQUIRE(ltResult.scalar(i) == testA.scalar(i) < SCALAR(64));                       \
				ltValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(ltValid);                                                                          \
                                                                                                   \
		auto leResult = (testA <= SCALAR(64)).eval();                                              \
		bool leValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(leResult.scalar(i) == testA.scalar(i) <= SCALAR(64))) {                          \
				REQUIRE(leResult.scalar(i) == testA.scalar(i) <= SCALAR(64));                      \
				leValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(leValid);                                                                          \
                                                                                                   \
		auto eqResult = (testA == SCALAR(64)).eval();                                              \
		bool eqValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(eqResult.scalar(i) == (testA.scalar(i) == SCALAR(64)))) {                        \
				REQUIRE(eqResult.scalar(i) == (testA.scalar(i) == SCALAR(64)));                    \
				eqValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(eqValid);                                                                          \
                                                                                                   \
		auto neResult = (testA != SCALAR(64)).eval();                                              \
		bool neValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(neResult.scalar(i) == (testA.scalar(i) != SCALAR(64)))) {                        \
				REQUIRE(neResult.scalar(i) == (testA.scalar(i) != SCALAR(64)));                    \
				neValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(neValid);                                                                          \
	}                                                                                              \
	do {                                                                                           \
	} while (false)

#define TEST_COMPARISONS_SCALAR_ARRAY(SCALAR, BACKEND)                                             \
	SECTION(fmt::format(                                                                           \
	  "Test Scalar-Array Comparisons [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {         \
		lrc::Array<SCALAR, BACKEND>::ShapeType shape({53, 79});                                    \
		lrc::Array<SCALAR, BACKEND> testA(shape); /* Prime-dimensioned to force wrapping */        \
                                                                                                   \
		for (int64_t i = 0; i < shape[0]; ++i) {                                                   \
			for (int64_t j = 0; j < shape[1]; ++j) {                                               \
				SCALAR a	= j + i * shape[1] + 1;                                                \
				testA[i][j] = a;                                                                   \
			}                                                                                      \
		}                                                                                          \
                                                                                                   \
		auto gtResult = (SCALAR(64) > testA).eval();                                               \
		bool gtValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(gtResult.scalar(i) == SCALAR(64) > testA.scalar(i))) {                           \
				REQUIRE(gtResult.scalar(i) == SCALAR(64) > testA.scalar(i));                       \
				gtValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(gtValid);                                                                          \
                                                                                                   \
		auto geResult = (SCALAR(64) >= testA).eval();                                              \
		bool geValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(geResult.scalar(i) == SCALAR(64) >= testA.scalar(i))) {                          \
				REQUIRE(geResult.scalar(i) == SCALAR(64) >= testA.scalar(i));                      \
				geValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(geValid);                                                                          \
                                                                                                   \
		auto ltResult = (SCALAR(64) < testA).eval();                                               \
		bool ltValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(ltResult.scalar(i) == SCALAR(64) < testA.scalar(i))) {                           \
				REQUIRE(ltResult.scalar(i) == SCALAR(64) < testA.scalar(i));                       \
				ltValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(ltValid);                                                                          \
                                                                                                   \
		auto leResult = (SCALAR(64) <= testA).eval();                                              \
		bool leValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(leResult.scalar(i) == SCALAR(64) <= testA.scalar(i))) {                          \
				REQUIRE(leResult.scalar(i) == SCALAR(64) <= testA.scalar(i));                      \
				leValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(leValid);                                                                          \
                                                                                                   \
		auto eqResult = (SCALAR(64) == testA).eval();                                              \
		bool eqValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(eqResult.scalar(i) == (SCALAR(64) == testA.scalar(i)))) {                        \
				REQUIRE(eqResult.scalar(i) == (SCALAR(64) == testA.scalar(i)));                    \
				eqValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(eqValid);                                                                          \
                                                                                                   \
		auto neResult = (SCALAR(64) != testA).eval();                                              \
		bool neValid  = true;                                                                      \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(neResult.scalar(i) == (SCALAR(64) != testA.scalar(i)))) {                        \
				REQUIRE(neResult.scalar(i) == (SCALAR(64) != testA.scalar(i)));                    \
				neValid = false;                                                                   \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(neValid);                                                                          \
	}                                                                                              \
	do {                                                                                           \
	} while (false)

#define TEST_ALL(SCALAR, BACKEND)                                                                  \
	TEST_COMPARISONS(SCALAR, BACKEND);                                                             \
	TEST_COMPARISONS_ARRAY_SCALAR(SCALAR, BACKEND);                                                \
	TEST_COMPARISONS_SCALAR_ARRAY(SCALAR, BACKEND);

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
