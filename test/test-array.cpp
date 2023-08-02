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
		REQUIRE(testI.str() == fmt::format("[[[{} {}]\n  [{} {}]]\n\n [[{} {}]\n  [{} {}]]]",      \
										   SCALAR(1),                                              \
										   SCALAR(2),                                              \
										   SCALAR(3),                                              \
										   SCALAR(4),                                              \
										   SCALAR(5),                                              \
										   SCALAR(6),                                              \
										   SCALAR(7),                                              \
										   SCALAR(8)));                                            \
                                                                                                   \
		auto testJ =                                                                               \
		  lrc::Array<SCALAR, BACKEND>::fromData(Vec({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));        \
		REQUIRE(testJ.str() == fmt::format("[[[{} {}]\n  [{} {}]]\n\n [[{} {}]\n  [{} {}]]]",      \
										   SCALAR(1),                                              \
										   SCALAR(2),                                              \
										   SCALAR(3),                                              \
										   SCALAR(4),                                              \
										   SCALAR(5),                                              \
										   SCALAR(6),                                              \
										   SCALAR(7),                                              \
										   SCALAR(8)));                                            \
	}

#define TEST_INDEXING(SCALAR, BACKEND)                                                             \
	SECTION(fmt::format("Test Indexing [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {       \
		lrc::Array<SCALAR, BACKEND> testA(lrc::Array<SCALAR, BACKEND>::ShapeType({5, 3}));         \
		testA << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;                                \
		std::string index0 = fmt::format("[{} {} {}]", SCALAR(1), SCALAR(2), SCALAR(3));           \
		std::string index1 = fmt::format("[{} {} {}]", SCALAR(4), SCALAR(5), SCALAR(6));           \
		std::string index2 = fmt::format("[{} {} {}]", SCALAR(7), SCALAR(8), SCALAR(9));           \
		std::string index3 = fmt::format("[{} {} {}]", SCALAR(10), SCALAR(11), SCALAR(12));        \
		std::string index4 = fmt::format("[{} {} {}]", SCALAR(13), SCALAR(14), SCALAR(15));        \
		REQUIRE(testA[0].str() == index0);                                                         \
		REQUIRE(testA[1].str() == index1);                                                         \
		REQUIRE(testA[2].str() == index2);                                                         \
		REQUIRE(testA[3].str() == index3);                                                         \
		REQUIRE(testA[4].str() == index4);                                                         \
		REQUIRE(testA[0][0].str() == fmt::format("{}", SCALAR(1)));                                \
		REQUIRE(testA[1][1].str() == fmt::format("{}", SCALAR(5)));                                \
		REQUIRE(testA[2][2].str() == fmt::format("{}", SCALAR(9)));                                \
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

#define TEST_STRING_FORMATTING(SCALAR, BACKEND)                                                    \
	SECTION(                                                                                       \
	  fmt::format("Test String Formatting [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(BACKEND))) {    \
		lrc::Array<SCALAR, BACKEND> testA(lrc::Array<SCALAR, BACKEND>::ShapeType({2, 3}));         \
		testA << 1, 2, 3, 4, 5, 6;                                                                 \
                                                                                                   \
		REQUIRE(testA.str() == fmt::format("[[{} {} {}]\n [{} {} {}]]",                            \
										   SCALAR(1),                                              \
										   SCALAR(2),                                              \
										   SCALAR(3),                                              \
										   SCALAR(4),                                              \
										   SCALAR(5),                                              \
										   SCALAR(6)));                                            \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testB(lrc::Array<SCALAR, BACKEND>::ShapeType({2, 3}));         \
		testB << 10, 2, 3, 4, 5, 6;                                                                \
                                                                                                   \
		REQUIRE(testB.str() == fmt::format("[[{} {} {}]\n [ {} {} {}]]",                           \
										   SCALAR(10),                                             \
										   SCALAR(2),                                              \
										   SCALAR(3),                                              \
										   SCALAR(4),                                              \
										   SCALAR(5),                                              \
										   SCALAR(6)));                                            \
                                                                                                   \
		lrc::Array<SCALAR, BACKEND> testC(lrc::Array<SCALAR, BACKEND>::ShapeType({2, 2, 2}));      \
		testC << 100, 2, 3, 4, 5, 6, 7, 8;                                                         \
		REQUIRE(testC.str() ==                                                                     \
				fmt::format("[[[{} {}]\n  [  {} {}]]\n\n [[  {} {}]\n  [  {} {}]]]",               \
							SCALAR(100),                                                           \
							SCALAR(2),                                                             \
							SCALAR(3),                                                             \
							SCALAR(4),                                                             \
							SCALAR(5),                                                             \
							SCALAR(6),                                                             \
							SCALAR(7),                                                             \
							SCALAR(8)));                                                           \
	}

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
	TEST_CONSTRUCTORS(SCALAR, BACKEND);                                                            \
	TEST_INDEXING(SCALAR, BACKEND);                                                                \
	TEST_STRING_FORMATTING(SCALAR, BACKEND);                                                       \
	TEST_ARITHMETIC(SCALAR, BACKEND);                                                              \
	TEST_ARITHMETIC_ARRAY_SCALAR(SCALAR, BACKEND);                                                 \
	TEST_ARITHMETIC_SCALAR_ARRAY(SCALAR, BACKEND);                                                 \
	TEST_COMPARISONS(SCALAR, BACKEND);                                                             \
	TEST_COMPARISONS_ARRAY_SCALAR(SCALAR, BACKEND);                                                \
	TEST_COMPARISONS_SCALAR_ARRAY(SCALAR, BACKEND);

TEST_CASE("Test Array -- int8_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int8_t, CPU);
	TEST_INDEXING(int8_t, CPU);
	TEST_STRING_FORMATTING(int8_t, CPU);
}

TEST_CASE("Test Array -- uint8_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint8_t, CPU);
	TEST_INDEXING(uint8_t, CPU);
	TEST_STRING_FORMATTING(uint8_t, CPU);
}

TEST_CASE("Test Array -- int16_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int16_t, CPU);
	TEST_INDEXING(int16_t, CPU);
	TEST_STRING_FORMATTING(int16_t, CPU);
}

TEST_CASE("Test Array -- uint16_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint16_t, CPU);
	TEST_INDEXING(uint16_t, CPU);
	TEST_STRING_FORMATTING(uint16_t, CPU);
}

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

TEST_CASE("Test Array -- int8_t CUDA", "[array-lib]") {
	TEST_CONSTRUCTORS(int8_t, CUDA);
	TEST_INDEXING(int8_t, CUDA);
	TEST_STRING_FORMATTING(int8_t, CUDA);
}

TEST_CASE("Test Array -- uint8_t CUDA", "[array-lib]") {
	TEST_CONSTRUCTORS(uint8_t, CUDA);
	TEST_INDEXING(uint8_t, CUDA);
	TEST_STRING_FORMATTING(uint8_t, CUDA);
}

TEST_CASE("Test Array -- int16_t CUDA", "[array-lib]") {
	TEST_CONSTRUCTORS(int16_t, CUDA);
	TEST_INDEXING(int16_t, CUDA);
	TEST_STRING_FORMATTING(int16_t, CUDA);
}

TEST_CASE("Test Array -- uint16_t CUDA", "[array-lib]") {
	TEST_CONSTRUCTORS(uint16_t, CUDA);
	TEST_INDEXING(uint16_t, CUDA);
	TEST_STRING_FORMATTING(uint16_t, CUDA);
}

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
