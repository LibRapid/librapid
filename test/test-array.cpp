#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

#define TEST_CONSTRUCTORS(SCALAR, DEVICE)                                                          \
	SECTION(fmt::format("Test Constructors [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(DEVICE))) {    \
		lrc::Array<SCALAR, DEVICE> testA;                                                          \
		REQUIRE(testA.shape() == lrc::Array<SCALAR, DEVICE>::ShapeType {0});                       \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testB(lrc::Array<SCALAR, DEVICE>::ShapeType {3, 4});            \
		REQUIRE(testB.shape() == lrc::Array<SCALAR, DEVICE>::ShapeType {3, 4});                    \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testC(lrc::Array<SCALAR, DEVICE>::ShapeType {3, 4}, 5);         \
		REQUIRE(testC.shape() == lrc::Array<SCALAR, DEVICE>::ShapeType {3, 4});                    \
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
		lrc::Array<SCALAR, DEVICE>::ShapeType tmpShape({2, 3});                                    \
		lrc::Array<SCALAR, DEVICE> testE(std::move(tmpShape));                                     \
		REQUIRE(testE.shape() == lrc::Array<SCALAR, DEVICE>::ShapeType {2, 3});                    \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testF(testC);                                                   \
		REQUIRE(testF.shape() == lrc::Array<SCALAR, DEVICE>::ShapeType {3, 4});                    \
		REQUIRE(testF.storage()[0] == 5);                                                          \
		REQUIRE(testF.storage()[1] == 5);                                                          \
		REQUIRE(testF.storage()[2] == 5);                                                          \
		REQUIRE(testF.storage()[9] == 5);                                                          \
		REQUIRE(testF.storage()[10] == 5);                                                         \
		REQUIRE(testF.storage()[11] == 5);                                                         \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testG(lrc::Array<SCALAR, DEVICE>::ShapeType {3, 4}, 10);        \
		testC = testG;                                                                             \
		REQUIRE(testC.storage()[0] == 10);                                                         \
		REQUIRE(testC.storage()[1] == 10);                                                         \
		REQUIRE(testC.storage()[2] == 10);                                                         \
		REQUIRE(testC.storage()[9] == 10);                                                         \
		REQUIRE(testC.storage()[10] == 10);                                                        \
		REQUIRE(testC.storage()[11] == 10);                                                        \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testH(lrc::Array<SCALAR, DEVICE>::ShapeType {3, 3});            \
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
	}

#define TEST_INDEXING(SCALAR, DEVICE)                                                              \
	SECTION(fmt::format("Test Indexing [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(DEVICE))) {        \
		lrc::Array<SCALAR, DEVICE> testA(lrc::Array<SCALAR, DEVICE>::ShapeType({5, 3}));           \
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
		REQUIRE(testA[0][0].get() == SCALAR(1));                                                   \
		REQUIRE(testA[1][1].get() == SCALAR(5));                                                   \
		REQUIRE(testA[2][2].get() == SCALAR(9));                                                   \
                                                                                                   \
		testA[0][0] = 123;                                                                         \
		testA[1][1] = 456;                                                                         \
		testA[2][2] = 789;                                                                         \
		REQUIRE((SCALAR)testA.storage()[0] == SCALAR(123));                                        \
		REQUIRE((SCALAR)testA.storage()[4] == SCALAR(456));                                        \
		REQUIRE((SCALAR)testA.storage()[8] == SCALAR(789));                                        \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testB(lrc::Array<SCALAR, DEVICE>::ShapeType({10}));             \
		testB << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;                                                    \
		REQUIRE(testB[0].get() == SCALAR(1));                                                      \
		REQUIRE(testB[9].get() == SCALAR(10));                                                     \
	}

#define TEST_STRING_FORMATTING(SCALAR, DEVICE)                                                     \
	SECTION(                                                                                       \
	  fmt::format("Test String Formatting [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(DEVICE))) {     \
		lrc::Array<SCALAR, DEVICE> testA(lrc::Array<SCALAR, DEVICE>::ShapeType({2, 3}));           \
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
		lrc::Array<SCALAR, DEVICE> testB(lrc::Array<SCALAR, DEVICE>::ShapeType({2, 3}));           \
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
		lrc::Array<SCALAR, DEVICE> testC(lrc::Array<SCALAR, DEVICE>::ShapeType({2, 2, 2}));        \
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

#define TEST_ARITHMETIC(SCALAR, DEVICE)                                                            \
	SECTION(fmt::format("Test Operations [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(DEVICE))) {      \
		lrc::Array<SCALAR, DEVICE>::ShapeType shape({37, 41});                                     \
		lrc::Array<SCALAR, DEVICE> testA(shape); /* Prime-dimensioned to force wrapping */         \
		lrc::Array<SCALAR, DEVICE> testB(shape);                                                   \
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
		auto sumResult = (testA + testB).eval();                                                   \
		bool sumValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(sumResult.scalar(i) == testA.scalar(i) + testB.scalar(i))) {                     \
				REQUIRE(sumResult.scalar(i) == testA.scalar(i) + testB.scalar(i));                 \
				sumValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(sumValid);                                                                         \
                                                                                                   \
		auto diffResult = (testA - testB).eval();                                                  \
		bool diffValid	= true;                                                                    \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(diffResult.scalar(i) == testA.scalar(i) - testB.scalar(i))) {                    \
				REQUIRE(diffResult.scalar(i) == testA.scalar(i) - testB.scalar(i));                \
				diffValid = false;                                                                 \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(diffValid);                                                                        \
                                                                                                   \
		auto prodResult = (testA * testB).eval();                                                  \
		bool prodValid	= true;                                                                    \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(prodResult.scalar(i) == testA.scalar(i) * testB.scalar(i))) {                    \
				REQUIRE(prodResult.scalar(i) == testA.scalar(i) * testB.scalar(i));                \
				prodValid = false;                                                                 \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(prodValid);                                                                        \
                                                                                                   \
		auto divResult = (testA / testB).eval();                                                   \
		bool divValid  = true;                                                                     \
		for (int64_t i = 0; i < shape[0] * shape[1]; ++i) {                                        \
			if (!(divResult.scalar(i) == testA.scalar(i) / testB.scalar(i))) {                     \
				REQUIRE(divResult.scalar(i) == testA.scalar(i) / testB.scalar(i));                 \
				divValid = false;                                                                  \
			}                                                                                      \
		}                                                                                          \
		REQUIRE(diffValid);                                                                        \
	}                                                                                              \
	do {                                                                                           \
	} while (false)

#define TEST_COMPARISONS(SCALAR, DEVICE)                                                           \
	SECTION(fmt::format("Test Operations [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(DEVICE))) {      \
		lrc::Array<SCALAR, DEVICE>::ShapeType shape({53, 79});                                     \
		lrc::Array<SCALAR, DEVICE> testA(shape); /* Prime-dimensioned to force wrapping */         \
		lrc::Array<SCALAR, DEVICE> testB(shape);                                                   \
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

TEST_CASE("Test Array -- int8_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int8_t, lrc::device::CPU);
	TEST_INDEXING(int8_t, lrc::device::CPU);
	TEST_STRING_FORMATTING(int8_t, lrc::device::CPU);
}

TEST_CASE("Test Array -- uint8_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint8_t, lrc::device::CPU);
	TEST_INDEXING(uint8_t, lrc::device::CPU);
	TEST_STRING_FORMATTING(uint8_t, lrc::device::CPU);
}

TEST_CASE("Test Array -- int16_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int16_t, lrc::device::CPU);
	TEST_INDEXING(int16_t, lrc::device::CPU);
	TEST_STRING_FORMATTING(int16_t, lrc::device::CPU);
}

TEST_CASE("Test Array -- uint16_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint16_t, lrc::device::CPU);
	TEST_INDEXING(uint16_t, lrc::device::CPU);
	TEST_STRING_FORMATTING(uint16_t, lrc::device::CPU);
}

TEST_CASE("Test Array -- int32_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int32_t, lrc::device::CPU);
	TEST_INDEXING(int32_t, lrc::device::CPU);
	TEST_STRING_FORMATTING(int32_t, lrc::device::CPU);
	TEST_ARITHMETIC(int32_t, lrc::device::CPU);
	TEST_COMPARISONS(int32_t, lrc::device::CPU);
}

TEST_CASE("Test Array -- uint32_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint32_t, lrc::device::CPU);
	TEST_INDEXING(uint32_t, lrc::device::CPU);
	TEST_ARITHMETIC(int32_t, lrc::device::CPU);
	TEST_STRING_FORMATTING(uint32_t, lrc::device::CPU);
	TEST_COMPARISONS(uint32_t, lrc::device::CPU);
}

TEST_CASE("Test Array -- int64_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int64_t, lrc::device::CPU);
	TEST_INDEXING(uint32_t, lrc::device::CPU);
	TEST_ARITHMETIC(int64_t, lrc::device::CPU);
	TEST_STRING_FORMATTING(int64_t, lrc::device::CPU);
	TEST_COMPARISONS(int64_t, lrc::device::CPU);
}

TEST_CASE("Test Array -- uint64_t CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint64_t, lrc::device::CPU);
	TEST_INDEXING(uint64_t, lrc::device::CPU);
	TEST_ARITHMETIC(uint64_t, lrc::device::CPU);
	TEST_STRING_FORMATTING(uint64_t, lrc::device::CPU);
	TEST_COMPARISONS(uint64_t, lrc::device::CPU);
}

TEST_CASE("Test Array -- float CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(float, lrc::device::CPU);
	TEST_INDEXING(float, lrc::device::CPU);
	TEST_ARITHMETIC(float, lrc::device::CPU);
	TEST_STRING_FORMATTING(float, lrc::device::CPU);
	TEST_COMPARISONS(float, lrc::device::CPU);
}

TEST_CASE("Test Array -- double CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(double, lrc::device::CPU);
	TEST_INDEXING(double, lrc::device::CPU);
	TEST_ARITHMETIC(double, lrc::device::CPU);
	TEST_STRING_FORMATTING(double, lrc::device::CPU);
	TEST_COMPARISONS(double, lrc::device::CPU);
}

#if defined(LIBRAPID_USE_MULTIPREC)

TEST_CASE("Test Array -- lrc::mpfr CPU", "[array-lib]") {
	TEST_CONSTRUCTORS(lrc::mpfr, lrc::device::CPU);
	TEST_INDEXING(lrc::mpfr, lrc::device::CPU);
	TEST_ARITHMETIC(lrc::mpfr, lrc::device::CPU);
	TEST_STRING_FORMATTING(lrc::mpfr, lrc::device::CPU);
	TEST_COMPARISONS(lrc::mpfr, lrc::device::CPU);
}

#endif // LIBRAPID_USE_MULTIPREC

#if defined(LIBRAPID_HAS_CUDA)

TEST_CASE("Test Array -- int8_t GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int8_t, lrc::device::GPU);
	TEST_INDEXING(int8_t, lrc::device::GPU);
	TEST_ARITHMETIC(int8_t, lrc::device::GPU);
	TEST_STRING_FORMATTING(int8_t, lrc::device::GPU);
	TEST_COMPARISONS(int8_t, lrc::device::GPU);
}

TEST_CASE("Test Array -- uint8_t GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint8_t, lrc::device::GPU);
	TEST_INDEXING(uint8_t, lrc::device::GPU);
	TEST_ARITHMETIC(uint8_t, lrc::device::GPU);
	TEST_STRING_FORMATTING(uint8_t, lrc::device::GPU);
	TEST_COMPARISONS(uint8_t, lrc::device::GPU);
}

TEST_CASE("Test Array -- int16_t GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int16_t, lrc::device::GPU);
	TEST_INDEXING(int16_t, lrc::device::GPU);
	TEST_ARITHMETIC(int16_t, lrc::device::GPU);
	TEST_STRING_FORMATTING(int16_t, lrc::device::GPU);
	TEST_COMPARISONS(int16_t, lrc::device::GPU);
	TEST_COMPARISONS(int16_t, lrc::device::GPU);
}

TEST_CASE("Test Array -- uint16_t GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint16_t, lrc::device::GPU);
	TEST_INDEXING(uint16_t, lrc::device::GPU);
	TEST_ARITHMETIC(uint16_t, lrc::device::GPU);
	TEST_STRING_FORMATTING(uint16_t, lrc::device::GPU);
	TEST_COMPARISONS(uint16_t, lrc::device::GPU);
}

TEST_CASE("Test Array -- int32_t GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int32_t, lrc::device::GPU);
	TEST_INDEXING(int32_t, lrc::device::GPU);
	TEST_ARITHMETIC(int32_t, lrc::device::GPU);
	TEST_STRING_FORMATTING(int32_t, lrc::device::GPU);
	TEST_COMPARISONS(int32_t, lrc::device::GPU);
}

TEST_CASE("Test Array -- uint32_t GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint32_t, lrc::device::GPU);
	TEST_INDEXING(uint32_t, lrc::device::GPU);
	TEST_ARITHMETIC(uint32_t, lrc::device::GPU);
	TEST_STRING_FORMATTING(uint32_t, lrc::device::GPU);
	TEST_COMPARISONS(uint32_t, lrc::device::GPU);
}

TEST_CASE("Test Array -- int64_t GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(int64_t, lrc::device::GPU);
	TEST_INDEXING(int64_t, lrc::device::GPU);
	TEST_ARITHMETIC(int64_t, lrc::device::GPU);
	TEST_STRING_FORMATTING(int64_t, lrc::device::GPU);
	TEST_COMPARISONS(int64_t, lrc::device::GPU);
}

TEST_CASE("Test Array -- uint64_t GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(uint64_t, lrc::device::GPU);
	TEST_INDEXING(uint64_t, lrc::device::GPU);
	TEST_ARITHMETIC(uint64_t, lrc::device::GPU);
	TEST_STRING_FORMATTING(uint64_t, lrc::device::GPU);
	TEST_COMPARISONS(uint64_t, lrc::device::GPU);
}

TEST_CASE("Test Array -- float GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(float, lrc::device::GPU);
	TEST_INDEXING(float, lrc::device::GPU);
	TEST_ARITHMETIC(float, lrc::device::GPU);
	TEST_STRING_FORMATTING(float, lrc::device::GPU);
	TEST_COMPARISONS(float, lrc::device::GPU);
}

TEST_CASE("Test Array -- double GPU", "[array-lib]") {
	TEST_CONSTRUCTORS(double, lrc::device::GPU);
	TEST_INDEXING(double, lrc::device::GPU);
	TEST_ARITHMETIC(double, lrc::device::GPU);
	TEST_STRING_FORMATTING(double, lrc::device::GPU);
	TEST_COMPARISONS(double, lrc::device::GPU);
}

#endif // LIBRAPID_HAS_CUDA
