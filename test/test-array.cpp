#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

#define TEST_SUITE(SCALAR, DEVICE)                                                                 \
	SECTION(fmt::format("Test Constructors [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(DEVICE))) {    \
		lrc::Array<SCALAR, DEVICE> testA;                                                          \
		REQUIRE(testA.shape() == lrc::Shape {0});                                                  \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testB(lrc::Shape {3, 4});                                       \
		REQUIRE(testB.shape() == lrc::Shape {3, 4});                                               \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testC(lrc::Shape {3, 4}, 5);                                    \
		REQUIRE(testC.shape() == lrc::Shape {3, 4});                                               \
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
		lrc::Shape tmpShape({2, 3});                                                               \
		lrc::Array<SCALAR, DEVICE> testE(std::move(tmpShape));                                     \
		REQUIRE(testE.shape() == lrc::Shape {2, 3});                                               \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testF(testC);                                                   \
		REQUIRE(testF.shape() == lrc::Shape {3, 4});                                               \
		REQUIRE(testF.storage()[0] == 5);                                                          \
		REQUIRE(testF.storage()[1] == 5);                                                          \
		REQUIRE(testF.storage()[2] == 5);                                                          \
		REQUIRE(testF.storage()[9] == 5);                                                          \
		REQUIRE(testF.storage()[10] == 5);                                                         \
		REQUIRE(testF.storage()[11] == 5);                                                         \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testG(lrc::Shape {3, 4}, 10);                                   \
		testC = testG;                                                                             \
		REQUIRE(testC.storage()[0] == 10);                                                         \
		REQUIRE(testC.storage()[1] == 10);                                                         \
		REQUIRE(testC.storage()[2] == 10);                                                         \
		REQUIRE(testC.storage()[9] == 10);                                                         \
		REQUIRE(testC.storage()[10] == 10);                                                        \
		REQUIRE(testC.storage()[11] == 10);                                                        \
                                                                                                   \
		lrc::Array<SCALAR, DEVICE> testH(lrc::Shape {3, 3});                                       \
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
	}                                                                                              \
                                                                                                   \
	SECTION(fmt::format("Test Indexing [{} | {}]", STRINGIFY(SCALAR), STRINGIFY(DEVICE))) {        \
		lrc::Array<SCALAR, DEVICE> testA(lrc::Shape({5, 3}));                                      \
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
		testA[0][0] = 123;                                                                         \
		testA[1][1] = 456;                                                                         \
		testA[2][2] = 789;                                                                         \
		REQUIRE(testA.storage()[0] == SCALAR(123));                                                \
		REQUIRE(testA.storage()[4] == SCALAR(456));                                                \
		REQUIRE(testA.storage()[8] == SCALAR(789));                                                \
	}

TEST_CASE("Test Array", "[array-lib]") {
	TEST_SUITE(int8_t, lrc::device::CPU)
	TEST_SUITE(uint8_t, lrc::device::CPU)
	TEST_SUITE(int16_t, lrc::device::CPU)
	TEST_SUITE(uint16_t, lrc::device::CPU)
	TEST_SUITE(int32_t, lrc::device::CPU)
	TEST_SUITE(uint32_t, lrc::device::CPU)
	TEST_SUITE(int64_t, lrc::device::CPU)
	TEST_SUITE(uint64_t, lrc::device::CPU)
	TEST_SUITE(float, lrc::device::CPU)
	TEST_SUITE(double, lrc::device::CPU)

	TEST_SUITE(int8_t, lrc::device::GPU)

	TEST_SUITE(lrc::mpz, lrc::device::CPU)
	TEST_SUITE(lrc::mpq, lrc::device::CPU)
	TEST_SUITE(lrc::mpf, lrc::device::CPU)
	TEST_SUITE(lrc::mpfr, lrc::device::CPU)
}
