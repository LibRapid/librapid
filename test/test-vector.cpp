#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

#define VEC_TYPE lrc::GenericVector
#define SCALAR	 double

#define FUNDAMENTAL_OPS(VEC_TYPE, SCALAR)                                                          \
	SECTION("Test Fundamental Operations") {                                                       \
		VEC_TYPE<SCALAR, 4> testA(1, 2, 3, 4);                                                     \
                                                                                                   \
		REQUIRE(testA.x() == 1);                                                                   \
		REQUIRE(testA.y() == 2);                                                                   \
		REQUIRE(testA.z() == 3);                                                                   \
		REQUIRE(testA.w() == 4);                                                                   \
                                                                                                   \
		REQUIRE(testA[0] == 1);                                                                    \
		REQUIRE(testA[1] == 2);                                                                    \
		REQUIRE(testA[2] == 3);                                                                    \
		REQUIRE(testA[3] == 4);                                                                    \
                                                                                                   \
		REQUIRE(testA == VEC_TYPE<SCALAR, 4>(1, 2, 3, 4));                                         \
		REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(1, 2, 3, 4)));                             \
	}

#define CONSTRUCTORS(VEC_TYPE, SCALAR)                                                             \
	SECTION("Test Constructors") {                                                                 \
		VEC_TYPE<SCALAR, 4> testA(1, 2, 3, 4);                                                     \
		VEC_TYPE<SCALAR, 4> testB(testA);                                                          \
		VEC_TYPE<SCALAR, 4> testC = testA;                                                         \
                                                                                                   \
		REQUIRE(testA == testB);                                                                   \
		REQUIRE(testA == testC);                                                                   \
                                                                                                   \
		VEC_TYPE<SCALAR, 4> testD(std::move(testA));                                               \
		VEC_TYPE<SCALAR, 4> testE = std::move(testB);                                              \
		REQUIRE(testD == VEC_TYPE<SCALAR, 4>(1, 2, 3, 4));                                         \
		REQUIRE(testE == VEC_TYPE<SCALAR, 4>(1, 2, 3, 4));                                         \
                                                                                                   \
		VEC_TYPE<SCALAR, 4> testF = {1, 2, 3, 4};                                                  \
		REQUIRE(testF == VEC_TYPE<SCALAR, 4>(1, 2, 3, 4));                                         \
                                                                                                   \
		VEC_TYPE<SCALAR, 4> testG = VEC_TYPE<SCALAR, 4>(1, 2, 3, 4);                               \
		REQUIRE(testG == VEC_TYPE<SCALAR, 4>(1, 2, 3, 4));                                         \
	}

#define ARITHMETIC(VEC_TYPE, SCALAR)                                                               \
	SECTION("Test Arithmetic") {                                                                   \
		SECTION("Test Inplace Arithmetic") {                                                       \
			VEC_TYPE<SCALAR, 4> testA(1, 2, 3, 4);                                                 \
			VEC_TYPE<SCALAR, 4> testB(2, 4, 6, 8);                                                 \
                                                                                                   \
			testA += testB;                                                                        \
			REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(3, 6, 9, 12)));                        \
                                                                                                   \
			testA -= testB;                                                                        \
			REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(1, 2, 3, 4)));                         \
                                                                                                   \
			testA *= testB;                                                                        \
			REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(2, 8, 18, 32)));                       \
                                                                                                   \
			testA /= testB;                                                                        \
			REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(1, 2, 3, 4)));                         \
                                                                                                   \
			testA += 1;                                                                            \
			REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(2, 3, 4, 5)));                         \
                                                                                                   \
			testA -= 1;                                                                            \
			REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(1, 2, 3, 4)));                         \
                                                                                                   \
			testA *= 2;                                                                            \
			REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(2, 4, 6, 8)));                         \
                                                                                                   \
			testA /= 2;                                                                            \
			REQUIRE(lrc::isClose(testA, VEC_TYPE<SCALAR, 4>(1, 2, 3, 4)));                         \
		}                                                                                          \
                                                                                                   \
		SECTION("Test General Arithmetic") {                                                       \
			VEC_TYPE<SCALAR, 4> testA(1, 2, 3, 4);                                                 \
			VEC_TYPE<SCALAR, 4> testB(2, 4, 6, 8);                                                 \
                                                                                                   \
			REQUIRE(lrc::isClose(testA + testB, VEC_TYPE<SCALAR, 4>(3, 6, 9, 12)));                \
			REQUIRE(lrc::isClose(testA - testB, VEC_TYPE<SCALAR, 4>(-1, -2, -3, -4)));             \
			REQUIRE(lrc::isClose(testA *testB, VEC_TYPE<SCALAR, 4>(2, 8, 18, 32)));                \
			REQUIRE(lrc::isClose(testA / testB, VEC_TYPE<SCALAR, 4>(0.5, 0.5, 0.5, 0.5)));         \
                                                                                                   \
			REQUIRE(lrc::isClose(testA + 1, VEC_TYPE<SCALAR, 4>(2, 3, 4, 5)));                     \
			REQUIRE(lrc::isClose(testA - 1, VEC_TYPE<SCALAR, 4>(0, 1, 2, 3)));                     \
			REQUIRE(lrc::isClose(testA * 2, VEC_TYPE<SCALAR, 4>(2, 4, 6, 8)));                     \
			REQUIRE(lrc::isClose(testA / 2, VEC_TYPE<SCALAR, 4>(0.5, 1, 1.5, 2)));                 \
                                                                                                   \
			REQUIRE(lrc::isClose(1 + testA, VEC_TYPE<SCALAR, 4>(2, 3, 4, 5)));                     \
			REQUIRE(lrc::isClose(1 - testA, VEC_TYPE<SCALAR, 4>(0, -1, -2, -3)));                  \
			REQUIRE(lrc::isClose(2 * testA, VEC_TYPE<SCALAR, 4>(2, 4, 6, 8)));                     \
			REQUIRE(lrc::isClose(2 / testA, VEC_TYPE<SCALAR, 4>(2, 1, 2.0 / 3.0, 0.5)));           \
		}                                                                                          \
	}

#define COMPARISONS(VEC_TYPE, SCALAR)                                                              \
	SECTION("Test Comparisons") {                                                                  \
		VEC_TYPE<SCALAR, 4> testA(1, 2, 3, 4);                                                     \
		VEC_TYPE<SCALAR, 4> testB(-1, 2, 4, 8);                                                    \
                                                                                                   \
		REQUIRE(testA.cmp(testB, "eq") == VEC_TYPE<SCALAR, 4>(0, 1, 0, 0));                        \
		REQUIRE(testA.cmp(testB, "ne") == VEC_TYPE<SCALAR, 4>(1, 0, 1, 1));                        \
		REQUIRE(testA.cmp(testB, "lt") == VEC_TYPE<SCALAR, 4>(0, 0, 1, 1));                        \
		REQUIRE(testA.cmp(testB, "le") == VEC_TYPE<SCALAR, 4>(0, 1, 1, 1));                        \
		REQUIRE(testA.cmp(testB, "gt") == VEC_TYPE<SCALAR, 4>(1, 0, 0, 0));                        \
		REQUIRE(testA.cmp(testB, "ge") == VEC_TYPE<SCALAR, 4>(1, 1, 0, 0));                        \
                                                                                                   \
		REQUIRE((testA == testB) == VEC_TYPE<SCALAR, 4>(0, 1, 0, 0));                              \
		REQUIRE((testA != testB) == VEC_TYPE<SCALAR, 4>(1, 0, 1, 1));                              \
		REQUIRE((testA < testB) == VEC_TYPE<SCALAR, 4>(0, 0, 1, 1));                               \
		REQUIRE((testA <= testB) == VEC_TYPE<SCALAR, 4>(0, 1, 1, 1));                              \
		REQUIRE((testA > testB) == VEC_TYPE<SCALAR, 4>(1, 0, 0, 0));                               \
		REQUIRE((testA >= testB) == VEC_TYPE<SCALAR, 4>(1, 1, 0, 0));                              \
                                                                                                   \
		REQUIRE((testA == 2) == VEC_TYPE<SCALAR, 4>(0, 1, 0, 0));                                  \
		REQUIRE((testA != 2) == VEC_TYPE<SCALAR, 4>(1, 0, 1, 1));                                  \
		REQUIRE((testA < 2) == VEC_TYPE<SCALAR, 4>(1, 0, 0, 0));                                   \
		REQUIRE((testA <= 2) == VEC_TYPE<SCALAR, 4>(1, 1, 0, 0));                                  \
		REQUIRE((testA > 2) == VEC_TYPE<SCALAR, 4>(0, 0, 1, 1));                                   \
		REQUIRE((testA >= 2) == VEC_TYPE<SCALAR, 4>(0, 1, 1, 1));                                  \
	}

#define VECTOR_OPS(VEC_TYPE, SCALAR)                                                               \
	SECTION("Test Vector Operations") {                                                            \
		VEC_TYPE<SCALAR, 4> testA(1, 2, 3, 4);                                                     \
		VEC_TYPE<SCALAR, 4> testB(2, 4, 6, 8);                                                     \
		SCALAR testASquareSum = 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4;                                     \
		SCALAR testBSquareSum = 2 * 2 + 4 * 4 + 6 * 6 + 8 * 8;                                     \
                                                                                                   \
		REQUIRE(testA.mag2() == testASquareSum);                                                   \
		REQUIRE(testA.mag() - lrc::sqrt(testASquareSum) < 1e-6);                                   \
                                                                                                   \
		REQUIRE(lrc::isClose(testA.norm(),                                                         \
							 VEC_TYPE<SCALAR, 4>(1.0 / lrc::sqrt(testASquareSum),                  \
												 2.0 / lrc::sqrt(testASquareSum),                  \
												 3.0 / lrc::sqrt(testASquareSum),                  \
												 4.0 / lrc::sqrt(testASquareSum))));               \
                                                                                                   \
		REQUIRE(testA.dot(testB) == 60);                                                           \
                                                                                                   \
		REQUIRE(VEC_TYPE<SCALAR, 3>(1, 0, 0).cross(VEC_TYPE<SCALAR, 3>(0, 1, 0)) ==                \
				VEC_TYPE<SCALAR, 3>(0, 0, 1));                                                     \
		REQUIRE(VEC_TYPE<SCALAR, 3>(0, 1, 0).cross(VEC_TYPE<SCALAR, 3>(1, 0, 0)) ==                \
				VEC_TYPE<SCALAR, 3>(0, 0, -1));                                                    \
		REQUIRE(VEC_TYPE<SCALAR, 3>(1, 0, 0).cross(VEC_TYPE<SCALAR, 3>(0, 0, 1)) ==                \
				VEC_TYPE<SCALAR, 3>(0, -1, 0));                                                    \
		REQUIRE(VEC_TYPE<SCALAR, 3>(0, 0, 1).cross(VEC_TYPE<SCALAR, 3>(1, 0, 0)) ==                \
				VEC_TYPE<SCALAR, 3>(0, 1, 0));                                                     \
		REQUIRE(VEC_TYPE<SCALAR, 3>(0, 1, 0).cross(VEC_TYPE<SCALAR, 3>(0, 0, 1)) ==                \
				VEC_TYPE<SCALAR, 3>(1, 0, 0));                                                     \
		REQUIRE(VEC_TYPE<SCALAR, 3>(0, 0, 1).cross(VEC_TYPE<SCALAR, 3>(0, 1, 0)) ==                \
				VEC_TYPE<SCALAR, 3>(-1, 0, 0));                                                    \
                                                                                                   \
		REQUIRE(VEC_TYPE<SCALAR, 4>(-1, 4, 2, 1).proj(VEC_TYPE<SCALAR, 4>(1, 0, 3, 2)) ==          \
				VEC_TYPE<SCALAR, 4>(                                                               \
				  SCALAR(1) / SCALAR(2), SCALAR(0), SCALAR(3) / SCALAR(2), SCALAR(1)));            \
		REQUIRE((bool)testA == true);                                                              \
		REQUIRE((bool)VEC_TYPE<SCALAR, 4>(0, 0, 0, 0) == false);                                   \
                                                                                                   \
		VEC_TYPE<SCALAR, 4> testC(1, 2, 3, 4);                                                     \
		REQUIRE(testC.xy() == VEC_TYPE<SCALAR, 2>(1, 2));                                          \
		REQUIRE(testC.yx() == VEC_TYPE<SCALAR, 2>(2, 1));                                          \
		REQUIRE(testC.xz() == VEC_TYPE<SCALAR, 2>(1, 3));                                          \
		REQUIRE(testC.zx() == VEC_TYPE<SCALAR, 2>(3, 1));                                          \
		REQUIRE(testC.yz() == VEC_TYPE<SCALAR, 2>(2, 3));                                          \
		REQUIRE(testC.zy() == VEC_TYPE<SCALAR, 2>(3, 2));                                          \
		REQUIRE(testC.xyz() == VEC_TYPE<SCALAR, 3>(1, 2, 3));                                      \
		REQUIRE(testC.xzy() == VEC_TYPE<SCALAR, 3>(1, 3, 2));                                      \
		REQUIRE(testC.yxz() == VEC_TYPE<SCALAR, 3>(2, 1, 3));                                      \
		REQUIRE(testC.yzx() == VEC_TYPE<SCALAR, 3>(2, 3, 1));                                      \
		REQUIRE(testC.zxy() == VEC_TYPE<SCALAR, 3>(3, 1, 2));                                      \
		REQUIRE(testC.zyx() == VEC_TYPE<SCALAR, 3>(3, 2, 1));                                      \
		REQUIRE(testC.xyw() == VEC_TYPE<SCALAR, 3>(1, 2, 4));                                      \
		REQUIRE(testC.xwy() == VEC_TYPE<SCALAR, 3>(1, 4, 2));                                      \
		REQUIRE(testC.yxw() == VEC_TYPE<SCALAR, 3>(2, 1, 4));                                      \
		REQUIRE(testC.ywx() == VEC_TYPE<SCALAR, 3>(2, 4, 1));                                      \
		REQUIRE(testC.wxy() == VEC_TYPE<SCALAR, 3>(4, 1, 2));                                      \
		REQUIRE(testC.wyx() == VEC_TYPE<SCALAR, 3>(4, 2, 1));                                      \
		REQUIRE(testC.xzw() == VEC_TYPE<SCALAR, 3>(1, 3, 4));                                      \
		REQUIRE(testC.xwz() == VEC_TYPE<SCALAR, 3>(1, 4, 3));                                      \
		REQUIRE(testC.zxw() == VEC_TYPE<SCALAR, 3>(3, 1, 4));                                      \
		REQUIRE(testC.zwx() == VEC_TYPE<SCALAR, 3>(3, 4, 1));                                      \
		REQUIRE(testC.wxz() == VEC_TYPE<SCALAR, 3>(4, 1, 3));                                      \
		REQUIRE(testC.wzx() == VEC_TYPE<SCALAR, 3>(4, 3, 1));                                      \
		REQUIRE(testC.yzw() == VEC_TYPE<SCALAR, 3>(2, 3, 4));                                      \
		REQUIRE(testC.ywz() == VEC_TYPE<SCALAR, 3>(2, 4, 3));                                      \
		REQUIRE(testC.zyw() == VEC_TYPE<SCALAR, 3>(3, 2, 4));                                      \
		REQUIRE(testC.zwy() == VEC_TYPE<SCALAR, 3>(3, 4, 2));                                      \
		REQUIRE(testC.wyz() == VEC_TYPE<SCALAR, 3>(4, 2, 3));                                      \
		REQUIRE(testC.wzy() == VEC_TYPE<SCALAR, 3>(4, 3, 2));                                      \
		REQUIRE(testC.xyzw() == VEC_TYPE<SCALAR, 4>(1, 2, 3, 4));                                  \
		REQUIRE(testC.xywz() == VEC_TYPE<SCALAR, 4>(1, 2, 4, 3));                                  \
		REQUIRE(testC.xzyw() == VEC_TYPE<SCALAR, 4>(1, 3, 2, 4));                                  \
		REQUIRE(testC.xzwy() == VEC_TYPE<SCALAR, 4>(1, 3, 4, 2));                                  \
		REQUIRE(testC.xwyz() == VEC_TYPE<SCALAR, 4>(1, 4, 2, 3));                                  \
		REQUIRE(testC.xwzy() == VEC_TYPE<SCALAR, 4>(1, 4, 3, 2));                                  \
		REQUIRE(testC.yxzw() == VEC_TYPE<SCALAR, 4>(2, 1, 3, 4));                                  \
		REQUIRE(testC.yxwz() == VEC_TYPE<SCALAR, 4>(2, 1, 4, 3));                                  \
		REQUIRE(testC.yzxw() == VEC_TYPE<SCALAR, 4>(2, 3, 1, 4));                                  \
		REQUIRE(testC.yzwx() == VEC_TYPE<SCALAR, 4>(2, 3, 4, 1));                                  \
		REQUIRE(testC.ywxz() == VEC_TYPE<SCALAR, 4>(2, 4, 1, 3));                                  \
		REQUIRE(testC.ywzx() == VEC_TYPE<SCALAR, 4>(2, 4, 3, 1));                                  \
		REQUIRE(testC.zxyw() == VEC_TYPE<SCALAR, 4>(3, 1, 2, 4));                                  \
		REQUIRE(testC.zxwy() == VEC_TYPE<SCALAR, 4>(3, 1, 4, 2));                                  \
		REQUIRE(testC.zyxw() == VEC_TYPE<SCALAR, 4>(3, 2, 1, 4));                                  \
		REQUIRE(testC.zywx() == VEC_TYPE<SCALAR, 4>(3, 2, 4, 1));                                  \
		REQUIRE(testC.zwxy() == VEC_TYPE<SCALAR, 4>(3, 4, 1, 2));                                  \
		REQUIRE(testC.zwyx() == VEC_TYPE<SCALAR, 4>(3, 4, 2, 1));                                  \
		REQUIRE(testC.wxyz() == VEC_TYPE<SCALAR, 4>(4, 1, 2, 3));                                  \
		REQUIRE(testC.wxzy() == VEC_TYPE<SCALAR, 4>(4, 1, 3, 2));                                  \
		REQUIRE(testC.wyxz() == VEC_TYPE<SCALAR, 4>(4, 2, 1, 3));                                  \
		REQUIRE(testC.wyzx() == VEC_TYPE<SCALAR, 4>(4, 2, 3, 1));                                  \
		REQUIRE(testC.wzxy() == VEC_TYPE<SCALAR, 4>(4, 3, 1, 2));                                  \
		REQUIRE(testC.wzyx() == VEC_TYPE<SCALAR, 4>(4, 3, 2, 1));                                  \
                                                                                                   \
		/*REQUIRE(testC.str() == "(1, 2, 3, 4)");                            */                    \
		/*REQUIRE(testC.str("{:.2f}") == "(1.00, 2.00, 3.00, 4.00)");        */                    \
                                                                                                   \
		VEC_TYPE<SCALAR, 4> testD = testA + VEC_TYPE<SCALAR, 4>({3, 4});                           \
		REQUIRE(lrc::dist(testA, testA) == 0);                                                     \
		REQUIRE(lrc::dist2(testA, testD) == 25);                                                   \
		REQUIRE(lrc::dist(testA, testD) == 5);                                                     \
                                                                                                   \
		VEC_TYPE<SCALAR, 4> testE(SCALAR(0.1), SCALAR(0.2), SCALAR(0.3), SCALAR(0.4));             \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::sin(testE),                                                                         \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			sin(SCALAR(0.1)), sin(SCALAR(0.2)), sin(SCALAR(0.3)), sin(SCALAR(0.4)))));             \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::cos(testE),                                                                         \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			cos(SCALAR(0.1)), cos(SCALAR(0.2)), cos(SCALAR(0.3)), cos(SCALAR(0.4)))));             \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::tan(testE),                                                                         \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			tan(SCALAR(0.1)), tan(SCALAR(0.2)), tan(SCALAR(0.3)), tan(SCALAR(0.4)))));             \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::asin(testE),                                                                        \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			asin(SCALAR(0.1)), asin(SCALAR(0.2)), asin(SCALAR(0.3)), asin(SCALAR(0.4)))));         \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::acos(testE),                                                                        \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			acos(SCALAR(0.1)), acos(SCALAR(0.2)), acos(SCALAR(0.3)), acos(SCALAR(0.4)))));         \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::atan(testE),                                                                        \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			atan(SCALAR(0.1)), atan(SCALAR(0.2)), atan(SCALAR(0.3)), atan(SCALAR(0.4)))));         \
		REQUIRE(lrc::isClose(lrc::atan2(testE, testE),                                             \
							 VEC_TYPE<SCALAR, 4>(atan2(SCALAR(0.1), SCALAR(0.1)),                  \
												 atan2(SCALAR(0.2), SCALAR(0.2)),                  \
												 atan2(SCALAR(0.3), SCALAR(0.3)),                  \
												 atan2(SCALAR(0.4), SCALAR(0.4)))));               \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::sinh(testE),                                                                        \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			sinh(SCALAR(0.1)), sinh(SCALAR(0.2)), sinh(SCALAR(0.3)), sinh(SCALAR(0.4)))));         \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::cosh(testE),                                                                        \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			cosh(SCALAR(0.1)), cosh(SCALAR(0.2)), cosh(SCALAR(0.3)), cosh(SCALAR(0.4)))));         \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::tanh(testE),                                                                        \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			tanh(SCALAR(0.1)), tanh(SCALAR(0.2)), tanh(SCALAR(0.3)), tanh(SCALAR(0.4)))));         \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::asinh(testE),                                                                       \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			asinh(SCALAR(0.1)), asinh(SCALAR(0.2)), asinh(SCALAR(0.3)), asinh(SCALAR(0.4)))));     \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::acosh(testE),                                                                       \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			acosh(SCALAR(0.1)), acosh(SCALAR(0.2)), acosh(SCALAR(0.3)), acosh(SCALAR(0.4)))));     \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::atanh(testE),                                                                       \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			atanh(SCALAR(0.1)), atanh(SCALAR(0.2)), atanh(SCALAR(0.3)), atanh(SCALAR(0.4)))));     \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::exp(testE),                                                                         \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			exp(SCALAR(0.1)), exp(SCALAR(0.2)), exp(SCALAR(0.3)), exp(SCALAR(0.4)))));             \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::log(testE),                                                                         \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			log(SCALAR(0.1)), log(SCALAR(0.2)), log(SCALAR(0.3)), log(SCALAR(0.4)))));             \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::log10(testE),                                                                       \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			log10(SCALAR(0.1)), log10(SCALAR(0.2)), log10(SCALAR(0.3)), log10(SCALAR(0.4)))));     \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::pow(testE, 2),                                                                      \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			pow(SCALAR(0.1), 2), pow(SCALAR(0.2), 2), pow(SCALAR(0.3), 2), pow(SCALAR(0.4), 2)))); \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::sqrt(testE),                                                                        \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			sqrt(SCALAR(0.1)), sqrt(SCALAR(0.2)), sqrt(SCALAR(0.3)), sqrt(SCALAR(0.4)))));         \
		REQUIRE(lrc::isClose(                                                                      \
		  lrc::cbrt(testE),                                                                        \
		  VEC_TYPE<SCALAR, 4>(                                                                     \
			cbrt(SCALAR(0.1)), cbrt(SCALAR(0.2)), cbrt(SCALAR(0.3)), cbrt(SCALAR(0.4)))));         \
	}

TEST_CASE("GenericVector<int16_t>", "[vector-lib]") {
  FUNDAMENTAL_OPS(lrc::GenericVector, int16_t) CONSTRUCTORS(lrc::GenericVector, int16_t)
	ARITHMETIC(lrc::GenericVector, int16_t) COMPARISONS(lrc::GenericVector, int16_t)
	  VECTOR_OPS(lrc::GenericVector, int16_t)}

TEST_CASE("GenericVector<int32_t>", "[vector-lib]") {
  FUNDAMENTAL_OPS(lrc::GenericVector, int32_t) CONSTRUCTORS(lrc::GenericVector, int32_t)
	ARITHMETIC(lrc::GenericVector, int32_t) COMPARISONS(lrc::GenericVector, int32_t)
	  VECTOR_OPS(lrc::GenericVector, int32_t)}

TEST_CASE("GenericVector<int64_t>", "[vector-lib]") {
  FUNDAMENTAL_OPS(lrc::GenericVector, int64_t) CONSTRUCTORS(lrc::GenericVector, int64_t)
	ARITHMETIC(lrc::GenericVector, int64_t) COMPARISONS(lrc::GenericVector, int64_t)
	  VECTOR_OPS(lrc::GenericVector, int64_t)}

TEST_CASE("GenericVector<float>", "[vector-lib]") {
  FUNDAMENTAL_OPS(lrc::GenericVector, float) CONSTRUCTORS(lrc::GenericVector, float)
	ARITHMETIC(lrc::GenericVector, float) COMPARISONS(lrc::GenericVector, float)
	  VECTOR_OPS(lrc::GenericVector, float)}

TEST_CASE("GenericVector<double>", "[vector-lib]") {
  FUNDAMENTAL_OPS(lrc::GenericVector, double) CONSTRUCTORS(lrc::GenericVector, double)
	ARITHMETIC(lrc::GenericVector, double) COMPARISONS(lrc::GenericVector, double)
	  VECTOR_OPS(lrc::GenericVector, double)}

TEST_CASE("SIMDVector<int16_t>", "[vector-lib]") {
  FUNDAMENTAL_OPS(lrc::SIMDVector, int16_t) CONSTRUCTORS(lrc::SIMDVector, int16_t)
	ARITHMETIC(lrc::SIMDVector, int16_t) COMPARISONS(lrc::SIMDVector, int16_t)
	  VECTOR_OPS(lrc::SIMDVector, int16_t)}

TEST_CASE("SIMDVector<int32_t>", "[vector-lib]") {
  FUNDAMENTAL_OPS(lrc::SIMDVector, int32_t) CONSTRUCTORS(lrc::SIMDVector, int32_t)
	ARITHMETIC(lrc::SIMDVector, int32_t) COMPARISONS(lrc::SIMDVector, int32_t)}

TEST_CASE("SIMDVector<float>", "[vector-lib]") {
  FUNDAMENTAL_OPS(lrc::SIMDVector, float) CONSTRUCTORS(lrc::SIMDVector, float)
	ARITHMETIC(lrc::SIMDVector, float) COMPARISONS(lrc::SIMDVector, float)
	  VECTOR_OPS(lrc::SIMDVector, float)}

TEST_CASE("SIMDVector<double>", "[vector-lib]") {
	FUNDAMENTAL_OPS(lrc::SIMDVector, double)
	CONSTRUCTORS(lrc::SIMDVector, double)
	ARITHMETIC(lrc::SIMDVector, double)
	COMPARISONS(lrc::SIMDVector, double)
	VECTOR_OPS(lrc::SIMDVector, double)
}
