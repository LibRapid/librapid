#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

TEST_CASE("Test Vectors", "[vector-lib]") {
	SECTION("Test Fundamental Operations") {
		lrc::Vec3d testA(1, 2, 3);

		REQUIRE(testA.x() == 1);
		REQUIRE(testA.y() == 2);
		REQUIRE(testA.z() == 3);

		REQUIRE(testA[0] == 1);
		REQUIRE(testA[1] == 2);
		REQUIRE(testA[2] == 3);

		REQUIRE(testA == lrc::Vec3d(1, 2, 3));
	}

	SECTION("Test Constructors") {
		lrc::Vec3d testA(1, 2, 3);
		lrc::Vec3d testB(testA);
		lrc::Vec3d testC = testA;

		REQUIRE(testA == testB);
		REQUIRE(testA == testC);

		lrc::Vec3d testD(std::move(testA));
		lrc::Vec3d testE = std::move(testB);

		REQUIRE(testD == lrc::Vec3d(1, 2, 3));
		REQUIRE(testE == lrc::Vec3d(1, 2, 3));

		lrc::Vec3d testF = {1, 2, 3};

		REQUIRE(testF == lrc::Vec3d(1, 2, 3));

		lrc::Vec3d testG = lrc::Vec3d(1, 2, 3);

		REQUIRE(testG == lrc::Vec3d(1, 2, 3));
	}

	SECTION("Test Arithmetic") {
		SECTION("Test Inplace Arithmetic") {
			lrc::Vec3d testA(1, 2, 3);
			lrc::Vec3d testB(2, 4, 6);

			testA += testB;
			REQUIRE(testA == lrc::Vec3d(3, 6, 9));

			testA -= testB;
			REQUIRE(testA == lrc::Vec3d(1, 2, 3));

			testA *= testB;
			REQUIRE(testA == lrc::Vec3d(2, 8, 18));

			testA /= testB;
			REQUIRE(testA == lrc::Vec3d(1, 2, 3));

			testA += 1;
			REQUIRE(testA == lrc::Vec3d(2, 3, 4));

			testA -= 1;
			REQUIRE(testA == lrc::Vec3d(1, 2, 3));

			testA *= 2;
			REQUIRE(testA == lrc::Vec3d(2, 4, 6));

			testA /= 2;
			REQUIRE(testA == lrc::Vec3d(1, 2, 3));
		}

		SECTION("Test General Arithmetic") {
			lrc::Vec3d testA(1, 2, 3);
			lrc::Vec3d testB(2, 4, 6);

			REQUIRE(testA + testB == lrc::Vec3d(3, 6, 9));
			REQUIRE(testA - testB == lrc::Vec3d(-1, -2, -3));
			REQUIRE(testA * testB == lrc::Vec3d(2, 8, 18));
			REQUIRE(testA / testB == lrc::Vec3d(0.5, 0.5, 0.5));

			REQUIRE(testA + 1 == lrc::Vec3d(2, 3, 4));
			REQUIRE(testA - 1 == lrc::Vec3d(0, 1, 2));
			REQUIRE(testA * 2 == lrc::Vec3d(2, 4, 6));
			REQUIRE(testA / 2 == lrc::Vec3d(0.5, 1, 1.5));

			REQUIRE(1 + testA == lrc::Vec3d(2, 3, 4));
			REQUIRE(1 - testA == lrc::Vec3d(0, -1, -2));
			REQUIRE(2 * testA == lrc::Vec3d(2, 4, 6));
			REQUIRE(2 / testA == lrc::Vec3d(2, 1, 2.0 / 3.0));
		}
	}

	SECTION("Test Comparisons") {
		lrc::Vec3i testA(1, 2, 3);
		lrc::Vec3i testB(-1, 2, 4);

		REQUIRE(testA.cmp(testB, "eq") == lrc::Vec3i(0, 1, 0));
		REQUIRE(testA.cmp(testB, "ne") == lrc::Vec3i(1, 0, 1));
		REQUIRE(testA.cmp(testB, "lt") == lrc::Vec3i(0, 0, 1));
		REQUIRE(testA.cmp(testB, "le") == lrc::Vec3i(0, 1, 1));
		REQUIRE(testA.cmp(testB, "gt") == lrc::Vec3i(1, 0, 0));
		REQUIRE(testA.cmp(testB, "ge") == lrc::Vec3i(1, 1, 0));

		REQUIRE((testA == testB) == lrc::Vec3i(0, 1, 0));
		REQUIRE((testA != testB) == lrc::Vec3i(1, 0, 1));
		REQUIRE((testA < testB) == lrc::Vec3i(0, 0, 1));
		REQUIRE((testA <= testB) == lrc::Vec3i(0, 1, 1));
		REQUIRE((testA > testB) == lrc::Vec3i(1, 0, 0));
		REQUIRE((testA >= testB) == lrc::Vec3i(1, 1, 0));

		REQUIRE((testA == 2) == lrc::Vec3i(0, 1, 0));
		REQUIRE((testA != 2) == lrc::Vec3i(1, 0, 1));
		REQUIRE((testA < 2) == lrc::Vec3i(1, 0, 0));
		REQUIRE((testA <= 2) == lrc::Vec3i(1, 1, 0));
		REQUIRE((testA > 2) == lrc::Vec3i(0, 0, 1));
		REQUIRE((testA >= 2) == lrc::Vec3i(0, 1, 1));
	}

	SECTION("Test Vector Operations") {
		lrc::Vec3d testA(1, 2, 3);
		lrc::Vec3d testB(2, 4, 6);

		REQUIRE(testA.mag2() == 14);
		REQUIRE(testA.mag() == lrc::sqrt(14));

		REQUIRE(testA.norm() ==
				lrc::Vec3d(1.0 / lrc::sqrt(14), 2.0 / lrc::sqrt(14), 3.0 / lrc::sqrt(14)));

		REQUIRE(testA.dot(testB) == 28);
		REQUIRE(testA.cross(lrc::Vec3d(4, 5, 6)) == lrc::Vec3d(-3, 6, -3));
		REQUIRE(testA.proj(lrc::Vec3d(-2, 3, 1)) == lrc::Vec3d(-1., 3. / 2., 1. / 2.));
		REQUIRE((bool)testA == true);
		REQUIRE((bool)lrc::Vec3d(0, 0, 0) == false);

		lrc::Vec4d testC(1, 2, 3, 4);
		REQUIRE(testC.xy() == lrc::Vec2d(1, 2));
		REQUIRE(testC.yx() == lrc::Vec2d(2, 1));
		REQUIRE(testC.xz() == lrc::Vec2d(1, 3));
		REQUIRE(testC.zx() == lrc::Vec2d(3, 1));
		REQUIRE(testC.yz() == lrc::Vec2d(2, 3));
		REQUIRE(testC.zy() == lrc::Vec2d(3, 2));
		REQUIRE(testC.xyz() == lrc::Vec3d(1, 2, 3));
		REQUIRE(testC.xzy() == lrc::Vec3d(1, 3, 2));
		REQUIRE(testC.yxz() == lrc::Vec3d(2, 1, 3));
		REQUIRE(testC.yzx() == lrc::Vec3d(2, 3, 1));
		REQUIRE(testC.zxy() == lrc::Vec3d(3, 1, 2));
		REQUIRE(testC.zyx() == lrc::Vec3d(3, 2, 1));
		REQUIRE(testC.xyw() == lrc::Vec3d(1, 2, 4));
		REQUIRE(testC.xwy() == lrc::Vec3d(1, 4, 2));
		REQUIRE(testC.yxw() == lrc::Vec3d(2, 1, 4));
		REQUIRE(testC.ywx() == lrc::Vec3d(2, 4, 1));
		REQUIRE(testC.wxy() == lrc::Vec3d(4, 1, 2));
		REQUIRE(testC.wyx() == lrc::Vec3d(4, 2, 1));
		REQUIRE(testC.xzw() == lrc::Vec3d(1, 3, 4));
		REQUIRE(testC.xwz() == lrc::Vec3d(1, 4, 3));
		REQUIRE(testC.zxw() == lrc::Vec3d(3, 1, 4));
		REQUIRE(testC.zwx() == lrc::Vec3d(3, 4, 1));
		REQUIRE(testC.wxz() == lrc::Vec3d(4, 1, 3));
		REQUIRE(testC.wzx() == lrc::Vec3d(4, 3, 1));
		REQUIRE(testC.yzw() == lrc::Vec3d(2, 3, 4));
		REQUIRE(testC.ywz() == lrc::Vec3d(2, 4, 3));
		REQUIRE(testC.zyw() == lrc::Vec3d(3, 2, 4));
		REQUIRE(testC.zwy() == lrc::Vec3d(3, 4, 2));
		REQUIRE(testC.wyz() == lrc::Vec3d(4, 2, 3));
		REQUIRE(testC.wzy() == lrc::Vec3d(4, 3, 2));
		REQUIRE(testC.xyzw() == lrc::Vec4d(1, 2, 3, 4));
		REQUIRE(testC.xywz() == lrc::Vec4d(1, 2, 4, 3));
		REQUIRE(testC.xzyw() == lrc::Vec4d(1, 3, 2, 4));
		REQUIRE(testC.xzwy() == lrc::Vec4d(1, 3, 4, 2));
		REQUIRE(testC.xwyz() == lrc::Vec4d(1, 4, 2, 3));
		REQUIRE(testC.xwzy() == lrc::Vec4d(1, 4, 3, 2));
		REQUIRE(testC.yxzw() == lrc::Vec4d(2, 1, 3, 4));
		REQUIRE(testC.yxwz() == lrc::Vec4d(2, 1, 4, 3));
		REQUIRE(testC.yzxw() == lrc::Vec4d(2, 3, 1, 4));
		REQUIRE(testC.yzwx() == lrc::Vec4d(2, 3, 4, 1));
		REQUIRE(testC.ywxz() == lrc::Vec4d(2, 4, 1, 3));
		REQUIRE(testC.ywzx() == lrc::Vec4d(2, 4, 3, 1));
		REQUIRE(testC.zxyw() == lrc::Vec4d(3, 1, 2, 4));
		REQUIRE(testC.zxwy() == lrc::Vec4d(3, 1, 4, 2));
		REQUIRE(testC.zyxw() == lrc::Vec4d(3, 2, 1, 4));
		REQUIRE(testC.zywx() == lrc::Vec4d(3, 2, 4, 1));
		REQUIRE(testC.zwxy() == lrc::Vec4d(3, 4, 1, 2));
		REQUIRE(testC.zwyx() == lrc::Vec4d(3, 4, 2, 1));
		REQUIRE(testC.wxyz() == lrc::Vec4d(4, 1, 2, 3));
		REQUIRE(testC.wxzy() == lrc::Vec4d(4, 1, 3, 2));
		REQUIRE(testC.wyxz() == lrc::Vec4d(4, 2, 1, 3));
		REQUIRE(testC.wyzx() == lrc::Vec4d(4, 2, 3, 1));
		REQUIRE(testC.wzxy() == lrc::Vec4d(4, 3, 1, 2));
		REQUIRE(testC.wzyx() == lrc::Vec4d(4, 3, 2, 1));

		REQUIRE(testC.str() == "(1, 2, 3, 4)");
		REQUIRE(testC.str("{:.2f}") == "(1.00, 2.00, 3.00, 4.00)");

		lrc::Vec3d testD = testA + lrc::Vec3d({3, 4});
		REQUIRE(lrc::dist(testA, testA) == 0);
		REQUIRE(lrc::dist2(testA, testD) == 25);
		REQUIRE(lrc::dist(testA, testD) == 5);

		lrc::Vec3d testE(0.1, 0.2, 0.3);
		REQUIRE((lrc::sin(testE)            - lrc::Vec3d(sin(0.1), lrc::sin(0.2), lrc::sin(0.3))) < 1e-6);
		REQUIRE((lrc::cos(testE)            - lrc::Vec3d(cos(0.1), lrc::cos(0.2), lrc::cos(0.3))).mag() < 1e-6);
		REQUIRE((lrc::tan(testE)            - lrc::Vec3d(tan(0.1), lrc::tan(0.2), lrc::tan(0.3))).mag() < 1e-6);
		REQUIRE((lrc::asin(testE)           - lrc::Vec3d(asin(0.1), lrc::asin(0.2), lrc::asin(0.3))).mag() < 1e-6);
		REQUIRE((lrc::acos(testE)           - lrc::Vec3d(acos(0.1), lrc::acos(0.2), lrc::acos(0.3))).mag() < 1e-6);
		REQUIRE((lrc::atan(testE)           - lrc::Vec3d(atan(0.1), lrc::atan(0.2), lrc::atan(0.3))).mag() < 1e-6);
		REQUIRE((lrc::atan2(testA, testE)   - lrc::Vec3d(atan2(1, 0.1), lrc::atan2(2, 0.2), lrc::atan2(3, 0.3))).mag() < 1e-6);
		REQUIRE((lrc::sinh(testE)           - lrc::Vec3d(sinh(0.1), lrc::sinh(0.2), lrc::sinh(0.3))).mag() < 1e-6);
		REQUIRE((lrc::cosh(testE)           - lrc::Vec3d(cosh(0.1), lrc::cosh(0.2), lrc::cosh(0.3))).mag() < 1e-6);
		REQUIRE((lrc::tanh(testE)           - lrc::Vec3d(tanh(0.1), lrc::tanh(0.2), lrc::tanh(0.3))).mag() < 1e-6);
		REQUIRE((lrc::asinh(testE)          - lrc::Vec3d(asinh(0.1), lrc::asinh(0.2), lrc::asinh(0.3))).mag() < 1e-6);
		REQUIRE((lrc::acosh(testE + 1)      - lrc::Vec3d(acosh(1.1), lrc::acosh(1.2), lrc::acosh(1.3))).mag() < 1e-6);
		REQUIRE((lrc::atanh(testE)          - lrc::Vec3d(atanh(0.1), lrc::atanh(0.2), lrc::atanh(0.3))).mag() < 1e-6);
		REQUIRE((lrc::exp(testE)            - lrc::Vec3d(exp(0.1), lrc::exp(0.2), lrc::exp(0.3))).mag() < 1e-6);
		REQUIRE((lrc::log(testE)      		- lrc::Vec3d(log(0.1), lrc::log(0.2), lrc::log(0.3))).mag() < 1e-6);
		REQUIRE((lrc::log10(testE)    		- lrc::Vec3d(log10(0.1), lrc::log10(0.2), lrc::log10(0.3))).mag() < 1e-6);
		REQUIRE((lrc::log2(testE)     	    - lrc::Vec3d(log2(0.1), lrc::log2(0.2), lrc::log2(0.3))).mag() < 1e-6);
		REQUIRE((lrc::pow(testE, 2)         - lrc::Vec3d(pow(0.1, 2), lrc::pow(0.2, 2), lrc::pow(0.3, 2))).mag() < 1e-6);
		REQUIRE((lrc::sqrt(testE)           - lrc::Vec3d(sqrt(0.1), lrc::sqrt(0.2), lrc::sqrt(0.3))).mag() < 1e-6);
		REQUIRE((lrc::cbrt(testE)           - lrc::Vec3d(cbrt(0.1), lrc::cbrt(0.2), lrc::cbrt(0.3))).mag() < 1e-6);
	}

	//	SECTION("Benchmarks") {
	//		BENCHMARK_CONSTRUCTORS(int, 123);
	//		BENCHMARK_CONSTRUCTORS(double, 456);
	//		BENCHMARK_CONSTRUCTORS(std::string, "Hello, World");
	//		BENCHMARK_CONSTRUCTORS(std::vector<int>, {1 COMMA 2 COMMA 3 COMMA 4});
	//		BENCHMARK_CONSTRUCTORS(std::vector<double>, {1 COMMA 2 COMMA 3 COMMA 4});
	//	}
}
