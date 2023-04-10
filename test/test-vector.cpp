#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

TEST_CASE("Test Generic Vector", "[vector-lib]") {
	SECTION("Test Fundamental Operations") {
		lrc::GenericVector<double, 3> testA(1, 2, 3);

		REQUIRE(testA.x() == 1);
		REQUIRE(testA.y() == 2);
		REQUIRE(testA.z() == 3);

		REQUIRE(testA[0] == 1);
		REQUIRE(testA[1] == 2);
		REQUIRE(testA[2] == 3);

		REQUIRE(testA == lrc::GenericVector<double, 3>(1, 2, 3));
	}

	SECTION("Test Constructors") {
		lrc::GenericVector<double, 3> testA(1, 2, 3);
		lrc::GenericVector<double, 3> testB(testA);
		lrc::GenericVector<double, 3> testC = testA;

		REQUIRE(testA == testB);
		REQUIRE(testA == testC);

		lrc::GenericVector<double, 3> testD(std::move(testA));
		lrc::GenericVector<double, 3> testE = std::move(testB);

		REQUIRE(testD == lrc::GenericVector<double, 3>(1, 2, 3));
		REQUIRE(testE == lrc::GenericVector<double, 3>(1, 2, 3));

		lrc::GenericVector<double, 3> testF = {1, 2, 3};

		REQUIRE(testF == lrc::GenericVector<double, 3>(1, 2, 3));

		lrc::GenericVector<double, 3> testG = lrc::GenericVector<double, 3>(1, 2, 3);

		REQUIRE(testG == lrc::GenericVector<double, 3>(1, 2, 3));
	}

	SECTION("Test Arithmetic") {
		SECTION("Test Inplace Arithmetic") {
			lrc::GenericVector<double, 3> testA(1, 2, 3);
			lrc::GenericVector<double, 3> testB(2, 4, 6);

			testA += testB;
			REQUIRE(testA == lrc::GenericVector<double, 3>(3, 6, 9));

			testA -= testB;
			REQUIRE(testA == lrc::GenericVector<double, 3>(1, 2, 3));

			testA *= testB;
			REQUIRE(testA == lrc::GenericVector<double, 3>(2, 8, 18));

			testA /= testB;
			REQUIRE(testA == lrc::GenericVector<double, 3>(1, 2, 3));

			testA += 1;
			REQUIRE(testA == lrc::GenericVector<double, 3>(2, 3, 4));

			testA -= 1;
			REQUIRE(testA == lrc::GenericVector<double, 3>(1, 2, 3));

			testA *= 2;
			REQUIRE(testA == lrc::GenericVector<double, 3>(2, 4, 6));

			testA /= 2;
			REQUIRE(testA == lrc::GenericVector<double, 3>(1, 2, 3));
		}

		SECTION("Test General Arithmetic") {
			lrc::GenericVector<double, 3> testA(1, 2, 3);
			lrc::GenericVector<double, 3> testB(2, 4, 6);

			REQUIRE(testA + testB == lrc::GenericVector<double, 3>(3, 6, 9));
			REQUIRE(testA - testB == lrc::GenericVector<double, 3>(-1, -2, -3));
			REQUIRE(testA * testB == lrc::GenericVector<double, 3>(2, 8, 18));
			REQUIRE(testA / testB == lrc::GenericVector<double, 3>(0.5, 0.5, 0.5));

			REQUIRE(testA + 1 == lrc::GenericVector<double, 3>(2, 3, 4));
			REQUIRE(testA - 1 == lrc::GenericVector<double, 3>(0, 1, 2));
			REQUIRE(testA * 2 == lrc::GenericVector<double, 3>(2, 4, 6));
			REQUIRE(testA / 2 == lrc::GenericVector<double, 3>(0.5, 1, 1.5));

			REQUIRE(1 + testA == lrc::GenericVector<double, 3>(2, 3, 4));
			REQUIRE(1 - testA == lrc::GenericVector<double, 3>(0, -1, -2));
			REQUIRE(2 * testA == lrc::GenericVector<double, 3>(2, 4, 6));
			REQUIRE(2 / testA == lrc::GenericVector<double, 3>(2, 1, 2.0 / 3.0));
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
		lrc::GenericVector<double, 3> testA(1, 2, 3);
		lrc::GenericVector<double, 3> testB(2, 4, 6);

		REQUIRE(testA.mag2() == 14);
		REQUIRE(testA.mag() == lrc::sqrt(14));

		REQUIRE(testA.norm() == lrc::GenericVector<double, 3>(
								  1.0 / lrc::sqrt(14), 2.0 / lrc::sqrt(14), 3.0 / lrc::sqrt(14)));

		REQUIRE(testA.dot(testB) == 28);
		REQUIRE(testA.cross(lrc::GenericVector<double, 3>(4, 5, 6)) ==
				lrc::GenericVector<double, 3>(-3, 6, -3));
		REQUIRE(testA.proj(lrc::GenericVector<double, 3>(-2, 3, 1)) ==
				lrc::GenericVector<double, 3>(-1., 3. / 2., 1. / 2.));
		REQUIRE((bool)testA == true);
		REQUIRE((bool)lrc::GenericVector<double, 3>(0, 0, 0) == false);

		lrc::GenericVector<double, 4> testC(1, 2, 3, 4);
		REQUIRE(testC.xy() == lrc::GenericVector<double, 2>(1, 2));
		REQUIRE(testC.yx() == lrc::GenericVector<double, 2>(2, 1));
		REQUIRE(testC.xz() == lrc::GenericVector<double, 2>(1, 3));
		REQUIRE(testC.zx() == lrc::GenericVector<double, 2>(3, 1));
		REQUIRE(testC.yz() == lrc::GenericVector<double, 2>(2, 3));
		REQUIRE(testC.zy() == lrc::GenericVector<double, 2>(3, 2));
		REQUIRE(testC.xyz() == lrc::GenericVector<double, 3>(1, 2, 3));
		REQUIRE(testC.xzy() == lrc::GenericVector<double, 3>(1, 3, 2));
		REQUIRE(testC.yxz() == lrc::GenericVector<double, 3>(2, 1, 3));
		REQUIRE(testC.yzx() == lrc::GenericVector<double, 3>(2, 3, 1));
		REQUIRE(testC.zxy() == lrc::GenericVector<double, 3>(3, 1, 2));
		REQUIRE(testC.zyx() == lrc::GenericVector<double, 3>(3, 2, 1));
		REQUIRE(testC.xyw() == lrc::GenericVector<double, 3>(1, 2, 4));
		REQUIRE(testC.xwy() == lrc::GenericVector<double, 3>(1, 4, 2));
		REQUIRE(testC.yxw() == lrc::GenericVector<double, 3>(2, 1, 4));
		REQUIRE(testC.ywx() == lrc::GenericVector<double, 3>(2, 4, 1));
		REQUIRE(testC.wxy() == lrc::GenericVector<double, 3>(4, 1, 2));
		REQUIRE(testC.wyx() == lrc::GenericVector<double, 3>(4, 2, 1));
		REQUIRE(testC.xzw() == lrc::GenericVector<double, 3>(1, 3, 4));
		REQUIRE(testC.xwz() == lrc::GenericVector<double, 3>(1, 4, 3));
		REQUIRE(testC.zxw() == lrc::GenericVector<double, 3>(3, 1, 4));
		REQUIRE(testC.zwx() == lrc::GenericVector<double, 3>(3, 4, 1));
		REQUIRE(testC.wxz() == lrc::GenericVector<double, 3>(4, 1, 3));
		REQUIRE(testC.wzx() == lrc::GenericVector<double, 3>(4, 3, 1));
		REQUIRE(testC.yzw() == lrc::GenericVector<double, 3>(2, 3, 4));
		REQUIRE(testC.ywz() == lrc::GenericVector<double, 3>(2, 4, 3));
		REQUIRE(testC.zyw() == lrc::GenericVector<double, 3>(3, 2, 4));
		REQUIRE(testC.zwy() == lrc::GenericVector<double, 3>(3, 4, 2));
		REQUIRE(testC.wyz() == lrc::GenericVector<double, 3>(4, 2, 3));
		REQUIRE(testC.wzy() == lrc::GenericVector<double, 3>(4, 3, 2));
		REQUIRE(testC.xyzw() == lrc::GenericVector<double, 4>(1, 2, 3, 4));
		REQUIRE(testC.xywz() == lrc::GenericVector<double, 4>(1, 2, 4, 3));
		REQUIRE(testC.xzyw() == lrc::GenericVector<double, 4>(1, 3, 2, 4));
		REQUIRE(testC.xzwy() == lrc::GenericVector<double, 4>(1, 3, 4, 2));
		REQUIRE(testC.xwyz() == lrc::GenericVector<double, 4>(1, 4, 2, 3));
		REQUIRE(testC.xwzy() == lrc::GenericVector<double, 4>(1, 4, 3, 2));
		REQUIRE(testC.yxzw() == lrc::GenericVector<double, 4>(2, 1, 3, 4));
		REQUIRE(testC.yxwz() == lrc::GenericVector<double, 4>(2, 1, 4, 3));
		REQUIRE(testC.yzxw() == lrc::GenericVector<double, 4>(2, 3, 1, 4));
		REQUIRE(testC.yzwx() == lrc::GenericVector<double, 4>(2, 3, 4, 1));
		REQUIRE(testC.ywxz() == lrc::GenericVector<double, 4>(2, 4, 1, 3));
		REQUIRE(testC.ywzx() == lrc::GenericVector<double, 4>(2, 4, 3, 1));
		REQUIRE(testC.zxyw() == lrc::GenericVector<double, 4>(3, 1, 2, 4));
		REQUIRE(testC.zxwy() == lrc::GenericVector<double, 4>(3, 1, 4, 2));
		REQUIRE(testC.zyxw() == lrc::GenericVector<double, 4>(3, 2, 1, 4));
		REQUIRE(testC.zywx() == lrc::GenericVector<double, 4>(3, 2, 4, 1));
		REQUIRE(testC.zwxy() == lrc::GenericVector<double, 4>(3, 4, 1, 2));
		REQUIRE(testC.zwyx() == lrc::GenericVector<double, 4>(3, 4, 2, 1));
		REQUIRE(testC.wxyz() == lrc::GenericVector<double, 4>(4, 1, 2, 3));
		REQUIRE(testC.wxzy() == lrc::GenericVector<double, 4>(4, 1, 3, 2));
		REQUIRE(testC.wyxz() == lrc::GenericVector<double, 4>(4, 2, 1, 3));
		REQUIRE(testC.wyzx() == lrc::GenericVector<double, 4>(4, 2, 3, 1));
		REQUIRE(testC.wzxy() == lrc::GenericVector<double, 4>(4, 3, 1, 2));
		REQUIRE(testC.wzyx() == lrc::GenericVector<double, 4>(4, 3, 2, 1));

		REQUIRE(testC.str() == "(1, 2, 3, 4)");
		REQUIRE(testC.str("{:.2f}") == "(1.00, 2.00, 3.00, 4.00)");

		lrc::GenericVector<double, 3> testD = testA + lrc::GenericVector<double, 3>({3, 4});
		REQUIRE(lrc::dist(testA, testA) == 0);
		REQUIRE(lrc::dist2(testA, testD) == 25);
		REQUIRE(lrc::dist(testA, testD) == 5);

		lrc::GenericVector<double, 3> testE(0.1, 0.2, 0.3);
		REQUIRE((lrc::sin(testE) -
				 lrc::GenericVector<double, 3>(sin(0.1), lrc::sin(0.2), lrc::sin(0.3))) < 1e-6);
		REQUIRE(
		  (lrc::cos(testE) - lrc::GenericVector<double, 3>(cos(0.1), lrc::cos(0.2), lrc::cos(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::tan(testE) - lrc::GenericVector<double, 3>(tan(0.1), lrc::tan(0.2), lrc::tan(0.3)))
			.mag() < 1e-6);
		REQUIRE((lrc::asin(testE) -
				 lrc::GenericVector<double, 3>(asin(0.1), lrc::asin(0.2), lrc::asin(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::acos(testE) -
				 lrc::GenericVector<double, 3>(acos(0.1), lrc::acos(0.2), lrc::acos(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::atan(testE) -
				 lrc::GenericVector<double, 3>(atan(0.1), lrc::atan(0.2), lrc::atan(0.3)))
				  .mag() < 1e-6);
		REQUIRE(
		  (lrc::atan2(testA, testE) -
		   lrc::GenericVector<double, 3>(atan2(1, 0.1), lrc::atan2(2, 0.2), lrc::atan2(3, 0.3)))
			.mag() < 1e-6);
		REQUIRE((lrc::sinh(testE) -
				 lrc::GenericVector<double, 3>(sinh(0.1), lrc::sinh(0.2), lrc::sinh(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::cosh(testE) -
				 lrc::GenericVector<double, 3>(cosh(0.1), lrc::cosh(0.2), lrc::cosh(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::tanh(testE) -
				 lrc::GenericVector<double, 3>(tanh(0.1), lrc::tanh(0.2), lrc::tanh(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::asinh(testE) -
				 lrc::GenericVector<double, 3>(asinh(0.1), lrc::asinh(0.2), lrc::asinh(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::acosh(testE + 1) -
				 lrc::GenericVector<double, 3>(acosh(1.1), lrc::acosh(1.2), lrc::acosh(1.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::atanh(testE) -
				 lrc::GenericVector<double, 3>(atanh(0.1), lrc::atanh(0.2), lrc::atanh(0.3)))
				  .mag() < 1e-6);
		REQUIRE(
		  (lrc::exp(testE) - lrc::GenericVector<double, 3>(exp(0.1), lrc::exp(0.2), lrc::exp(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::log(testE) - lrc::GenericVector<double, 3>(log(0.1), lrc::log(0.2), lrc::log(0.3)))
			.mag() < 1e-6);
		REQUIRE((lrc::log10(testE) -
				 lrc::GenericVector<double, 3>(log10(0.1), lrc::log10(0.2), lrc::log10(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::log2(testE) -
				 lrc::GenericVector<double, 3>(log2(0.1), lrc::log2(0.2), lrc::log2(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::pow(testE, 2) -
				 lrc::GenericVector<double, 3>(pow(0.1, 2), lrc::pow(0.2, 2), lrc::pow(0.3, 2)))
				  .mag() < 1e-6);
		REQUIRE((lrc::sqrt(testE) -
				 lrc::GenericVector<double, 3>(sqrt(0.1), lrc::sqrt(0.2), lrc::sqrt(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::cbrt(testE) -
				 lrc::GenericVector<double, 3>(cbrt(0.1), lrc::cbrt(0.2), lrc::cbrt(0.3)))
				  .mag() < 1e-6);
	}
}

// =================================================================

TEST_CASE("Test SIMD Vector", "[vector-lib]") {
	SECTION("Test Fundamental Operations") {
		lrc::SIMDVector<double, 3> testA(1, 2, 3);

		REQUIRE(testA.x() == 1);
		REQUIRE(testA.y() == 2);
		REQUIRE(testA.z() == 3);

		REQUIRE(testA[0] == 1);
		REQUIRE(testA[1] == 2);
		REQUIRE(testA[2] == 3);

		REQUIRE(testA == lrc::SIMDVector<double, 3>(1, 2, 3));
	}

	SECTION("Test Constructors") {
		lrc::SIMDVector<double, 3> testA(1, 2, 3);
		lrc::SIMDVector<double, 3> testB(testA);
		lrc::SIMDVector<double, 3> testC = testA;

		REQUIRE(testA == testB);
		REQUIRE(testA == testC);

		lrc::SIMDVector<double, 3> testD(std::move(testA));
		lrc::SIMDVector<double, 3> testE = std::move(testB);

		REQUIRE(testD == lrc::SIMDVector<double, 3>(1, 2, 3));
		REQUIRE(testE == lrc::SIMDVector<double, 3>(1, 2, 3));

		lrc::SIMDVector<double, 3> testF = {1, 2, 3};

		REQUIRE(testF == lrc::SIMDVector<double, 3>(1, 2, 3));

		lrc::SIMDVector<double, 3> testG = lrc::SIMDVector<double, 3>(1, 2, 3);

		REQUIRE(testG == lrc::SIMDVector<double, 3>(1, 2, 3));
	}

	SECTION("Test Arithmetic") {
		SECTION("Test Inplace Arithmetic") {
			lrc::SIMDVector<double, 3> testA(1, 2, 3);
			lrc::SIMDVector<double, 3> testB(2, 4, 6);

			testA += testB;
			REQUIRE(testA == lrc::SIMDVector<double, 3>(3, 6, 9));

			testA -= testB;
			REQUIRE(testA == lrc::SIMDVector<double, 3>(1, 2, 3));

			testA *= testB;
			REQUIRE(testA == lrc::SIMDVector<double, 3>(2, 8, 18));

			testA /= testB;
			REQUIRE(testA == lrc::SIMDVector<double, 3>(1, 2, 3));

			testA += 1;
			REQUIRE(testA == lrc::SIMDVector<double, 3>(2, 3, 4));

			testA -= 1;
			REQUIRE(testA == lrc::SIMDVector<double, 3>(1, 2, 3));

			testA *= 2;
			REQUIRE(testA == lrc::SIMDVector<double, 3>(2, 4, 6));

			testA /= 2;
			REQUIRE(testA == lrc::SIMDVector<double, 3>(1, 2, 3));
		}

		SECTION("Test General Arithmetic") {
			lrc::SIMDVector<double, 3> testA(1, 2, 3);
			lrc::SIMDVector<double, 3> testB(2, 4, 6);

			REQUIRE(testA + testB == lrc::SIMDVector<double, 3>(3, 6, 9));
			REQUIRE(testA - testB == lrc::SIMDVector<double, 3>(-1, -2, -3));
			REQUIRE(testA * testB == lrc::SIMDVector<double, 3>(2, 8, 18));
			REQUIRE(testA / testB == lrc::SIMDVector<double, 3>(0.5, 0.5, 0.5));

			REQUIRE(testA + 1 == lrc::SIMDVector<double, 3>(2, 3, 4));
			REQUIRE(testA - 1 == lrc::SIMDVector<double, 3>(0, 1, 2));
			REQUIRE(testA * 2 == lrc::SIMDVector<double, 3>(2, 4, 6));
			REQUIRE(testA / 2 == lrc::SIMDVector<double, 3>(0.5, 1, 1.5));

			REQUIRE(1 + testA == lrc::SIMDVector<double, 3>(2, 3, 4));
			REQUIRE(1 - testA == lrc::SIMDVector<double, 3>(0, -1, -2));
			REQUIRE(2 * testA == lrc::SIMDVector<double, 3>(2, 4, 6));
			REQUIRE(2 / testA == lrc::SIMDVector<double, 3>(2, 1, 2.0 / 3.0));
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
		lrc::SIMDVector<double, 3> testA(1, 2, 3);
		lrc::SIMDVector<double, 3> testB(2, 4, 6);

		REQUIRE(testA.mag2() == 14);
		REQUIRE(testA.mag() == lrc::sqrt(14));

		REQUIRE(testA.norm() == lrc::SIMDVector<double, 3>(
								  1.0 / lrc::sqrt(14), 2.0 / lrc::sqrt(14), 3.0 / lrc::sqrt(14)));

		REQUIRE(testA.dot(testB) == 28);
		REQUIRE(testA.cross(lrc::SIMDVector<double, 3>(4, 5, 6)) ==
				lrc::SIMDVector<double, 3>(-3, 6, -3));
		REQUIRE(testA.proj(lrc::SIMDVector<double, 3>(-2, 3, 1)) ==
				lrc::SIMDVector<double, 3>(-1., 3. / 2., 1. / 2.));
		REQUIRE((bool)testA == true);
		REQUIRE((bool)lrc::SIMDVector<double, 3>(0, 0, 0) == false);

		lrc::SIMDVector<double, 4> testC(1, 2, 3, 4);
		REQUIRE(testC.xy() == lrc::SIMDVector<double, 2>(1, 2));
		REQUIRE(testC.yx() == lrc::SIMDVector<double, 2>(2, 1));
		REQUIRE(testC.xz() == lrc::SIMDVector<double, 2>(1, 3));
		REQUIRE(testC.zx() == lrc::SIMDVector<double, 2>(3, 1));
		REQUIRE(testC.yz() == lrc::SIMDVector<double, 2>(2, 3));
		REQUIRE(testC.zy() == lrc::SIMDVector<double, 2>(3, 2));
		REQUIRE(testC.xyz() == lrc::SIMDVector<double, 3>(1, 2, 3));
		REQUIRE(testC.xzy() == lrc::SIMDVector<double, 3>(1, 3, 2));
		REQUIRE(testC.yxz() == lrc::SIMDVector<double, 3>(2, 1, 3));
		REQUIRE(testC.yzx() == lrc::SIMDVector<double, 3>(2, 3, 1));
		REQUIRE(testC.zxy() == lrc::SIMDVector<double, 3>(3, 1, 2));
		REQUIRE(testC.zyx() == lrc::SIMDVector<double, 3>(3, 2, 1));
		REQUIRE(testC.xyw() == lrc::SIMDVector<double, 3>(1, 2, 4));
		REQUIRE(testC.xwy() == lrc::SIMDVector<double, 3>(1, 4, 2));
		REQUIRE(testC.yxw() == lrc::SIMDVector<double, 3>(2, 1, 4));
		REQUIRE(testC.ywx() == lrc::SIMDVector<double, 3>(2, 4, 1));
		REQUIRE(testC.wxy() == lrc::SIMDVector<double, 3>(4, 1, 2));
		REQUIRE(testC.wyx() == lrc::SIMDVector<double, 3>(4, 2, 1));
		REQUIRE(testC.xzw() == lrc::SIMDVector<double, 3>(1, 3, 4));
		REQUIRE(testC.xwz() == lrc::SIMDVector<double, 3>(1, 4, 3));
		REQUIRE(testC.zxw() == lrc::SIMDVector<double, 3>(3, 1, 4));
		REQUIRE(testC.zwx() == lrc::SIMDVector<double, 3>(3, 4, 1));
		REQUIRE(testC.wxz() == lrc::SIMDVector<double, 3>(4, 1, 3));
		REQUIRE(testC.wzx() == lrc::SIMDVector<double, 3>(4, 3, 1));
		REQUIRE(testC.yzw() == lrc::SIMDVector<double, 3>(2, 3, 4));
		REQUIRE(testC.ywz() == lrc::SIMDVector<double, 3>(2, 4, 3));
		REQUIRE(testC.zyw() == lrc::SIMDVector<double, 3>(3, 2, 4));
		REQUIRE(testC.zwy() == lrc::SIMDVector<double, 3>(3, 4, 2));
		REQUIRE(testC.wyz() == lrc::SIMDVector<double, 3>(4, 2, 3));
		REQUIRE(testC.wzy() == lrc::SIMDVector<double, 3>(4, 3, 2));
		REQUIRE(testC.xyzw() == lrc::SIMDVector<double, 4>(1, 2, 3, 4));
		REQUIRE(testC.xywz() == lrc::SIMDVector<double, 4>(1, 2, 4, 3));
		REQUIRE(testC.xzyw() == lrc::SIMDVector<double, 4>(1, 3, 2, 4));
		REQUIRE(testC.xzwy() == lrc::SIMDVector<double, 4>(1, 3, 4, 2));
		REQUIRE(testC.xwyz() == lrc::SIMDVector<double, 4>(1, 4, 2, 3));
		REQUIRE(testC.xwzy() == lrc::SIMDVector<double, 4>(1, 4, 3, 2));
		REQUIRE(testC.yxzw() == lrc::SIMDVector<double, 4>(2, 1, 3, 4));
		REQUIRE(testC.yxwz() == lrc::SIMDVector<double, 4>(2, 1, 4, 3));
		REQUIRE(testC.yzxw() == lrc::SIMDVector<double, 4>(2, 3, 1, 4));
		REQUIRE(testC.yzwx() == lrc::SIMDVector<double, 4>(2, 3, 4, 1));
		REQUIRE(testC.ywxz() == lrc::SIMDVector<double, 4>(2, 4, 1, 3));
		REQUIRE(testC.ywzx() == lrc::SIMDVector<double, 4>(2, 4, 3, 1));
		REQUIRE(testC.zxyw() == lrc::SIMDVector<double, 4>(3, 1, 2, 4));
		REQUIRE(testC.zxwy() == lrc::SIMDVector<double, 4>(3, 1, 4, 2));
		REQUIRE(testC.zyxw() == lrc::SIMDVector<double, 4>(3, 2, 1, 4));
		REQUIRE(testC.zywx() == lrc::SIMDVector<double, 4>(3, 2, 4, 1));
		REQUIRE(testC.zwxy() == lrc::SIMDVector<double, 4>(3, 4, 1, 2));
		REQUIRE(testC.zwyx() == lrc::SIMDVector<double, 4>(3, 4, 2, 1));
		REQUIRE(testC.wxyz() == lrc::SIMDVector<double, 4>(4, 1, 2, 3));
		REQUIRE(testC.wxzy() == lrc::SIMDVector<double, 4>(4, 1, 3, 2));
		REQUIRE(testC.wyxz() == lrc::SIMDVector<double, 4>(4, 2, 1, 3));
		REQUIRE(testC.wyzx() == lrc::SIMDVector<double, 4>(4, 2, 3, 1));
		REQUIRE(testC.wzxy() == lrc::SIMDVector<double, 4>(4, 3, 1, 2));
		REQUIRE(testC.wzyx() == lrc::SIMDVector<double, 4>(4, 3, 2, 1));

		REQUIRE(testC.str() == "(1, 2, 3, 4)");
		REQUIRE(testC.str("{:.2f}") == "(1.00, 2.00, 3.00, 4.00)");

		lrc::SIMDVector<double, 3> testD = testA + lrc::SIMDVector<double, 3>({3, 4});
		REQUIRE(lrc::dist(testA, testA) == 0);
		REQUIRE(lrc::dist2(testA, testD) == 25);
		REQUIRE(lrc::dist(testA, testD) == 5);

		lrc::SIMDVector<double, 3> testE(0.1, 0.2, 0.3);
		REQUIRE((lrc::sin(testE) -
				 lrc::SIMDVector<double, 3>(sin(0.1), lrc::sin(0.2), lrc::sin(0.3))) < 1e-6);
		REQUIRE(
		  (lrc::cos(testE) - lrc::SIMDVector<double, 3>(cos(0.1), lrc::cos(0.2), lrc::cos(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::tan(testE) - lrc::SIMDVector<double, 3>(tan(0.1), lrc::tan(0.2), lrc::tan(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::asin(testE) - lrc::SIMDVector<double, 3>(asin(0.1), lrc::asin(0.2), lrc::asin(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::acos(testE) - lrc::SIMDVector<double, 3>(acos(0.1), lrc::acos(0.2), lrc::acos(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::atan(testE) - lrc::SIMDVector<double, 3>(atan(0.1), lrc::atan(0.2), lrc::atan(0.3)))
			.mag() < 1e-6);
		REQUIRE((lrc::atan2(testA, testE) -
				 lrc::SIMDVector<double, 3>(atan2(1, 0.1), lrc::atan2(2, 0.2), lrc::atan2(3, 0.3)))
				  .mag() < 1e-6);
		REQUIRE(
		  (lrc::sinh(testE) - lrc::SIMDVector<double, 3>(sinh(0.1), lrc::sinh(0.2), lrc::sinh(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::cosh(testE) - lrc::SIMDVector<double, 3>(cosh(0.1), lrc::cosh(0.2), lrc::cosh(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::tanh(testE) - lrc::SIMDVector<double, 3>(tanh(0.1), lrc::tanh(0.2), lrc::tanh(0.3)))
			.mag() < 1e-6);
		REQUIRE((lrc::asinh(testE) -
				 lrc::SIMDVector<double, 3>(asinh(0.1), lrc::asinh(0.2), lrc::asinh(0.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::acosh(testE + 1) -
				 lrc::SIMDVector<double, 3>(acosh(1.1), lrc::acosh(1.2), lrc::acosh(1.3)))
				  .mag() < 1e-6);
		REQUIRE((lrc::atanh(testE) -
				 lrc::SIMDVector<double, 3>(atanh(0.1), lrc::atanh(0.2), lrc::atanh(0.3)))
				  .mag() < 1e-6);
		REQUIRE(
		  (lrc::exp(testE) - lrc::SIMDVector<double, 3>(exp(0.1), lrc::exp(0.2), lrc::exp(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::log(testE) - lrc::SIMDVector<double, 3>(log(0.1), lrc::log(0.2), lrc::log(0.3)))
			.mag() < 1e-6);
		REQUIRE((lrc::log10(testE) -
				 lrc::SIMDVector<double, 3>(log10(0.1), lrc::log10(0.2), lrc::log10(0.3)))
				  .mag() < 1e-6);
		REQUIRE(
		  (lrc::log2(testE) - lrc::SIMDVector<double, 3>(log2(0.1), lrc::log2(0.2), lrc::log2(0.3)))
			.mag() < 1e-6);
		REQUIRE((lrc::pow(testE, 2) -
				 lrc::SIMDVector<double, 3>(pow(0.1, 2), lrc::pow(0.2, 2), lrc::pow(0.3, 2)))
				  .mag() < 1e-6);
		REQUIRE(
		  (lrc::sqrt(testE) - lrc::SIMDVector<double, 3>(sqrt(0.1), lrc::sqrt(0.2), lrc::sqrt(0.3)))
			.mag() < 1e-6);
		REQUIRE(
		  (lrc::cbrt(testE) - lrc::SIMDVector<double, 3>(cbrt(0.1), lrc::cbrt(0.2), lrc::cbrt(0.3)))
			.mag() < 1e-6);
	}
}
