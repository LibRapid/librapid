#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

/*
 * Very incomplete testing. I'm going to assume that the mpfr library is correct :)
 */

namespace lrc = librapid;

#if defined(LIBRAPID_USE_MULTIPREC)

TEST_CASE("Test Multiprecision", "[multiprecision]") {
	lrc::prec(16);
	REQUIRE(lrc::mpz(1) == 1);
	REQUIRE(lrc::mpq(1) == 1);
	REQUIRE(lrc::mpf(1) == 1);
	REQUIRE(lrc::mpfr(1) == 1);

	REQUIRE(lrc::mpz(1) == lrc::mpz(1));
	REQUIRE(lrc::mpq(1) == lrc::mpq(1));
	REQUIRE(lrc::mpf(1) == lrc::mpf(1));
	REQUIRE(lrc::mpfr(1) == lrc::mpfr(1));

	REQUIRE(fmt::format("{}", lrc::mpz(1)) == "1");
	REQUIRE(fmt::format("{}", lrc::mpq(1)) == "1");
	REQUIRE(fmt::format("{}", lrc::mpf(1)) == "1.0");

	REQUIRE(fmt::format("{}", lrc::mpz(1234)) == "1234");
	REQUIRE(fmt::format("{}", lrc::mpq(1234)) == "1234");
	REQUIRE(fmt::format("{}", lrc::mpf(1234)) == "1234.0");

	REQUIRE(lrc::mpz("1234") == 1234);
	REQUIRE(lrc::mpq("1234") == 1234);
	REQUIRE(lrc::mpq("1234/617") == 2);
	REQUIRE(lrc::mpf("1234") == 1234);
	REQUIRE(lrc::mpfr("1234") == 1234);

	REQUIRE(lrc::abs(lrc::sin(lrc::constPi()) - 0) < lrc::exp10(-15));
	REQUIRE(lrc::abs(lrc::cos(lrc::constPi()) + 1) < lrc::exp10(-15));
	REQUIRE(lrc::abs(lrc::tan(lrc::constPi()) - 0) < lrc::exp10(-15));

	REQUIRE(lrc::mpz("10") + lrc::mpz("100") == 110);
	REQUIRE(lrc::mpq("10") + lrc::mpq("100") == 110);
	REQUIRE(lrc::mpf("10") + lrc::mpf("100") == 110);
	REQUIRE(lrc::mpfr("10") + lrc::mpfr("100") == 110);

	lrc::prec(500);
	REQUIRE(
	  lrc::constPi() ==
	  "3."
	  "14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534"
	  "21170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446"
	  "22948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326"
	  "64821339360726024914127372458700660631558817488152092096282925409171536436789259036001133053"
	  "05488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799"
	  "627495673518857527248912279381830119491304781324640566153215447811706812394506641467563306");

	SECTION("Benchmarks") {
		for (int64_t prec = 128; prec <= 1 << 24; prec <<= 1) {
			lrc::prec2(prec);

			lrc::mpz bigInt(1);
			bigInt <<= prec;
			BENCHMARK(fmt::format("Integer Addition\n[{} bits]", prec)) { return bigInt + bigInt; };

			BENCHMARK(fmt::format("Integer Multiplication\n[{} bits]", prec)) {
				return bigInt * bigInt;
			};

			lrc::mpq bigRat = lrc::mpq(bigInt) / lrc::mpq(bigInt + 1);
			BENCHMARK(fmt::format("Rational Addition\n[{} bits]", prec)) {
				return bigRat + bigRat;
			};

			lrc::mpfr bigFloat = lrc::constPi();
			BENCHMARK(fmt::format("Floating Point Addition\n[{} bits]", prec)) {
				return bigFloat + bigFloat;
			};

			BENCHMARK(fmt::format("Floating Point Multiplication\n[{} bits]", prec)) {
				return bigFloat * bigFloat;
			};

			BENCHMARK(fmt::format("Pi Calculation\n[{} bits]", prec)) { return lrc::constPi(); };
		}
	}
}

#else

TEST_CASE("INVALID -- MultiPrecision not Enabled", "[multiprecision]") { REQUIRE(false); }

#endif // LIBRAPID_USE_MULTIPRECISION
