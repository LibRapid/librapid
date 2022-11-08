#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

TEST_CASE("Test Multiprecision", "[multiprecision]") {
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
	REQUIRE(fmt::format("{}", lrc::mpfr(1)) == "1.00000000000000");

	REQUIRE(fmt::format("{}", lrc::mpz(1234)) == "1234");
	REQUIRE(fmt::format("{}", lrc::mpq(1234)) == "1234");
	REQUIRE(fmt::format("{}", lrc::mpf(1234)) == "1234");
	REQUIRE(fmt::format("{}", lrc::mpfr(1234)) == "1234");
	REQUIRE(fmt::format("{}", lrc::mpfr(3.1415926)) == "3.1415926");

	REQUIRE(lrc::mpz("1234") == 1234);
	REQUIRE(lrc::mpq("1234") == 1234);
	REQUIRE(lrc::mpq("1234/617") == 2);
	REQUIRE(lrc::mpf("1234") == 1234);
	REQUIRE(lrc::mpfr("1234") == 1234);

	lrc::prec(500);
	REQUIRE(lrc::constPi() ==
			"3."
			"14159265358979323846264338327950288419716939937510582097494459230781640628620899862803"
			"48253421170679821480865132823066470938446095505822317253594081284811174502841027019385"
			"21105559644622948954930381964428810975665933446128475648233786783165271201909145648566"
			"92346034861045432664821339360726024914127372458700660631558817488152092096282925409171"
			"53643678925903600113305305488204665213841469519415116094330572703657595919530921861173"
			"819326117931051185480744623799627495673518857527248912279381830119491");
}
