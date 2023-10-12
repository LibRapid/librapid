#include <librapid>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace lrc = librapid;

TEST_CASE("Integer Sets") {
	lrc::Set<int> a = {1, 3, 5, 7, 9, 12, 13, 14};
	lrc::Set<int> b = {2, 4, 6, 8, 10, 12, 13, 14};

	REQUIRE(a.size() == 8);
	REQUIRE(b.size() == 8);

	REQUIRE(a.contains(1));
	REQUIRE(a.contains(3));
	REQUIRE(a.contains(5));
	REQUIRE(a.contains(7));
	REQUIRE(a.contains(9));
	REQUIRE(a.contains(12));
	REQUIRE(a.contains(13));
	REQUIRE(a.contains(14));

	REQUIRE(b.contains(2));
	REQUIRE(b.contains(4));
	REQUIRE(b.contains(6));
	REQUIRE(b.contains(8));
	REQUIRE(b.contains(10));
	REQUIRE(b.contains(12));
	REQUIRE(b.contains(13));
	REQUIRE(b.contains(14));

	REQUIRE(!a.contains(2));
	REQUIRE(!a.contains(4));
	REQUIRE(!a.contains(6));
	REQUIRE(!a.contains(8));
	REQUIRE(!a.contains(10));

	REQUIRE(!b.contains(1));
	REQUIRE(!b.contains(3));
	REQUIRE(!b.contains(5));
	REQUIRE(!b.contains(7));
	REQUIRE(!b.contains(9));

	REQUIRE(a == a);
	REQUIRE(b == b);
	REQUIRE(a != b);
	REQUIRE(b != a);

	lrc::Set<int> setUnion = a | b;
	lrc::Set<int> setIntersection = a & b;
	lrc::Set<int> setDifference = a - b;
	lrc::Set<int> setSymmetricDifference = a ^ b;

	REQUIRE(setUnion.size() == 13);
	REQUIRE(setIntersection.size() == 3);
	REQUIRE(setDifference.size() == 5);
	REQUIRE(setSymmetricDifference.size() == 10);

	REQUIRE(setUnion == lrc::Set<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14}));
	REQUIRE(setIntersection == lrc::Set<int>({12, 13, 14}));
	REQUIRE(setDifference == lrc::Set<int>({1, 3, 5, 7, 9}));
	REQUIRE(setSymmetricDifference == lrc::Set<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

	setUnion.discard(1);
	REQUIRE(setUnion.size() == 12);
	REQUIRE(setUnion == lrc::Set<int>({2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14}));
	REQUIRE(!setUnion.contains(1));

	setUnion.remove(2);
	REQUIRE(setUnion.size() == 11);
	REQUIRE(setUnion == lrc::Set<int>({3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14}));

	lrc::Set<int> arrSet(lrc::linspace(0, 10, 11));
	REQUIRE(arrSet.size() == 11);
	REQUIRE(arrSet == lrc::Set<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}
