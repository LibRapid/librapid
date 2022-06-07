/*
 * A very simple program demonstrating how an array can be created, indexed, and some of the basic
 * operations that can be applied to it
 */

// You can replace this with "#include <librapid/librapid.hpp>"
#include "../../src/librapid/librapid.hpp"

namespace lrc = librapid;

int main() {
	lrc::Array myArr = lrc::Array<int>(lrc::Extent(3, 5));

	for (int i = 0; i < myArr.extent()[0]; ++i) {
		for (int j = 0; j < myArr.extent()[1]; j++) { myArr[i][j] = lrc::randint(1, 10); }
	}

	fmt::print("My array:\n");
	fmt::print("{}\n", myArr.str());

	fmt::print("\nGetting a single row of the matrix\n");
	auto row = myArr[0];
	fmt::print("{}\n", row.str());

	fmt::print("\nGetting a single column of the matrix\n");
	auto col = myArr.transposed()[0];
	fmt::print("{}\n", col.str());

	fmt::print("\nAdding two arrays together:\n");
	auto sum = myArr + myArr;
	fmt::print("{}\n", sum.str());

	fmt::print("\nCasting to a float array\n");
	auto floatArr = sum.cast<float>();
	fmt::print("{}\n", floatArr.str());

	fmt::print("\nCalculating the reciprocal\n");
	auto reciprocal = 1 / floatArr;
	fmt::print("{}\n", reciprocal.str());

	fmt::print("\n\n\n");
	fmt::print("Running a short benchmark to whet your appetite :)\n");
	int rows, cols;
	scn::prompt("Enter '<rows>x<cols>' >>>", "{}x{}", rows, cols);
	fmt::print("Lazily evaluated result:\n");
	lrc::Array<float> bench(lrc::Extent(rows, cols));
	lrc::timeFunction([&]() { auto res = bench + bench; });
	fmt::print("Actually evaluating the restlt:\n");
	lrc::Array<float> res;
	lrc::timeFunction([&]() { res = bench + bench; });

	return 0;
}
