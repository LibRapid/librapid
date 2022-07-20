/*
 * A very simple program demonstrating how an array can be created, indexed, and some of the basic
 * operations that can be applied to it
 */

#include <librapid>

namespace lrc = librapid;

int main() {
	lrc::numThreads = 8;

	lrc::Array myArr = lrc::Array<int>(lrc::Extent(3, 5));

	for (int i = 0; i < myArr.extent()[0]; ++i) {
		for (int j = 0; j < myArr.extent()[1]; j++) { myArr[i][j] = lrc::randint(1, 10); }
	}

	fmt::print("My array:\n");
	fmt::print("{}\n", myArr);

	fmt::print("\nGetting a single row of the matrix\n");
	auto row = myArr[0];
	fmt::print("{}\n", row);

	/*
	 * Please note that in the following example, the call to operator[] forces the evaluation of
	 * the lazy transpose object. This is merely a proof of concept, and you should avoid indexing
	 * temporary objects where possible. If you need a single value, you can index the array with
	 * operator() instead, which is much more efficient.
	 */

	fmt::print("\nGetting a single column of the matrix\n");
	auto col = myArr.transposed()[0];
	fmt::print("{}\n", col);

	fmt::print("\nAdding two arrays together:\n");
	auto sum = myArr + myArr;
	fmt::print("{}\n", sum);

	fmt::print("\nCasting to a float array\n");
	auto floatArr = sum.cast<float>();
	fmt::print("{}\n", floatArr);

	fmt::print("\nCalculating the reciprocal\n");
	auto reciprocal = 1 / floatArr;
	fmt::print("{}\n", reciprocal);

	fmt::print("\n\n\n");
	fmt::print("Running a short benchmark\n");
	int rows, cols;
retry:
	fmt::print("Enter <rows>x<cols>:");
	if (!scn::input("{}x{}", rows, cols)) goto retry;
	fmt::print("Lazily evaluated result:\n");
	lrc::Array<float> bench(lrc::Extent(rows, cols));
	lrc::timeFunction([&]() { auto res = bench + bench; });
	fmt::print("Actually evaluating the result:\n");
	lrc::Array<float> res;
	lrc::timeFunction([&]() { res = bench + bench; });

	return 0;
}
