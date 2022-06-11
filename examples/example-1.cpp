/*
 * A very simple program demonstrating how an array can be created, indexed, and some of the basic
 * operations that can be applied to it
 */

#include <librapid/librapid.hpp>

namespace lrc = librapid;

int main() {
	lrc::numThreads = 8;

	lrc::Array myArr = lrc::Array<int>(lrc::ExtentType(3, 5));

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
	fmt::print("Running a short benchmark\n");
	int rows = 1000, cols = 1000;
	// if (!scn::prompt("Enter '<rows>x<cols>' >>>", "{}x{}", rows, cols)) exit(1);
	fmt::print("Lazily evaluated result:\n");
	lrc::Array<float> bench(lrc::ExtentType(rows, cols));
	lrc::timeFunction([&]() { auto res = bench + bench; });
	fmt::print("Actually evaluating the result:\n");
	lrc::Array<float> res;
	lrc::timeFunction([&]() { res = bench + bench; });
	/*
	 */

	std::cout << "Hello, World\n";

	int64_t size  = 1000 * 1000;
	int64_t iters = 10000;

	auto data123 = new float[size];
	auto res123	 = new float[size];

	double start = (double)std::chrono::high_resolution_clock::now().time_since_epoch().count();
	for (int64_t loop = 0; loop < iters; ++loop) {
		for (int64_t i = 0; i < size; ++i) { res123[i] = data123[i] + data123[i]; }
	}
	double end = (double)std::chrono::high_resolution_clock::now().time_since_epoch().count();

	std::cout << "Result: " << data123[lrc::randint<int64_t>(0, size - 1)] << "\n";

	delete[] data123;
	delete[] res123;

	std::cout << "Elapsed: " << (end - start) / 1000000000 << "s\n";
	std::cout << "Average: " << ((end - start) / (double)iters) / 1000 << "us\n";

	return 0;
}
