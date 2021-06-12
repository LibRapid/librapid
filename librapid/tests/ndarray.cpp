#include <iostream>
#include <chrono>

#define ND_MAX_DIMS 10
#define ND_NUM_THREADS 32
// #define LIBRAPID_CBLAS

#include "ndarray_benchmarks.hpp"
#include <librapid/ndarray/ndarray.hpp>

int main()
{
	auto lhs = librapid::ndarray::basic_ndarray<double>(librapid::ndarray::extent({2, 2}), 0);
	auto rhs = librapid::ndarray::basic_ndarray<double>(librapid::ndarray::extent({2, 1}), 0);

	for (nd_int i = 0; i < lhs.size(); i++)
		lhs.set_value(i, i + 1);

	for (nd_int i = 0; i < rhs.size(); i++)
		rhs.set_value(i, i + 1);

	std::cout << lhs.str() << "\n";
	std::cout << rhs.str() << "\n";
	auto start = TIME;
	auto res = lhs.dot(rhs);
	auto end = TIME;

	std::cout << end - start << "ms\n";

	std::cout << res.str() << "\n";

	benchmark_ndarray();

	return 0;
}
