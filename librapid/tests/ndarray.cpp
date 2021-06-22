#include <iostream>
#include <chrono>

#define LIBRAPID_MAX_DIMS 10
// #define LIBRAPID_CBLAS

#include "ndarray_benchmarks.hpp"
#include <librapid/ndarray/ndarray.hpp>

int main()
{
	auto lhs = librapid::ndarray(librapid::extent({10, 10}), 0);
	auto rhs = librapid::ndarray(librapid::extent({10, 10}), 0);

	for (lr_int i = 0; i < lhs.size(); i++)
		lhs.set_value(i, i + 1);

	for (lr_int i = 0; i < rhs.size(); i++)
		rhs.set_value(i, i + 1);

	std::cout << lhs.str() << "\n";
	std::cout << rhs.str() << "\n";
	auto start = TIME;
	auto res = lhs.dot(rhs);
	res = lhs + rhs;
	auto end = TIME;

	std::cout << end - start << "ms\n";

	std::cout << res.str() << "\n";

	// benchmark_ndarray();

	std::cout << librapid::math::round(-0.05, 1) << "\n";
	std::cout << librapid::math::round(-0.5) << "\n";
	std::cout << librapid::math::round(0.44, 1) << "\n";

	return 0;
}
