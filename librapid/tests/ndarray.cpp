#include <iostream>
#include <chrono>

#define ND_MAX_DIMS 10
#define ND_NUM_THREADS 1
// #define LIBRAPID_CBLAS

#include <librapid/ndarray/ndarray.hpp>

int main()
{
	auto lhs = librapid::ndarray::basic_ndarray<double>(librapid::ndarray::extent({1000, 1000}));
	auto rhs = librapid::ndarray::basic_ndarray<double>(librapid::ndarray::extent({1000, 1000}));

	for (nd_int i = 0; i < librapid::ndarray::math::product(lhs.get_extent().get_extent(), lhs.get_extent().ndim()); i++)
		// 	lhs.set_value(i, i + 1);
		lhs.set_value(i, 1);

	for (nd_int i = 0; i < librapid::ndarray::math::product(rhs.get_extent().get_extent(), rhs.get_extent().ndim()); i++)
		// 	rhs.set_value(i, i + 1);
		rhs.set_value(i, 1);

	std::cout << "LHS:\n" << lhs.str() << "\n\n";
	std::cout << "RHS:\n" << rhs.str() << "\n\n";

	try
	{
		std::cout << "Result:\n" << (lhs.dot(rhs)).str() << "\n\n";
	}
	catch (std::exception &e)
	{
		std::cout << "Error occurred: " << e.what() << "\n";
	}

	long long iters = 100;
	auto start = (double) std::chrono::high_resolution_clock().now().time_since_epoch().count() / 1000000;
	for (long long i = 0; i < iters; i++)
	{
		auto res = lhs.dot(rhs);
	}
	auto end = (double) std::chrono::high_resolution_clock().now().time_since_epoch().count() / 1000000;
	std::cout << "Time: " << (end - start) / (double) iters << "ms\n";

	return 0;
}
