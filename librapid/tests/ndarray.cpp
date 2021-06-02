#include <iostream>
#include <chrono>

#define ND_MAX_DIMS 32
#define ND_NUM_THREADS 5
#define ULM_BLOCKED

#include <librapid/ndarray/ndarray.hpp>

int main()
{
	auto lhs = librapid::ndarray::basic_ndarray<double>(librapid::ndarray::extent({100, 100}));
	auto rhs = librapid::ndarray::basic_ndarray<double>(librapid::ndarray::extent({100, 100}));

	for (nd_int i = 0; i < librapid::ndarray::math::product(lhs.get_extent().get_extent(), lhs.get_extent().ndim()); i++)
		// lhs.set_value(i, i + 1);
		lhs.set_value(i, 1);

	for (nd_int i = 0; i < librapid::ndarray::math::product(rhs.get_extent().get_extent(), rhs.get_extent().ndim()); i++)
		// rhs.set_value(i, i + 1);
		rhs.set_value(i, 1);

	std::cout << "LHS:\n" << lhs.str() << "\n\n";
	std::cout << "RHS:\n" << rhs.str() << "\n\n";

	try
	{
		std::cout << "Result:\n" << (lhs.dot(rhs)).str() << "\n\n";
	}
	catch (std::exception &e)
	{
		std::cout << "Error occurred while adding arrays: " << e.what() << "\n";
	}

	long long iters = 1000;
	auto start = (double) std::chrono::high_resolution_clock().now().time_since_epoch().count() / 1000000;
	for (long long i = 0; i < iters; i++)
		auto res = lhs.dot(rhs);
	auto end = (double) std::chrono::high_resolution_clock().now().time_since_epoch().count() / 1000000;
	std::cout << "Time: " << (end - start) / (double) iters << "ms\n";

	// 	std::cout << "Breaking the Matrix\n";
	// 
	// #pragma omp parallel for shared(rhs, lhs)
	// 	for (nd_int outer = 0; outer < 1000; outer++)
	// 	{
	// 		for (nd_int inner = 0; inner < 1000; inner++)
	// 		{
	// 			// res[outer][inner] = tmp_this[outer].dot(tmp_other[inner]);
	// 			auto new_thing = lhs[0].clone();
	// 
	// 			if (*new_thing.m_origin_references == 0)
	// 				std::cout << "Matrix corrupted\n";
	// 		}
	// 	}
	// 
	// 	std::cout << "The Matrix has Held\n";
	// 
	// 	std::cout << lhs.str() << "\n";

	return 0;
}
