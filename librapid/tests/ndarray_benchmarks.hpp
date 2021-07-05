#pragma once

// #define LIBRAPID_CBLAS
#include <librapid/librapid.hpp>

#include <iostream>
#include <chrono>

inline int benchmark_ndarray()
{
	// Benchmark simple arithmetic functions
	lr_int iters = 1000;
	lr_int min = 10;
	lr_int max = 1000;
	lr_int inc = 10;

	for (lr_int size = min; size <= max; size += inc)
	{
		auto matrix = librapid::basic_ndarray<float>(librapid::extent({size, size}));
		auto vector = librapid::basic_ndarray<float>(librapid::extent({size, size}));
		matrix.fill(0);
		vector.fill(0);

		auto start = TIME;
		for (lr_int iter = 0; iter < iters; iter++)
		{
			auto res = matrix.dot(vector);
		}
		auto end = TIME;
		std::cout << (end - start) / (double) iters << "\n";
	}

	return 0;
}