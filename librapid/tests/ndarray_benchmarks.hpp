#pragma once

// #define LIBRAPID_CBLAS
#include <librapid/librapid.hpp>

#include <iostream>
#include <chrono>

// Get the time in milliseconds
#define TIME ((double) std::chrono::high_resolution_clock().now().time_since_epoch().count() / 1000000)

inline int benchmark_ndarray()
{
	// Benchmark simple arithmetic functions
	nd_int iters = 1000;
	nd_int min = 10;
	nd_int max = 1000;
	nd_int inc = 10;

	for (nd_int size = min; size <= max; size += inc)
	{
		auto matrix = librapid::ndarray::basic_ndarray<float>(librapid::ndarray::extent({size, size}));
		auto vector = librapid::ndarray::basic_ndarray<float>(librapid::ndarray::extent({size, 1ll}));
		matrix.fill(0);
		vector.fill(0);

		auto start = TIME;
		for (nd_int iter = 0; iter < iters; iter++)
		{
			auto res = matrix.dot(vector);
		}
		auto end = TIME;
		std::cout << (end - start) / (double) iters << "\n";
	}

	return 0;
}