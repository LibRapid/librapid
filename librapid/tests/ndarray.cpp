#include <iostream>
#include <chrono>
#include <map>

// #define LIBRAPID_CBLAS

#include "ndarray_benchmarks.hpp"
#include <librapid/ndarray/ndarray.hpp>

int main()
{
	double start, end;

	librapid::network_config<double> config = {
		{"input", 2},
		{"output", 1}
	};

	auto test_network = librapid::network(config);

	return 0;
}