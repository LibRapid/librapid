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
		{"input", librapid::named_param({
										{"pos", 2},
										{"vel", 2},
										{"thing", 5}})},
		{"output", librapid::named_param({
										 {"pos", 3},
										 {"vel", 4},
										 {"thing", 5}})}
	};

	auto test_network = librapid::network(config);

	return 0;
}