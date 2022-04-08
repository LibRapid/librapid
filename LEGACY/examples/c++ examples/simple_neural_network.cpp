#include <iostream>
#include <chrono>
#include <map>

// #define LIBRAPID_CBLAS

#include <librapid/librapid.hpp>

int main() {
	double start, end;

	librapid::network_config<float> config = {
			{"input",         2},
			{"hidden",        {3}},
			{"output",        1},
			{"learning rate", {0.01, 0.01}},
			{"activation",    "leaky relu"},
			{"optimizer",     "adam"}
	};

	librapid::config_container<float> test;

	test = librapid::config_container<float>("hello", 1234);

	auto test_network = librapid::network(config);

	auto test_network2 = librapid::network(config);

	test_network = test_network2;

	// The training data input for XOR
	auto train_input = librapid::from_data(
			VEC < VEC < float >> {
					{0, 0},
					{0, 1},
					{1, 0},
					{1, 1}
			});

	// The labeled data (x XOR y)
	auto train_output = librapid::from_data(
			VEC < VEC < float >> {
					{0},
					{1},
					{1},
					{0}
			});

	// Reshape the data so it can be used by the neural network
	train_input.reshape({4, 2, 1});
	train_output.reshape({4, 1, 1});

	test_network.compile();

	start = TIME;
	for (int64_t i = 0; i < 3000 * 4; i++) // 3000 * 4 because there are 4 bits of data
	{
		int64_t index = librapid::math::random(0, 3);
		test_network.backpropagate(train_input[index], train_output[index]);
	}
	end = TIME;
	std::cout << "Time: " << end - start << "\n";

	for (int64_t i = 0; i < 4; i++) {
		std::cout << "Input: " << train_input[i].str(7) << "\n";
		std::cout << "Output: " << test_network.forward(train_input[i]) << "\n";
	}

	return 0;
}