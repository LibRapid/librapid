#include <iostream>
#include <chrono>
#include <map>

// #define LIBRAPID_CBLAS

#include <librapid/librapid.hpp>

int main()
{
	auto my_array = librapid::ndarray({{{{1, 2, 3}}, {{1, 2, 3}}}, {{{1, 2, 3}}, {{1, 2, 3}}}});
	std::cout << my_array << "\n";

	double start, end;

	librapid::network_config<float> config = {
	{"input", D_{{"x", 1}, {"y", 1}}},
	{"hidden", {3}},
	{"output", D_{{"o", 1}}},
	{"learning rate", {0.05, 0.05}},
	{"activation", {"leaky relu", "sigmoid"}},
	{"optimizer", "adam"}
	};

	auto test_network = librapid::network<float>(config);

	auto test_network2 = librapid::network<float>(config);

	test_network = test_network2;

	// // The training data input for XOR
	// auto train_input = librapid::from_data(
	// 	VEC<VEC<float>>{
	// 			{0, 0},
	// 			{0, 1},
	// 			{1, 0},
	// 			{1, 1}
	// });

	// Input is a vector << All inputs
	// of vectors		 << Values for input on neural network
	// of dictionaries	 << KEY ( name , value )

	using input_type = std::unordered_map<std::string, librapid::basic_ndarray<float>>;

	auto train_input = std::vector<input_type>{
	{{"x", 0},
	{"y", 0}},

	{{"x", 1},
	{"y", 0}},

	{{"x", 0},
	{"y", 1}},

	{{"x", 1},
	{"y", 1}}
	};

	// The labeled data (x XOR y)
	auto train_output = std::vector<input_type>{
		{{"o", 0}},

		{{"o", 1}},

		{{"o", 1}},

		{{"o", 0}}
	};

	test_network.compile();

	start = TIME;
	for (lr_int i = 0; i < 3000 * 4; i++) // 3000 * 4 because there are 4 bits of data
	{
		lr_int index = librapid::math::random(0, 3);
		test_network.backpropagate(train_input[index], train_output[index]);
	}
	end = TIME;
	std::cout << "Time: " << end - start << "\n";

	for (lr_int i = 0; i < 4; i++)
	{
		std::cout << "Input: " << train_input[i]["x"] << " ^ " << train_input[i]["y"] << "\n";
		std::cout << "Output: " << test_network.forward(train_input[i]) << "\n";
	}

	return 0;
}
