#include <iostream>
#include <chrono>
#include <map>

// #define LIBRAPID_CBLAS

#include <librapid/librapid.hpp>

int main()
{
	auto top = librapid::range(1., 10.).reshaped({3, 3});
	auto bottom = librapid::range(10., 19.).reshaped({3, 3});

	std::cout << top << "\n\n\n";
	std::cout << bottom << "\n\n\n";

	try
	{
		std::cout << librapid::concatenate({top, bottom}, 0) << "\n\n\n\n";
	}
	catch (std::exception &e)
	{
		std::cout << "Failed\n";
	}

	try
	{
		std::cout << librapid::concatenate({top, bottom}, 1) << "\n\n\n\n";
	}
	catch (std::exception &e)
	{
		std::cout << "Failed\n";
	}

	try
	{
		std::cout << librapid::concatenate({top, bottom}, 2) << "\n\n\n\n";
	}
	catch (std::exception &e)
	{
		std::cout << "Failed\n";
	}

	try
	{
		std::cout << librapid::stack({top, bottom}, 2).str(0, true) << "\n";
	}
	catch (std::exception &e)
	{
		std::cout << "Error: " << e.what() << "\n";
		return 1;
	}

	double start, end;

	librapid::network_config<float> config = {
	{"input", D_{{"x", 1}, {"y", 1}}},
	{"hidden", {3}},
	{"output", D_{{"o", 1}}},
	{"learning rate", {0.05, 0.05}},
	{"activation", {"leaky relu", "sigmoid"}},
	{"optimizer", "adam"}
	};

	auto test_network = librapid::network(config);

	auto test_network2 = librapid::network(config);

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
	{{"x", librapid::from_data(0.f)},
	{"y", librapid::from_data(0.f)}},

	{{"x", librapid::from_data(1.f)},
	{"y", librapid::from_data(0.f)}},

	{{"x", librapid::from_data(0.f)},
	{"y", librapid::from_data(1.f)}},

	{{"x", librapid::from_data(1.f)},
	{"y", librapid::from_data(1.f)}}
	};

	// The labeled data (x XOR y)
	auto train_output = std::vector<input_type>{
		{{"o", librapid::from_data(0.f)}},

		{{"o", librapid::from_data(1.f)}},

		{{"o", librapid::from_data(1.f)}},

		{{"o", librapid::from_data(0.f)}}
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
