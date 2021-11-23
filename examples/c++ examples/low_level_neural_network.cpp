#include <iostream>
#include <chrono>

// Uncomment this to use CBlas (note, it must be linked at compile time)
// #define LIBRAPID_CBLAS
#include <librapid/librapid.hpp>

int main() {
	// The training data input for XOR
	auto train_input = librapid::ndarray::from_data(
			VEC < VEC < int >> {
					{0, 0},
					{0, 1},
					{1, 0},
					{1, 1}
			});

	// The labeled data (x XOR y)
	auto train_output = librapid::ndarray::from_data(
			VEC < VEC < int >> {
					{0},
					{1},
					{1},
					{0}
			});

	// Reshape the data so it can be used by the neural network
	train_input.reshape({4, 2, 1});
	train_output.reshape({4, 1, 1});

	// Create the activations
	auto activation1 = new librapid::activations::sigmoid<double>();
	auto activation2 = new librapid::activations::sigmoid<double>();

	// Create the optimizers
	auto optimizer1 = new librapid::optimizers::sgd<double>(0.1);
	auto output_optimizer = new librapid::optimizers::sgd<double>(0.1);

	// Create the layers
	auto input_layer = new librapid::layers::input<double>(2);
	auto hidden_layer_1 = new librapid::layers::affine<double>(3, activation1, optimizer1);
	auto output_layer = new librapid::layers::affine<double>(1, activation2, output_optimizer);

	// Create the network and add the layers
	auto network = librapid::network<double>();
	network.add_layer(input_layer);
	network.add_layer(hidden_layer_1);
	network.add_layer(output_layer);

	// Compile the network
	network.compile();

	// Print the data
	std::cout << "Inputs\n" << train_input << "\n\n";
	std::cout << "Outputs\n" << train_output << "\n\n";

	std::cout << "\n\n\n";

	// Time the training process over 3000 epochs
	auto start = TIME;
	for (int64_t i = 0; i < 3000 * 4; i++) // 3000 * 4 because there are 4 bits of data
	{
		int64_t index = librapid::math::random(0, 3);
		network.backpropagate(train_input[index], train_output[index]);
	}
	auto end = TIME;
	std::cout << "Time: " << end - start << "\n";

	// Print the output of the neural network
	for (int64_t i = 0; i < 4; i++) {
		std::cout << "Input: " << train_input[i].str(7) << "\n";
		std::cout << "Output: " << network.forward(train_input[i]) << "\n";
	}


	// Everything below here is for the gradiented neural network output.
	// Don't bother reading or changing anything below here unless you have to

	std::string space(10, ' ');

	std::cout << " Neural Network Output                  \"True Output\"\n";
	std::cout << "/" << std::string(22, '=') << "\\";
	std::cout << space;
	std::cout << "/" << std::string(22, '=') << "\\\n";

	for (int y = 0; y < 20; y++) {
		std::cout << "||";
		for (int x = 0; x < 20; x++) {
			double x_coord = x / 20.;
			double y_coord = y / 20.;

			auto res = network.forward(
					librapid::ndarray::from_data(std::vector<double>{x_coord, y_coord}).reshaped({2, 1}));
			double val = res[0][0].to_scalar();

			//  . - * % & #
			if (val > 0.86)
				std::cout << "#";
			else if (val > 0.71)
				std::cout << "&";
			else if (val > 0.57)
				std::cout << "%";
			else if (val > 0.43)
				std::cout << "*";
			else if (val > 0.32)
				std::cout << "-";
			else if (val > 0.14)
				std::cout << ".";
			else
				std::cout << " ";
		}
		std::cout << "||";

		std::cout << space;

		std::cout << "||";
		for (int x = 0; x < 20; x++) {
			double x_coord = x / 20.;
			double y_coord = y / 20.;

			if (abs(x_coord - y_coord) > 0.5)
				std::cout << "#";
			else
				std::cout << " ";
		}
		std::cout << "||\n";
	}

	std::cout << "\\" << std::string(22, '=') << "/";
	std::cout << space;
	std::cout << "\\" << std::string(22, '=') << "/\n";

	std::cout << "Press enter to quit\n";
	std::getchar();

	return 0;
}
