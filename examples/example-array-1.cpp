#include <librapid>

namespace lrc = librapid;

template<typename... Args>
void printHelper(Args... args) {
	fmt::print(args...);
	std::cout << std::flush; // Flush the output buffer
}

auto main() -> int {
	fmt::print("LibRapid Example -- Array 1\n");

	// Create a vector with 10 elements
	printHelper("Creating Vector");
	lrc::Array<int> myVector(lrc::Shape({5}));

	// Fill the vector with values
	printHelper("Filling Vector");
	for (int i = 0; i < 5; i++) { myVector[i] = i; }

	// Print the vector
	printHelper("Printing Vector");
	fmt::print("Vector: {}\n", myVector);

	// Create a matrix with 3x5 elements
	printHelper("Creating Matrix");
	lrc::Array<int> myMatrix(lrc::Shape({3, 5}));

	printHelper("Filling Matrix");
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 5; j++) { myMatrix[i][j] = i * 5 + j; }
	}

	printHelper("Printing Matrix");
	fmt::print("Matrix:\n{}\n", myMatrix);

	// Do some simple calculations
	printHelper("Adding Vectors");
	fmt::print("My Vector + My Vector: {}\n", myVector + myVector);

	printHelper("Adding Elements");
	fmt::print("[0] + [4]: {}\n", myVector[0].get() + myVector[4].get());
	fmt::print("\n");
	printHelper("Combined Operations");
	fmt::print("M + M * M:\n{}\n", myMatrix + myMatrix * myMatrix);

	// Add a vector to a row of a matrix
	printHelper("Vector + Matrix[2]");
	fmt::print("My Vector + My matrix [2]: {}\n", myVector + myMatrix[2]);

	// Compare two arrays
	printHelper("Creating Vectors");
	lrc::Array<int> leftVector(lrc::Shape({5}));
	lrc::Array<int> rightVector(lrc::Shape({5}));
	printHelper("Comma Initializing Vectors");
	leftVector << 1, 2, 3, 4, 5;
	rightVector << 5, 4, 3, 2, 1;
	printHelper("Less Than");
	fmt::print("{} < {}  -->  {}\n", leftVector, rightVector, leftVector < rightVector);
	printHelper("Greater or Equal");
	fmt::print("{} >= {}  -->  {}\n", leftVector, rightVector, leftVector >= rightVector);

	return 0;
}