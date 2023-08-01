#include <librapid>

namespace lrc = librapid;

auto main() -> int {
	fmt::print("LibRapid Example -- Array 1\n");

	// Create a vector with 10 elements
	lrc::Array<int> myVector(lrc::Shape({5}));

	// Fill the vector with values
	for (int i = 0; i < 5; i++) { myVector[i] = i; }

	// Print the vector
	fmt::print("Vector: {}\n", myVector);

	// Create a matrix with 3x5 elements
	lrc::Array<int> myMatrix(lrc::Shape({3, 5}));

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 5; j++) { myMatrix[i][j] = i * 5 + j; }
	}

	fmt::print("Matrix:\n{}\n", myMatrix);

	// Do some simple calculations
	fmt::print("My Vector + My Vector: {}\n", myVector + myVector);
	fmt::print("[0] + [4]: {}\n", myVector[0].get() + myVector[4].get());
	fmt::print("\n");
	fmt::print("M + M * M:\n{}\n", myMatrix + myMatrix * myMatrix);

	// Add a vector to a row of a matrix
	fmt::print("My Vector + My matrix [2]: {}\n", myVector + myMatrix[2]);

	// Compare two arrays
	lrc::Array<int> leftVector(lrc::Shape({5}));
	lrc::Array<int> rightVector(lrc::Shape({5}));
	leftVector << 1, 2, 3, 4, 5;
	rightVector << 5, 4, 3, 2, 1;
	fmt::print("{} < {}  -->  {}\n", leftVector, rightVector, leftVector < rightVector);
	fmt::print("{} >= {}  -->  {}\n", leftVector, rightVector, leftVector >= rightVector);

	return 0;
}