/*
 * A very simple program demonstrating how an array can be created, indexed, and some of the basic
 * operations that can be applied to it
 */

#include <librapid>

namespace lrc = librapid;

int main() {
	// Create an array of 3x3 integers
	lrc::Array<int> arr(lrc::Extent(3, 3));

	// Fill it with random values
	for (int i = 0; i < arr.extent()[0]; ++i) {
		for (int j = 0; j < arr.extent()[1]; ++j) { arr(i, j) = lrc::randint(0, 100); }
	}

	fmt::print("Array:\n{}\n", arr);

	// Set all values below 50 to 0, otherwise set them to 100
	auto res = lrc::map<false>(
	  [](auto val) {
		  if (val >= 50) {
			  return 100;
		  } else {
			  return 0;
		  }
	  },
	  arr);

	fmt::print("Result:\n{}\n", res);

	return 0;
}
