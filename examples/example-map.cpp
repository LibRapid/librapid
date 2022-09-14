/*
 * A very simple program demonstrating how an array can be created, indexed, and some of the basic
 * operations that can be applied to it
 */

#include <librapid>

namespace lrc = librapid;

struct MyOperation {
	// CPU implementation
	// (This should almost always take a templated type. See the docs for more info on why)
	template<typename A, typename B>
	auto operator()(A a, B b) const {
		return (a > b) ? a : b;
	}

	// CUDA kernel generators
	std::vector<std::string> args() const { return {"a", "b"}; }
	std::string kernel() const { return "return (a > b) ? a : b;"; }
};

int main() {
	// Create a 3x3 array of doubles
	lrc::Array<double> arr(lrc::Extent(3, 3));

	// Fill it with random values
	for (int i = 0; i < arr.extent()[0]; ++i) {
		for (int j = 0; j < arr.extent()[1]; ++j) { arr(i, j) = lrc::random(0, 1); }
	}

	fmt::print("Array:\n{}\n", arr);

	// The map function defaults to "map<true>", but by overriding that, you can specify that
	// this function should not use SIMD operations. Try setting this to "map<true>" and see
	// that the compiler will error :)
	auto resSISD = lrc::map<false>(
	  [](auto val) {
		  // Set all values below 0.5 to 0, otherwise set them to 1
		  if (val >= 0.5) {
			  return 1;
		  } else {
			  return 0;
		  }
	  },
	  arr);

	// Same calculation as above, except the "map<true>" version uses SIMD instructions.
	// SIMD instructions accelerate the calculation significantly, in some cases, but
	// not all calculations will support them. That's why you can specify whether to use them.
	auto resSIMD = lrc::map<true>([](auto val) { return lrc::floor(val + 0.5); }, arr);

	fmt::print("Result:\n{}\n", resSISD);
	fmt::print("Result:\n{}\n", resSIMD);

	return 0;
}
