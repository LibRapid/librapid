#include <librapid>

namespace lrc = librapid;

auto main() -> int {
#if defined(LIBRAPID_HAS_OPENCL)
	// Must be called to enable OpenCL. Passing `true` logs the devices
	// available and the one selected. Set to false for a cleaner output.
	// (You can pass (true, true) to select the device manually)
	lrc::configureOpenCL(true);

	auto openclArray = lrc::Array<float, lrc::backend::OpenCL>({{1, 2, 3}, {4, 5, 6}});
	fmt::print("OpenCL Array:\n{}\n", openclArray);

	// Operations on OpenCL arrays work exactly the same as on CPU arrays
	auto sum  = openclArray + openclArray;
	auto prod = sum * sum * 10;
	fmt::print("(x + x) * (x + x) * 10:\n{}\n", prod);

	// All accessing methods work as well (though some are faster than others)
	// Note that you MUST use `auto` or `auto &` (NOT `const auto &`). This is because of how
	// the data is represented internally and how our iterators work. For more information,
	// check out the documentation:
	// https://librapid.readthedocs.io/en/latest/topics/arrayIterators.html#implicit-iteration
	fmt::print("Accessing elements: ");
	for (auto val : prod) {
		for (auto v : val) { fmt::print("{} ", v); }
	}
	fmt::print("\n");

	// Linear algebra operations also work
	fmt::print("Transposed OpenCL Array:\n{}\n", lrc::transpose(prod));

	auto vector = lrc::Array<float, lrc::backend::OpenCL>({{1, 2, 3}});
	fmt::print("Array: \n{}\n", openclArray);
	fmt::print("Vector: \n{}\n", vector);
	fmt::print("Matrix dot Vector^T:\n{}\n", lrc::dot(openclArray, lrc::transpose(vector)));
#else
	fmt::print("OpenCL not enabled in this build of librapid\n");
	fmt::print("Check the documentation for more information on enabling OpenCL\n");
	fmt::print("https://librapid.readthedocs.io/en/latest/cmakeIntegration.html#librapid-use-opencl\n");
#endif

	return 0;
}
