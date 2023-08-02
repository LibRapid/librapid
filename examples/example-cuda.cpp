#include <librapid>

namespace lrc = librapid;

auto main() -> int {
#if defined(LIBRAPID_HAS_CUDA)
	auto cudaArray = lrc::Array<float, lrc::backend::CUDA>::fromData({{1, 2, 3}, {4, 5, 6}});
	fmt::print("CUDA Array:\n{}\n", cudaArray);

	// Operations on CUDA arrays work exactly the same as on CPU arrays
	auto sum  = cudaArray + cudaArray;
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
	fmt::print("Transposed CUDA Array:\n{}\n", lrc::transpose(prod));

	auto vector = lrc::Array<float, lrc::backend::CUDA>::fromData({{1, 2, 3}});
	fmt::print("Array: \n{}\n", cudaArray);
	fmt::print("Vector: \n{}\n", vector);
	fmt::print("Matrix dot Vector^T:\n{}\n", lrc::dot(cudaArray, lrc::transpose(vector)));
#else
	fmt::print("OpenCL not enabled in this build of librapid\n");
	fmt::print("Check the documentation for more information on enabling OpenCL\n");
	fmt::print("https://librapid.readthedocs.io/en/latest/cmakeIntegration.html#librapid-use-cuda\n");
#endif // LIBRAPID_HAS_CUDA

	return 0;
}
