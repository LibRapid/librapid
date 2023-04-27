#include <librapid/librapid.hpp>

namespace librapid::detail {
	bool preMainRun = false;

	PreMain::PreMain() {
		if (!preMainRun) {
#if defined(LIBRAPID_WINDOWS) && !defined(LIBRAPID_NO_WINDOWS_H)
			// Force the terminal to accept ANSI characters
			system(("chcp " + std::to_string(CP_UTF8)).c_str());
#endif // LIBRAPID_WINDOWS

			preMainRun			  = true;
			global::cacheLineSize = cacheLineSize();

#if defined(LIBRAPID_HAS_CUDA)
			cudaStreamCreate(&global::cudaStream);
			cublasCreate(&global::cublasHandle);
#endif // LIBRAPID_HAS_CUDA
		}
	}
} // namespace librapid::detail
