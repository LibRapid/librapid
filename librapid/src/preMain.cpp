#include <librapid/librapid.hpp>

namespace librapid::detail {
	bool preMainRun = false;

	PreMain::PreMain() {
		if (!preMainRun) {
#if defined(LIBRAPID_WINDOWS)
			// Force the terminal to accept ANSI characters
			system(("chcp " + std::to_string(CP_UTF8)).c_str());
#endif // LIBRAPID_WINDOWS

			preMainRun = true;
			global::cacheLineSize = cacheLineSize();
		}
	}
} // namespace librapid::detail
