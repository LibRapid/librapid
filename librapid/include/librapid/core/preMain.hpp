#ifndef LIBRAPID_CORE_PREMAIN
#define LIBRAPID_CORE_PREMAIN

/*
 * This file defines internal functions which must run *before* main() is called.
 */

namespace librapid::detail {
	class PreMain {
	public:
		PreMain() {
#if defined(LIBRAPID_WINDOWS)
			// Force the terminal to accept ANSI characters
			system(("chcp " + std::to_string(CP_UTF8)).c_str());
#endif // LIBRAPID_WINDOWS
		}

	private:
	};

	// These must be declared here for use in ASSERT functions
	template<typename T>
	T internalMax(T val) {
		return val;
	}

	template<typename T, typename... Tn>
	T internalMax(T val, Tn... vals) {
		auto maxOther = internalMax(vals...);
		return val < maxOther ? maxOther : val;
	}

	[[maybe_unused]] static PreMain preMain = PreMain();
} // namespace librapid::detail

#endif // LIBRAPID_CORE_PREMAIN