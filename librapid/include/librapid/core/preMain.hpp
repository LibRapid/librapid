#ifndef LIBRAPID_CORE_PREMAIN
#define LIBRAPID_CORE_PREMAIN

/*
 * This file defines internal functions which must run *before* main() is called.
 */

namespace librapid::detail {
	class PreMain {
	public:
		PreMain();
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

	extern PreMain preMain;
} // namespace librapid::detail

#endif // LIBRAPID_CORE_PREMAIN