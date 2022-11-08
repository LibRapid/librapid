#include <librapid/librapid.hpp>

#if defined(LIBRAPID_USE_MULTIPREC)

namespace librapid {
	mpfr floor(const mpfr &val) { return ::mpfr::floor(val); }
	mpfr ceil(const mpfr &val) { return ::mpfr::ceil(val); }
}

#else

[[maybe_unused]] LIBRAPID_NODISCARD int patch(int x) { return x; }

#endif // LIBRAPID_USE_MULTIPREC
