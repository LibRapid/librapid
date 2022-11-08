#include <librapid/librapid.hpp>

# if defined(LIBRAPID_USE_MULTIPREC)

namespace librapid {
	mpfr hypot(const mpfr &a, const mpfr &b) { return ::mpfr::hypot(a, b); }
} // namespace librapid

#else

[[maybe_unused]] LIBRAPID_NODISCARD int patch(int x) { return x; }

#endif // LIBRAPID_USE_MULTIPREC
