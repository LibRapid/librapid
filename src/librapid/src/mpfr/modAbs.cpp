#include <librapid/math/mpfr.hpp>

namespace librapid {
	mpfr abs(const mpfr &val) { return ::mpfr::abs(val); }
	mpfr mod(const mpfr &val, const mpfr &mod) { return ::mpfr::fmod(val, mod); }
} // namespace librapid
