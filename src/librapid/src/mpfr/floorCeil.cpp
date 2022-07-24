#include <librapid/math/mpir.hpp>

namespace librapid {
	mpfr floor(const mpfr &val) { return ::mpfr::floor(val); }
	mpfr ceil(const mpfr &val) { return ::mpfr::ceil(val); }
}
