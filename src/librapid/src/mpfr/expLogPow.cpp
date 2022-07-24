#include <librapid/math/mpir.hpp>

namespace librapid {
	mpfr sqrt(const mpfr &val) { return ::mpfr::sqrt(val); }
	mpfr pow(const mpfr &base, const mpfr &pow) { return ::mpfr::pow(base, pow); }
	mpfr exp(const mpfr &val) { return ::mpfr::exp(val); }
	mpfr exp2(const mpfr &val) { return ::mpfr::exp2(val); }
	mpfr exp10(const mpfr &val) { return ::mpfr::exp10(val); }
	mpfr ln(const mpfr &val) { return ::mpfr::log(val); }
	mpfr log(const mpfr &val, const mpfr &base) { return ::mpfr::log(val) / ::mpfr::log(base); }
	mpfr log2(const mpfr &val) { return ::mpfr::log2(val); }
	mpfr log10(const mpfr &val) { return ::mpfr::log10(val); }
} // namespace librapid
