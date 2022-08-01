#include <librapid/math/mpfr.hpp>

namespace librapid {
	mpfr sin(const mpfr &val) { return ::mpfr::sin(val); }
	mpfr cos(const mpfr &val) { return ::mpfr::cos(val); }
	mpfr tan(const mpfr &val) { return ::mpfr::tan(val); }

	mpfr asin(const mpfr &val) { return ::mpfr::asin(val); }
	mpfr acos(const mpfr &val) { return ::mpfr::acos(val); }
	mpfr atan(const mpfr &val) { return ::mpfr::atan(val); }
	mpfr atan2(const mpfr &dy, const mpfr &dx) { return ::mpfr::atan2(dy, dx); }

	mpfr csc(const mpfr &val) { return ::mpfr::csc(val); }
	mpfr sec(const mpfr &val) { return ::mpfr::sec(val); }
	mpfr cot(const mpfr &val) { return ::mpfr::cot(val); }

	mpfr acsc(const mpfr &val) { return ::mpfr::acsc(val); }
	mpfr asec(const mpfr &val) { return ::mpfr::asec(val); }
	mpfr acot(const mpfr &val) { return ::mpfr::acot(val); }

	mpfr sinh(const mpfr &val) { return ::mpfr::sinh(val); }
	mpfr cosh(const mpfr &val) { return ::mpfr::cosh(val); }
	mpfr tanh(const mpfr &val) { return ::mpfr::tanh(val); }

	mpfr asinh(const mpfr &val) { return ::mpfr::asinh(val); }
	mpfr acosh(const mpfr &val) { return ::mpfr::acosh(val); }
	mpfr atanh(const mpfr &val) { return ::mpfr::atanh(val); }

	mpfr csch(const mpfr &val) { return ::mpfr::csch(val); }
	mpfr sech(const mpfr &val) { return ::mpfr::sech(val); }
	mpfr coth(const mpfr &val) { return ::mpfr::coth(val); }

	mpfr acsch(const mpfr &val) { return ::mpfr::acsch(val); }
	mpfr asech(const mpfr &val) { return ::mpfr::asech(val); }
	mpfr acoth(const mpfr &val) { return ::mpfr::acoth(val); }
}
