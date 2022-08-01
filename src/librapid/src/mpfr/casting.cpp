#include <librapid/math/mpfr.hpp>

namespace librapid {
	mpz toMpz(const mpz &other) { return other; }
	mpz toMpz(const mpf &other) { return mpz(other); }
	mpz toMpz(const mpq &other) { return mpz(other); }
	mpz toMpz(const mpfr &other) { return mpz(mpf_class(str(other))); }

	mpq toMpq(const mpz &other) { return {other}; }
	mpq toMpq(const mpf &other) { return mpq(other); }
	mpq toMpq(const mpq &other) { return other; }
	mpq toMpq(const mpfr &other) { return mpq(mpf_class(str(other))); }

	mpfr toMpfr(const mpz &other) { return {str(other)}; }
	mpfr toMpfr(const mpf &other) { return {str(other)}; }
	mpfr toMpfr(const mpq &other) { return {str(mpf_class(other))}; }
	mpfr toMpfr(const mpfr &other) { return other; }
} // namespace librapid