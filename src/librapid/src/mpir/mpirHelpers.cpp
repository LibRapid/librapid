#include "librapid/math/mpir.hpp"

namespace librapid {
	mpf epsilon(const mpf &val) {
		auto prec = val.get_prec();
		mpf ep;
		mpf two(2);
		mpf_pow_ui(ep.get_mpf_t(), two.get_mpf_t(), -prec + 1);
		return ep;
	}

	mpf fmod(const mpf &val, const mpf &mod) {
		auto div	  = val / mod;
		auto floordiv = floor(div + epsilon(val));
		return val - (mod * floordiv);
	}
} // namespace librapid