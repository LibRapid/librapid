#include <librapid/utils/toString.hpp>
#include <librapid/math/mpir.hpp>

namespace librapid {
#if defined(LIBRAPID_USE_MULTIPREC)
	std::string str(const mpz &val, const StrOpt &options) {
		return val.get_str((int)options.base);
	}

	std::string str(const mpf &val, const StrOpt &options) {
		mp_exp_t exp;
		auto res = val.get_str(exp, (int)options.base);

		if (exp > 0) {
			if (exp >= res.length()) res += std::string(exp - res.length() + 1, '0');
			res.insert(exp, ".");
			return res;
		} else {
			std::string tmp(-exp + 1, '0');
			tmp += res;
			tmp.insert(1, ".");
			return tmp;
		}
	}

	std::string str(const mpq &val, const StrOpt &options) {
		return val.get_str((int)options.base);
	}
#endif
} // namespace librapid