#include <librapid/librapid.hpp>

namespace librapid::literals {
#if defined(LIBRAPID_USE_MULTIPREC)
    ::librapid::mpfr operator""_f(const char *str, size_t) { return {str}; }
#endif // LIBRAPID_USE_MULTIPREC
} // namespace librapid::literals
