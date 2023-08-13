#ifndef LIBRAPID_CORE_LITERALS_HPP
#define LIBRAPID_CORE_LITERALS_HPP

namespace librapid::literals {
#if defined(LIBRAPID_USE_MULTIPREC)
    /// \brief Creates a multiprecision floating point number from a string literal
    /// \param str The string literal to convert
    /// \return The multiprecision floating point number
    ::librapid::mpfr operator""_f(const char *str, size_t);
#endif // LIBRAPID_USE_MULTIPREC
} // namespace librapid::literals

#endif // LIBRAPID_CORE_LITERALS_HPP
