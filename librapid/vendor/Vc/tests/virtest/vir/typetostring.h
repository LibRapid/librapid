/*{{{
Copyright © 2014-2017 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VIR_TYPETOSTRING_H_
#define VIR_TYPETOSTRING_H_

#ifdef __has_include
#  if __has_include(<Vc/fwddecl.h>)
#    include <Vc/fwddecl.h>
#  endif
#elif defined COMPILE_FOR_UNIT_TESTS
#  include <Vc/fwddecl.h>
#endif
#include <array>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>  // for __cpp_lib_integer_sequence
#include <vector>
#include "typelist.h"

#ifdef __has_include
#  if __has_include(<cxxabi.h>)
#    include <cxxabi.h>
#    define VIR_HAVE_CXXABI_H 1
#  endif
#elif defined HAVE_CXX_ABI_H
#  include <cxxabi.h>
#  define VIR_HAVE_CXXABI_H 1
#endif // __has_include

#if defined __cpp_return_type_deduction && __cpp_return_type_deduction &&                \
    __cpp_lib_integer_sequence
#define VIR_HAVE_CONSTEXPR_TYPESTRINGS 1
#define VIR_AUTO_OR_STRING constexpr auto
#define VIR_CONSTEXPR_STRING_RET(N_) constexpr constexpr_string<N_>
#else
#define VIR_AUTO_OR_STRING inline std::string
#define VIR_CONSTEXPR_STRING_RET(N_) inline std::string
#endif

namespace vir
{
namespace detail
{
template <typename T> VIR_AUTO_OR_STRING typeToStringRecurse();

template <int N> class constexpr_string
{
#ifdef VIR_HAVE_CONSTEXPR_TYPESTRINGS
  const std::array<char, N + 1> chars;
  static constexpr std::make_index_sequence<N> index_seq{};

  template <int M, std::size_t... Ls, std::size_t... Rs>
  constexpr constexpr_string<N + M> concat(const constexpr_string<M> &rhs,
                                           std::index_sequence<Ls...>,
                                           std::index_sequence<Rs...>) const
  {
    return {chars[Ls]..., rhs[Rs]...};
  }

public:
  constexpr constexpr_string(const char c) : chars{{c, '\0'}} {}
  constexpr constexpr_string(const std::initializer_list<char> &c)
      : constexpr_string(&*c.begin(), index_seq)
  {
  }
  constexpr constexpr_string(const char str[N]) : constexpr_string(str, index_seq) {}
  template <std::size_t... Is>
  constexpr constexpr_string(const char str[N], std::index_sequence<Is...>)
      : chars{{str[Is]..., '\0'}}
  {
  }

  template <int M>
  constexpr constexpr_string<N + M> operator+(const constexpr_string<M> &rhs) const
  {
    return concat(rhs, std::make_index_sequence<N>(), std::make_index_sequence<M>());
  }

  constexpr char operator[](size_t i) const { return chars[i]; }
  operator std::string() const { return {&chars[0], N}; }
  const char *c_str() const { return chars.data(); }

  friend std::string operator+(const std::string &lhs, const constexpr_string &rhs)
  {
    return lhs + rhs.c_str();
  }
  friend std::string operator+(const constexpr_string &lhs, const std::string &rhs)
  {
    return lhs.c_str() + rhs;
  }

  friend std::ostream &operator<<(std::ostream &lhs, const constexpr_string &rhs)
  {
    return lhs.write(&rhs.chars[0], N);
  }
#else
  constexpr_string(std::string &&tmp) : s(std::move(tmp)) {}
public:
  std::string s;
  constexpr_string(const char c) : s(1, c) {}
  constexpr_string(const std::initializer_list<char> &c) : s(&*c.begin(), c.size()) {}
  constexpr_string(const char str[N]) : s(str, N) {}
  template <int M>
  constexpr constexpr_string<N + M> operator+(const constexpr_string<M> &rhs) const
  {
    return s + rhs.s;
  }
  constexpr char operator[](size_t i) const { return s[i]; }
  operator std::string() const { return s; }
  const char *c_str() const { return s.c_str(); }
  friend std::string operator+(const std::string &lhs, const constexpr_string &rhs)
  {
    return lhs + rhs.s;
  }
  friend std::string operator+(const constexpr_string &lhs, const std::string &rhs)
  {
    return lhs.s + rhs;
  }

  friend std::ostream &operator<<(std::ostream &lhs, const constexpr_string &rhs)
  {
    return lhs << rhs.s;
  }
#endif
};

template <std::size_t N> using CStr = const char[N];
template <std::size_t N> VIR_CONSTEXPR_STRING_RET(N - 1) cs(const CStr<N> &str)
{
  return str;
}
VIR_CONSTEXPR_STRING_RET(1) cs(const char c) { return constexpr_string<1>(c); }

template <class T, T N>
VIR_CONSTEXPR_STRING_RET(1)
number_to_string(std::integral_constant<T, N>,
                 typename std::enable_if<(N >= 0 && N <= 9)>::type * = nullptr)
{
  return cs('0' + N);
}

template <class T, T N>
VIR_AUTO_OR_STRING number_to_string(std::integral_constant<T, N>,
                                    typename std::enable_if<(N >= 10)>::type * = nullptr)
{
  return number_to_string(std::integral_constant<T, N / 10>()) + cs('0' + N % 10);
}

template <class T, T N>
VIR_AUTO_OR_STRING number_to_string(std::integral_constant<T, N>,
                                    typename std::enable_if<(N < 0)>::type * = nullptr)
{
  return cs('-') + number_to_string(std::integral_constant<T, -N>());
}

// std::array<T, N> {{{1
template <typename T, std::size_t N>
VIR_AUTO_OR_STRING typeToString_impl(std::array<T, N> *)
{
  return cs("array<") + typeToStringRecurse<T>() + cs(", ") +
         number_to_string(std::integral_constant<int, N>()) + cs('>');
}

// std::vector<T> {{{1
template <typename T> VIR_AUTO_OR_STRING typeToString_impl(std::vector<T> *)
{
  return cs("vector<") + typeToStringRecurse<T>() + cs('>');
}

// std::integral_constant<T, N> {{{1
template <typename T, T N>
VIR_AUTO_OR_STRING typeToString_impl(std::integral_constant<T, N> *)
{
  return cs("integral_constant<") + typeToStringRecurse<T>() + cs(", ") +
         number_to_string(std::integral_constant<T, N>()) + cs('>');
}

// template parameter pack to a comma separated string {{{1
VIR_AUTO_OR_STRING typelistToStringRecursive()
{
  return cs('}');
}
template <typename T0, typename... Ts>
VIR_AUTO_OR_STRING typelistToStringRecursive(T0 *, Ts *...)
{
  return cs(", ") + typeToStringRecurse<T0>() +
         typelistToStringRecursive(typename std::add_pointer<Ts>::type()...);
}

template <typename T0, typename... Ts>
VIR_AUTO_OR_STRING typeToString_impl(Typelist<T0, Ts...> *)
{
  return cs('{') + typeToStringRecurse<T0>() +
         typelistToStringRecursive(typename std::add_pointer<Ts>::type()...);
}

VIR_CONSTEXPR_STRING_RET(2) typeToString_impl(Typelist<> *) { return "{}"; }

// Vc::simd to string {{{1
#ifdef VC_FWDDECL_H_
template <int N> VIR_AUTO_OR_STRING typeToString_impl(Vc::simd_abi::fixed_size<N> *)
{
  return number_to_string(std::integral_constant<int, N>());
}
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(Vc::simd_abi::scalar *) { return "scalar"; }
VIR_CONSTEXPR_STRING_RET(3) typeToString_impl(Vc::simd_abi::__sse *) { return "SSE"; }
VIR_CONSTEXPR_STRING_RET(3) typeToString_impl(Vc::simd_abi::__avx *) { return "AVX"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(Vc::simd_abi::__avx512 *) { return "AVX512"; }
VIR_CONSTEXPR_STRING_RET(4) typeToString_impl(Vc::simd_abi::__neon *) { return "NEON"; }
template <class T, class A> VIR_AUTO_OR_STRING typeToString_impl(Vc::simd<T, A> *)
{
  return cs("simd<") + typeToStringRecurse<T>() + cs(", ") + typeToStringRecurse<A>() +
         cs('>');
}
template <class T, class A> VIR_AUTO_OR_STRING typeToString_impl(Vc::simd_mask<T, A> *)
{
  return cs("simd_mask<") + typeToStringRecurse<T>() + cs(", ") + typeToStringRecurse<A>() +
         cs('>');
}
#endif  // VC_FWDDECL_H_

// generic fallback (typeid::name) {{{1
template <typename T> inline std::string typeToString_impl(T *)
{
#ifdef VIR_HAVE_CXXABI_H
  char buf[1024];
  size_t size = 1024;
  abi::__cxa_demangle(typeid(T).name(), buf, &size, nullptr);
  return std::string{buf};
#else
  return typeid(T).name();
#endif
}

VIR_CONSTEXPR_STRING_RET(0) typeToString_impl(void *);// { return ""; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(long double *) { return "ldoubl"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(double *) { return "double"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(float *) { return " float"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(long long *) { return " llong"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(unsigned long long *) { return "ullong"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(long *) { return "  long"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(unsigned long *) { return " ulong"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(int *) { return "   int"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(unsigned int *) { return "  uint"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(short *) { return " short"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(unsigned short *) { return "ushort"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(char *) { return "  char"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(unsigned char *) { return " uchar"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(signed char *) { return " schar"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(bool *)     { return "  bool"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(wchar_t *)  { return " wchar"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(char16_t *) { return "char16"; }
VIR_CONSTEXPR_STRING_RET(6) typeToString_impl(char32_t *) { return "char32"; }

template <typename T> VIR_AUTO_OR_STRING typeToStringRecurse()
{
  using tag = T *;
  return typeToString_impl(tag());
}
//}}}1
}  // namespace detail

// typeToString specializations {{{1
template <typename T> inline std::string typeToString()
{
  using tag = T *;
  return detail::typeToString_impl(tag());
}

//}}}1
}  // namespace vir

// vim: foldmethod=marker
#endif  // VIR_TYPETOSTRING_H_
