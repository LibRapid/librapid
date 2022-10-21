/*{{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VIR_TESTS_METAHELPERS_H_
#define VIR_TESTS_METAHELPERS_H_

namespace vir
{
namespace test
{
// operator_is_substitution_failure {{{1
template <class A, class B, class Op>
constexpr bool operator_is_substitution_failure_impl(float)
{
  return true;
}

template <class A, class B, class Op>
constexpr
    typename std::conditional<true, bool,
                              decltype(Op()(std::declval<A>(), std::declval<B>()))>::type
    operator_is_substitution_failure_impl(int)
{
  return false;
}

template <class... Ts> constexpr bool operator_is_substitution_failure()
{
  return operator_is_substitution_failure_impl<Ts...>(int());
}

// sfinae_is_callable{{{1
#ifdef Vc_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
#endif
template <class... Args, class F>
constexpr auto sfinae_is_callable_impl(int, F &&f) -> typename std::conditional<
    true, std::true_type, decltype(std::forward<F>(f)(std::declval<Args>()...))>::type;
template <class... Args, class F> constexpr std::false_type sfinae_is_callable_impl(float, const F &);
template <class... Args, class F> constexpr bool sfinae_is_callable(F &&)
{
  return decltype(sfinae_is_callable_impl<Args...>(int(), std::declval<F>()))::value;
}
template <class... Args, class F>
constexpr auto sfinae_is_callable_t(F &&f)
    -> decltype(sfinae_is_callable_impl<Args...>(int(), std::declval<F>()));

#ifdef Vc_CLANG
#pragma clang diagnostic pop
#endif

// traits {{{1
template <class A, class B> constexpr bool has_less_bits()
{
  return std::numeric_limits<A>::digits < std::numeric_limits<B>::digits;
}

//}}}1

}  // namespace test
}  // namespace vir
#endif  // VIR_TESTS_METAHELPERS_H_
// vim: foldmethod=marker
