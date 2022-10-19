/*{{{
Copyright © 2011-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VIR_DETAIL_ULP_H_
#define VIR_DETAIL_ULP_H_

#include <cmath>
#include <limits>

namespace vir
{
namespace detail
{
template <class T> struct where_expr {
  bool c;
  T &v;
  void operator=(const T &x)
  {
    if (c) {
      v = x;
    }
  }
  void operator+=(const T &x)
  {
    if (c) {
      v += x;
    }
  }
};

template <class T> where_expr<T> where(bool c, T &v) { return {c, v}; }

template <class T, class R = typename T::value_type> R value_type_impl(int);
template <class T> T value_type_impl(float);
template <class T> using value_type_t = decltype(value_type_impl<T>(int()));

template <
    class T,
    class = typename std::enable_if<std::is_floating_point<value_type_t<T>>::value>::type>
inline T ulpDiffToReference(const T &val_, const T &ref_)
{
  T val = val_;
  T ref = ref_;

  T diff = T();

  using std::abs;
  using std::frexp;
  using std::ldexp;
  using std::isnan;
  using std::fpclassify;
  using limits = std::numeric_limits<value_type_t<T>>;

  where(ref == 0, val) = abs(val);
  where(ref == 0, diff) = 1;
  where(ref == 0, ref) = limits::min();

  where(val == 0, ref) = abs(ref);
  where(val == 0, diff) += 1;
  where(val == 0, val) = limits::min();

  decltype(fpclassify(std::declval<T>())) exp = {};
  frexp(ref, &exp);
  diff += ldexp(abs(ref - val), limits::digits - exp);
  where(val_ == ref_ || (isnan(val_) && isnan(ref_)), diff) = T();
  return diff;
}

template <typename T> inline T ulpDiffToReferenceSigned(const T &_val, const T &_ref)
{
  using std::copysign;
  return copysign(ulpDiffToReference(_val, _ref), _val - _ref);
}

}  // namespace detail
}  // namespace vir

#endif  // VIR_DETAIL_ULP_H_
// vim: sw=2 et sts=2 foldmethod=marker
