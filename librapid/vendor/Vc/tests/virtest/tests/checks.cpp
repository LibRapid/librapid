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

#include <vir/testassert.h>
#include <vir/test.h>
#include <vir/metahelpers.h>

#include <cmath>
#include <limits>

TEST(sanity_checks)  //{{{1
{
  VERIFY(true);
  COMPARE(1, 1);

  struct NotComparable {
    unsigned int x = 0xdeadbeef;
  };
  COMPARE(NotComparable(), NotComparable());  // compares with memcmp

  ADD_PASS() << "extra PASS\nwith newline";
}

TEST(skip)  //{{{1
{
  SKIP() << "\"output string\nwith newline\"";
}

TEST(xfail)  //{{{1
{
  vir::test::expect_failure();
  COMPARE(1, 2) << "\"output string\nwith newline\"";
}

TEST_CATCH(test_catch, int)  //{{{1
{
  throw int();
}

TEST(test_assert)  //{{{1
{
  expect_assert_failure([&]() { assert(1 == 2); });
}

TEST(type_to_string)  //{{{1
{
  COMPARE((vir::typeToString<std::array<int, 3>>()), "array<   int, 3>");
  COMPARE((vir::typeToString<std::vector<float>>()), "vector< float>");
  COMPARE((vir::typeToString<std::integral_constant<int, 3>>()), "integral_constant<   int, 3>");
  COMPARE((vir::typeToString<std::integral_constant<int, -15>>()), "integral_constant<   int, -15>");
  COMPARE((vir::typeToString<std::integral_constant<unsigned long long, 281474976710655>>()), "integral_constant<ullong, 281474976710655>");
  COMPARE((vir::typeToString<vir::Typelist<>>()), "{}");
  COMPARE((vir::typeToString<vir::Typelist<int, float>>()), "{   int,  float}");
}

struct Test1 {
  template <class T> auto operator()(T x) -> decltype(std::sin(x)) {}
};
struct Test2 {
  template <class T>
  auto operator()(T x) -> decltype(vir::detail::ulpDiffToReference(x, x))
  {
  }
};

TEST(sfinae_checks)  //{{{1
{
  VERIFY( sfinae_is_callable<float>(Test1()));  // std::sin(float) is callable
  VERIFY(!sfinae_is_callable<Test1>(Test1()));  // std::sin(Test1) is ill-formed

  VERIFY( sfinae_is_callable<float>(Test2()));  // ulpDiffToReference(float) is callable
  VERIFY(!sfinae_is_callable<Test2>(Test2()));  // ulpDiffToReference(  int) is ill-formed
}

TEST(Typelist)  //{{{1
{
  using namespace vir;
  using _1 = std::integral_constant<int, 1>;
  using _2 = std::integral_constant<int, 2>;
  using _3 = std::integral_constant<int, 3>;
  using _4 = std::integral_constant<int, 4>;
  COMPARE(typeid(outer_product<Typelist<_1, _2>, Typelist<_3, _4>>),
          typeid(Typelist<Typelist<_1, _3>, Typelist<_1, _4>, Typelist<_2, _3>,
                          Typelist<_2, _4>>));
}

// test_types[_check]  {{{1
std::vector<std::string> seen_types;
TEST_TYPES(T, test_types, int, float, char)
{
  seen_types.push_back(vir::typeToString<T>());
}
TEST(test_types_check)
{
  COMPARE(seen_types.size(), 3u);
  COMPARE(seen_types[0], "   int");
  COMPARE(seen_types[1], " float");
  COMPARE(seen_types[2], "  char");
}

TEST_TYPES(T, testUlpDiff, double, float, long double)  //{{{1
{
  using std::numeric_limits;
  using vir::detail::ulpDiffToReference;
  using vir::detail::ulpDiffToReferenceSigned;

  const auto zero = T();
  const auto min = numeric_limits<T>::min();
  const auto epsilon = numeric_limits<T>::epsilon();

  COMPARE(ulpDiffToReference(zero, zero), 0);

  // if 0 is expected, the closest normalized value representation is considered 1 ULP away, even if
  // 2 ULP away from 0 is almost the same. But hitting exactly 0 is quite hard sometimes.
  COMPARE(ulpDiffToReferenceSigned(min, zero), 1);
  COMPARE(ulpDiffToReferenceSigned(-min, zero), -1);
  COMPARE(ulpDiffToReferenceSigned(std::nextafter(min, T(1)), zero), 2);
  COMPARE(ulpDiffToReferenceSigned(-std::nextafter(min, T(1)), zero), -2);

  // if the expectation is nonzero, it is not clear that 0 should be considered 1 ULP away. This may
  // be reconsidered given an good argument.
  COMPARE(ulpDiffToReferenceSigned(zero, min), -1);

  // epsilon is the ULP relative to [1, 2)
  COMPARE(ulpDiffToReferenceSigned(T(1), T(1) + epsilon), -1);
  COMPARE(ulpDiffToReferenceSigned(T(1) + epsilon, T(1)), 1);

  // expected * epsilon(1) = 2 - 1
  // => expected = 1 / epsilon
  COMPARE(ulpDiffToReferenceSigned(T(2), T(1)), T(1) / epsilon);

  // expected * epsilon(2) = expected * epsilon(1) * 2 = 2 - 1
  // => expected = .5 / epsilon
  COMPARE(ulpDiffToReferenceSigned(T(1), T(2)), T(-.5) / epsilon);

  // change of sign - considered far far away
  COMPARE(ulpDiffToReferenceSigned(-min, min), -T(2) / epsilon);
  COMPARE(ulpDiffToReferenceSigned(min, -min), +T(2) / epsilon);
}

//}}}1
// vim: sw=2 et sts=2 foldmethod=marker
