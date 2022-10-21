/* Test mp*_class unary expressions.

Copyright 2001, 2002, 2003 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MP Library; see the file COPYING.LIB.  If not, write to
the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
MA 02110-1301, USA. */

#include <iostream>

#include "mpir.h"
#include "mpirxx.h"
#include "gmp-impl.h"
#include "tests.h"

using namespace std;


void
check_mpz (void)
{
  // template <class T, class Op>
  // __gmp_expr<T, __gmp_unary_expr<__gmp_expr<T, T>, Op> >
  {
    mpz_class a(1);
    mpz_class b(+a); ASSERT_ALWAYS(b == 1);
  }
  {
    mpz_class a(2);
    mpz_class b;
    b = -a; ASSERT_ALWAYS(b == -2);
  }
  {
    mpz_class a(3);
    mpz_class b;
    b = ~a; ASSERT_ALWAYS(b == -4);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_unary_expr<__gmp_expr<T, U>, Op> >
  {
    mpz_class a(1);
    mpz_class b(-(-a)); ASSERT_ALWAYS(b == 1);
  }
  {
    mpz_class a(2);
    mpz_class b;
    b = -(-(-a)); ASSERT_ALWAYS(b == -2);
  }
}

void
check_mpq (void)
{
  // template <class T, class Op>
  // __gmp_expr<T, __gmp_unary_expr<__gmp_expr<T, T>, Op> >
  {
    mpq_class a(1);
    mpq_class b(+a); ASSERT_ALWAYS(b == 1);
  }
  {
    mpq_class a(2);
    mpq_class b;
    b = -a; ASSERT_ALWAYS(b == -2);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_unary_expr<__gmp_expr<T, U>, Op> >
  {
    mpq_class a(1);
    mpq_class b(-(-a)); ASSERT_ALWAYS(b == 1);
  }
  {
    mpq_class a(2);
    mpq_class b;
    b = -(-(-a)); ASSERT_ALWAYS(b == -2);
  }
}

void
check_mpf (void)
{
  // template <class T, class Op>
  // __gmp_expr<T, __gmp_unary_expr<__gmp_expr<T, T>, Op> >
  {
    mpf_class a(1);
    mpf_class b(+a); ASSERT_ALWAYS(b == 1);
  }
  {
    mpf_class a(2);
    mpf_class b;
    b = -a; ASSERT_ALWAYS(b == -2);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_unary_expr<__gmp_expr<T, U>, Op> >
  {
    mpf_class a(1);
    mpf_class b(-(-a)); ASSERT_ALWAYS(b == 1);
  }
  {
    mpf_class a(2);
    mpf_class b;
    b = -(-(-a)); ASSERT_ALWAYS(b == -2);
  }
}


int
main (void)
{
  tests_start();

  check_mpz();
  check_mpq();
  check_mpf();

  tests_end();
  return 0;
}
