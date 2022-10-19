/* mpz_cmp_si(u,v) -- Compare an integer U with a single-word int V.
   Return positive, zero, or negative based on if U > V, U == V, or U < V.

Copyright 1991, 1993, 1994, 1995, 1996, 2000, 2001, 2002 Free Software
Foundation, Inc.

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

#include "mpir.h"
#include "gmp-impl.h"

int
_mpz_cmp_si (mpz_srcptr u, mpir_si v_digit)
{
  mp_size_t usize = u->_mp_size;
  mp_size_t vsize;
  mp_limb_t u_digit;

#if GMP_NAIL_BITS != 0
  /* FIXME.  This isn't very pretty.  */
  mpz_t tmp;
  mp_limb_t tt[2];
  PTR(tmp) = tt;
  ALLOC(tmp) = 2;
  mpz_set_si (tmp, v_digit);
  return mpz_cmp (u, tmp);
#endif

  vsize = 0;
  if (v_digit > 0)
    vsize = 1;
  else if (v_digit < 0)
    {
      vsize = -1;
      v_digit = -v_digit;
    }

  if (usize != vsize)
    return usize - vsize;

  if (usize == 0)
    return 0;

  u_digit = u->_mp_d[0];

  if (u_digit == (mp_limb_t) (mpir_ui) v_digit)
    return 0;

  if (u_digit > (mp_limb_t) (mpir_ui) v_digit)
    return usize;
  else
    return -usize;
}
