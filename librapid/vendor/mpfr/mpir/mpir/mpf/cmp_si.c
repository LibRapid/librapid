/* mpf_cmp_si -- Compare a float with a signed integer.

Copyright 1993, 1994, 1995, 1999, 2000, 2001, 2002, 2004 Free Software
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
mpf_cmp_si (mpf_srcptr u, mpir_si vval)
{
  mp_srcptr up;
  mp_size_t usize;
  mp_exp_t uexp;
  mp_limb_t ulimb;
  int usign;

  uexp = u->_mp_exp;
  usize = u->_mp_size;

  /* 1. Are the signs different?  */
  if ((usize < 0) == (vval < 0)) /* don't use xor, type size may differ */
    {
      /* U and V are both non-negative or both negative.  */
      if (usize == 0)
	/* vval >= 0 */
	return -(vval != 0);
      if (vval == 0)
	/* usize >= 0 */
	return usize != 0;
      /* Fall out.  */
    }
  else
    {
      /* Either U or V is negative, but not both.  */
      return usize >= 0 ? 1 : -1;
    }

  /* U and V have the same sign and are both non-zero.  */

  usign = usize >= 0 ? 1 : -1;
  usize = ABS (usize);
  vval = ABS (vval);

  /* 2. Are the exponents different (V's exponent == 1)?  */
#if GMP_NAIL_BITS != 0
  if (uexp > 1 + ((mpir_ui) vval > GMP_NUMB_MAX))
    return usign;
  if (uexp < 1 + ((mpir_ui) vval > GMP_NUMB_MAX))
    return -usign;
#else
  if (uexp > 1)
    return usign;
  if (uexp < 1)
    return -usign;
#endif

  up = u->_mp_d;

  ulimb = up[usize - 1];
#if GMP_NAIL_BITS != 0
  if (usize >= 2 && uexp == 2)
    {
      if ((ulimb >> GMP_NAIL_BITS) != 0)
	return usign;
      ulimb = (ulimb << GMP_NUMB_BITS) | up[usize - 2];
      usize--;
    }
#endif
  usize--;

  /* 3. Compare the most significant mantissa limb with V.  */
  if (ulimb > (mpir_ui) vval)
    return usign;
  else if (ulimb < (mpir_ui) vval)
    return -usign;

  /* Ignore zeroes at the low end of U.  */
  while (*up == 0)
    {
      up++;
      usize--;
    }

  /* 4. Now, if the number of limbs are different, we have a difference
     since we have made sure the trailing limbs are not zero.  */
  if (usize > 0)
    return usign;

  /* Wow, we got zero even if we tried hard to avoid it.  */
  return 0;
}
