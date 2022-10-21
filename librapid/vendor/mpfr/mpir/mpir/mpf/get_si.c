/* mpf_get_si -- mpf to long conversion

Copyright 2001, 2002, 2004 Free Software Foundation, Inc.

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
MA 02110-1301, USA.
*/

#include "mpir.h"
#include "gmp-impl.h"


/* Any fraction bits are truncated, meaning simply discarded.

   For values bigger than a long, the low bits are returned, like
   mpz_get_si, but this isn't documented.

   Notice this is equivalent to mpz_set_f + mpz_get_si.


   Implementation:

   fl is established in basically the same way as for mpf_get_ui, see that
   code for explanations of the conditions.

   However unlike mpf_get_ui we need an explicit return 0 for exp<=0.  When
   f is a negative fraction (ie. size<0 and exp<=0) we can't let fl==0 go
   through to the zany final "~ ((fl - 1) & LONG_MAX)", that would give
   -0x80000000 instead of the desired 0.  */

mpir_si
mpf_get_si (mpf_srcptr f)
{
  mp_exp_t exp;
  mp_size_t size, abs_size;
  mp_srcptr fp;
  mp_limb_t fl;

  exp = EXP (f);
  size = SIZ (f);
  fp = PTR (f);

  /* fraction alone truncates to zero
     this also covers zero, since we have exp==0 for zero */
  if (exp <= 0)
    return 0L;

  /* there are some limbs above the radix point */

  fl = 0;
  abs_size = ABS (size);
  if (abs_size >= exp)
    fl = fp[abs_size-exp];

#if BITS_PER_UI > GMP_NUMB_BITS
  if (exp > 1 && abs_size+1 >= exp)
    fl |= fp[abs_size - exp + 1] << GMP_NUMB_BITS;
#endif

  if (size > 0)
    return (mpir_si)(fl & GMP_SI_MAX);
  else
    /* this form necessary to correctly handle -0x80..00 */
    return (mpir_si)(~((fl - 1) & GMP_SI_MAX));
}
