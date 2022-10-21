/* mpz_mul_ui/si (product, multiplier, small_multiplicand) -- Set PRODUCT to
   MULTIPLICATOR times SMALL_MULTIPLICAND.

Copyright 1991, 1993, 1994, 1996, 2000, 2001, 2002, 2005 Free Software
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


#ifdef OPERATION_mul_si
#define FUNCTION               mpz_mul_si
#define MULTIPLICAND_UNSIGNED  mpir_si
#define MULTIPLICAND_ABS(x)    ((mpir_ui) ABS(x))
#define LT_ZERO(x)             (x < 0)
#endif

#ifdef OPERATION_mul_ui
#define FUNCTION               mpz_mul_ui
#define MULTIPLICAND_UNSIGNED  mpir_ui
#define MULTIPLICAND_ABS(x)    x
#define LT_ZERO(x)             0
#endif

#ifndef FUNCTION
Error, error, unrecognised OPERATION
#endif


void
FUNCTION (mpz_ptr prod, mpz_srcptr mult,
          MULTIPLICAND_UNSIGNED small_mult)
{
  mp_size_t size = SIZ(mult);
  mp_size_t sign_product = size;
  mp_limb_t sml;
  mp_limb_t cy;
  mp_ptr pp;

  if (size == 0 || small_mult == 0)
    {
      SIZ(prod) = 0;
      return;
    }

  size = ABS (size);

  sml = MULTIPLICAND_ABS (small_mult);

  if (small_mult <= GMP_NUMB_MAX)
    {
      MPZ_REALLOC (prod, size + 1);
      pp = PTR(prod);
      cy = mpn_mul_1 (pp, PTR(mult), size, sml & GMP_NUMB_MASK);
      pp[size] = cy;
      size += cy != 0;
    }
#if GMP_NAIL_BITS != 0
  else
    {
      /* Operand too large for the current nails size.  Use temporary for
	 intermediate products, to allow prod and mult being identical.  */
      mp_ptr tp;
      TMP_DECL;
      TMP_MARK;

      tp = TMP_ALLOC_LIMBS (size + 2);

      cy = mpn_mul_1 (tp, PTR(mult), size, sml & GMP_NUMB_MASK);
      tp[size] = cy;
      cy = mpn_addmul_1 (tp + 1, PTR(mult), size, sml >> GMP_NUMB_BITS);
      tp[size + 1] = cy;
      size += 2;
      MPN_NORMALIZE_NOT_ZERO (tp, size); /* too general, need to trim one or two limb */
      MPZ_REALLOC (prod, size);
      pp = PTR(prod);
      MPN_COPY (pp, tp, size);
      TMP_FREE;
    }
#endif

  SIZ(prod) = ((sign_product < 0) ^ LT_ZERO(small_mult)) ? -size : size;
}
