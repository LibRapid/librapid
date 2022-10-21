/* mpz_cdiv_qr -- Division rounding the quotient towards +infinity.  The
   remainder gets the opposite sign as the denominator.

Copyright 1994, 1995, 1996, 2000, 2001, 2005 Free Software Foundation, Inc.

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

void
mpz_cdiv_qr (mpz_ptr quot, mpz_ptr rem, mpz_srcptr dividend, mpz_srcptr divisor)
{
  mp_size_t divisor_size = divisor->_mp_size;
  mp_size_t xsize;
  mpz_t temp_divisor;		/* N.B.: lives until function returns! */
  TMP_DECL;

  TMP_MARK;

  /* We need the original value of the divisor after the quotient and
     remainder have been preliminary calculated.  We have to copy it to
     temporary space if it's the same variable as either QUOT or REM.  */
  if (quot == divisor || rem == divisor)
    {
      MPZ_TMP_INIT (temp_divisor, ABS (divisor_size));
      mpz_set (temp_divisor, divisor);
      divisor = temp_divisor;
    }

  xsize = dividend->_mp_size ^ divisor_size;;
  mpz_tdiv_qr (quot, rem, dividend, divisor);

  if (xsize >= 0 && rem->_mp_size != 0)
    {
      mpz_add_ui (quot, quot, 1L);
      mpz_sub (rem, rem, divisor);
    }

  TMP_FREE;
}
