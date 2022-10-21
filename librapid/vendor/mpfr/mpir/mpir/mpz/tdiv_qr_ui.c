/* mpz_tdiv_qr_ui(quot,rem,dividend,short_divisor) --
   Set QUOT to DIVIDEND / SHORT_DIVISOR
   and REM to DIVIDEND mod SHORT_DIVISOR.

Copyright 1991, 1993, 1994, 1996, 1998, 2001, 2002, 2004 Free Software
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

mpir_ui
mpz_tdiv_qr_ui (mpz_ptr quot, mpz_ptr rem, mpz_srcptr dividend, mpir_ui divisor)
{
  mp_size_t ns, nn, qn;
  mp_ptr np, qp;
  mp_limb_t rl;

  if (divisor == 0)
    DIVIDE_BY_ZERO;

  ns = SIZ(dividend);
  if (ns == 0)
    {
      SIZ(quot) = 0;
      SIZ(rem) = 0;
      return 0;
    }

  nn = ABS(ns);
  MPZ_REALLOC (quot, nn);
  qp = PTR(quot);
  np = PTR(dividend);

#if BITS_PER_UI > GMP_NUMB_BITS  /* avoid warnings about shift amount */
  if (divisor > GMP_NUMB_MAX)
    {
      mp_limb_t dp[2];
      mp_ptr rp;
      mp_size_t rn;

      if (nn == 1)		/* tdiv_qr requirements; tested above for 0 */
	{
	  SIZ(quot) = 0;
	  rl = np[0];
	  SIZ(rem) = ns >= 0 ? 1 : -1;
	  PTR(rem)[0] = rl;
	  return rl;
	}

      MPZ_REALLOC (rem, 2);
      rp = PTR(rem);

      dp[0] = divisor & GMP_NUMB_MASK;
      dp[1] = divisor >> GMP_NUMB_BITS;
      mpn_tdiv_qr (qp, rp, (mp_size_t) 0, np, nn, dp, (mp_size_t) 2);
      rl = rp[0] + (rp[1] << GMP_NUMB_BITS);
      qn = nn - 2 + 1; qn -= qp[qn - 1] == 0; qn -= qn != 0 && qp[qn - 1] == 0;
      rn = 2 - (rp[1] == 0);  rn -= (rp[rn - 1] == 0);
      SIZ(rem) = ns >= 0 ? rn : -rn;
    }
  else
#endif
    {
      rl = mpn_divrem_1 (qp, 0, np, nn, (mp_limb_t) divisor);
      if (rl == 0)
	SIZ(rem) = 0;
      else
	{
	  /* Store the single-limb remainder.  We don't check if there's space
	     for just one limb, since no function ever makes zero space.  */
	  SIZ(rem) = ns >= 0 ? 1 : -1;
	  PTR(rem)[0] = rl;
	}
      qn = nn - (qp[nn - 1] == 0);
    }

  SIZ(quot) = ns >= 0 ? qn : -qn;
  return rl;
}
