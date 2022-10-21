/* mpz_tdiv_r(rem, dividend, divisor) -- Set REM to DIVIDEND mod DIVISOR.

Copyright 1991, 1993, 1994, 2000, 2001, 2005 Free Software Foundation, Inc.

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
#include "longlong.h"

void
mpz_tdiv_r (mpz_ptr rem, mpz_srcptr num, mpz_srcptr den)
{
  mp_size_t ql;
  mp_size_t ns, ds, nl, dl;
  mp_ptr np, dp, qp, rp;
  TMP_DECL;

  ns = SIZ (num);
  ds = SIZ (den);
  nl = ABS (ns);
  dl = ABS (ds);
  ql = nl - dl + 1;

  if (dl == 0)
    DIVIDE_BY_ZERO;

  MPZ_REALLOC (rem, dl);

  if (ql <= 0)
    {
      if (num != rem)
	{
	  mp_ptr np, rp;
	  np = PTR (num);
	  rp = PTR (rem);
	  MPN_COPY (rp, np, nl);
	  SIZ (rem) = SIZ (num);
	}
      return;
    }

  TMP_MARK;
  qp = (mp_ptr) TMP_ALLOC (ql * BYTES_PER_MP_LIMB);
  rp = PTR (rem);
  np = PTR (num);
  dp = PTR (den);

  /* FIXME: We should think about how to handle the temporary allocation.
     Perhaps mpn_tdiv_qr should handle it, since it anyway often needs to
     allocate temp space.  */

  /* Copy denominator to temporary space if it overlaps with the remainder.  */
  if (dp == rp)
    {
      mp_ptr tp;
      tp = (mp_ptr) TMP_ALLOC (dl * BYTES_PER_MP_LIMB);
      MPN_COPY (tp, dp, dl);
      dp = tp;
    }
  /* Copy numerator to temporary space if it overlaps with the remainder.  */
  if (np == rp)
    {
      mp_ptr tp;
      tp = (mp_ptr) TMP_ALLOC (nl * BYTES_PER_MP_LIMB);
      MPN_COPY (tp, np, nl);
      np = tp;
    }

  mpn_tdiv_qr (qp, rp, 0L, np, nl, dp, dl);

  MPN_NORMALIZE (rp, dl);

  SIZ (rem) = ns >= 0 ? dl : -dl;
  TMP_FREE;
}
