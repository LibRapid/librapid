/* mpz_tdiv_q -- divide two integers and produce a quotient.

Copyright 1991, 1993, 1994, 1996, 2000, 2001, 2005 Free Software Foundation,
Inc.

Copyright 2009, William Hart

This file is part of the MPIR Library.

The MPIR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The MPIR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the MPIR Library; see the file COPYING.LIB.  If not, write to
the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
MA 02110-1301, USA. */

#include "mpir.h"
#include "gmp-impl.h"
#include "longlong.h"

void
mpz_tdiv_q (mpz_ptr quot, mpz_srcptr num, mpz_srcptr den)
{
  mp_size_t ql;
  mp_size_t ns, ds, nl, dl;
  mp_ptr np, dp, qp;
  TMP_DECL;

  ns = SIZ (num);
  ds = SIZ (den);
  nl = ABS (ns);
  dl = ABS (ds);
  ql = nl - dl + 1;

  if (dl == 0)
    DIVIDE_BY_ZERO;

  if (ql <= 0)
    {
      SIZ (quot) = 0;
      return;
    }

  MPZ_REALLOC (quot, ql);

  TMP_MARK;
  qp = PTR (quot);
  np = PTR (num);
  dp = PTR (den);

  /* FIXME: We should think about how to handle the temporary allocation.
     Perhaps mpn_tdiv_qr should handle it, since it anyway often needs to
     allocate temp space.  */

  /* Copy denominator to temporary space if it overlaps with the quotient.  */
  if (dp == qp)
    {
      mp_ptr tp;
      tp = (mp_ptr) TMP_ALLOC (dl * BYTES_PER_MP_LIMB);
      MPN_COPY (tp, dp, dl);
      dp = tp;
    }
  /* Copy numerator to temporary space if it overlaps with the quotient.  */
  if (np == qp)
    {
      mp_ptr tp;
      tp = (mp_ptr) TMP_ALLOC (nl * BYTES_PER_MP_LIMB);
      MPN_COPY (tp, np, nl);
      np = tp;
    }

  mpn_tdiv_q (qp, np, nl, dp, dl);

  ql -=  qp[ql - 1] == 0;

  SIZ (quot) = (ns ^ ds) >= 0 ? ql : -ql;
  TMP_FREE;
}
