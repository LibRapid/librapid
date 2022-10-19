/* mpn_dc_bdiv_q -- divide-and-conquer Hensel division with precomputed
   inverse, returning quotient.

   Contributed to the GNU project by Niels M�ller and Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2006, 2007, 2009, 2010 Free Software Foundation, Inc.

Copyright 2010 William Hart (minor modifications)

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MP Library.  If not, see http://www.gnu.org/licenses/.  */

#include "mpir.h"
#include "gmp-impl.h"

/* Computes Q = N / D mod B^nn, destroys N.

   N = {np,nn}
   D = {dp,dn}
*/

void
mpn_dc_bdiv_q (mp_ptr qp,
		  mp_ptr np, mp_size_t nn,
		  mp_srcptr dp, mp_size_t dn,
		  mp_limb_t dinv)
{
  mp_size_t qn;
  mp_limb_t cy, wp[2];
  mp_ptr tp;
  TMP_DECL;

  TMP_MARK;

  ASSERT (dn >= 6);
  ASSERT (nn - dn >= 0);
  ASSERT (dp[0] & 1);

  tp = TMP_ALLOC_LIMBS (MAX(dn, DC_BDIV_Q_N_ITCH(dn)));

  qn = nn;

  if (qn > dn)
    {
      /* Reduce qn mod dn in a super-efficient manner.  */
      do
	qn -= dn;
      while (qn > dn);

      /* Perform the typically smaller block first.  */
      if (BELOW_THRESHOLD (qn, DC_BDIV_QR_THRESHOLD))
	cy = mpn_sb_bdiv_qr (qp, np, 2 * qn, dp, qn, dinv);
      else
	cy = mpn_dc_bdiv_qr_n (qp, np, dp, qn, dinv, tp);

      if (qn != dn)
	{
	  if (qn > dn - qn)
	    mpn_mul (tp, qp, qn, dp + qn, dn - qn);
	  else
	    mpn_mul (tp, dp + qn, dn - qn, qp, qn);
	  mpn_incr_u (tp + qn, cy);

	  mpn_sub (np + qn, np + qn, nn - qn, tp, dn);
	  cy = 0;
	}

      np += qn;
      qp += qn;

      qn = nn - qn;
      while (qn > dn)
	{
	  mpn_sub_1 (np + dn, np + dn, qn, cy);
	  cy = mpn_dc_bdiv_qr_n (qp, np, dp, dn, dinv, tp);
	  qp += dn;
	  np += dn;
	  qn -= dn;
	}
      mpn_dc_bdiv_q_n (qp, wp, np, dp, dn, dinv, tp);
    }
  else
    {
      if (BELOW_THRESHOLD (qn, DC_BDIV_Q_THRESHOLD))
	mpn_sb_bdiv_q (qp, wp, np, qn, dp, qn, dinv);
      else
	mpn_dc_bdiv_q_n (qp, wp, np, dp, qn, dinv, tp);
    }

  TMP_FREE;
}
