/* mpn_inv_divappr_q -- divide-and-conquer division, returning approximate
   quotient.  The quotient returned is either correct, or one too large.

   Derived from a file orignally contributed to the GMP project by Torbjorn 
   Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2006, 2007, 2009, 2010 Free Software Foundation, Inc.

Copyright 2010 William Hart.

This file is part of the MPIR Library.

The MPIR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The MPIR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the MPIR Library.  If not, see http://www.gnu.org/licenses/.  */

#include "mpir.h"
#include "gmp-impl.h"
#include "longlong.h"

mp_limb_t
mpn_inv_divappr_q (mp_ptr qp, mp_ptr np, mp_size_t nn,
		     mp_srcptr dp, mp_size_t dn, mp_srcptr dinv)
{
  mp_size_t qn;
  mp_limb_t qh, cy, qsave, dinv2;
  mp_ptr tp, temp;
  TMP_DECL;

  TMP_MARK;

  ASSERT (dn >= 6);
  ASSERT (nn > dn);
  ASSERT (dp[dn-1] & GMP_NUMB_HIGHBIT);

  qn = nn - dn;
  qp += qn;
  np += nn;
  dp += dn;

  if (qn >= dn)
    {
      qn++;			/* pretend we'll need an extra limb */
      /* Reduce qn mod dn without division, optimizing small operations.  */
      do
	qn -= dn;
      while (qn > dn);

      qp -= qn;			/* point at low limb of next quotient block */
      np -= qn;			/* point in the middle of partial remainder */

      tp = TMP_ALLOC_LIMBS (DC_DIVAPPR_Q_N_ITCH(dn));

      /* Perform the typically smaller block first.  */
      if (qn == 1)
	{
	  mp_limb_t q, n2, n1, n0, d1, d0, d11, d01;

	  /* Handle qh up front, for simplicity. */
	  qh = mpn_cmp (np - dn + 1, dp - dn, dn) >= 0;
	  if (qh)
	    ASSERT_NOCARRY (mpn_sub_n (np - dn + 1, np - dn + 1, dp - dn, dn));

	  /* A single iteration of schoolbook: One 3/2 division,
	     followed by the bignum update and adjustment. */

	  n2 = np[0];
	  n1 = np[-1];
	  n0 = np[-2];
	  d1 = dp[-1];
	  d0 = dp[-2];
     d01 = d0 + 1;
     d11 = d1 + (d01 < d0);

	  ASSERT (n2 < d1 || (n2 == d1 && n1 <= d0));

	  if (UNLIKELY (n2 == d1) && n1 == d0)
	    {
	      q = GMP_NUMB_MASK;
	      cy = mpn_submul_1 (np - dn, dp - dn, dn, q);
	      ASSERT (cy == n2);
	    }
	  else
	    {
	      mpir_invert_pi1(dinv2, d1, d0);
         udiv_qr_3by2 (q, n1, n0, n2, n1, n0, d1, d0, dinv2);

	      if (dn > 2)
		{
		  mp_limb_t cy, cy1;
		  cy = mpn_submul_1 (np - dn, dp - dn, dn - 2, q);

		  cy1 = n0 < cy;
		  n0 = (n0 - cy) & GMP_NUMB_MASK;
		  cy = n1 < cy1;
		  n1 = (n1 - cy1) & GMP_NUMB_MASK;
		  np[-2] = n0;

		  if (UNLIKELY (cy != 0))
		    {
		      n1 += d1 + mpn_add_n (np - dn, np - dn, dp - dn, dn - 1);
		      qh -= (q == 0);
		      q = (q - 1) & GMP_NUMB_MASK;
		    }
		}
	      else
		np[-2] = n0;

	      np[-1] = n1;
	    }
	  qp[0] = q;
	}
      else
	{
	  mpir_invert_pi1(dinv2, dp[-1], dp[-2]);
     if (qn == 2)
	    qh = mpn_divrem_2 (qp, 0L, np - 2, 4, dp - 2);
	  else if (BELOW_THRESHOLD (qn, DC_DIV_QR_THRESHOLD))
        qh = mpn_sb_div_qr (qp, np - qn, 2 * qn, dp - qn, qn, dinv2);
     else if (BELOW_THRESHOLD (qn, INV_DIV_QR_THRESHOLD))
        qh = mpn_dc_div_qr_n (qp, np - qn, dp - qn, qn, dinv2, tp);
     else
        {
   	    mpn_invert_trunc(tp, qn, dinv, dn, dp - dn);
           qh = mpn_inv_div_qr_n (qp, np - qn, dp - qn, qn, tp);
        }

	  
      if (qn != dn)
	    {
	      if (qn > dn - qn)
		mpn_mul (tp, qp, qn, dp - dn, dn - qn);
	      else
		mpn_mul (tp, dp - dn, dn - qn, qp, qn);

	      cy = mpn_sub_n (np - dn, np - dn, tp, dn);
	      if (qh != 0)
		cy += mpn_sub_n (np - dn + qn, np - dn + qn, dp - dn, dn - qn);

	      while (cy != 0)
		{
		  qh -= mpn_sub_1 (qp, qp, qn, 1);
		  cy -= mpn_add_n (np - dn, np - dn, dp - dn, dn);
		}
	    }
	}
      qn = nn - dn - qn + 1;
      while (qn > dn)
	{
	  qp -= dn;
	  np -= dn;
      ASSERT_NOCARRY(mpn_inv_div_qr_n (qp, np - dn, dp - dn, dn, dinv));
	  qn -= dn;
	}

      /* Since we pretended we'd need an extra quotient limb before, we now
	 have made sure the code above left just dn-1=qn quotient limbs to
	 develop.  Develop that plus a guard limb. */
      qn--;
      qp -= qn;
      np -= dn;
      qsave = qp[qn];
      cy = mpn_inv_divappr_q_n (qp, np - dn, dp - dn, dn, dinv);
      if (UNLIKELY(cy)) mpn_sub_1(qp, qp, dn, 1);
      MPN_COPY_INCR (qp, qp + 1, qn);
      qp[qn] = qsave;
    }
  else    /* (qn < dn) */
    {
      mp_ptr q2p;

      qp -= qn;			/* point at low limb of next quotient block */
      np -= qn;			/* point in the middle of partial remainder */

      q2p = TMP_ALLOC_LIMBS (qn + 1);
      
      mpir_invert_pi1(dinv2, dp[-1], dp[-2]);
      
       if (qn == 1)
        {
          qh = mpn_divrem_2(q2p, 0, np - 3, 4, dp - 2);
        }
      else if (BELOW_THRESHOLD (qn, DC_DIVAPPR_Q_THRESHOLD))
	{
	  qh = mpn_sb_divappr_q (q2p, np - qn - 2, 2 * (qn + 1),
				    dp - (qn + 1), qn + 1, dinv2);
	}
      else if (BELOW_THRESHOLD (qn, INV_DIVAPPR_Q_N_THRESHOLD))
	{
	  /* It is tempting to use qp for recursive scratch and put quotient in
	     tp, but the recursive scratch needs one limb too many.  */
	  tp = TMP_ALLOC_LIMBS (DC_DIVAPPR_Q_N_ITCH(qn + 1));
	  qh = mpn_dc_divappr_q (q2p, np - qn - 2, 2 * (qn + 1), dp - (qn + 1), qn + 1, dinv2);
       } 
      else 
	{
   	    tp = TMP_ALLOC_LIMBS (qn + 1);
	    mpn_invert_trunc(tp, qn + 1, dinv, dn, dp - dn);
           qh = mpn_inv_divappr_q_n (q2p, np - qn - 2, dp - (qn + 1), qn + 1, tp);
       }


      MPN_COPY (qp, q2p + 1, qn);
    }

  TMP_FREE;
  return qh;
}
