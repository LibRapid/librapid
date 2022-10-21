/* mpn/bdivmod.c: mpn_bdivmod for computing U/V mod 2^d.

Copyright 1991, 1993, 1994, 1995, 1996, 1999, 2000, 2001, 2002 Free Software
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

/* q_high = mpn_bdivmod (qp, up, usize, vp, vsize, d).

   Puts the low d/BITS_PER_MP_LIMB limbs of Q = U / V mod 2^d at qp, and
   returns the high d%BITS_PER_MP_LIMB bits of Q as the result.

   Also, U - Q * V mod 2^(usize*BITS_PER_MP_LIMB) is placed at up.  Since the
   low d/BITS_PER_MP_LIMB limbs of this difference are zero, the code allows
   the limb vectors at qp to overwrite the low limbs at up, provided qp <= up.

   Preconditions:
   1.  V is odd.
   2.  usize * BITS_PER_MP_LIMB >= d.
   3.  If Q and U overlap, qp <= up.

   Ken Weber (kweber@mat.ufrgs.br, kweber@mcs.kent.edu)

   Funding for this work has been partially provided by Conselho Nacional
   de Desenvolvimento Cienti'fico e Tecnolo'gico (CNPq) do Brazil, Grant
   301314194-2, and was done while I was a visiting reseacher in the Instituto
   de Matema'tica at Universidade Federal do Rio Grande do Sul (UFRGS).

   References:
       T. Jebelean, An algorithm for exact division, Journal of Symbolic
       Computation, v. 15, 1993, pp. 169-180.

       K. Weber, The accelerated integer GCD algorithm, ACM Transactions on
       Mathematical Software, v. 21 (March), 1995, pp. 111-122.  */

#include "mpir.h"
#include "gmp-impl.h"
#include "longlong.h"


mp_limb_t
mpn_bdivmod (mp_ptr qp, mp_ptr up, mp_size_t usize,
	     mp_srcptr vp, mp_size_t vsize, mpir_ui d)
{
  mp_limb_t v_inv;

  ASSERT (usize >= 1);
  ASSERT (vsize >= 1);
  ASSERT (usize * GMP_NUMB_BITS >= d);
  ASSERT (! MPN_OVERLAP_P (up, usize, vp, vsize));
  ASSERT (! MPN_OVERLAP_P (qp, d/GMP_NUMB_BITS, vp, vsize));
  ASSERT (MPN_SAME_OR_INCR2_P (qp, d/GMP_NUMB_BITS, up, usize));
  ASSERT_MPN (up, usize);
  ASSERT_MPN (vp, vsize);

  /* 1/V mod 2^GMP_NUMB_BITS. */
  modlimb_invert (v_inv, vp[0]);

  /* Fast code for two cases previously used by the accel part of mpn_gcd.
     (Could probably remove this now it's inlined there.) */
  if (usize == 2 && vsize == 2 &&
      (d == GMP_NUMB_BITS || d == 2*GMP_NUMB_BITS))
    {
      mp_limb_t hi, lo;
      mp_limb_t q = (up[0] * v_inv) & GMP_NUMB_MASK;
      umul_ppmm (hi, lo, q, vp[0] << GMP_NAIL_BITS);
      up[0] = 0;
      up[1] -= hi + q*vp[1];
      qp[0] = q;
      if (d == 2*GMP_NUMB_BITS)
        {
          q = (up[1] * v_inv) & GMP_NUMB_MASK;
          up[1] = 0;
          qp[1] = q;
        }
      return 0;
    }

  /* Main loop.  */
  while (d >= GMP_NUMB_BITS)
    {
      mp_limb_t q = (up[0] * v_inv) & GMP_NUMB_MASK;
      mp_limb_t b = mpn_submul_1 (up, vp, MIN (usize, vsize), q);
      if (usize > vsize)
	mpn_sub_1 (up + vsize, up + vsize, usize - vsize, b);
      d -= GMP_NUMB_BITS;
      up += 1, usize -= 1;
      *qp++ = q;
    }

  if (d)
    {
      mp_limb_t b;
      mp_limb_t q = (up[0] * v_inv) & (((mp_limb_t)1<<d) - 1);
      if (q <= 1)
	{
	  if (q == 0)
	    return 0;
	  else
	    b = mpn_sub_n (up, up, vp, MIN (usize, vsize));
	}
      else
	b = mpn_submul_1 (up, vp, MIN (usize, vsize), q);

      if (usize > vsize)
	mpn_sub_1 (up + vsize, up + vsize, usize - vsize, b);
      return q;
    }

  return 0;
}
