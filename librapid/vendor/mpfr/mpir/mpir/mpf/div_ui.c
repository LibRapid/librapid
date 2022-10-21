/* mpf_div_ui -- Divide a float with an unsigned integer.

Copyright 1993, 1994, 1996, 2000, 2001, 2002, 2004, 2005 Free Software
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
#include "longlong.h"

void
mpf_div_ui (mpf_ptr r, mpf_srcptr u, mpir_ui v)
{
  mp_srcptr up;
  mp_ptr rp, tp, rtp;
  mp_size_t usize;
  mp_size_t rsize, tsize;
  mp_size_t sign_quotient;
  mp_size_t prec;
  mp_limb_t q_limb;
  mp_exp_t rexp;
  TMP_DECL;

#if BITS_PER_UI > GMP_NUMB_BITS  /* avoid warnings about shift amount */
  if (v > GMP_NUMB_MAX)
    {
      mpf_t vf;
      mp_limb_t vl[2];
      SIZ(vf) = 2;
      EXP(vf) = 2;
      PTR(vf) = vl;
      vl[0] = v & GMP_NUMB_MASK;
      vl[1] = v >> GMP_NUMB_BITS;
      mpf_div (r, u, vf);
      return;
    }
#endif

  usize = u->_mp_size;
  sign_quotient = usize;
  usize = ABS (usize);
  prec = r->_mp_prec;

  if (v == 0)
    DIVIDE_BY_ZERO;

  if (usize == 0)
    {
      r->_mp_size = 0;
      r->_mp_exp = 0;
      return;
    }

  TMP_MARK;

  rp = r->_mp_d;
  up = u->_mp_d;

  tsize = 1 + prec;
  tp = (mp_ptr) TMP_ALLOC ((tsize + 1) * BYTES_PER_MP_LIMB);

  if (usize > tsize)
    {
      up += usize - tsize;
      usize = tsize;
      rtp = tp;
    }
  else
    {
      MPN_ZERO (tp, tsize - usize);
      rtp = tp + (tsize - usize);
    }

  /* Move the dividend to the remainder.  */
  MPN_COPY (rtp, up, usize);

  mpn_divmod_1 (rp, tp, tsize, (mp_limb_t) v);
  q_limb = rp[tsize - 1];

  rsize = tsize - (q_limb == 0);
  rexp = u->_mp_exp - (q_limb == 0);
  r->_mp_size = sign_quotient >= 0 ? rsize : -rsize;
  r->_mp_exp = rexp;
  TMP_FREE;
}
