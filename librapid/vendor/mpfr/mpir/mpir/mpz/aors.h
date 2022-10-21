/* mpz_add, mpz_sub -- add or subtract integers.

Copyright 1991, 1993, 1994, 1996, 2000, 2001 Free Software Foundation, Inc.

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


#ifdef OPERATION_add
#define FUNCTION     mpz_add
#define VARIATION
#endif
#ifdef OPERATION_sub
#define FUNCTION     mpz_sub
#define VARIATION    -
#endif
#define ARGUMENTS    mpz_ptr w, mpz_srcptr u, mpz_srcptr v

#ifndef FUNCTION
Error, need OPERATION_add or OPERATION_sub
#endif


void
FUNCTION (ARGUMENTS)
{
  mp_srcptr up, vp;
  mp_ptr wp;
  mp_size_t usize, vsize, wsize;
  mp_size_t abs_usize;
  mp_size_t abs_vsize;

  usize = u->_mp_size;
  vsize = VARIATION v->_mp_size;
  abs_usize = ABS (usize);
  abs_vsize = ABS (vsize);

  if (abs_usize < abs_vsize)
    {
      /* Swap U and V. */
      MPZ_SRCPTR_SWAP (u, v);
      MP_SIZE_T_SWAP (usize, vsize);
      MP_SIZE_T_SWAP (abs_usize, abs_vsize);
    }

  /* True: ABS_USIZE >= ABS_VSIZE.  */

  /* If not space for w (and possible carry), increase space.  */
  wsize = abs_usize + 1;
  if (w->_mp_alloc < wsize)
    _mpz_realloc (w, wsize);

  /* These must be after realloc (u or v may be the same as w).  */
  up = u->_mp_d;
  vp = v->_mp_d;
  wp = w->_mp_d;

  if ((usize ^ vsize) < 0)
    {
      /* U and V have different sign.  Need to compare them to determine
	 which operand to subtract from which.  */

      /* This test is right since ABS_USIZE >= ABS_VSIZE.  */
      if (abs_usize != abs_vsize)
	{
	  mpn_sub (wp, up, abs_usize, vp, abs_vsize);
	  wsize = abs_usize;
	  MPN_NORMALIZE (wp, wsize);
	  if (usize < 0)
	    wsize = -wsize;
	}
      else if (mpn_cmp (up, vp, abs_usize) < 0)
	{
	  mpn_sub_n (wp, vp, up, abs_usize);
	  wsize = abs_usize;
	  MPN_NORMALIZE (wp, wsize);
	  if (usize >= 0)
	    wsize = -wsize;
	}
      else
	{
	  mpn_sub_n (wp, up, vp, abs_usize);
	  wsize = abs_usize;
	  MPN_NORMALIZE (wp, wsize);
	  if (usize < 0)
	    wsize = -wsize;
	}
    }
  else
    {
      /* U and V have same sign.  Add them.  */
      mp_limb_t cy_limb = mpn_add (wp, up, abs_usize, vp, abs_vsize);
      wp[abs_usize] = cy_limb;
      wsize = abs_usize + cy_limb;
      if (usize < 0)
	wsize = -wsize;
    }

  w->_mp_size = wsize;
}
