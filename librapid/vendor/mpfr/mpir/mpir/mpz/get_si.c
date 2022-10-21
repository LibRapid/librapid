/* mpz_get_si(integer) -- Return the least significant digit from INTEGER.

Copyright 1991, 1993, 1994, 1995, 2000, 2001, 2002, 2006 Free Software
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

mpir_si
mpz_get_si (mpz_srcptr z)
{
  mp_ptr zp = z->_mp_d;
  mp_size_t size = z->_mp_size;
  mp_limb_t zl = zp[0];

#if GMP_NAIL_BITS != 0
  if (GMP_UI_MAX > GMP_NUMB_MAX && ABS (size) >= 2)
    zl |= zp[1] << GMP_NUMB_BITS;
#endif

  if (size > 0)
    return (mpir_si) zl & GMP_SI_MAX;
  else if (size < 0)
    /* This expression is necessary to properly handle 0x80000000 */
    return ~(((mpir_si) zl - 1L) & GMP_SI_MAX);
  else
    return 0;
}
