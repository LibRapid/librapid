/* Old function entrypoints retained for binary compatibility.

Copyright 2000, 2001 Free Software Foundation, Inc.

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

#include <stdio.h>
#include "mpir.h"
#include "gmp-impl.h"


/* mpn_divmod_1 was a function in gmp 3.0.1 and earlier, but marked obsolete
   in both gmp 2 and 3.  As of gmp 3.1 it's a macro calling mpn_divrem_1. */
mp_limb_t
__MPN (divmod_1) (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t divisor)
{
  return mpn_divmod_1 (dst, src, size, divisor);
}


/* mpz_legendre was a separate function in gmp 3.1.1 and earlier, but as of
   4.0 it's a #define alias for mpz_jacobi.  */
int
__gmpz_legendre (mpz_srcptr a, mpz_srcptr b)
{
  return mpz_jacobi (a, b);
}
