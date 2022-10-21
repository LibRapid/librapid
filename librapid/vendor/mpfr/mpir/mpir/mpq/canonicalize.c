/* mpq_canonicalize(op) -- Remove common factors of the denominator and
   numerator in OP.

Copyright 1991, 1994, 1995, 1996, 2000, 2001, 2005 Free Software Foundation,
Inc.

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

void
mpq_canonicalize (mpq_ptr op)
{
  mpz_t gcd;
  TMP_DECL;

  if (op->_mp_den._mp_size == 0)
    DIVIDE_BY_ZERO;

  TMP_MARK;

  /* ??? Dunno if the 1+ is needed.  */
  MPZ_TMP_INIT (gcd, 1 + MAX (ABS (op->_mp_num._mp_size),
			      ABS (op->_mp_den._mp_size)));

  mpz_gcd (gcd, &(op->_mp_num), &(op->_mp_den));
  if (! MPZ_EQUAL_1_P (gcd))
    {
      mpz_divexact_gcd (&(op->_mp_num), &(op->_mp_num), gcd);
      mpz_divexact_gcd (&(op->_mp_den), &(op->_mp_den), gcd);
    }

  if (op->_mp_den._mp_size < 0)
    {
      op->_mp_num._mp_size = -op->_mp_num._mp_size;
      op->_mp_den._mp_size = -op->_mp_den._mp_size;
    }
  TMP_FREE;
}
