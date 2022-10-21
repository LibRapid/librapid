/* mpq_set_num(dest,num) -- Set the numerator of DEST from NUM.

Copyright 1991, 1994, 1995, 2001 Free Software Foundation, Inc.

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
mpq_set_num (mpq_ptr dest, mpz_srcptr num)
{
  mp_size_t size = num->_mp_size;
  mp_size_t abs_size = ABS (size);

  if (dest->_mp_num._mp_alloc < abs_size)
    _mpz_realloc (&(dest->_mp_num), abs_size);

  MPN_COPY (dest->_mp_num._mp_d, num->_mp_d, abs_size);
  dest->_mp_num._mp_size = size;
}
