/* mpq_set(dest,src) -- Set DEST to SRC.

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
mpq_set (mpq_ptr dest, mpq_srcptr src)
{
  mp_size_t num_size, den_size;
  mp_size_t abs_num_size;

  num_size = src->_mp_num._mp_size;
  abs_num_size = ABS (num_size);
  if (dest->_mp_num._mp_alloc < abs_num_size)
    _mpz_realloc (&(dest->_mp_num), abs_num_size);
  MPN_COPY (dest->_mp_num._mp_d, src->_mp_num._mp_d, abs_num_size);
  dest->_mp_num._mp_size = num_size;

  den_size = src->_mp_den._mp_size;
  if (dest->_mp_den._mp_alloc < den_size)
    _mpz_realloc (&(dest->_mp_den), den_size);
  MPN_COPY (dest->_mp_den._mp_d, src->_mp_den._mp_d, den_size);
  dest->_mp_den._mp_size = den_size;
}
