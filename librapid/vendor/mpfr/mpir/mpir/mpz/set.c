/* mpz_set (dest_integer, src_integer) -- Assign DEST_INTEGER from SRC_INTEGER.

Copyright 1991, 1993, 1994, 1995, 2000 Free Software Foundation, Inc.

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


#define FUNCTION   mpz_set
#define ARGUMENTS  mpz_ptr w, mpz_srcptr u

void
FUNCTION (ARGUMENTS)
{
  mp_ptr wp, up;
  mp_size_t usize, size;

  usize = u->_mp_size;
  size = ABS (usize);

  if (w->_mp_alloc < size)
    _mpz_realloc (w, size);

  wp = w->_mp_d;
  up = u->_mp_d;

  MPN_COPY (wp, up, size);
  w->_mp_size = usize;
}
