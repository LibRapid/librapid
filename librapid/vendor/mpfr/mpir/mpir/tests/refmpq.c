/* Reference rational routines.

Copyright 2001 Free Software Foundation, Inc.

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
#include "tests.h"


void
refmpq_add (mpq_ptr w, mpq_srcptr x, mpq_srcptr y)
{
  mpz_mul    (mpq_numref(w), mpq_numref(x), mpq_denref(y));
  mpz_addmul (mpq_numref(w), mpq_denref(x), mpq_numref(y));
  mpz_mul    (mpq_denref(w), mpq_denref(x), mpq_denref(y));
  mpq_canonicalize (w);
}

void
refmpq_sub (mpq_ptr w, mpq_srcptr x, mpq_srcptr y)
{
  mpz_mul    (mpq_numref(w), mpq_numref(x), mpq_denref(y));
  mpz_submul (mpq_numref(w), mpq_denref(x), mpq_numref(y));
  mpz_mul    (mpq_denref(w), mpq_denref(x), mpq_denref(y));
  mpq_canonicalize (w);
}
