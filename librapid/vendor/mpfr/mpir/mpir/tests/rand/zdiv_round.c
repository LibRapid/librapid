/* zdiv_round() -- divide integers, round to nearest */

/*
Copyright 1999 Free Software Foundation, Inc.

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
MA 02110-1301, USA.
*/

#include "mpir.h"

void
zdiv_round (mpz_t rop, mpz_t n, mpz_t d)
{
  mpf_t f_n, f_d;

  mpf_init (f_n);
  mpf_init (f_d);

  mpf_set_z (f_d, d);
  mpf_set_z (f_n, n);

  mpf_div (f_n, f_n, f_d);
  mpf_set_d (f_d, .5);
  if (mpf_sgn (f_n) < 0)
    mpf_neg (f_d, f_d);
  mpf_add (f_n, f_n, f_d);
  mpz_set_f (rop, f_n);

  mpf_clear (f_n);
  mpf_clear (f_d);
  return;
}
