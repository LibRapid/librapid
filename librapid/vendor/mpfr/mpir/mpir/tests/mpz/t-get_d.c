/* Test mpz_get_d.

Copyright 2002 Free Software Foundation, Inc.

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
#include <stdlib.h>
#include "mpir.h"
#include "gmp-impl.h"
#include "tests.h"


void
check_onebit (void)
{
  int     i;
  mpz_t   z;
  double  got, want;
  /* FIXME: It'd be better to base this on the float format. */
  int     limit = 512;

  mpz_init (z);

  mpz_set_ui (z, 1L);
  want = 1.0;

  for (i = 0; i < limit; i++)
    {
      got = mpz_get_d (z);

      if (got != want)
        {
          printf    ("mpz_get_d wrong on 2**%d\n", i);
          mpz_trace ("   z    ", z);
          printf    ("   want  %.20g\n", want);
          printf    ("   got   %.20g\n", got);
          abort();
        }

      mpz_mul_2exp (z, z, 1L);
      want *= 2.0;
    }
  mpz_clear (z);
}


int
main (void)
{
  tests_start ();

  check_onebit ();

  tests_end ();
  exit (0);
}
