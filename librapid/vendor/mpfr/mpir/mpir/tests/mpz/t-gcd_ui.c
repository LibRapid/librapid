/* Test mpz_gcd_ui.

Copyright 2003 Free Software Foundation, Inc.

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

#define printf gmp_printf

/* Check mpz_gcd_ui doesn't try to return a value out of range.
   This was wrong in gmp 4.1.2 with a long long limb.  */
static void
check_ui_range (void)
{
  mpir_ui got;
  mpz_t  x;
  int  i;

  mpz_init_set_ui (x, GMP_UI_MAX);

  for (i = 0; i < 20; i++)
    {
      mpz_mul_2exp (x, x, 1L);
      got = mpz_gcd_ui (NULL, x, 0L);
      if (got != 0)
        {
          printf ("mpz_gcd_ui (GMP_UI_MAX*2^%d, 0)\n", i);
          printf ("   return %#Mx\n", got);
          printf ("   should be 0\n");
          abort ();
        }
    }

  mpz_clear (x);
}

int
main (void)
{
  tests_start ();

  check_ui_range ();

  tests_end ();
  exit (0);
}
