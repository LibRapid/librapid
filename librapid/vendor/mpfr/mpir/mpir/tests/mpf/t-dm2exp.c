/* Test mpf_div, mpf_div_2exp, mpf_mul_2exp.

Copyright 1996, 2000, 2001 Free Software Foundation, Inc.

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

#ifndef SIZE
#define SIZE 16
#endif

int
main (int argc, char **argv)
{
  int reps = 100000;
  int i;
  mpf_t u, v, w1, w2, w3;
  mp_size_t bprec = 100;
  mpf_t rerr, limit_rerr;
  mp_size_t un;
  mp_exp_t ue;
  gmp_randstate_t rands;
  
  tests_start ();
  gmp_randinit_default(rands);
  
  if (argc > 1)
    {
      reps = strtol (argv[1], 0, 0);
      if (argc > 2)
	bprec = strtol (argv[2], 0, 0);
    }

  mpf_set_default_prec (bprec);

  mpf_init (rerr);
  mpf_init (limit_rerr);

  mpf_init (u);
  mpf_init (v);
  mpf_init (w1);
  mpf_init (w2);
  mpf_init (w3);

  for (i = 0; i < reps; i++)
    {
      unsigned long int res_prec;
      unsigned long int pow2;

      res_prec = urandom (rands) % (bprec + 100);
      mpf_set_prec (w1, res_prec);
      mpf_set_prec (w2, res_prec);
      mpf_set_prec (w3, res_prec);

      mpf_set_ui (limit_rerr, 1);
      mpf_div_2exp (limit_rerr, limit_rerr, res_prec);

      pow2 = urandom (rands) % 0x10000;
      mpf_set_ui (v, 1);
      mpf_mul_2exp (v, v, pow2);

      un = urandom (rands) % (2 * SIZE) - SIZE;
      ue = urandom (rands) % SIZE;
      mpf_rrandomb (u, rands, un, ue);

      mpf_div_2exp (w1, u, pow2);
      mpf_div (w2, u, v);
      mpf_reldiff (rerr, w1, w2);
      if (mpf_cmp (rerr, limit_rerr) > 0)
	{
	  printf ("ERROR in mpf_div or mpf_div_2exp after %d tests\n", i);
	  printf ("   u = "); mpf_dump (u);
	  printf ("   v = "); mpf_dump (v);
	  printf ("  w1 = "); mpf_dump (w1);
	  printf ("  w2 = "); mpf_dump (w2);
	  abort ();
	}
      mpf_mul_2exp (w3, w1, pow2);
      mpf_reldiff (rerr, u, w3);
      if (mpf_cmp (rerr, limit_rerr) > 0)
	{
	  printf ("ERROR in mpf_mul_2exp after %d tests\n", i);
	  printf ("   u = "); mpf_dump (u);
	  printf ("   v = "); mpf_dump (v);
	  printf ("  w1 = "); mpf_dump (w1);
	  printf ("  w3 = "); mpf_dump (w3);
	  abort ();
	}
    }

  mpf_clear (rerr);
  mpf_clear (limit_rerr);

  mpf_clear (u);
  mpf_clear (v);
  mpf_clear (w1);
  mpf_clear (w2);
  mpf_clear (w3);
  gmp_randclear(rands);
  tests_end ();
  exit (0);
}
