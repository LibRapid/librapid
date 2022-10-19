/* Test mpf_mul, mpf_div, mpf_ui_div, and mpf_div_ui.

Copyright 1996, 2000, 2001, 2003 Free Software Foundation, Inc.

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
  mp_size_t size;
  mp_exp_t exp;
  int reps = 10000;
  int i;
  mpf_t u, v, w, x;
  mp_size_t bprec = SIZE * BITS_PER_MP_LIMB;
  mpf_t rerr, limit_rerr;
  unsigned long ulimb, vlimb;
  int single_flag;
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
  mpf_init (w);
  mpf_init (x);

  for (i = 0; i < reps; i++)
    {
      mp_size_t res_prec;

      res_prec = urandom (rands) % bprec + 1;
      mpf_set_prec (w, res_prec);
      mpf_set_prec (x, res_prec);

      mpf_set_ui (limit_rerr, 1);
      mpf_div_2exp (limit_rerr, limit_rerr, res_prec - 1);

      single_flag = 0;

      if ((urandom (rands) & 1) != 0)
	{
	  size = urandom (rands) % (2 * SIZE) - SIZE;
	  exp = urandom (rands) % SIZE;
	  mpf_rrandomb (u, rands, size, exp);
	}
      else
	{
	  ulimb = urandom (rands);
	  mpf_set_ui (u, ulimb);
	  single_flag = 1;
	}

      if ((urandom (rands) & 1) != 0)
	{
	  size = urandom (rands) % (2 * SIZE) - SIZE;
	  exp = urandom (rands) % SIZE;
	  mpf_rrandomb (v, rands, size, exp);
	}
      else
	{
	  vlimb = urandom (rands);
	  mpf_set_ui (v, vlimb);
	  single_flag = 2;
	}

      if (mpf_sgn (v) == 0)
	continue;

      mpf_div (w, u, v);
      mpf_mul (x, w, v);
      mpf_reldiff (rerr, u, x);
      if (mpf_cmp (rerr, limit_rerr) > 0)
	{
	  printf ("ERROR in mpf_mul or mpf_div after %d tests\n", i);
	  printf ("   u = "); mpf_dump (u);
	  printf ("   v = "); mpf_dump (v);
	  printf ("   x = "); mpf_dump (x);
	  printf ("   w = "); mpf_dump (w);
	  abort ();
	}

      if (single_flag == 2)
	{
	  mpf_div_ui (x, u, vlimb);
	  mpf_reldiff (rerr, w, x);
	  if (mpf_cmp (rerr, limit_rerr) > 0)
	    {
	      printf ("ERROR in mpf_div or mpf_div_ui after %d tests\n", i);
	      printf ("   u = "); mpf_dump (u);
	      printf ("   v = "); mpf_dump (v);
	      printf ("   x = "); mpf_dump (x);
	      printf ("   w = "); mpf_dump (w);
	      abort ();
	    }
	}

      if (single_flag == 1)
	{
	  mpf_ui_div (x, ulimb, v);
	  mpf_reldiff (rerr, w, x);
	  if (mpf_cmp (rerr, limit_rerr) > 0)
	    {
	      printf ("ERROR in mpf_div or mpf_ui_div after %d tests\n", i);
	      printf ("   u = "); mpf_dump (u);
	      printf ("   v = "); mpf_dump (v);
	      printf ("   x = "); mpf_dump (x);
	      printf ("   w = "); mpf_dump (w);
	      abort ();
	    }
	}
    }

  mpf_clear (rerr);
  mpf_clear (limit_rerr);

  mpf_clear (u);
  mpf_clear (v);
  mpf_clear (w);
  mpf_clear (x);
  gmp_randclear(rands); 
  tests_end ();
  exit (0);
}
