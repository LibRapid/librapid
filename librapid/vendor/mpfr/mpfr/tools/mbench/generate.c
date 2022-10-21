/*
Copyright 2005-2022 Free Software Foundation, Inc.
Contributed by Patrick Pelissier, INRIA.

This file is part of the MPFR Library.

The MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the MPFR Library; see the file COPYING.LESSER.  If not, see
https://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdio.h>
#include <stdlib.h>

#ifndef __MPFR_H
# define NEED_MAIN
# include "gmp.h"
# include "mpfr.h"
#endif

#define fp_t mpfr_t
#define fp_set_fr(dest,src) mpfr_set (dest, src, MPFR_RNDN)

#define MAX_PREC 10000
#define BASE 16

void gnumb_read (const char *filename, fp_t *dest, int n)
{
  mpfr_t x;
  int i;
  FILE *f = fopen(filename, "r");
  if (!f) {printf ("File not found: %s\n", filename); exit (1);}

  mpfr_init2 (x, MAX_PREC);
  for (i = 0 ; i < n ; i++)
    {
      if (mpfr_inp_str (x, f, 16, MPFR_RNDN) == 0)
	{
	  printf ("Error reading entry %d/%d\n", i, n);
	  mpfr_clear (x);
	  fclose (f);
	  exit (2);
	}
      fp_set_fr (*dest++, x);
    }
  mpfr_clear (x);
  fclose (f);
}

void gnumb_generate (const char *filename, int n, unsigned long seed)
{
  mpfr_t x;
  gmp_randstate_t state;
  int i;
  FILE *f;

  f = fopen (filename, "w");
  if (!f) {printf ("Can't create file: %s\n", filename); exit (1);}
  mpfr_init2 (x, MAX_PREC);
  gmp_randinit_lc_2exp_size (state, 128);
  gmp_randseed_ui (state, seed);

  for (i = 0 ; i < n ; i++)
    {
      mpfr_urandomb (x, state);
      if ((i & 2) == 0)
        mpfr_mul_2si (x, x, (rand()%(2*GMP_NUMB_BITS))-GMP_NUMB_BITS,
                      MPFR_RNDN);
      mpfr_out_str (f, 16, 0, x, MPFR_RNDN);
      fputc ('\n', f);
    }

  gmp_randclear(state);
  mpfr_clear (x);
  fclose (f);
}

#ifdef NEED_MAIN
int main (int argc, char *argv[])
{
  const char *filename = "float.data";
  int num = 1100;
  unsigned long seed = 12345679;
  if (argc >= 2)
    filename = argv[1];
  if (argc >= 3)
    num = atoi (argv[2]);
  if (argc >= 4)
    seed = atol (argv[3]);
  gnumb_generate (filename, num, seed);
  return 0;
}
#endif
