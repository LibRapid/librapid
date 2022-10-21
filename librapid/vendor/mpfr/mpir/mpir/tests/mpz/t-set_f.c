/* Test mpz_set_f.

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

#include <stdio.h>
#include <stdlib.h>
#include "mpir.h"
#include "gmp-impl.h"
#include "tests.h"


void
check_one (mpz_srcptr z)
{
  static const int shift[] = {
    0, 1, BITS_PER_MP_LIMB, 2*BITS_PER_MP_LIMB, 5*BITS_PER_MP_LIMB
  };

  int    sh, shneg, neg;
  mpf_t  f;
  mpz_t  got, want;

  mpf_init2 (f, mpz_sizeinbase(z,2));
  mpz_init (got);
  mpz_init (want);

  for (sh = 0; sh < numberof(shift); sh++)
    {
      for (shneg = 0; shneg <= 1; shneg++)
        {
          for (neg = 0; neg <= 1; neg++)
            {
              mpf_set_z (f, z);
              mpz_set (want, z);
            
              if (neg)
                {
                  mpf_neg (f, f);
                  mpz_neg (want, want);
                }

              if (shneg)
                {
                  mpz_tdiv_q_2exp (want, want, shift[sh]);
                  mpf_div_2exp (f, f, shift[sh]);
                }
              else
                {
                  mpz_mul_2exp (want, want, shift[sh]);
                  mpf_mul_2exp (f, f, shift[sh]);
                }

              mpz_set_f (got, f);
              MPZ_CHECK_FORMAT (got);

              if (mpz_cmp (got, want) != 0)
                {
                  printf ("wrong result\n");
                  printf ("  shift  %d\n", shneg ? -shift[sh] : shift[sh]);
                  printf ("  neg    %d\n", neg);
                  mpf_trace ("     f", f);
                  mpz_trace ("   got", got);
                  mpz_trace ("  want", want);
                  abort ();
                }
            }
        }
    }

  mpf_clear (f);
  mpz_clear (got);
  mpz_clear (want);
}


void
check_various (void)
{gmp_randstate_t rands;
  mpz_t  z;

  mpz_init (z);
  gmp_randinit_default(rands);

  mpz_set_ui (z, 0L);
  check_one (z);

  mpz_set_si (z, 123L);
  check_one (z);

  mpz_rrandomb (z, rands, 2*BITS_PER_MP_LIMB);
  check_one (z);

  mpz_rrandomb (z, rands, 5*BITS_PER_MP_LIMB);
  check_one (z);

  mpz_clear (z);
  gmp_randclear(rands);
}


int
main (int argc, char *argv[])
{
#if GMP_NAIL_BITS == 0
  tests_start ();
  mp_trace_base = 16;

  check_various ();

  tests_end ();
#endif
  exit (0);
}
