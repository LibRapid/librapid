/* test mpz_congruent_p and mpz_congruent_ui_p

Copyright 2001, 2002 Free Software Foundation, Inc.

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
check_one (mpz_srcptr a, mpz_srcptr c, mpz_srcptr d, int want)
{
  int   got;
  int   swap;

  for (swap = 0; swap <= 1; swap++)
    {
      got = (mpz_congruent_p (a, c, d) != 0);
      if (want != got)
        {
          printf ("mpz_congruent_p wrong\n");
          printf ("   expected %d got %d\n", want, got);
          mpz_trace ("   a", a);
          mpz_trace ("   c", c);
          mpz_trace ("   d", d);
          mp_trace_base = -16;
          mpz_trace ("   a", a);
          mpz_trace ("   c", c);
          mpz_trace ("   d", d);
          abort ();
        }

      if (mpz_fits_ulong_p (c) && mpz_fits_ulong_p (d))
        {
          unsigned long  uc = mpz_get_ui (c);
          unsigned long  ud = mpz_get_ui (d);
          got = (mpz_congruent_ui_p (a, uc, ud) != 0);
          if (want != got)
            {
              printf    ("mpz_congruent_ui_p wrong\n");
              printf    ("   expected %d got %d\n", want, got);
              mpz_trace ("   a", a);
              printf    ("   c=%lu\n", uc);
              printf    ("   d=%lu\n", ud);
              mp_trace_base = -16;
              mpz_trace ("   a", a);
              printf    ("   c=0x%lX\n", uc);
              printf    ("   d=0x%lX\n", ud);
              abort ();
            }
        }

      MPZ_SRCPTR_SWAP (a, c);
    }
}


void
check_data (void)
{
  static const struct {
    const char *a;
    const char *c;
    const char *d;
    int        want;

  } data[] = {

    /* anything congruent mod 1 */
    { "0", "0", "1", 1 },
    { "1", "0", "1", 1 },
    { "0", "1", "1", 1 },
    { "123", "456", "1", 1 },
    { "0x123456789123456789", "0x987654321987654321", "1", 1 },

    /* csize==1, dsize==2 changing to 1 after stripping 2s */
    { "0x3333333333333333",  "0x33333333",
      "0x180000000", 1 },
    { "0x33333333333333333333333333333333", "0x3333333333333333",
      "0x18000000000000000", 1 },

    /* another dsize==2 becoming 1, with opposite signs this time */
    {  "0x444444441",
      "-0x22222221F",
       "0x333333330", 1 },
    {  "0x44444444444444441",
      "-0x2222222222222221F",
       "0x33333333333333330", 1 },
  };

  mpz_t   a, c, d;
  int     i;

  mpz_init (a);
  mpz_init (c);
  mpz_init (d);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (a, data[i].a, 0);
      mpz_set_str_or_abort (c, data[i].c, 0);
      mpz_set_str_or_abort (d, data[i].d, 0);
      check_one (a, c, d, data[i].want);
    }

  mpz_clear (a);
  mpz_clear (c);
  mpz_clear (d);
}


void
check_random (int argc, char *argv[])
{
  gmp_randstate_t rands;
  mpz_t   a, c, d, ra, rc;
  int     i;
  int     want;
  int     reps = 2000;

  if (argc >= 2)
    reps = atoi (argv[1]);

  mpz_init (a);
  mpz_init (c);
  mpz_init (d);
  mpz_init (ra);
  mpz_init (rc);
  gmp_randinit_default(rands);

  for (i = 0; i < reps; i++)
    {
      mpz_errandomb (a, rands, 8*BITS_PER_MP_LIMB);
      MPZ_CHECK_FORMAT (a);
      mpz_errandomb (c, rands, 8*BITS_PER_MP_LIMB);
      MPZ_CHECK_FORMAT (c);
      mpz_errandomb_nonzero (d, rands, 8*BITS_PER_MP_LIMB);

      mpz_negrandom (a, rands);
      MPZ_CHECK_FORMAT (a);
      mpz_negrandom (c, rands);
      MPZ_CHECK_FORMAT (c);
      mpz_negrandom (d, rands);

      mpz_fdiv_r (ra, a, d);
      mpz_fdiv_r (rc, c, d);

      want = (mpz_cmp (ra, rc) == 0);
      check_one (a, c, d, want);

      mpz_sub (ra, ra, rc);
      mpz_sub (a, a, ra);
      MPZ_CHECK_FORMAT (a);
      check_one (a, c, d, 1);

      if (! mpz_pow2abs_p (d))
        {
          refmpz_combit (a, urandom(rands) % (8*BITS_PER_MP_LIMB));
          check_one (a, c, d, 0);
        }
    }

  mpz_clear (a);
  mpz_clear (c);
  mpz_clear (d);
  mpz_clear (ra);
  mpz_clear (rc);
  gmp_randclear(rands);
}


int
main (int argc, char *argv[])
{
  tests_start ();

  check_data ();
  check_random (argc, argv);

  tests_end ();
  exit (0);
}
