/* Test file for mpfr_nrandom

Copyright 2011-2022 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
https://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include "mpfr-test.h"

static void
test_special (mpfr_prec_t p)
{
  mpfr_t x;
  int inexact;

  mpfr_init2 (x, p);

  inexact = mpfr_nrandom (x, RANDS, MPFR_RNDN);
  if (inexact == 0)
    {
      printf ("Error: mpfr_nrandom() returns a zero ternary value.\n");
      exit (1);
    }

  mpfr_clear (x);
}

#define NRES 10

#ifndef MPFR_USE_MINI_GMP
static const char *res[NRES] = {
  "d.045d0ff20f5ba6d8702391be8d38e3b82023bb445efd47af60b9a16dd42b91ccb6fb4b9c93ac4134570583b079ac575df695ec570@-1",
  "9.c8ab7e45a0f79cfbb5486d44c56e99e69e33cfb58729a7ce72cf34270a8b751c0e65269bf9c122ac5192d6d0bb15c03230b1c4600@-1",
  "7.f82ae1b380e448b35216920cd4a1e20f3390cf8aa06a419c8fcb18abc0057220b4d4170574654606f6d3ef664523ce1bd2fbc0508@-1",
  "-4.a86e702fe0c829f547b489d39f11283a52ea70e1a44ee34d621cc62ca44b02c9a55d7754b011b934281c1da2bab2e94f80ad079b0@-1",
  "e.16dacf5086c47676d70dc41a9c9e05d2d7cd55e15c4f92b37838812f995a4a4242197f334769313ccd414d3137bc7833d1c200e40@-1",
  "f.3581a7f831e2ef4c4c5f2ba21583a599ee722e64c017e9d9bd11f6065243d777c8dcd82e4658001b7f7115077eff5d8dbaaad2040@-1",
  "d.57e17bebe2a23b24a1bb6b294779406a09590c011baf3c66157a944c182bcbb89ac301c35db8703ce220d9e0a5cd10344a202de90@-1",
  "-a.55d67f858fb3fd92c440ee27c1dfebae2b71a915abd87bd4801967abcfa662b0e28edf3d5ea311dc8ba465b0ec5b4a190b1e55850@-1",
  "-1.00f594aa573376a1ac4be1bbc4850738a4ac7ee805408dfd07a96b7edd42773a1ede75a5f371f607f41f2aff72eee7fb2b6f13138@0",
  "-3.275e0ceb2a81bc9387cccf6eb3404aed9e275e03fe9f0745e2cf3967616a37479768ba61bee1aa02120f527a320460a616980ea94@-1" };
#endif /* MPFR_USE_MINI_GMP */

static void
test_nrandom (long nbtests, mpfr_prec_t prec, mpfr_rnd_t rnd,
              int verbose)
{
  mpfr_t *t;
  int i, inexact;

  t = (mpfr_t *) tests_allocate (nbtests * sizeof (mpfr_t));

  for (i = 0; i < nbtests; ++i)
    mpfr_init2 (t[i], prec);

  for (i = 0; i < nbtests; i++)
    {
      inexact = mpfr_nrandom (t[i], RANDS, MPFR_RNDN);

#ifndef MPFR_USE_MINI_GMP
      if (i < NRES && mpfr_cmp_str (t[i], res[i], 16, MPFR_RNDN) != 0)
        {
          printf ("Unexpected value in test_nrandom().\n"
                  "Expected %s\n"
                  "Got      ", res[i]);
          mpfr_out_str (stdout, 16, 0, t[i], MPFR_RNDN);
          printf ("\n");
          exit (1);
        }
#endif /* MPFR_USE_MINI_GMP */

      if (inexact == 0)
        {
          /* one call in the loop pretended to return an exact number! */
          printf ("Error: mpfr_nrandom() returns a zero ternary value.\n");
          exit (1);
        }
    }

#if defined(HAVE_STDARG) && !defined(MPFR_USE_MINI_GMP)
  if (verbose)
    {
      mpfr_t av, va, tmp;

      mpfr_init2 (av, prec);
      mpfr_init2 (va, prec);
      mpfr_init2 (tmp, prec);

      mpfr_set_ui (av, 0, MPFR_RNDN);
      mpfr_set_ui (va, 0, MPFR_RNDN);
      for (i = 0; i < nbtests; ++i)
        {
          mpfr_add (av, av, t[i], MPFR_RNDN);
          mpfr_sqr (tmp, t[i], MPFR_RNDN);
          mpfr_add (va, va, tmp, MPFR_RNDN);
        }
      mpfr_div_ui (av, av, nbtests, MPFR_RNDN);
      mpfr_div_ui (va, va, nbtests, MPFR_RNDN);
      mpfr_sqr (tmp, av, MPFR_RNDN);
      mpfr_sub (va, va, av, MPFR_RNDN);

      mpfr_printf ("Average = %.5Rf\nVariance = %.5Rf\n", av, va);
      mpfr_clear (av);
      mpfr_clear (va);
      mpfr_clear (tmp);
    }
#endif /* HAVE_STDARG */

  for (i = 0; i < nbtests; ++i)
    mpfr_clear (t[i]);
  tests_free (t, nbtests * sizeof (mpfr_t));
  return;
}


int
main (int argc, char *argv[])
{
  long nbtests;
  int verbose;

  tests_start_mpfr ();

  verbose = 0;
  nbtests = 10;
  if (argc > 1)
    {
      long a = atol (argv[1]);
      verbose = 1;
      if (a != 0)
        nbtests = a;
    }

  test_nrandom (nbtests, 420, MPFR_RNDN, verbose);
  test_special (2);
  test_special (42000);

  tests_end_mpfr ();
  return 0;
}
