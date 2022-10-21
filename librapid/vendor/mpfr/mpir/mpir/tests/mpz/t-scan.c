/* Tests of mpz_scan0 and mpz_scan1.

Copyright 2000, 2001, 2002, 2003 Free Software Foundation, Inc.

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


unsigned long
refmpz_scan (mpz_srcptr z, unsigned long i, int sought)
{
  unsigned long  z_bits = (unsigned long) ABSIZ(z) * GMP_NUMB_BITS;

  do
    {
      if (mpz_tstbit (z, i) == sought)
        return i;
      i++;
    }
  while (i <= z_bits);

  return ULONG_MAX;
}

unsigned long
refmpz_scan0 (mpz_srcptr z, unsigned long starting_bit)
{
  return refmpz_scan (z, starting_bit, 0);
}

unsigned long
refmpz_scan1 (mpz_srcptr z, unsigned long starting_bit)
{
  return refmpz_scan (z, starting_bit, 1);
}


void
check_ref (void)
{
  static const int offset[] = {
    -2, -1, 0, 1, 2, 3
  };

  mpz_t          z;
  int            test, neg, sought, oindex, o;
  mp_size_t      size, isize;
  unsigned long  start, got, want;
  gmp_randstate_t rands;

  mpz_init (z);
  gmp_randinit_default(rands);
  for (test = 0; test < 5; test++)
    {
      for (size = 0; size < 5; size++)
        {
          mpz_urandomb (z, rands, (unsigned long) (ABS (size) * GMP_NUMB_BITS));
          if (size < 0)SIZ(z) = -SIZ(z);               

          for (neg = 0; neg <= 1; neg++)
            {
              if (neg)
                mpz_neg (z, z);

              for (isize = 0; isize <= size; isize++)
                {
                  for (oindex = 0; oindex < numberof (offset); oindex++)
                    {
                      o = offset[oindex];
                      if ((int) isize*GMP_NUMB_BITS < -o)
                        continue;  /* start would be negative */

                      start = isize*GMP_NUMB_BITS + o;

                      for (sought = 0; sought <= 1; sought++)
                        {
                          if (sought == 0)
                            {
                              got = mpz_scan0 (z, start);
                              want = refmpz_scan0 (z, start);
                            }
                          else
                            {
                              got = mpz_scan1 (z, start);
                              want = refmpz_scan1 (z, start);
                            }

                          if (got != want)
                            {
                              printf ("wrong at test=%d, size=%ld, neg=%d, start=%lu, sought=%d\n",
                                      test, size, neg, start, sought);
                              printf ("   z 0x");
                              mpz_out_str (stdout, -16, z);
                              printf ("\n");
                              printf ("   got=%lu, want=%lu\n", got, want);
                              exit (1);
                            }
                        }
                    }
                }
            }
        }
    }
  mpz_clear (z);gmp_randclear(rands);
}


int
main (int argc, char *argv[])
{
  tests_start ();

  check_ref ();

  tests_end ();
  exit (0);
}
