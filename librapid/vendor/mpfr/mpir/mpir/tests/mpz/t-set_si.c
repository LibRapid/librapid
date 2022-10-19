/* Test mpz_set_si and mpz_init_set_si.

Copyright 2000, 2001, 2002 Free Software Foundation, Inc.

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
check_data (void)
{
#if GMP_NUMB_BITS <= BITS_PER_UI
#define ENTRY(n)   { n, { n, 0 } }
#else
#define ENTRY(n)   { n, { (n) & GMP_NUMB_MASK, (n) >> GMP_NUMB_BITS } }
#endif

  static const struct {
    mpir_si       n;
    mp_size_t  want_size;
    mp_limb_t  want_data[2];
  } data[] = {

    {  0L,  0 },
    {  1L,  1, { 1 } },
    { -1L, -1, { 1 } },

#if GMP_NUMB_BITS >= BITS_PER_UI - 1
    { GMP_SI_MAX,  1, { GMP_SI_MAX, 0 } },
    { -GMP_SI_MAX,  -1, { GMP_SI_MAX, 0 } },
#else
    { GMP_SI_MAX,  2, { GMP_SI_MAX & GMP_NUMB_MASK, GMP_SI_MAX >> GMP_NUMB_BITS } },
    { -GMP_SI_MAX,  -2, { GMP_SI_MAX & GMP_NUMB_MASK, GMP_SI_MAX >> GMP_NUMB_BITS } },
#endif

#if GMP_NUMB_BITS >= BITS_PER_UI
    { GMP_SI_MIN,  -1, { GMP_UI_HIBIT, 0 } },
#else
    { GMP_SI_MIN,  -2, { 0, GMP_UI_HIBIT >> GMP_NUMB_BITS } },
#endif
  };

  mpz_t  n;
  int    i;

  for (i = 0; i < numberof (data); i++)
    {
      mpz_init (n);
      mpz_set_si (n, data[i].n);
      MPZ_CHECK_FORMAT (n);
      if (n->_mp_size != data[i].want_size
          || refmpn_cmp_allowzero (n->_mp_d, data[i].want_data,
                                   ABS (data[i].want_size)) != 0)
        {
          printf ("mpz_set_si wrong on data[%d]\n", i);
          abort();
        }
      mpz_clear (n);

      mpz_init_set_si (n, data[i].n);
      MPZ_CHECK_FORMAT (n);
      if (n->_mp_size != data[i].want_size
          || refmpn_cmp_allowzero (n->_mp_d, data[i].want_data,
                                   ABS (data[i].want_size)) != 0)
        {
          printf ("mpz_init_set_si wrong on data[%d]\n", i);
          abort();
        }
      mpz_clear (n);
    }
}


int
main (void)
{
  tests_start ();

  check_data ();

  tests_end ();
  exit (0);
}
