/* Test BSWAP_LIMB.

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


int
main (void)
{
  mp_limb_t  src, want, got;
  int        i;
  gmp_randstate_t rands;
  
  tests_start ();
  gmp_randinit_default(rands);
  mp_trace_base = -16;

  for (i = 0; i < 1000; i++)
    {
      mpn_randomb (&src, rands, (mp_size_t) 1);

      want = ref_bswap_limb (src);

      BSWAP_LIMB (got, src);
      if (got != want)
        {
          printf ("BSWAP_LIMB wrong result\n");
        error:
          mpn_trace ("  src ", &src,  (mp_size_t) 1);
          mpn_trace ("  want", &want, (mp_size_t) 1);
          mpn_trace ("  got ", &got,  (mp_size_t) 1);
          abort ();
        }

      BSWAP_LIMB_FETCH (got, &src);
      if (got != want)
        {
          printf ("BSWAP_LIMB_FETCH wrong result\n");
          goto error;
        }

      BSWAP_LIMB_STORE (&got, src);
      if (got != want)
        {
          printf ("BSWAP_LIMB_STORE wrong result\n");
          goto error;
        }
    }
  gmp_randclear(rands);
  tests_end ();
  exit (0);
}
