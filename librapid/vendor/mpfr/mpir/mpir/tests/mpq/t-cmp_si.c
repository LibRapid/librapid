/* Test mpq_cmp_si.

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
#include <limits.h>

#include "mpir.h"
#include "gmp-impl.h"
#include "tests.h"

#define printf gmp_printf

#define SGN(x)   ((x)<0 ? -1 : (x) != 0)

void
check_data (void)
{
  static const struct {
    const char     *q;
    mpir_si         n;
    mpir_ui         d;
    int            want;
  } data[] = {
    { "0", 0, 1, 0 },
    { "0", 0, 123, 0 },
    { "0", 0, GMP_UI_MAX, 0 },
    { "1", 0, 1, 1 },
    { "1", 0, 123, 1 },
    { "1", 0, GMP_UI_MAX, 1 },
    { "-1", 0, 1, -1 },
    { "-1", 0, 123, -1 },
    { "-1", 0, GMP_UI_MAX, -1 },

    { "123", 123, 1, 0 },
    { "124", 123, 1, 1 },
    { "122", 123, 1, -1 },

    { "-123", 123, 1, -1 },
    { "-124", 123, 1, -1 },
    { "-122", 123, 1, -1 },

    { "123", -123, 1, 1 },
    { "124", -123, 1, 1 },
    { "122", -123, 1, 1 },

    { "-123", -123, 1, 0 },
    { "-124", -123, 1, -1 },
    { "-122", -123, 1, 1 },

    { "5/7", 3,4, -1 },
    { "5/7", -3,4, 1 },
    { "-5/7", 3,4, -1 },
    { "-5/7", -3,4, 1 },
  };

  mpq_t  q;
  int    i, got;

  mpq_init (q);

  for (i = 0; i < numberof (data); i++)
    {
      mpq_set_str_or_abort (q, data[i].q, 0);
      MPQ_CHECK_FORMAT (q);

      got = mpq_cmp_si (q, data[i].n, data[i].d);
      if (SGN(got) != data[i].want)
        {
          printf ("mpq_cmp_si wrong\n");
        error:
          mpq_trace ("  q", q);
          printf ("  n=%Md\n", data[i].n);
          printf ("  d=%Mu\n", data[i].d);
          printf ("  got=%d\n", got);
          printf ("  want=%Md\n", data[i].want);
          abort ();
        }

      if (data[i].n == 0)
        {
          got = mpq_cmp_si (q, 0L, data[i].d);
          if (SGN(got) != data[i].want)
            {
              printf ("mpq_cmp_si wrong\n");
              goto error;
            }
        }
    }

  mpq_clear (q);
}

int
main (int argc, char **argv)
{
  tests_start ();

  check_data ();

  tests_end ();
  exit (0);
}
