/* Test mpf_get_d_2exp.

Copyright 2002, 2003 Free Software Foundation, Inc.

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


static void
check_onebit (void)
{
  static const long data[] = {
    -513, -512, -511, -65, -64, -63, -32, -1,
    0, 1, 32, 53, 54, 64, 128, 256, 511, 512, 513
  };
  mpf_t   f;
  double  got, want;
  signed long got_exp, want_exp;
  int     i,sign;

  mpf_init2(f,1024L);
  for(sign=-1;sign<=1;sign+=2){
  
  for (i = 0; i < numberof (data); i++)
    {
      mpf_set_ui (f, 1L);if(sign==-1)mpf_neg(f,f);
      if (data[i] >= 0)
        mpf_mul_2exp (f, f, data[i]);
      else
        mpf_div_2exp (f, f, -data[i]);
      want = 0.5*sign;
      want_exp = data[i] + 1;

      got = mpf_get_d_2exp (&got_exp, f);
      if (got != want || got_exp != want_exp)
        {
          printf    ("mpf_get_d_2exp wrong on 2**%ld\n", data[i]);
          mpf_trace ("   f    ", f);
          d_trace   ("   want ", want);
          d_trace   ("   got  ", got);
          printf    ("   want exp %ld\n", want_exp);
          printf    ("   got exp  %ld\n", got_exp);
          abort();
        }
    }}
  mpf_clear (f);
}

/* Check that hardware rounding doesn't make mpf_get_d_2exp return a value
   outside its defined range. */
static void
check_round (void)
{
  static const unsigned long data[] = { 1, 32, 53, 54, 64, 128, 256, 512 };
  mpf_t   f;
  double  got;
  mpir_si got_exp;
  int     i, rnd_mode, old_rnd_mode;

  mpf_init2 (f, 1024L);
  old_rnd_mode = tests_hardware_getround ();

  for (rnd_mode = 0; rnd_mode < 4; rnd_mode++)
    {
      tests_hardware_setround (rnd_mode);

      for (i = 0; i < numberof (data); i++)
        {
          mpf_set_ui (f, 1L);
          mpf_mul_2exp (f, f, data[i]);
          mpf_sub_ui (f, f, 1L);

          got = mpf_get_d_2exp (&got_exp, f);
          if (got < 0.5 || got >= 1.0)
            {
              printf    ("mpf_get_d_2exp bad on 2**%lu-1\n", data[i]);
              printf    ("result out of range, expect 0.5 <= got < 1.0\n");
              printf    ("   rnd_mode = %d\n", rnd_mode);
              printf    ("   data[i]  = %lu\n", data[i]);
              mpf_trace ("   f    ", f);
              d_trace   ("   got  ", got);
              printf    ("   got exp  %ld\n", got_exp);
              abort();
            }
        }
    }

  mpf_clear (f);
  tests_hardware_setround (old_rnd_mode);
}

static void
check_large_exponent(void)
{
#ifdef _WIN64
  static const mpir_si data[] = { 0x1F789ABCDE, 0x1FFFFFFFBF, -0x1FEDCBA987, -0x1FFFFFFFC1 };
  mpf_t   f;
  double  got;
  mpir_si got_exp;
  unsigned int i;

  mpf_init2 (f, 1024L);

  for (i = 0; i < numberof (data); i++)
  {
    mpf_set_ui (f, 1L);
    if (data[i] > 0)
    {
      mpf_mul_2exp(f, f, data[i]);
    }
    else
    {
      mpf_div_2exp(f, f, -data[i]);
    }
    got_exp = mpf_get_2exp_d (&got, f);
    if (got != 0.5 || got_exp != data[i] + 1)
    {
      printf    ("mpf_get_2exp_d bad on 2**%lld\n", data[i]);
      printf    ("result incorrect, expect mantissa = 0.5, exp = %lld\n", data[i] + 1);
      mpf_trace ("   f    ", f);
      d_trace   ("   got  ", got);
      printf    ("   got exp  %lld\n", got_exp);
      abort();
    }
  }
  mpf_clear (f);
#endif
}

int
main (void)
{
  tests_start ();
  mp_trace_base = 16;

  check_onebit ();
  check_round ();
  check_large_exponent ();

  tests_end ();
  exit (0);
}
