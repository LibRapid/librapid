/* Reference mpz functions.

Copyright 1997, 1999, 2000, 2001, 2002 Free Software Foundation, Inc.

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

/* always do assertion checking */
#define WANT_ASSERT  1

#include <stdio.h>
#include <stdlib.h> /* for free */
#include "mpir.h"
#include "gmp-impl.h"
#include "longlong.h"
#include "tests.h"


/* Change this to "#define TRACE(x) x" for some traces. */
#define TRACE(x) 


/* FIXME: Shouldn't use plain mpz functions in a reference routine. */
void
refmpz_combit (mpz_ptr r, unsigned long bit)
{
  if (mpz_tstbit (r, bit))
    mpz_clrbit (r, bit);
  else
    mpz_setbit (r, bit);
}


unsigned long
refmpz_hamdist (mpz_srcptr x, mpz_srcptr y)
{
  mp_size_t      xsize, ysize, tsize;
  mp_ptr         xp, yp;
  unsigned long  ret;

  if ((SIZ(x) < 0 && SIZ(y) >= 0)
      || (SIZ(y) < 0 && SIZ(x) >= 0))
    return ULONG_MAX;

  xsize = ABSIZ(x);
  ysize = ABSIZ(y);
  tsize = MAX (xsize, ysize);

  xp = refmpn_malloc_limbs (tsize);
  refmpn_zero (xp, tsize);
  refmpn_copy (xp, PTR(x), xsize);

  yp = refmpn_malloc_limbs (tsize);
  refmpn_zero (yp, tsize);
  refmpn_copy (yp, PTR(y), ysize);

  if (SIZ(x) < 0)
    refmpn_neg_n (xp, xp, tsize);

  if (SIZ(x) < 0)
    refmpn_neg_n (yp, yp, tsize);

  ret = refmpn_hamdist (xp, yp, tsize);

  free (xp);
  free (yp);
  return ret;
}


/* (0/b), with mpz b; is 1 if b=+/-1, 0 otherwise */
#define JACOBI_0Z(b)  JACOBI_0LS (PTR(b)[0], SIZ(b))

/* (a/b) effect due to sign of b: mpz/mpz */
#define JACOBI_BSGN_ZZ_BIT1(a, b)   JACOBI_BSGN_SS_BIT1 (SIZ(a), SIZ(b))

/* (a/b) effect due to sign of a: mpz/unsigned-mpz, b odd;
   is (-1/b) if a<0, or +1 if a>=0 */
#define JACOBI_ASGN_ZZU_BIT1(a, b)  JACOBI_ASGN_SU_BIT1 (SIZ(a), PTR(b)[0])

int
refmpz_kronecker (mpz_srcptr a_orig, mpz_srcptr b_orig)
{
  unsigned long  twos;
  mpz_t  a, b;
  int    result_bit1 = 0;

  if (mpz_sgn (b_orig) == 0)
    return JACOBI_Z0 (a_orig);  /* (a/0) */

  if (mpz_sgn (a_orig) == 0)
    return JACOBI_0Z (b_orig);  /* (0/b) */

  if (mpz_even_p (a_orig) && mpz_even_p (b_orig))
    return 0;

  if (mpz_cmp_ui (b_orig, 1) == 0)
    return 1;

  mpz_init_set (a, a_orig);
  mpz_init_set (b, b_orig);

  if (mpz_sgn (b) < 0)
    {
      result_bit1 ^= JACOBI_BSGN_ZZ_BIT1 (a, b);
      mpz_neg (b, b);
    }
  if (mpz_even_p (b))
    {
      twos = mpz_scan1 (b, 0L);
      mpz_tdiv_q_2exp (b, b, twos);
      result_bit1 ^= JACOBI_TWOS_U_BIT1 (twos, PTR(a)[0]);
    }

  if (mpz_sgn (a) < 0)
    {
      result_bit1 ^= JACOBI_N1B_BIT1 (PTR(b)[0]);
      mpz_neg (a, a);
    }
  if (mpz_even_p (a))
    {
      twos = mpz_scan1 (a, 0L);
      mpz_tdiv_q_2exp (a, a, twos);
      result_bit1 ^= JACOBI_TWOS_U_BIT1 (twos, PTR(b)[0]);
    }

  for (;;)
    {
      ASSERT (mpz_odd_p (a));
      ASSERT (mpz_odd_p (b));
      ASSERT (mpz_sgn (a) > 0);
      ASSERT (mpz_sgn (b) > 0);

      TRACE (printf ("top\n");
             mpz_trace (" a", a);
             mpz_trace (" b", b));

      if (mpz_cmp (a, b) < 0)
        {
          TRACE (printf ("swap\n"));
          mpz_swap (a, b);
          result_bit1 ^= JACOBI_RECIP_UU_BIT1 (PTR(a)[0], PTR(b)[0]);
        }

      if (mpz_cmp_ui (b, 1) == 0)
        break;

      mpz_sub (a, a, b);
      TRACE (printf ("sub\n");
             mpz_trace (" a", a));
      if (mpz_sgn (a) == 0)
        goto zero;

      twos = mpz_scan1 (a, 0L);
      mpz_fdiv_q_2exp (a, a, twos);
      TRACE (printf ("twos %lu\n", twos);
             mpz_trace (" a", a));
      result_bit1 ^= JACOBI_TWOS_U_BIT1 (twos, PTR(b)[0]);
    }

  mpz_clear (a);
  mpz_clear (b);
  return JACOBI_BIT1_TO_PN (result_bit1);

 zero:
  mpz_clear (a);
  mpz_clear (b);
  return 0;
}

/* Same as mpz_kronecker, but ignoring factors of 2 on b */
int
refmpz_jacobi (mpz_srcptr a, mpz_srcptr b)
{
  mpz_t  b_odd;
  mpz_init_set (b_odd, b);
  if (mpz_sgn (b_odd) != 0)
    mpz_fdiv_q_2exp (b_odd, b_odd, mpz_scan1 (b_odd, 0L));
  return refmpz_kronecker (a, b_odd);
}

int
refmpz_legendre (mpz_srcptr a, mpz_srcptr b)
{
  return refmpz_jacobi (a, b);
}


int
refmpz_kronecker_ui (mpz_srcptr a, unsigned long b)
{
  mpz_t  bz;
  int    ret;
  mpz_init_set_ui (bz, b);
  ret = refmpz_kronecker (a, bz);
  mpz_clear (bz);
  return ret;
}

int
refmpz_kronecker_si (mpz_srcptr a, long b)
{
  mpz_t  bz;
  int    ret;
  mpz_init_set_si (bz, b);
  ret = refmpz_kronecker (a, bz);
  mpz_clear (bz);
  return ret;
}

int
refmpz_ui_kronecker (unsigned long a, mpz_srcptr b)
{
  mpz_t  az;
  int    ret;
  mpz_init_set_ui (az, a);
  ret = refmpz_kronecker (az, b);
  mpz_clear (az);
  return ret;
}

int
refmpz_si_kronecker (long a, mpz_srcptr b)
{
  mpz_t  az;
  int    ret;
  mpz_init_set_si (az, a);
  ret = refmpz_kronecker (az, b);
  mpz_clear (az);
  return ret;
}


void
refmpz_pow_ui (mpz_ptr w, mpz_srcptr b, unsigned long e)
{
  mpz_t          s, t;
  unsigned long  i;

  mpz_init_set_ui (t, 1L);
  mpz_init_set (s, b);

  if ((e & 1) != 0)
    mpz_mul (t, t, s);

  for (i = 2; i <= e; i <<= 1)
    {
      mpz_mul (s, s, s);
      if ((i & e) != 0)
	mpz_mul (t, t, s);
    }

  mpz_set (w, t);

  mpz_clear (s);
  mpz_clear (t);
}
