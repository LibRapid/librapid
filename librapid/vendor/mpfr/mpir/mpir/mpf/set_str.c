/* mpf_set_str (dest, string, base) -- Convert the string STRING
   in base BASE to a float in dest.  If BASE is zero, the leading characters
   of STRING is used to figure out the base.

Copyright 1993, 1994, 1995, 1996, 1997, 2000, 2001, 2002, 2003, 2005 Free
Software Foundation, Inc.

Copyright 2009 B R Gladman

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

/*
  This still needs work, as suggested by some FIXME comments.
  1. Don't depend on superfluous mantissa digits.
  2. Allocate temp space more cleverly.
  3. Use mpn_tdiv_qr instead of mpn_lshift+mpn_divrem.
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE    /* for DECIMAL_POINT in langinfo.h */
#endif

#include "config.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#if HAVE_LANGINFO_H
#include <langinfo.h>  /* for nl_langinfo */
#endif

#if HAVE_LOCALE_H
#include <locale.h>    /* for localeconv */
#endif

#include "mpir.h"
#include "gmp-impl.h"
#include "longlong.h"

static mp_limb_t mpn_intdivrem (mp_ptr qp, mp_size_t qxn,
	    mp_ptr np, mp_size_t nn,
	    mp_srcptr dp, mp_size_t dn)
{
  ASSERT (qxn >= 0);
  ASSERT (nn >= dn);
  ASSERT (dn >= 1);
  ASSERT (dp[dn-1] & GMP_NUMB_HIGHBIT);
  ASSERT (! MPN_OVERLAP_P (np, nn, dp, dn));
  ASSERT (! MPN_OVERLAP_P (qp, nn-dn+qxn, np, nn) || qp==np+dn+qxn);
  ASSERT (! MPN_OVERLAP_P (qp, nn-dn+qxn, dp, dn));
  ASSERT_MPN (np, nn);
  ASSERT_MPN (dp, dn);

  if (dn == 1)
    {
      mp_limb_t ret;
      mp_ptr q2p;
      mp_size_t qn;
      TMP_DECL;

      TMP_MARK;
      q2p = (mp_ptr) TMP_ALLOC ((nn + qxn) * BYTES_PER_MP_LIMB);

      np[0] = mpn_divrem_1 (q2p, qxn, np, nn, dp[0]);
      qn = nn + qxn - 1;
      MPN_COPY (qp, q2p, qn);
      ret = q2p[qn];

      TMP_FREE;
      return ret;
    }
  else if (dn == 2)
    {
      return mpn_divrem_2 (qp, qxn, np, nn, dp);
    }
  else
    {
      mp_ptr rp, q2p;
      mp_limb_t qhl;
      mp_size_t qn;
      TMP_DECL;

      TMP_MARK;
      if (UNLIKELY (qxn != 0))
	{
	  mp_ptr n2p;
	  n2p = (mp_ptr) TMP_ALLOC ((nn + qxn) * BYTES_PER_MP_LIMB);
	  MPN_ZERO (n2p, qxn);
	  MPN_COPY (n2p + qxn, np, nn);
	  q2p = (mp_ptr) TMP_ALLOC ((nn - dn + qxn + 1) * BYTES_PER_MP_LIMB);
	  rp = (mp_ptr) TMP_ALLOC (dn * BYTES_PER_MP_LIMB);
	  mpn_tdiv_qr (q2p, rp, 0L, n2p, nn + qxn, dp, dn);
	  MPN_COPY (np, rp, dn);
	  qn = nn - dn + qxn;
	  MPN_COPY (qp, q2p, qn);
	  qhl = q2p[qn];
	}
      else
	{
	  q2p = (mp_ptr) TMP_ALLOC ((nn - dn + 1) * BYTES_PER_MP_LIMB);
	  rp = (mp_ptr) TMP_ALLOC (dn * BYTES_PER_MP_LIMB);
	  mpn_tdiv_qr (q2p, rp, 0L, np, nn, dp, dn);
	  MPN_COPY (np, rp, dn);	/* overwrite np area with remainder */
	  qn = nn - dn;
	  MPN_COPY (qp, q2p, qn);
	  qhl = q2p[qn];
	}
      TMP_FREE;
      return qhl;
    }
}


extern const unsigned char __gmp_digit_value_tab[];
#define digit_value_tab __gmp_digit_value_tab

/* Compute base^exp and return the most significant prec limbs in rp[].
   Put the count of omitted low limbs in *ign.
   Return the actual size (which might be less than prec).  */
static mp_size_t
mpn_pow_1_highpart (mp_ptr rp, mp_size_t *ignp,
		    mp_limb_t base, mp_exp_t exp,
		    mp_size_t prec, mp_ptr tp)
{
  mp_size_t ign;		/* counts number of ignored low limbs in r */
  mp_size_t off;		/* keeps track of offset where value starts */
  mp_ptr passed_rp = rp;
  mp_size_t rn;
  int cnt;
  int i;

  rp[0] = base;
  rn = 1;
  off = 0;
  ign = 0;
  count_leading_zeros (cnt, exp);
  for (i = GMP_LIMB_BITS - cnt - 2; i >= 0; i--)
    {
      mpn_sqr (tp, rp + off, rn);
      rn = 2 * rn;
      rn -= tp[rn - 1] == 0;
      ign <<= 1;

      off = 0;
      if (rn > prec)
	{
	  ign += rn - prec;
	  off = rn - prec;
	  rn = prec;
	}
      MP_PTR_SWAP (rp, tp);

      if (((exp >> i) & 1) != 0)
	{
	  mp_limb_t cy;
	  cy = mpn_mul_1 (rp, rp + off, rn, base);
	  rp[rn] = cy;
	  rn += cy != 0;
	  off = 0;
	}
    }

  if (rn > prec)
    {
      ign += rn - prec;
      rp += rn - prec;
      rn = prec;
    }

  MPN_COPY_INCR (passed_rp, rp + off, rn);
  *ignp = ign;
  return rn;
}

int
mpf_set_str (mpf_ptr x, const char *str, int base)
{
  size_t str_size;
  char *s, *begs;
  size_t i, j;
  int c;
  int negative;
  char *dotpos = 0;
  const char *expptr;
  int exp_base;
  const char  *point = GMP_DECIMAL_POINT;
  size_t      pointlen = strlen (point);
  const unsigned char *digit_value;
  TMP_DECL;

  c = (unsigned char) *str;

  /* Skip whitespace.  */
  while (isspace (c))
    c = (unsigned char) *++str;

  negative = 0;
  if (c == '-')
    {
      negative = 1;
      c = (unsigned char) *++str;
    }

  digit_value = digit_value_tab;
  exp_base = base;
  if (base <= 0)
    {
      exp_base = 10;
      base = base ? -base : 10;
    }

  if(base < 2 || base > 62)
      return -1;
  else if(base > 36)
    {
      /* For bases > 36, use the collating sequence
	 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.  */
      digit_value += 224;
    }

  /* Require at least one digit, possibly after an initial decimal point.  */
  if (digit_value[c] >= base)
    {
      /* not a digit, must be a decimal point */
      for (i = 0; i < pointlen; i++)
        if (str[i] != point[i])
          return -1;
      if (digit_value[(unsigned char) str[pointlen]] >= (base == 0 ? 10 : base))
	return -1;
    }

  /* Locate exponent part of the input.  Look from the right of the string,
     since the exponent is usually a lot shorter than the mantissa.  */
  expptr = NULL;
  str_size = strlen (str);
  for (i = str_size - 1; i > 0; i--)
    {
      c = (unsigned char) str[i];
      if (c == '@' || (base <= 10 && (c == 'e' || c == 'E')))
	{
	  expptr = str + i + 1;
	  str_size = i;
	  break;
	}
    }

  TMP_MARK;
  s = begs = (char *) TMP_ALLOC (str_size + 1);

  /* Loop through mantissa, converting it from ASCII to raw byte values.  */
  for (i = 0; i < str_size; i++)
    {
      c = (unsigned char) *str;
      if (!isspace (c))
	{
	  int dig;

          for (j = 0; j < pointlen; j++)
            if (str[j] != point[j])
              goto not_point;
          if (1)
	    {
	      if (dotpos != 0)
		{
		  /* already saw a decimal point, another is invalid */
		  TMP_FREE;
		  return -1;
		}
	      dotpos = s;
	      str += pointlen - 1;
	      i += pointlen - 1;
	    }
	  else
	    {
            not_point:
	      dig = digit_value[c];
	      if (dig >= base)
		{
		  TMP_FREE;
		  return -1;
		}
	      *s++ = dig;
	    }
	}
      c = (unsigned char) *++str;
    }

  str_size = s - begs;

  {
    long exp_in_base;
    mp_size_t ra, ma, rn, mn;
    int cnt;
    mp_ptr mp, tp, rp;
    mp_exp_t exp_in_limbs;
    mp_size_t prec = PREC(x) + 1;
    int divflag;
    mp_size_t madj, radj;

#if 0
    size_t n_chars_needed;

    /* This breaks things like 0.000...0001.  To safely ignore superfluous
       digits, we need to skip over leadng zeros.  */
    /* Just consider the relevant leading digits of the mantissa.  */
    n_chars_needed = 2 + (size_t)
      (((size_t) prec * GMP_NUMB_BITS) * mp_bases[base].chars_per_bit_exactly);
    if (str_size > n_chars_needed)
      str_size = n_chars_needed;
#endif

    ma = (mp_size_t) (str_size / mp_bases[base].chars_per_bit_exactly);
    mp = TMP_ALLOC_LIMBS (ma / GMP_NUMB_BITS + 2);
    mn = mpn_set_str (mp, (unsigned char *) begs, str_size, base);

    if (mn == 0)
      {
	SIZ(x) = 0;
	EXP(x) = 0;
	TMP_FREE;
	return 0;
      }

    madj = 0;
    /* Ignore excess limbs in MP,MSIZE.  */
    if (mn > prec)
      {
	madj = mn - prec;
	mp += mn - prec;
	mn = prec;
      }

    exp_in_base = 0;
    if (expptr != 0)
    {   char sgn = '+';
        int digit = 0, cnt = -1;
        
        if(*expptr == '+' || *expptr == '-')
            sgn = *expptr++;

        do
        {
            exp_in_base = exp_in_base * exp_base + digit;
            digit = digit_value[*(unsigned char*)expptr++];
            cnt++;
        }
        while
            (digit < exp_base);

        if(!cnt)
        {
            TMP_FREE;
            return -1;
        }

        if(sgn == '-')
            exp_in_base = -exp_in_base;
    }

    if (dotpos != 0)
      exp_in_base -= s - dotpos;
    divflag = exp_in_base < 0;
    exp_in_base = ABS (exp_in_base);

    if (exp_in_base == 0)
      {
	MPN_COPY (PTR(x), mp, mn);
	SIZ(x) = negative ? -mn : mn;
	EXP(x) = mn + madj;
	TMP_FREE;
	return 0;
      }

    ra = 2 * (prec + 1);
    rp = TMP_ALLOC_LIMBS (ra);
    tp = TMP_ALLOC_LIMBS (ra);
    rn = mpn_pow_1_highpart (rp, &radj, (mp_limb_t) base, exp_in_base, prec, tp);

    if (divflag)
      {
#if 0
	/* FIXME: Should use mpn_tdiv here.  */
	mpn_tdiv_qr (qp, mp, 0L, mp, mn, rp, rn);
#else
	mp_ptr qp;
	mp_limb_t qlimb;
	if (mn < rn)
	  {
	    /* Pad out MP,MSIZE for current divrem semantics.  */
	    mp_ptr tmp = TMP_ALLOC_LIMBS (rn + 1);
	    MPN_ZERO (tmp, rn - mn);
	    MPN_COPY (tmp + rn - mn, mp, mn);
	    mp = tmp;
	    madj -= rn - mn;
	    mn = rn;
	  }
	if ((rp[rn - 1] & GMP_NUMB_HIGHBIT) == 0)
	  {
	    mp_limb_t cy;
	    count_leading_zeros (cnt, rp[rn - 1]);
	    cnt -= GMP_NAIL_BITS;
	    mpn_lshift (rp, rp, rn, cnt);
	    cy = mpn_lshift (mp, mp, mn, cnt);
	    if (cy)
	      mp[mn++] = cy;
	  }

	qp = TMP_ALLOC_LIMBS (prec + 1);
	qlimb = mpn_intdivrem (qp, prec - (mn - rn), mp, mn, rp, rn);
	tp = qp;
	exp_in_limbs = qlimb + (mn - rn) + (madj - radj);
	rn = prec;
	if (qlimb != 0)
	  {
	    tp[prec] = qlimb;
	    /* Skip the least significant limb not to overrun the destination
	       variable.  */
	    tp++;
	  }
#endif
      }
    else
      {
	tp = TMP_ALLOC_LIMBS (rn + mn);
	if (rn > mn)
	  mpn_mul (tp, rp, rn, mp, mn);
	else
	  mpn_mul (tp, mp, mn, rp, rn);
	rn += mn;
	rn -= tp[rn - 1] == 0;
	exp_in_limbs = rn + madj + radj;

	if (rn > prec)
	  {
	    tp += rn - prec;
	    rn = prec;
	    exp_in_limbs += 0;
	  }
      }

    MPN_COPY (PTR(x), tp, rn);
    SIZ(x) = negative ? -rn : rn;
    EXP(x) = exp_in_limbs;
    TMP_FREE;
    return 0;
  }
}
