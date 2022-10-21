/* mpz_inp_str(dest_integer, stream, base) -- Input a number in base
   BASE from stdio stream STREAM and store the result in DEST_INTEGER.

   OF THE FUNCTIONS IN THIS FILE, ONLY mpz_inp_str IS FOR EXTERNAL USE, THE
   REST ARE INTERNALS AND ARE ALMOST CERTAIN TO BE SUBJECT TO INCOMPATIBLE
   CHANGES OR DISAPPEAR COMPLETELY IN FUTURE GNU MP RELEASES.

Copyright 1991, 1993, 1994, 1996, 1998, 2000, 2001, 2002, 2003 Free Software
Foundation, Inc.

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
#include <ctype.h>
#include "mpir.h"
#include "gmp-impl.h"

extern const unsigned char __gmp_digit_value_tab[];
#define digit_value_tab __gmp_digit_value_tab

/* shared by mpq_inp_str */
size_t
mpz_inp_str_nowhite (mpz_ptr x, FILE *stream, int base, int c, size_t nread)
{
  char *str;
  size_t alloc_size, str_size;
  int negative;
  mp_size_t xsize;
  const unsigned char *digit_value;

  ASSERT_ALWAYS (EOF == -1);	/* FIXME: handle this by adding explicit */
         			/* comparisons of c and EOF before each  */
				/* read of digit_value[].  */

  digit_value = digit_value_tab;
  if (base > 36)
    {
      /* For bases > 36, use the collating sequence
	 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.  */
      digit_value += 224;
      if (base > 62)
	return 0;		/* too large base */
    }

  negative = 0;
  if (c == '-')
    {
      negative = 1;
      c = getc (stream);
      nread++;
    }

  if (c == EOF || digit_value[c] >= (base == 0 ? 10 : base))
    return 0;			/* error if no digits */

  /* If BASE is 0, try to find out the base by looking at the initial
     characters.  */
  if (base == 0)
    {
      base = 10;
      if (c == '0')
	{
	  base = 8;
	  c = getc (stream);
	  nread++;
	  if (c == 'x' || c == 'X')
	    {
	      base = 16;
	      c = getc (stream);
	      nread++;
	    }
	  else if (c == 'b' || c == 'B')
	    {
	      base = 2;
	      c = getc (stream);
	      nread++;
	    }
	}
    }

  /* Skip leading zeros.  */
  while (c == '0')
    {
      c = getc (stream);
      nread++;
    }

  alloc_size = 100;
  str = (char *) (*__gmp_allocate_func) (alloc_size);
  str_size = 0;

  while (c != EOF)
    {
      int dig;
      dig = digit_value[c];
      if (dig >= base)
	break;
      if (str_size >= alloc_size)
	{
	  size_t old_alloc_size = alloc_size;
	  alloc_size = alloc_size * 3 / 2;
	  str = (char *) (*__gmp_reallocate_func) (str, old_alloc_size, alloc_size);
	}
      str[str_size++] = dig;
      c = getc (stream);
    }
  nread += str_size;

  ungetc (c, stream);
  nread--;

  /* Make sure the string is not empty, mpn_set_str would fail.  */
  if (str_size == 0)
    {
      x->_mp_size = 0;
    }
  else
    {
      xsize = (((mp_size_t)
                (str_size / __mp_bases[base].chars_per_bit_exactly))
               / GMP_NUMB_BITS + 2);
      MPZ_REALLOC (x, xsize);

      /* Convert the byte array in base BASE to our bignum format.  */
      xsize = mpn_set_str (x->_mp_d, (unsigned char *) str, str_size, base);
      x->_mp_size = negative ? -xsize : xsize;
    }
  (*__gmp_free_func) (str, alloc_size);
  return nread;
}

size_t
mpz_inp_str (mpz_ptr x, FILE *stream, int base)
{
  int c;
  size_t nread;

  if (stream == 0)
    stream = stdin;

  nread = 0;

  /* Skip whitespace.  */
  do
    {
      c = getc (stream);
      nread++;
    }
  while (isspace (c));

  return mpz_inp_str_nowhite (x, stream, base, c, nread);
}
