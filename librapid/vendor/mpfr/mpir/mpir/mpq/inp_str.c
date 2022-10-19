/* mpq_inp_str -- read an mpq from a FILE.

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
#include <ctype.h>
#include "mpir.h"
#include "gmp-impl.h"


size_t
mpq_inp_str (mpq_ptr q, FILE *fp, int base)
{
  size_t  nread;
  int     c;

  if (fp == NULL)
    fp = stdin;

  q->_mp_den._mp_size = 1;
  q->_mp_den._mp_d[0] = 1;

  nread = mpz_inp_str (mpq_numref(q), fp, base);
  if (nread == 0)
    return 0;

  c = getc (fp);
  nread++;

  if (c == '/')
    {
      c = getc (fp);
      nread++;

      nread = mpz_inp_str_nowhite (mpq_denref(q), fp, base, c, nread);
      if (nread == 0)
        {
          q->_mp_num._mp_size = 0;
          q->_mp_den._mp_size = 1;
          q->_mp_den._mp_d[0] = 1;
        }
    }
  else
    {
      ungetc (c, fp);
      nread--;
    }

  return nread;
}
