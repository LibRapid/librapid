/* _mpq_cmp_si -- compare mpq and long/ulong fraction.

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

#include "mpir.h"
#include "gmp-impl.h"


/* Something like mpq_cmpabs_ui would be more useful for the neg/neg case,
   and perhaps a version accepting a parameter to reverse the test, to make
   it a tail call here.  */

int
_mpq_cmp_si (mpq_srcptr q, mpir_si n, mpir_ui d)
{
  /* need canonical sign to get right result */
  ASSERT (q->_mp_den._mp_size > 0);

  if (q->_mp_num._mp_size >= 0)
    {
      if (n >= 0)
        return _mpq_cmp_ui (q, n, d);            /* >=0 cmp >=0 */
      else
        return 1;                                /* >=0 cmp <0 */
    }
  else
    {
      if (n >= 0)
        return -1;                               /* <0 cmp >=0 */
      else
        {
          mpq_t  qabs;
          qabs->_mp_num._mp_size = ABS (q->_mp_num._mp_size);
          qabs->_mp_num._mp_d    = q->_mp_num._mp_d;
          qabs->_mp_den._mp_size = q->_mp_den._mp_size;
          qabs->_mp_den._mp_d    = q->_mp_den._mp_d;

          return - _mpq_cmp_ui (qabs, -n, d);    /* <0 cmp <0 */
        }
    }
}
