/* mpz_sqrt(root, u) --  Set ROOT to floor(sqrt(U)).

Copyright 1991, 1993, 1994, 1996, 2000, 2001, 2005 Free Software Foundation,
Inc.

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

#include <stdio.h> /* for NULL */
#include "mpir.h"
#include "gmp-impl.h"

void
mpz_sqrt (mpz_ptr root, mpz_srcptr op)
{
  mp_size_t op_size, root_size;
  mp_ptr root_ptr, op_ptr;
  mp_ptr free_me = NULL;
  mp_size_t free_me_size;
  TMP_DECL;

  TMP_MARK;
  op_size = op->_mp_size;
  if (op_size <= 0)
    {
      if (op_size < 0)
        SQRT_OF_NEGATIVE;
      SIZ(root) = 0;
      return;
    }

  /* The size of the root is accurate after this simple calculation.  */
  root_size = (op_size + 1) / 2;

  root_ptr = root->_mp_d;
  op_ptr = op->_mp_d;

  if (root->_mp_alloc < root_size)
    {
      if (root_ptr == op_ptr)
	{
	  free_me = root_ptr;
	  free_me_size = root->_mp_alloc;
	}
      else
	(*__gmp_free_func) (root_ptr, root->_mp_alloc * BYTES_PER_MP_LIMB);

      root->_mp_alloc = root_size;
      root_ptr = (mp_ptr) (*__gmp_allocate_func) (root_size * BYTES_PER_MP_LIMB);
      root->_mp_d = root_ptr;
    }
  else
    {
      /* Make OP not overlap with ROOT.  */
      if (root_ptr == op_ptr)
	{
	  /* ROOT and OP are identical.  Allocate temporary space for OP.  */
	  op_ptr = (mp_ptr) TMP_ALLOC (op_size * BYTES_PER_MP_LIMB);
	  /* Copy to the temporary space.  Hack: Avoid temporary variable
	     by using ROOT_PTR.  */
	  MPN_COPY (op_ptr, root_ptr, op_size);
	}
    }

  mpn_sqrtrem (root_ptr, NULL, op_ptr, op_size);

  root->_mp_size = root_size;

  if (free_me != NULL)
    (*__gmp_free_func) (free_me, free_me_size * BYTES_PER_MP_LIMB);
  TMP_FREE;
}
