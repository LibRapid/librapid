/* gmp_vsnprintf -- formatted output to an fixed size buffer.

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

#include "config.h"

#if HAVE_STDARG
#include <stdarg.h>
#else
#include <varargs.h>
#endif

#include <string.h>    /* for strlen */

#include "mpir.h"
#include "gmp-impl.h"


int
gmp_vsnprintf (char *buf, size_t size, const char *fmt, va_list ap)
{
  struct gmp_snprintf_t d;

  ASSERT (! MEM_OVERLAP_P (buf, size, fmt, strlen(fmt)+1));

  d.buf = buf;
  d.size = size;
  return __gmp_doprnt (&__gmp_snprintf_funs, &d, fmt, ap);
}
