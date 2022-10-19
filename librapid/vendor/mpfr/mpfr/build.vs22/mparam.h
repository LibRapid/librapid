/* Various Thresholds of MPFR, not exported.  -*- mode: C -*-

Copyright 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012 Free Software Foundation, Inc.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#ifndef __MPFR_IMPL_H__
# error "MPFR Internal not included"
#endif

// Check for 32bit vs 64bit
// Check windows
#if _WIN32 || _WIN64
#	if _WIN64
#		define LIBRAPID_64BIT
#	else
#		define LIBRAPID_32BIT
#	endif
#endif

// Check GCC
#if __GNUC__
#	if __x86_64__ || __ppc64__
#		define LIBRAPID_64BIT
#	else
#		define LIBRAPID_32BIT
#	endif
#endif

#if defined( LIBRAPID_64BIT )

#  if 0
#    include "../src/x86_64/pentium4/mparam.h"
#    define MPFR_TUNE_CASE "src/x86_64/pentium4/mparam.h"
#  elif 0
#    include "../src/x86_64/core2/mparam.h"
#    define MPFR_TUNE_CASE "src/x86_64/core2/mparam.h"
#  elif 1
#    include "../src/x86_64/mparam.h"
#    define MPFR_TUNE_CASE "src/x86_64/mparam.h"
#  elif 0
#    include "../src/amd/mparam.h"
#    define MPFR_TUNE_CASE "src/amd/mparam.h"
#  elif 0
#    include "../src/amdmparam.h"
#    define MPFR_TUNE_CASE "src/amdmparam.h"
#  elif 0
#    include "../src/amd/mparam.h"
#    define MPFR_TUNE_CASE "src/amd/mparam.h"
#  else
#  endif

#elif defined( LIBRAPID_32BIT )

#  if 0
#    include "../src/x86/core2/mparam.h"
#    define MPFR_TUNE_CASE "src/x86/core2/mparam.h"
#  elif 1
#    include "../src/x86/mparam.h"
#    define MPFR_TUNE_CASE "src/x86/mparam.h"
#  else
#  endif

#else
#  define MPFR_TUNE_CASE "default"
#endif

/****************************************************************
 * Default values of Threshold.                                 *
 * Must be included in any case: it checks, for every constant, *
 * if it has been defined, and it sets it to a default value if *
 * it was not previously defined.                               *
 ****************************************************************/
#include "../src/generic/mparam.h"
#include "mpfr_endian.h"
