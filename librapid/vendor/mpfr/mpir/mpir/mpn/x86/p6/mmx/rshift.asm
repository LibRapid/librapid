dnl  Intel Pentium-II mpn_rshift -- mpn left shift.

dnl  Copyright 2001 Free Software Foundation, Inc.
dnl
dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or
dnl  modify it under the terms of the GNU Lesser General Public License as
dnl  published by the Free Software Foundation; either version 2.1 of the
dnl  License, or (at your option) any later version.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful,
dnl  but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
dnl  Lesser General Public License for more details.
dnl
dnl  You should have received a copy of the GNU Lesser General Public
dnl  License along with the GNU MP Library; see the file COPYING.LIB.  If
dnl  not, write to the Free Software Foundation, Inc., 51 Franklin Street,
dnl  Fifth Floor, Boston, MA 02110-1301, USA.


dnl  The P55 code runs well on P-II/III, but could stand some minor tweaks
dnl  at some stage probably.

include(`../config.m4')

include_mpn(`x86/pentium/mmx/rshift.asm')
