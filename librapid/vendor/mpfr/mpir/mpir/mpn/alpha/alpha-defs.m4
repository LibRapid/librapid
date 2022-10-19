divert(-1)

dnl  m4 macros for Alpha assembler.

dnl  Copyright 2003, 2004 Free Software Foundation, Inc.
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


dnl  Usage: ASSERT([reg] [,code])
dnl
dnl  Require that the given reg is non-zero after executing the test code.
dnl  For example,
dnl
dnl         ASSERT(r8,
dnl         `       cmpult r16, r17, r8')
dnl
dnl  If the register argument is empty then nothing is tested, the code is
dnl  just executed.  This can be used for setups required by later ASSERTs.
dnl  If the code argument is omitted then the register is just tested, with
dnl  no special setup code.

define(ASSERT,
m4_assert_numargs_range(1,2)
m4_assert_defined(`WANT_ASSERT')
`ifelse(WANT_ASSERT,1,
`ifelse(`$2',,,`$2')
ifelse(`$1',,,
`	bne	$1, L(ASSERTok`'ASSERT_label_counter)
	.long	0	C halt
L(ASSERTok`'ASSERT_label_counter):
define(`ASSERT_label_counter',eval(ASSERT_label_counter+1))
')
')')
define(`ASSERT_label_counter',1)


dnl  Usage: bigend(`code')
dnl
dnl  Emit the given code only for a big-endian system, like Unicos.  This
dnl  can be used for instance for extra stuff needed by extwl.

define(bigend,
m4_assert_numargs(1)
`ifdef(`HAVE_LIMB_BIG_ENDIAN',`$1',
`ifdef(`HAVE_LIMB_LITTLE_ENDIAN',`',
`m4_error(`Cannot assemble, unknown limb endianness')')')')

dnl  Usage: unop
dnl
dnl  The Cray Unicos assembler lacks unop, so give the equivalent ldq_u
dnl  explicitly.

define(unop,
m4_assert_numargs(-1)
`ldq_u	r31, 0(r30)')


divert
