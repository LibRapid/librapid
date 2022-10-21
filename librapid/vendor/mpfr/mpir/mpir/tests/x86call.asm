dnl  x86 calling conventions checking.

dnl  Copyright 2000, 2003 Free Software Foundation, Inc.
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


include(`../config.m4')


dnl  Instrumented profiling doesn't come out quite right below, since we
dnl  don't do an actual "ret".  There's only a few instructions here, so
dnl  there's no great need to get them separately accounted, just let them
dnl  get attributed to the caller.

ifelse(WANT_PROFILING,instrument,
`define(`WANT_PROFILING',no)')


C int calling_conventions (...);
C
C The global variable "calling_conventions_function" is the function to
C call, with the arguments as passed here.
C
C Perhaps the finit should be done only if the tags word isn't clear, but
C nothing uses the rounding mode or anything at the moment.

define(G,
m4_assert_numargs(1)
`GSYM_PREFIX`'$1')

	.text
	ALIGN(8)
PROLOGUE(calling_conventions)
	movl	(%esp), %eax
	movl	%eax, G(calling_conventions_retaddr)

	movl	$L(return), (%esp)

	movl	%ebx, G(calling_conventions_save_ebx)
	movl	%esi, G(calling_conventions_save_esi)
	movl	%edi, G(calling_conventions_save_edi)
	movl	%ebp, G(calling_conventions_save_ebp)

	movl	$0x01234567, %ebx
	movl	$0x89ABCDEF, %esi
	movl	$0xFEDCBA98, %edi
	movl	$0x76543210, %ebp

	C try to provoke a problem by starting with junk in the registers,
	C especially in %eax and %edx which will be return values
	movl	$0x70246135, %eax
	movl	$0x8ACE9BDF, %ecx
	movl	$0xFDB97531, %edx

	jmp	*G(calling_conventions_function)

L(return):
	movl	%ebx, G(calling_conventions_ebx)
	movl	%esi, G(calling_conventions_esi)
	movl	%edi, G(calling_conventions_edi)
	movl	%ebp, G(calling_conventions_ebp)

	pushf
	popl	%ebx
	movl	%ebx, G(calling_conventions_eflags)

	fstenv	G(calling_conventions_fenv)
	finit

	movl	G(calling_conventions_save_ebx), %ebx
	movl	G(calling_conventions_save_esi), %esi
	movl	G(calling_conventions_save_edi), %edi
	movl	G(calling_conventions_save_ebp), %ebp

	jmp	*G(calling_conventions_retaddr)

EPILOGUE()

C void gmp_x86check_workaround_apple_ld_bug() 
C 
C Apple ld has an annoying bug that causes it to only load members from 
C static archives that satisfy text symbol dependencies.  This procedure 
C creates a bogus dependency on a text symbol in x86check.o (in libtests.a) 
C to ensure that ld loads it, also making all of the needed non-text 
C symbols available. 
PROLOGUE(gmp_x86check_workaround_apple_ld_bug) 
       jmp     *G(calling_conventions_check) 
EPILOGUE() 
