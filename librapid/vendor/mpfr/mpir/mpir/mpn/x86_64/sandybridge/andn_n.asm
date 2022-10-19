dnl  mpn_andn_n

dnl  Copyright 2009 Jason Moxham

dnl  This file is part of the MPIR Library.

dnl  The MPIR Library is free software; you can redistribute it and/or modify
dnl  it under the terms of the GNU Lesser General Public License as published
dnl  by the Free Software Foundation; either version 2.1 of the License, or (at
dnl  your option) any later version.

dnl  The MPIR Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
dnl  License for more details.

dnl  You should have received a copy of the GNU Lesser General Public License
dnl  along with the MPIR Library; see the file COPYING.LIB.  If not, write
dnl  to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
dnl  Boston, MA 02110-1301, USA.

include(`../config.m4')

C	ret mpn_andn_n(mp_ptr,mp_srcptr,mp_srcptr,mp_size_t)
C	rax             rdi,   rsi,      rdx,     rcx

ASM_START()
PROLOGUE(mpn_andn_n)
mov $3,%r8
lea -24(%rsi,%rcx,8),%rsi
lea -24(%rdx,%rcx,8),%rdx
lea -24(%rdi,%rcx,8),%rdi
sub %rcx,%r8
jnc L(skiplp)
ALIGN(16)
L(lp):
	movdqu (%rdx,%r8,8),%xmm0
	movdqu 16(%rdx,%r8,8),%xmm1
	movdqu 16(%rsi,%r8,8),%xmm3
	movdqu (%rsi,%r8,8),%xmm2
	pandn %xmm2,%xmm0
	movdqu %xmm0,(%rdi,%r8,8)
	pandn %xmm3,%xmm1
	add $4,%r8
	movdqu %xmm1,16-32(%rdi,%r8,8)
	jnc L(lp)
L(skiplp):
cmp $2,%r8
ja L(case0)
je L(case1)
jp L(case2)	
L(case3):	movdqu (%rdx,%r8,8),%xmm0
	mov 16(%rdx,%r8,8),%rax
	mov 16(%rsi,%r8,8),%rcx
	movdqu (%rsi,%r8,8),%xmm2
	pandn %xmm2,%xmm0
	movdqu %xmm0,(%rdi,%r8,8)
	not %rax
	and %rcx,%rax
	mov %rax,16(%rdi,%r8,8)
L(case0):	ret
L(case2):	movdqu (%rdx,%r8,8),%xmm0
	movdqu (%rsi,%r8,8),%xmm2
	pandn %xmm2,%xmm0
	movdqu %xmm0,(%rdi,%r8,8)
	ret
L(case1):	mov (%rdx,%r8,8),%rax
	mov (%rsi,%r8,8),%rcx
	not %rax
	and %rcx,%rax
	mov %rax,(%rdi,%r8,8)
	ret
EPILOGUE()
