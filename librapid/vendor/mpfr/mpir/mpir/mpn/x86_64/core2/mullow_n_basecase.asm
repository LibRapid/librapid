dnl  AMD64 mpn_mullo_basecase optimised for Conroe/Wolfdale/Nehalem/Westmere.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2008, 2009, 2011-2013 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of either:
dnl
dnl    * the GNU Lesser General Public License as published by the Free
dnl      Software Foundation; either version 3 of the License, or (at your
dnl      option) any later version.
dnl
dnl  or
dnl
dnl    * the GNU General Public License as published by the Free Software
dnl      Foundation; either version 2 of the License, or (at your option) any
dnl      later version.
dnl
dnl  or both in parallel, as here.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
dnl  for more details.
dnl
dnl  You should have received copies of the GNU General Public License and the
dnl  GNU Lesser General Public License along with the GNU MP Library.  If not,
dnl  see https://www.gnu.org/licenses/.

include(`../config.m4')

C cycles/limb	mul_2		addmul_2
C AMD K8,K9
C AMD K10
C AMD bull
C AMD pile
C AMD steam
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel core	 4.0		4.18-4.25
C Intel NHM	 3.75		4.06-4.2
C Intel SBR
C Intel IBR
C Intel HWL
C Intel BWL
C Intel atom
C VIA nano

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C   * Implement proper cor2, replacing current cor0.
C   * Offset n by 2 in order to avoid the outer loop cmp.  (And sqr_basecase?)
C   * Micro-optimise.

C When playing with pointers, set this to $2 to fall back to conservative
C indexing in wind-down code.
define(`I',`$1')

define(`rp',       `%rdi')
define(`up',       `%rsi')
define(`vp_param', `%rdx')
define(`n_param',  `%rcx')
define(`n_param8',  `%cl')

define(`v0',       `%r10')
define(`v1',       `%r11')
define(`w0',       `%rbx')
define(`w032',       `%ebx')
define(`w1',       `%rcx')
define(`w132',       `%ecx')
define(`w2',       `%rbp')
define(`w232',       `%ebp')
define(`w3',       `%r12')
define(`w332',       `%r12d')
define(`n',        `%r9')
define(`n32',        `%r9d')
define(`n8',        `%r9b')
define(`i',        `%r13')
define(`vp',       `%r8')

define(`X0',       `%r14')
define(`X1',       `%r15')

C rax rbx rcx rdx rdi rsi rbp r8 r9 r10 r11 r12 r13 r14 r15

define(`ALIGNx', `ALIGN(16)')

define(`N', 85)
ifdef(`N',,`define(`N',0)')
define(`MOV', `ifelse(eval(N & $3),0,`mov	$1, $2',`lea	($1), $2')')

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mullow_n_basecase)

	mov	(up), %rax
	mov	vp_param, vp

	cmp	$4, n_param
	jb	L(lsmall)

	mov	(vp_param), v0
	push	%rbx
	lea	(rp,n_param,8), rp	C point rp at R[un]
	push	%rbp
	lea	(up,n_param,8), up	C point up right after U's end
	push	%r12
	mov	$0, n32		C FIXME
	sub	n_param, n
	push	%r13
	mul	v0
	mov	8(vp), v1

	test	$1, n_param8
	jnz	L(lm2x1)

L(lm2x0):test	$2, n_param8
	jnz	L(lm2b2)

L(lm2b0):lea	(n), i
	mov	%rax, (rp,n,8)
	mov	%rdx, w1
	mov	(up,n,8), %rax
	xor	w232, w232
	jmp	L(lm2e0)

L(lm2b2):lea	-2(n), i
	mov	%rax, w2
	mov	(up,n,8), %rax
	mov	%rdx, w3
	xor	w032, w032
	jmp	L(lm2e2)

L(lm2x1):test	$2, n_param8
	jnz	L(lm2b3)

L(lm2b1):lea	1(n), i
	mov	%rax, (rp,n,8)
	mov	(up,n,8), %rax
	mov	%rdx, w0
	xor	w132, w132
	jmp	L(lm2e1)

L(lm2b3):lea	-1(n), i
	xor	w332, w332
	mov	%rax, w1
	mov	%rdx, w2
	mov	(up,n,8), %rax
	jmp	L(lm2e3)

	ALIGNx
L(lm2tp):mul	v0
	add	%rax, w3
	mov	-8(up,i,8), %rax
	mov	w3, -8(rp,i,8)
	adc	%rdx, w0
	adc	$0, w132
L(lm2e1):mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	$0, w232
	mov	(up,i,8), %rax
	mul	v0
	add	%rax, w0
	mov	w0, (rp,i,8)
	adc	%rdx, w1
	mov	(up,i,8), %rax
	adc	$0, w232
L(lm2e0):mul	v1
	add	%rax, w1
	adc	%rdx, w2
	mov	8(up,i,8), %rax
	mul	v0
	mov	$0, w332
	add	%rax, w1
	adc	%rdx, w2
	adc	$0, w332
	mov	8(up,i,8), %rax
L(lm2e3):mul	v1
	add	%rax, w2
	mov	w1, 8(rp,i,8)
	adc	%rdx, w3
	mov	$0, w032
	mov	16(up,i,8), %rax
	mul	v0
	add	%rax, w2
	mov	16(up,i,8), %rax
	adc	%rdx, w3
	adc	$0, w032
L(lm2e2):mul	v1
	mov	$0, w132		C FIXME: dead in last iteration
	add	%rax, w3
	mov	24(up,i,8), %rax
	mov	w2, 16(rp,i,8)
	adc	%rdx, w0		C FIXME: dead in last iteration
	add	$4, i
	js	L(lm2tp)

L(lm2ed):imul	v0, %rax
	add	w3, %rax
	mov	%rax, I(-8(rp),-8(rp,i,8))

	add	$2, n
	lea	16(vp), vp
	lea	-16(up), up
	cmp	$-2, n
	jge	L(lcor1)

	push	%r14
	push	%r15

L(louter):
	mov	(vp), v0
	mov	8(vp), v1
	mov	(up,n,8), %rax
	mul	v0
	test	$1, n8
	jnz	L(la1x1)

L(la1x0):mov	%rax, X1
	MOV(	%rdx, X0, 8)
	mov	(up,n,8), %rax
	mul	v1
	test	$2, n8
	jnz	L(la110)

L(la100):lea	(n), i
	mov	(rp,n,8), w3
	mov	%rax, w0
	MOV(	%rdx, w1, 16)
	jmp	L(llo0)

L(la110):lea	2(n), i
	mov	(rp,n,8), w1
	mov	%rax, w2
	mov	8(up,n,8), %rax
	MOV(	%rdx, w3, 1)
	jmp	L(llo2)

L(la1x1):mov	%rax, X0
	MOV(	%rdx, X1, 2)
	mov	(up,n,8), %rax
	mul	v1
	test	$2, n8
	jz	L(la111)

L(la101):lea	1(n), i
	MOV(	%rdx, w0, 4)
	mov	(rp,n,8), w2
	mov	%rax, w3
	jmp	L(llo1)

L(la111):lea	-1(n), i
	MOV(	%rdx, w2, 64)
	mov	%rax, w1
	mov	(rp,n,8), w0
	mov	8(up,n,8), %rax
	jmp	L(llo3)

	ALIGNx
L(ltop):	mul	v1
	add	w0, w1
	adc	%rax, w2
	mov	-8(up,i,8), %rax
	MOV(	%rdx, w3, 1)
	adc	$0, w3
L(llo2):	mul	v0
	add	w1, X1
	mov	X1, -16(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 2)
	adc	$0, X1
	mov	-8(up,i,8), %rax
	mul	v1
	MOV(	%rdx, w0, 4)
	mov	-8(rp,i,8), w1
	add	w1, w2
	adc	%rax, w3
	adc	$0, w0
L(llo1):	mov	(up,i,8), %rax
	mul	v0
	add	w2, X0
	adc	%rax, X1
	mov	X0, -8(rp,i,8)
	MOV(	%rdx, X0, 8)
	adc	$0, X0
	mov	(up,i,8), %rax
	mov	(rp,i,8), w2
	mul	v1
	add	w2, w3
	adc	%rax, w0
	MOV(	%rdx, w1, 16)
	adc	$0, w1
L(llo0):	mov	8(up,i,8), %rax
	mul	v0
	add	w3, X1
	mov	X1, (rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	8(rp,i,8), w3
	adc	$0, X1
	mov	8(up,i,8), %rax
	mul	v1
	add	w3, w0
	MOV(	%rdx, w2, 64)
	adc	%rax, w1
	mov	16(up,i,8), %rax
	adc	$0, w2
L(llo3):	mul	v0
	add	w0, X0
	mov	X0, 8(rp,i,8)
	MOV(	%rdx, X0, 128)
	adc	%rax, X1
	mov	16(up,i,8), %rax
	mov	16(rp,i,8), w0
	adc	$0, X0
	add	$4, i
	jnc	L(ltop)

L(lend):	imul	v1, %rax
	add	w0, w1
	adc	%rax, w2
	mov	I(-8(up),-8(up,i,8)), %rax
	imul	v0, %rax
	add	w1, X1
	mov	X1, I(-16(rp),-16(rp,i,8))
	adc	X0, %rax
	mov	I(-8(rp),-8(rp,i,8)), w1
	add	w1, w2
	add	w2, %rax
	mov	%rax, I(-8(rp),-8(rp,i,8))

	add	$2, n
	lea	16(vp), vp
	lea	-16(up), up
	cmp	$-2, n
	jl	L(louter)

	pop	%r15
	pop	%r14

	jnz	L(lcor0)

L(lcor1):mov	(vp), v0
	mov	8(vp), v1
	mov	-16(up), %rax
	mul	v0			C u0 x v2
	add	-16(rp), %rax		C FIXME: rp[0] still available in reg?
	adc	-8(rp), %rdx		C FIXME: rp[1] still available in reg?
	mov	-8(up), %rbx
	imul	v0, %rbx
	mov	-16(up), %rcx
	imul	v1, %rcx
	mov	%rax, -16(rp)
	add	%rbx, %rcx
	add	%rdx, %rcx
	mov	%rcx, -8(rp)
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret

L(lcor0):mov	(vp), %r11
	imul	-8(up), %r11
	add	%rax, %r11
	mov	%r11, -8(rp)
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret

	ALIGN(16)
L(lsmall):
	cmp	$2, n_param
	jae	L(lgt1)
L(ln1):	imul	(vp_param), %rax
	mov	%rax, (rp)
	ret
L(lgt1):	ja	L(lgt2)
L(ln2):	mov	(vp_param), %r9
	mul	%r9
	mov	%rax, (rp)
	mov	8(up), %rax
	imul	%r9, %rax
	add	%rax, %rdx
	mov	8(vp), %r9
	mov	(up), %rcx
	imul	%r9, %rcx
	add	%rcx, %rdx
	mov	%rdx, 8(rp)
	ret
L(lgt2):
L(ln3):	mov	(vp_param), %r9
	mul	%r9		C u0 x v0
	mov	%rax, (rp)
	mov	%rdx, %r10
	mov	8(up), %rax
	mul	%r9		C u1 x v0
	imul	16(up), %r9	C u2 x v0
	add	%rax, %r10
	adc	%rdx, %r9
	mov	8(vp), %r11
	mov	(up), %rax
	mul	%r11		C u0 x v1
	add	%rax, %r10
	adc	%rdx, %r9
	imul	8(up), %r11	C u1 x v1
	add	%r11, %r9
	mov	%r10, 8(rp)
	mov	16(vp), %r10
	mov	(up), %rax
	imul	%rax, %r10	C u0 x v2
	add	%r10, %r9
	mov	%r9, 16(rp)
	ret
EPILOGUE()
