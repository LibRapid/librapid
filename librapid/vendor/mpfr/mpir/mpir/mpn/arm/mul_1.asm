dnl  ARM mpn_mul_1 -- Multiply a limb vector with a limb and store the result
dnl  in a second limb vector.
dnl  Contributed by Robert Harley.

dnl  Copyright 1998, 2000, 2001, 2003 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.

dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of the GNU Lesser General Public License as published
dnl  by the Free Software Foundation; either version 2.1 of the License, or (at
dnl  your option) any later version.

dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
dnl  License for more details.

dnl  You should have received a copy of the GNU Lesser General Public License
dnl  along with the GNU MP Library; see the file COPYING.LIB.  If not, write
dnl  to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
dnl  Boston, MA 02110-1301, USA.

include(`../config.m4')

C            cycles/limb
C StrongARM:     6-8  (dependent on vl value)
C XScale:        ?-?

C We should rewrite this along the lines of addmul_1.asm.  That should save a
C cycle on StrongARM, and several cycles on XScale.

define(`rp',`r0')
define(`up',`r1')
define(`n',`r2')
define(`vl',`r3')


ASM_START()
PROLOGUE(mpn_mul_1)
	stmfd	sp!, { r8, r9, lr }
	ands	r12, n, #1
	beq	L(skip1)
	ldr	lr, [up], #4
	umull	r9, r12, lr, vl
	str	r9, [rp], #4
L(skip1):
	tst	n, #2
	beq	L(skip2)
	mov	r8, r12
	ldmia	up!, { r12, lr }
	mov	r9, #0
	umlal	r8, r9, r12, vl
	mov	r12, #0
	umlal	r9, r12, lr, vl
	stmia	rp!, { r8, r9 }
L(skip2):
	bics	n, n, #3
	beq	L(return)
	stmfd	sp!, { r6, r7 }
L(loop):
	mov	r6, r12
	ldmia	up!, { r8, r9, r12, lr }
	ldr	r7, [rp, #12]			C cache allocate
	mov	r7, #0
	umlal	r6, r7, r8, vl
	mov	r8, #0
	umlal	r7, r8, r9, vl
	mov	r9, #0
	umlal	r8, r9, r12, vl
	mov	r12, #0
	umlal	r9, r12, lr, vl
	subs	n, n, #4
	stmia	rp!, { r6, r7, r8, r9 }
	bne	L(loop)
	ldmfd	sp!, { r6, r7 }
L(return):
	mov	r0, r12
	ldmfd	sp!, { r8, r9, pc }
EPILOGUE(mpn_mul_1)
