dnl  SPARC v9 mpn_lshift

dnl  Copyright 1996, 2000, 2001, 2002, 2003 Free Software Foundation, Inc.

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

C		   cycles/limb
C UltraSPARC 1&2:     2
C UltraSPARC 3:	      3.25

C INPUT PARAMETERS
define(`rp',`%i0')
define(`up',`%i1')
define(`n',`%i2')
define(`cnt',`%i3')

define(`u0',`%l0')
define(`u1',`%l2')
define(`u2',`%l4')
define(`u3',`%l6')

define(`tnc',`%i4')

define(`fanop',`fitod %f0,%f2')		dnl  A quasi nop running in the FA pipe
define(`fmnop',`fmuld %f0,%f0,%f4')	dnl  A quasi nop running in the FM pipe

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_lshift)
	save	%sp,-160,%sp

	sllx	n,3,%g1
	sub	%g0,cnt,tnc		C negate shift count
	add	up,%g1,up		C make %o1 point at end of src
	add	rp,%g1,rp		C make %o0 point at end of res
	ldx	[up-8],u3		C load first limb
	subcc	n,5,n
	srlx	u3,tnc,%i5		C compute function result
	sllx	u3,cnt,%g3
	bl,pn	%icc,.Lend1234
	fanop

	subcc	n,4,n
	ldx	[up-16],u0
	ldx	[up-24],u1
	add	up,-32,up
	ldx	[up-0],u2
	ldx	[up-8],u3
	srlx	u0,tnc,%g2

	bl,pn	%icc,.Lend5678
	fanop

	b,a	.Loop
	.align	16
.Loop:
	sllx	u0,cnt,%g1
	or	%g3,%g2,%g3
	ldx	[up-16],u0
	fanop
C --
	srlx	u1,tnc,%g2
	subcc	n,4,n
	stx	%g3,[rp-8]
	fanop
C --
	sllx	u1,cnt,%g3
	or	%g1,%g2,%g1
	ldx	[up-24],u1
	fanop
C --
	srlx	u2,tnc,%g2
	stx	%g1,[rp-16]
	add	up,-32,up
	fanop
C --
	sllx	u2,cnt,%g1
	or	%g3,%g2,%g3
	ldx	[up-0],u2
	fanop
C --
	srlx	u3,tnc,%g2
	stx	%g3,[rp-24]
	add	rp,-32,rp
	fanop
C --
	sllx	u3,cnt,%g3
	or	%g1,%g2,%g1
	ldx	[up-8],u3
	fanop
C --
	srlx	u0,tnc,%g2
	stx	%g1,[rp-0]
	bge,pt	%icc,.Loop
	fanop
C --
.Lend5678:
	sllx	u0,cnt,%g1
	or	%g3,%g2,%g3
	srlx	u1,tnc,%g2
	stx	%g3,[rp-8]
	sllx	u1,cnt,%g3
	or	%g1,%g2,%g1
	srlx	u2,tnc,%g2
	stx	%g1,[rp-16]
	sllx	u2,cnt,%g1
	or	%g3,%g2,%g3
	srlx	u3,tnc,%g2
	stx	%g3,[rp-24]
	add	rp,-32,rp
	sllx	u3,cnt,%g3		C carry...
	or	%g1,%g2,%g1
	stx	%g1,[rp-0]

.Lend1234:
	addcc	n,4,n
	bz,pn	%icc,.Lret
	fanop
.Loop0:
	add	rp,-8,rp
	subcc	n,1,n
	ldx	[up-16],u3
	add	up,-8,up
	srlx	u3,tnc,%g2
	or	%g3,%g2,%g3
	stx	%g3,[rp]
	sllx	u3,cnt,%g3
	bnz,pt	%icc,.Loop0
	fanop
.Lret:
	stx	%g3,[rp-8]
	mov	%i5,%i0
	ret
	restore
EPILOGUE(mpn_lshift)
