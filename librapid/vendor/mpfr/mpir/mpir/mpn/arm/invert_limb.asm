dnl  ARM mpn_invert_limb -- Invert a normalized limb.

dnl  Copyright 2001 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
define(`d',`r0')	C number to be inverted


PROLOGUE(mpn_invert_limb)
	stmfd	sp!, {r4, lr}
	mov	r3, d, lsr #23
	sub	r3, r3, #256
	add	r2, pc, #invtab-.-8
	mov	r3, r3, lsl #1
	ldrh	r1, [r2, r3]		C get initial approximation from table
	mov	r2, r1, lsl #6		C start iteration 1
	mul	ip, r2, r2
	umull	lr, r4, ip, d
	mov	r2, r4, lsl #1
	rsb	r2, r2, r1, lsl #23	C iteration 1 complete
	umull	ip, r3, r2, r2		C start iteration 2
	umull	lr, r4, r3, d
	umull	r3, r1, ip, d
	adds	lr, lr, r1
	addcs	r4, r4, #1
	mov	r3, lr, lsr #30
	orr	r4, r3, r4, lsl #2
	mov	lr, lr, lsl #2
	cmn	lr, #1
	rsc	r2, r4, r2, lsl #2	C iteration 2 complete
	umull	ip, r1, d, r2		C start adjustment step
	add	r1, r1, d
	cmn	r1, #1
	beq	L(1)
	adds	ip, ip, d
	adc	r1, r1, #0
	add	r2, r2, #1
L(1):
	adds	r3, ip, d
	adcs	r1, r1, #0
	moveq	r0, r2
	addne	r0, r2, #1
	ldmfd	sp!, {r4, pc}

invtab:
	.short	1023,1020,1016,1012,1008,1004,1000,996
	.short	992,989,985,981,978,974,970,967
	.short	963,960,956,953,949,946,942,939
	.short	936,932,929,926,923,919,916,913
	.short	910,907,903,900,897,894,891,888
	.short	885,882,879,876,873,870,868,865
	.short	862,859,856,853,851,848,845,842
	.short	840,837,834,832,829,826,824,821
	.short	819,816,814,811,809,806,804,801
	.short	799,796,794,791,789,787,784,782
	.short	780,777,775,773,771,768,766,764
	.short	762,759,757,755,753,751,748,746
	.short	744,742,740,738,736,734,732,730
	.short	728,726,724,722,720,718,716,714
	.short	712,710,708,706,704,702,700,699
	.short	697,695,693,691,689,688,686,684
	.short	682,680,679,677,675,673,672,670
	.short	668,667,665,663,661,660,658,657
	.short	655,653,652,650,648,647,645,644
	.short	642,640,639,637,636,634,633,631
	.short	630,628,627,625,624,622,621,619
	.short	618,616,615,613,612,611,609,608
	.short	606,605,604,602,601,599,598,597
	.short	595,594,593,591,590,589,587,586
	.short	585,583,582,581,579,578,577,576
	.short	574,573,572,571,569,568,567,566
	.short	564,563,562,561,560,558,557,556
	.short	555,554,553,551,550,549,548,547
	.short	546,544,543,542,541,540,539,538
	.short	537,536,534,533,532,531,530,529
	.short	528,527,526,525,524,523,522,521
	.short	520,519,518,517,516,515,514,513
EPILOGUE(mpn_invert_limb)
