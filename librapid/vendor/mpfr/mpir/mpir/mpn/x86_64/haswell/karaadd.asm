dnl  mpn_karaadd

dnl  Copyright 2011 The Code Cavern

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

ASM_START()
PROLOGUE(mpn_karaadd)
# requires n>=8
push %rbx
push %rbp
push %r12
push %r13
push %r14
push %r15
#rp is rdi
#tp is rsi
#n is rdx and put it on the stack
push %rdx
shr $1,%rdx
#n2 is rdx
lea (%rdx,%rdx,1),%rcx
# 2*n2 is rcx
# L is rdi
# H is rbp
# tp is rsi
lea (%rdi,%rcx,8),%rbp
xor %rax,%rax
xor %rbx,%rbx
# rax rbx are the carrys
lea -24(%rdi,%rdx,8),%rdi
lea -24(%rsi,%rdx,8),%rsi
lea -24(%rbp,%rdx,8),%rbp
mov $3,%ecx
sub %rdx,%rcx
mov $3,%edx
.align 16
L(p):	bt $2,%rbx
	mov (%rdi,%rdx,8),%r8
	adc (%rbp,%rcx,8),%r8
	mov %r8,%r12
	mov 8(%rdi,%rdx,8),%r9
	adc 8(%rbp,%rcx,8),%r9
	mov 16(%rdi,%rdx,8),%r10
	adc 16(%rbp,%rcx,8),%r10
	mov 24(%rdi,%rdx,8),%r11
	adc 24(%rbp,%rcx,8),%r11
	adc %rbx,%rbx
	bt $1,%rax
	mov %r11,%r15
	adc (%rdi,%rcx,8),%r8
	mov %r9,%r13
	adc 8(%rdi,%rcx,8),%r9
	mov %r10,%r14
	adc 16(%rdi,%rcx,8),%r10
	adc 24(%rdi,%rcx,8),%r11
	adc %rax,%rax
	bt $2,%rbx
	adc (%rbp,%rdx,8),%r12
	adc 8(%rbp,%rdx,8),%r13
	adc 16(%rbp,%rdx,8),%r14
	adc 24(%rbp,%rdx,8),%r15
	adc %rbx,%rbx
	bt $1,%rax
	adc (%rsi,%rcx,8),%r8
	adc 8(%rsi,%rcx,8),%r9
	adc 16(%rsi,%rcx,8),%r10
	adc 24(%rsi,%rcx,8),%r11
	mov %r10,16(%rdi,%rdx,8)
	mov %r11,24(%rdi,%rdx,8)
	adc %rax,%rax
	bt $2,%rbx
	mov %r8,(%rdi,%rdx,8)
	mov %r9,8(%rdi,%rdx,8)
	adc (%rsi,%rdx,8),%r12
	adc 8(%rsi,%rdx,8),%r13
	adc 16(%rsi,%rdx,8),%r14
	adc 24(%rsi,%rdx,8),%r15
	adc %rbx,%rbx
	add $4,%rdx
	mov %r12,(%rbp,%rcx,8)
	mov %r13,8(%rbp,%rcx,8)
	mov %r14,16(%rbp,%rcx,8)
	mov %r15,24(%rbp,%rcx,8)
	add $4,%rcx
	jnc L(p)
cmp $2,%rcx
jg	L(case0)
jz	L(case1)
jp	L(case2)
L(case3):	#rcx=0
	bt $2,%rbx
	mov (%rdi,%rdx,8),%r8
	adc (%rbp),%r8
	mov %r8,%r12
	mov 8(%rdi,%rdx,8),%r9
	adc 8(%rbp),%r9
	mov 16(%rdi,%rdx,8),%r10
	adc 16(%rbp),%r10
	adc %rbx,%rbx
	bt $1,%rax
	adc (%rdi),%r8
	mov %r9,%r13
	adc 8(%rdi),%r9
	mov %r10,%r14
	adc 16(%rdi),%r10
	adc %rax,%rax
	bt $2,%rbx
	adc (%rbp,%rdx,8),%r12
	adc 8(%rbp,%rdx,8),%r13
	adc 16(%rbp,%rdx,8),%r14
	adc %rbx,%rbx
	bt $1,%rax
	adc (%rsi),%r8
	adc 8(%rsi),%r9
	adc 16(%rsi),%r10
	mov %r10,16(%rdi,%rdx,8)
	adc %rax,%rax
	bt $2,%rbx
	mov %r8,(%rdi,%rdx,8)
	mov %r9,8(%rdi,%rdx,8)
	adc (%rsi,%rdx,8),%r12
	adc 8(%rsi,%rdx,8),%r13
	adc 16(%rsi,%rdx,8),%r14
	adc %rbx,%rbx
	add $3,%rdx
	mov %r12,(%rbp)
	mov %r13,8(%rbp)
	mov %r14,16(%rbp)
	jmp L(fin)
L(case2):	#rcx=1
	bt $2,%rbx
	mov (%rdi,%rdx,8),%r8
	adc 8(%rbp),%r8
	mov %r8,%r12
	mov 8(%rdi,%rdx,8),%r9
	adc 16(%rbp),%r9
	adc %rbx,%rbx
	bt $1,%rax
	adc 8(%rdi),%r8
	mov %r9,%r13
	adc 16(%rdi),%r9
	adc %rax,%rax
	bt $2,%rbx
	adc (%rbp,%rdx,8),%r12
	adc 8(%rbp,%rdx,8),%r13
	adc %rbx,%rbx
	bt $1,%rax
	adc 8(%rsi),%r8
	adc 16(%rsi),%r9
	adc %rax,%rax
	bt $2,%rbx
	mov %r8,(%rdi,%rdx,8)
	mov %r9,8(%rdi,%rdx,8)
	adc (%rsi,%rdx,8),%r12
	adc 8(%rsi,%rdx,8),%r13
	adc %rbx,%rbx
	add $2,%rdx
	mov %r12,8(%rbp)
	mov %r13,16(%rbp)
	jmp L(fin)
L(case1):	#rcx=2
	bt $2,%rbx
	mov (%rdi,%rdx,8),%r8
	adc 16(%rbp),%r8
	mov %r8,%r12
	adc %rbx,%rbx
	bt $1,%rax
	adc 16(%rdi),%r8
	adc %rax,%rax
	bt $2,%rbx
	adc (%rbp,%rdx,8),%r12
	adc %rbx,%rbx
	bt $1,%rax
	adc 16(%rsi),%r8
	adc %rax,%rax
	bt $2,%rbx
	mov %r8,(%rdi,%rdx,8)
	adc (%rsi,%rdx,8),%r12
	adc %rbx,%rbx
	inc %rdx
	mov %r12,(%rbp,%rcx,8)
L(fin):	mov $3,%rcx
L(case0): #rcx=3
	pop %r8
	bt $0,%r8
	jnc L(notodd)
	xor %r10,%r10
	mov (%rbp,%rdx,8),%r8
	mov 8(%rbp,%rdx,8),%r9
	add (%rsi,%rdx,8),%r8
	adc 8(%rsi,%rdx,8),%r9
	adc %r10,%r10
	add %r8,24(%rbp)
	adc %r9,32(%rbp)
	adc %r10,40(%rbp)
L(l7):	adcq $0,24(%rbp,%rcx,8)
	inc %rcx
	jc L(l7)
	mov $3,%rcx
L(notodd):and $3,%rax
	popcnt %rax,%r8
	bt $2,%rbx
	adc $0,%r8
	adc %r8,(%rdi,%rdx,8)
L(l1):	adcq $0,8(%rdi,%rdx,8)
	inc %rdx
	jc L(l1)
	and $7,%rbx
	popcnt %rbx,%r8
	add %r8,24(%rbp)
L(l2):	adcq $0,8(%rbp,%rcx,8)
	inc %rcx
	jc L(l2)
pop %r15
pop %r14
pop %r13
pop %r12
pop %rbp
pop %rbx
ret
EPILOGUE()
