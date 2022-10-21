;  Copyright 1999, 2000, 2001, 2002 Free Software Foundation, Inc.
;
;  Copyright 2005, 2006 Pierrick Gaudry
;
;  Copyright 2008 Brian Gladman, William Hart
;
;  This file is part of the MPIR Library.
;
;  The MPIR Library is free software; you can redistribute it and/or
;  modify it under the terms of the GNU Lesser General Public License as
;  published by the Free Software Foundation; either version 2.1 of the
;  License, or (at your option) any later version.
;
;  The MPIR Library is distributed in the hope that it will be useful,
;  but WITHOUT ANY WARRANTY; without even the implied warranty of
;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;  Lesser General Public License for more details.
;
;  You should have received a copy of the GNU Lesser General Public
;  License along with the MPIR Library; see the file COPYING.LIB.  If
;  not, write to the Free Software Foundation, Inc., 51 Franklin Street,
;  Fifth Floor, Boston, MA 02110-1301, USA.
;
;  Adapted by Brian Gladman AMD64 using the Microsoft VC++ v8 64-bit
;  compiler and the YASM assembler.

;  AMD64 mpn_add_n/mpn_sub_n -- mpn add or subtract.
;
;  Calling interface:
;
;  mp_limb_t __gmpn_<op>_n(    <op> = add OR sub
;     mp_ptr dst,              rdi
;     mp_srcptr src1,          rsi
;     mp_srcptr src2,          rdx
;     mp_size_t  len           rcx
;  )
;
;  mp_limb_t __gmpn_<op>_nc(   <op> = add OR sub
;     mp_ptr dst,              rdi
;     mp_srcptr src1,          rsi
;     mp_srcptr src2,          rdx
;     mp_size_t len,           rcx
;     mp_limb_t carry           r8 
;  )
;
;  Calculate src1[size] plus(minus) src2[size] and store the result in
;  dst[size].  The return value is the carry bit from the top of the result
;  (1 or 0).  The _nc version accepts 1 or 0 for an initial carry into the
;  low limb of the calculation.  Note values other than 1 or 0 here will
;  lead to garbage results.

%include 'yasm_mac.inc'

%define dst       rdi   ; destination pointer
%define sr1       rsi   ; source 1 pointer
%define sr2       rdx   ; source 2 pointer
%define len       rcx   ; number of limbs
%define lend      ecx   ; number of limbs
%define cy         r8   ; carry value

%define r_jmp     r10   ; temporary for jump table entry
%define r_cnt     r11   ; temporary for loop count

%define UNROLL_LOG2         4
%define UNROLL_COUNT        (1 << UNROLL_LOG2)
%define UNROLL_MASK         (UNROLL_COUNT - 1)
%define UNROLL_BYTES        (8 * UNROLL_COUNT)
%define UNROLL_THRESHOLD    8

%if UNROLL_BYTES >= 256
%error unroll count is too large
%elif UNROLL_BYTES >= 128
%define off 128
%else
%define off 0
%endif

%macro  mac_sub  3

;LOBAL_FUNC mpn_add_nc
;    mov     rax,cy
;    jmp     %%0
GLOBAL_FUNC mpn_add_n
    xor     rax,rax
%%0:
    movsxd  len,lend
    cmp     len,UNROLL_THRESHOLD
    jae     %%2
    lea     sr1,[sr1+len*8]
    lea     sr2,[sr2+len*8]
    lea     dst,[dst+len*8]
    neg     len
    shr     rax,1
%%1:
    mov     rax,[sr1+len*8]
    mov     r10,[sr2+len*8]
    %1      rax,r10
    mov     [dst+len*8],rax
    inc     len
    jnz     %%1
    mov     rax,dword 0
    setc    al
    ret
%%2:
    mov     r_cnt,1
    and     r_cnt,len
    push    r_cnt
    and     len,-2
    mov     r_cnt,len
    dec     r_cnt
    shr     r_cnt,UNROLL_LOG2
    neg     len
    and     len,UNROLL_MASK
    lea     r_jmp,[len*4]
    neg     len
    lea     sr1,[sr1+len*8+off]
    lea     sr2,[sr2+len*8+off]
    lea     dst,[dst+len*8+off]
    shr     rax,1
    lea     r_jmp,[r_jmp+r_jmp*2]

%ifdef PIC
    call    .pic_calc
.unroll_here:
..@unroll_here1:

%else
    lea     rax,[rel %%3]
%endif

    lea     r_jmp,[r_jmp+rax]
    jmp     r_jmp

%ifdef PIC

.pic_calc:

	mov     rax, ..@unroll_entry1 - ..@unroll_here1
	add     rax, [rsp]
	ret

%endif

    align 32

.unroll_entry1:
..@unroll_entry1:
%%3:

%define CHUNK_COUNT  2
%assign i 0

%rep  UNROLL_COUNT / CHUNK_COUNT
%assign  disp0 8 * i * CHUNK_COUNT - off

    mov     r_jmp,[byte sr1+disp0]      ; len and r_jmp registers
    mov     len,[byte sr1+disp0+8]      ; now not needed
    %1      r_jmp,[byte sr2+disp0]
    mov     [byte dst+disp0],r_jmp
    %1      len,[byte sr2+disp0+8]
    mov     [byte dst+disp0+8],len

%assign i i + 1
%endrep

%if UNROLL_BYTES > 64
    lea     sr1,[byte sr1+127]
    inc     sr1
%else
    lea     sr1,[byte sr1+UNROLL_BYTES]
%endif
    dec     r_cnt
    lea     sr2,[sr2+UNROLL_BYTES]
    lea     dst,[dst+UNROLL_BYTES]
    jns     %%3

    pop     rax
    dec     rax
    js      %%5
    mov     len,[sr1-off]
    %1      len,[sr2-off]
    mov     [dst-off],len
%%5:mov     rax,dword 0
    setc    al
    ret

%endmacro

    BITS    64

    mac_sub adc,mpn_add_n,mpn_add_nc
