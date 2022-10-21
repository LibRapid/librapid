/* How to do an addition with carry in C */

void g (unsigned long a, unsigned long b, unsigned long c);

void f1 (unsigned long a, unsigned long b, unsigned long c, unsigned long i)
{
  a += i;
  b += (a < i);
  c += (b == 0) && (a < i);
  g (a, b, c);
}

#define ADD_LIMB(u,v,c) ((u) += (v), (c) = (u) < (v))

void f2 (unsigned long a, unsigned long b, unsigned long c, unsigned long i)
{
  unsigned long carry1, carry2;

  ADD_LIMB (a, i, carry1);
  ADD_LIMB (b, carry1, carry2);
  c += carry2;
  g (a, b, c);
}

/* Generated code on x86_64...

*** With GCC 6.3.0 (-O3) ***

For f1:
        .cfi_startproc
        addq    %rcx, %rdi
        setc    %r8b
        xorl    %eax, %eax
        movzbl  %r8b, %ecx
        addq    %rcx, %rsi
        sete    %al
        andq    %r8, %rax
        addq    %rax, %rdx
        jmp     g@PLT
        .cfi_endproc

0000000000000000 <f1>:
   0:   48 01 cf                add    %rcx,%rdi
   3:   41 0f 92 c0             setb   %r8b
   7:   31 c0                   xor    %eax,%eax
   9:   41 0f b6 c8             movzbl %r8b,%ecx
   d:   48 01 ce                add    %rcx,%rsi
  10:   0f 94 c0                sete   %al
  13:   4c 21 c0                and    %r8,%rax
  16:   48 01 c2                add    %rax,%rdx
  19:   e9 00 00 00 00          jmpq   1e <f1+0x1e>

For f2:
        .cfi_startproc
        xorl    %eax, %eax
        addq    %rcx, %rdi
        setc    %al
        addq    %rax, %rsi
        setc    %al
        movzbl  %al, %eax
        addq    %rax, %rdx
        jmp     g@PLT
        .cfi_endproc

0000000000000020 <f2>:
  20:   31 c0                   xor    %eax,%eax
  22:   48 01 cf                add    %rcx,%rdi
  25:   0f 92 c0                setb   %al
  28:   48 01 c6                add    %rax,%rsi
  2b:   0f 92 c0                setb   %al
  2e:   0f b6 c0                movzbl %al,%eax
  31:   48 01 c2                add    %rax,%rdx
  34:   e9 00 00 00 00          jmpq   39 <f2+0x19>

*** With Clang 3.9.1 (-O3) ***

0000000000000000 <f1>:
   0:   31 c0                   xor    %eax,%eax
   2:   48 01 cf                add    %rcx,%rdi
   5:   0f 92 c0                setb   %al
   8:   48 01 c6                add    %rax,%rsi
   b:   0f 94 c1                sete   %cl
   e:   20 c8                   and    %cl,%al
  10:   0f b6 c0                movzbl %al,%eax
  13:   48 01 c2                add    %rax,%rdx
  16:   e9 00 00 00 00          jmpq   1b <f1+0x1b>

0000000000000020 <f2>:
  20:   48 01 cf                add    %rcx,%rdi
  23:   48 19 c0                sbb    %rax,%rax
  26:   83 e0 01                and    $0x1,%eax
  29:   48 01 c6                add    %rax,%rsi
  2c:   48 83 d2 00             adc    $0x0,%rdx
  30:   e9 00 00 00 00          jmpq   35 <f2+0x15>

The f2 solution is better in both cases! But Clang could have done better.
Bug report: https://llvm.org/bugs/show_bug.cgi?id=31755
*/
