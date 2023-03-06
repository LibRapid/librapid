# Performance and Benchmarks

LibRapid is high-performance library and is fast by default, but there are still ways to make your code even faster.

## Lazy Evaluation

Operations performed on Arrays are evaluated only when needed, meaning functions can be chained together and evaluated
in one go. In many cases, the compiler can optimise these chained calls into a single loop, resulting in much faster
code.

Look at the example below:

```cpp
lrc::Array<float> A, B, C, D:
A = lrc::fromData({{1, 2}, {3, 4}});
B = lrc::fromData({{5, 6}, {7, 8}});
C = lrc::fromData({{9, 10}, {11, 12}});
D = A + B * C;
```

Without lazy-evaluation, the operation `A+B*C` must be performed in multiple stages:

```cpp
auto tmp1 = B * C;    // First operation and temporary object
auto tmp2 = A + tmp1; // Second operation and ANOTHER temporary object
D = tmp2;             // Unnecessary copy
```

This is clearly suboptimal.

With lazy-evaluation, however, the compiler can generate a loop similar to the pseudocode below:

```
FOR index IN A.size DO
    D[i] = A[i] + B[i] * C[i]
ENDFOR 
```

This has no unnecessary copies, no temporary variables, no additional memory allocation, etc. and is substantially
quicker.

### Making Use of LibRapid's Lazy Evaluation

To make use of LibRapid's lazy evaluation, try to avoid creating temporary objects and always assign results directly
to an existing array object, instead of creating a new one. This means no heap allocations are performed, which is a
very costly operation.

:::{warning}
Be very careful not to reference invalid memory. This is, unfortunately, an unavoidable side effect of returning
lazy-objects. See :Caution: <project:../caution/caution> for more information.
:::

Note that, sometimes, it is faster to evaluate intermediate results than to use the combined operation. To do this,
you can call ``eval()`` on the result of any operation to generate an Array object directly from it.
