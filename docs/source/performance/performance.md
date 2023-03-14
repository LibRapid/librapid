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
lazy-objects. See [Caution](../caution) for more information.
:::

Note that, sometimes, it is faster to evaluate intermediate results than to use the combined operation. To do this,
you can call ``eval()`` on the result of any operation to generate an Array object directly from it.

## Linear Algebra

Linear algebra methods in LibRapid also return temporary objects, meaning they are not evaluated fully until they are
needed. One implication of this is that expressions involving ***more than one operation*** will be evaluated
***very slowly***.

:::{danger}
Be careful when calling `eval` on the result of a linear algebra operation. Sometimes, LibRapid will be able to combine
multiple operations into a single function call, which can lead to much better performance. Check the documentation
for that specific function to see what further optimisations it supports.
:::

### Solution

To get around this issue, it'll often be quicker to simply evaluate (`myExpression.eval()`) the result of any linear
algebra operations inside the larger expression.

```cpp
auto slowExpression = a + b * c.dot(d);
auto fastExpression = a + b * c.dot(d).eval();
```

### Explanation

Since `c.dot(d)` is a lazy object, the lazy evaluator will calculate each element of the resulting array independently
as and when it is required by the rest of the expression. This means it is not possible to make use of the extremely
fast BLAS and LAPACK functions.

By forcing the result to be evaluated independently of the rest of the expression, LibRapid can call `gemm`, for
example, making the program significantly faster.
