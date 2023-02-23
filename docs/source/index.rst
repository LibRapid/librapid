LibRapid
########

What is LibRapid?
-----------------

LibRapid is a high performance Array library, supporting a wide range of optimised calculations which can be performed
on the CPU or GPU (via CUDA). All calculations are vectorised with SIMD instructions and are run on multiple threads (if
necessary) to make them as fast as possible on any given machine.

Why use LibRapid?
-----------------

LibRapid aims to provide a cohesive ecosystem of functions that interoperate with each other, allowing for faster
development and faster code execution.

For example, LibRapid implements a wide range of mathematical functions which can operate on primitive types,
multi-precision types, vectors, and arrays. Due to the way these functions are implemented, a single function call can
be used to operate on all of these types, reducing code duplication.

Licencing
=========

LibRapid is produced under the MIT License, so you are free to use the library
how you like for personal and commercial purposes, though this is subject to
some conditions, which can be found in full here: `LibRapid License`_

.. _LibRapid License: https://github.com/Pencilcaseman/librapid/blob/master/LICENSE
