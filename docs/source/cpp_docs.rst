LibRapid for C++
################

Download LibRapid
=================

To download the LibRapid library for C++ use, it is recommended that you download
the source code from GitHub and move the relevant files somewhere useful. The
recommended way of doing this is as follows:

.. code-block:: shell

	git clone https://github.com/Pencilcaseman/librapid.git

Then move the file to ``C:\opt\librapid`` (or equivelent)

This directory should then be added to the compiler's include path, so that you
can include the libary with ``#include <librapid/librapid.hpp>``.

Performance
===========

The C++ library has been heavily optimized, and due to overheads in Python and
language performance differences in C#, it is likely to be the fast method of
using LibRapid. Additionally, the ability to link with any CBlas compatible interface
makes the C++ side of LibRapid incredibly powerful.


Contents
========

.. toctree::
	:maxdepth: 2
	:glob:

	librapid_ndarray_api

Indices and tables
==================

* :ref:`search`
