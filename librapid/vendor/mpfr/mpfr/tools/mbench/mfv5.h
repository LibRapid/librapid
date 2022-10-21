/*
Copyright 2005-2022 Free Software Foundation, Inc.
Contributed by Patrick Pelissier, INRIA.

This file is part of the MPFR Library.

The MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the MPFR Library; see the file COPYING.LESSER.  If not, see
https://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#ifndef __MPFR_T_LOW_BENCHMARCH_H__
#define __MPFR_T_LOW_BENCHMARCH_H__

#include <iostream>
#include <stdio.h> /* for printf and putchar */
#include <cstring>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <string>
#include <fstream>

#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "mpfr.h"

struct option_test {
  unsigned long prec;
  unsigned long seed;
  unsigned long stat;
  long max_exp;       /* exponent is in [-max_exp/2, max_exp/2] */
  long exp_diff;      /* difference of exponents (for mpfr_add, mpfr_sub) */
  bool verbose;
  mpfr_rnd_t rnd;
  std::string export_base;
  std::string import_base;
  option_test () : prec (53), seed (14528596), stat (100), max_exp (1), exp_diff (-1), verbose (false), rnd(MPFR_RNDN), export_base("") {}
};

class registered_test;
extern registered_test *first_registered_test;

class registered_test {
 private:
  const char *name;
  registered_test *next_test;
 public:
  registered_test (const char *n) : name (n) {
    next_test = first_registered_test;
    first_registered_test = this;
  }
  virtual ~registered_test () {}
  registered_test *next (void) {
    return next_test;
  }
  const char *get_name (void) {
    return name;
  }
  bool is (const char *n) {
    return strcmp (n, name) == 0;
  }
  virtual bool test (const std::vector<std::string> &base, const option_test &opt) {
    return false;
  }
};

class timming {
 private:
  unsigned long size;
  unsigned long long *besttime;

 public:
  timming (unsigned long s) : size (s) {
    besttime = new unsigned long long[size];
    for (unsigned long i = 0 ; i < size ; i++)
      besttime[i] = 0xFFFFFFFFFFFFFFFFLL;
  }

  ~timming () {
    delete[] besttime;
  }

  bool update (unsigned long i, unsigned long long m) {
    if (size <= i)
      abort ();
    if (m < besttime[i]) {
      besttime[i] = m;
      return true;
    } else
      return false;
  }

  void print (const char *name, const option_test &opt) {
    unsigned long long min, max, moy;
    unsigned long imin = 0, imax = 0;
    min = 0xFFFFFFFFFFFFFFFFLL;
    max = moy = 0;
    for(unsigned long i = 0 ; i < (size-1) ; i++) {
      if (besttime[i] < min)
	{ min = besttime[i]; imin = i; }
      if (besttime[i] > max)
	{ max = besttime[i]; imax = i; }
      moy += besttime[i];
    }
    printf (" %s:\t %5Lu / %5Lu.%02Lu / %5Lu", name,
	    min, moy/(size-1), (moy*100/(size-1))%100, max);
    if (opt.verbose)
      printf ("\t Imin=%3lu Imax=%3lu", imin, imax);
    putchar ('\n');
  }
};

#endif
