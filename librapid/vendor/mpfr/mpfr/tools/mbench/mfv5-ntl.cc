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

#include "mfv5.h"

#define NTL_STD_CXX
#include "NTL/RR.h"

#include "timp.h"

using namespace std;
using namespace NTL;

/* Register New Test */
template <class T>
class ntl_test : public registered_test {
private:
  unsigned long size;
  RR *table;
  RR a, b, c;
  timming *tim;
public:
  ntl_test (const char *n) : registered_test (n), size (0) {}
  ~ntl_test () {
    if (size != 0) {
      delete tim;
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class ntl_add_test {
public:
  void func (RR &a, const RR &b, const RR &c) {
    a = b + c;
  }
};
class ntl_sub_test {
public:
  void func (RR &a, const RR &b, const RR &c) {
    a = b - c;
  }
};
class ntl_mul_test {
public:
  void func (RR &a, const RR &b, const RR &c) {
    a = b * c;
  }
};
class ntl_div_test {
public:
  void func (RR &a, const RR &b, const RR &c) {
    a = b / c;
  }
};

static ntl_test<ntl_add_test> test1 ("ntl_add");
static ntl_test<ntl_sub_test> test2 ("ntl_sub");
static ntl_test<ntl_mul_test> test3 ("ntl_mul");
static ntl_test<ntl_div_test> test4 ("ntl_div");

/* Do the test */
template <class T>
bool ntl_test<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;

  /* Init and set tables if first call */
  if (size == 0) {
    size = base.size ();
    tim = new timming (size);
    table = new RR[size];
    for (i = 0 ; i < size ; i++)
      {
	table[i].SetPrecision (opt.prec);
	table[i] = to_RR (base[i].c_str());
      }
    a.SetPrecision (opt.prec);
    b.SetPrecision (opt.prec);
    c.SetPrecision (opt.prec);
  }

  /* Do Measure */
  for(i = 0 ; i < (size-1) ; i++) {
    b = table[i];
    c = table[i+1];
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE (f.func (a, b, c) );
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}
