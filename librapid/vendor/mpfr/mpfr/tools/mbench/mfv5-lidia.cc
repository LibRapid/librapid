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

#include "LiDIA/bigfloat.h"

#include "timming.h"

using namespace std;
using namespace LiDIA;

/* Register New Test */
template <class T>
class lidia_test : public registered_test {
private:
  unsigned long size;
  bigfloat *table;
  bigfloat a, b, c;
  timming *tim;
public:
  lidia_test (const char *n) : registered_test (n), size (0) {}
  ~lidia_test () {
    if (size != 0) {
      delete tim;
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class lidia_add_test {
public:
  void func (bigfloat &a, const bigfloat &b, const bigfloat &c) {
    add (a, b, c);
  }
};
class lidia_sub_test {
public:
  void func (bigfloat &a, const bigfloat &b, const bigfloat &c) {
    subtract (a, b, c);
  }
};
class lidia_mul_test {
public:
  void func (bigfloat &a, const bigfloat &b, const bigfloat &c) {
    multiply (a, b, c);
  }
};
class lidia_div_test {
public:
  void func (bigfloat &a, const bigfloat &b, const bigfloat &c) {
    divide (a, b, c);
  }
};
class lidia_sqrt_test {
public:
  void func (bigfloat &a, const bigfloat &b, const bigfloat &c) {
    sqrt (a, b);
  }
};
class lidia_exp_test {
public:
  void func (bigfloat &a, const bigfloat &b, const bigfloat &c) {
    exp (a, b);
  }
};


static lidia_test<lidia_add_test> test1 ("lidia_add");
static lidia_test<lidia_sub_test> test2 ("lidia_sub");
static lidia_test<lidia_mul_test> test3 ("lidia_mul");
static lidia_test<lidia_div_test> test4 ("lidia_div");
static lidia_test<lidia_sqrt_test> test5 ("lidia_sqrt");
static lidia_test<lidia_exp_test> test6 ("lidia_exp");


/* Do the test */
template <class T>
bool lidia_test<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;

  /* Init and set tables if first call */
  if (size == 0) {
    unsigned long prec=(unsigned long)(((double)opt.prec)*0.3010299956639811);
    bigfloat::set_mode (MP_RND);
    bigfloat::set_precision (prec);
    size = base.size ();
    tim = new timming (size);
    table = new bigfloat[size];
    for (i = 0 ; i < size ; i++)
      string_to_bigfloat ((char*) base[i].c_str(), table[i]);
    a.set_precision (prec);
    b.set_precision (prec);
    c.set_precision (prec);
  }

  /* Do Measure */
  CALCUL_OVERHEAD;
  for(i = 0 ; i < (size-1) ; i++) {
    b = table[i];
    c = table[i+1];
    CALCUL_OVERHEAD ;
    m = MEASURE(f.func (a, b, c) );
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}
