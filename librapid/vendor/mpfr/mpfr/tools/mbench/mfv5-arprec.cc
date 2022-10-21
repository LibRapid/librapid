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

#include "mp/mpreal.h"

#include "timp.h"

using namespace std;

/* Register New Test */
template <class T>
class arprec_test : public registered_test {
private:
  unsigned long size;
  mp_real *table;
  timming *tim;
public:
  arprec_test (const char *n) : registered_test (n), size (0) {}
  ~arprec_test () {
    if (size != 0) {
      delete tim;
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class arprec_add_test {
public:
  void func (mp_real &a, const mp_real &b, const mp_real &c) {
    a = b + c;
  }
};
class arprec_sub_test {
public:
  void func (mp_real &a, const mp_real &b, const mp_real &c) {
    a = b - c;
  }
};
class arprec_mul_test {
public:
  void func (mp_real &a, const mp_real &b, const mp_real &c) {
    a = b * c;
  }
};
class arprec_div_test {
public:
  void func (mp_real &a, const mp_real &b, const mp_real &c) {
    a = b / c;
  }
};

static arprec_test<arprec_add_test> test1 ("arprec_add");
static arprec_test<arprec_sub_test> test2 ("arprec_sub");
static arprec_test<arprec_mul_test> test3 ("arprec_mul");
static arprec_test<arprec_div_test> test4 ("arprec_div");


/* Do the test */
template <class T> bool
arprec_test<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;

  /* Init and set tables if first call */
  if (size == 0) {
    size = base.size ();
    mp::mp_init (opt.prec);
    tim = new timming (size);
    table = new mp_real[size];
    for (i = 0 ; i < size ; i++)
      table[i] = base[i].c_str ();
  }
  mp_real a, b, c;

  /* Do Measure */
  for(i = 0 ; i < (size-1) ; i++) {
    b = table[i];
    c = table[i+1];
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE(f.func (a, b, c) );
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}
