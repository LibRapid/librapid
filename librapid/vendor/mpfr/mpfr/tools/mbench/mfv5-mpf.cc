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
#include "gmp.h"
#include "timp.h"

using namespace std;

/* Register New Test */
template <class T>
class mpf_test : public registered_test {
private:
  unsigned long size;
  mpf_t *table;
  mpf_t a, b, c;
  timming *tim;
public:
  mpf_test (const char *n) : registered_test (n), size (0) {}
  ~mpf_test () {
    if (size != 0) {
      unsigned long i;
      delete tim;
      mpf_clear (a);
      mpf_clear (b);
      mpf_clear (c);
      for (i = 0 ; i < size ; i++)
	mpf_clear (table[i]);
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class mpf_add_test {
public:
  void func (mpf_ptr a, mpf_srcptr b, mpf_srcptr c) {
    return mpf_add (a,b,c);
  }
};

class mpf_sub_test {
public:
  void func (mpf_ptr a, mpf_srcptr b, mpf_srcptr c) {
    return mpf_sub (a,b,c);
  }
};

class mpf_mul_test {
public:
  void func (mpf_ptr a, mpf_srcptr b, mpf_srcptr c) {
    return mpf_mul (a,b,c);
  }
};

class mpf_div_test {
public:
  void func (mpf_ptr a, mpf_srcptr b, mpf_srcptr c) {
    return mpf_div (a,b,c);
  }
};

class mpf_set_test {
public:
  void func (mpf_ptr a, mpf_srcptr b, mpf_srcptr c) {
    return mpf_set (a,b);
  }
};

class mpf_sqrt_test {
public:
  void func (mpf_ptr a, mpf_srcptr b, mpf_srcptr c) {
    return mpf_sqrt (a,b);
  }
};

static mpf_test<mpf_add_test> test1 ("mpf_add");
static mpf_test<mpf_sub_test> test2 ("mpf_sub");
static mpf_test<mpf_mul_test> test3 ("mpf_mul");
static mpf_test<mpf_div_test> test4 ("mpf_div");
static mpf_test<mpf_set_test> test5 ("mpf_set");
static mpf_test<mpf_sqrt_test> test6 ("mpf_sqrt");


/* Do the test */
template <class T>
bool mpf_test<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;

  /* Init and set tables if first call */
  if (size == 0) {
    size = base.size ();
    tim = new timming (size);
    table = new mpf_t[size];
    for (i = 0 ; i < size ; i++) {
      mpf_init2 (table[i], opt.prec);
      mpf_set_str (table[i], base[i].c_str(), 10);
    }
    mpf_init2 (a, opt.prec);
    mpf_init2 (b, opt.prec);
    mpf_init2 (c, opt.prec);
  }

  /* Do Measure */
  for(i = 0 ; i < (size-1) ; i++) {
    mpf_set (b, table[i]);
    mpf_set (c, table[i+1]);
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE(f.func (a, b, c) );
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}
