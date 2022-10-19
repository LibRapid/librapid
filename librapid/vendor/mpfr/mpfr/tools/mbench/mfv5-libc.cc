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

#include <math.h>
#include <float.h>

#include "mpfr.h"

#include "mfv5.h"
#include "timp.h"


using namespace std;

/* Register New Test */
template <class T>
class libc_test : public registered_test {
private:
  unsigned long size;
  double *table;
  timming *tim;
public:
  libc_test (const char *n) : registered_test (n), size (0) {}
  ~libc_test () {
    if (size != 0) {
      delete tim;
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class libc_sqrt_test {
public:
  inline double func (double a) {
    return sqrt (a);
  }
};
class libc_exp_test {
public:
  inline double func (double a) {
    return exp (a);
  }
};
class libc_log_test {
public:
  inline double func (double a) {
    return log (a);
  }
};
class libc_cos_test {
public:
  inline double func (double a) {
    return cos (a);
  }
};
class libc_sin_test {
public:
  inline double func (double a) {
    return sin (a);
  }
};
class libc_tan_test {
public:
  inline double func (double a) {
    return tan (a);
  }
};
class libc_acos_test {
public:
  inline double func (double a) {
    return acos (a);
  }
};
class libc_asin_test {
public:
  inline double func (double a) {
    return asin (a);
  }
};
class libc_atan_test {
public:
  inline double func (double a) {
    return atan (a);
  }
};
class libc_cosh_test {
public:
  inline double func (double a) {
    return cosh (a);
  }
};
class libc_sinh_test {
public:
  inline double func (double a) {
    return sinh (a);
  }
};
class libc_tanh_test {
public:
  inline double func (double a) {
    return tanh (a);
  }
};
class libc_acosh_test {
public:
  inline double func (double a) {
    return acosh (a);
  }
};
class libc_asinh_test {
public:
  inline double func (double a) {
    return asinh (a);
  }
};
class libc_atanh_test {
public:
  inline double func (double a) {
    return atanh (a);
  }
};


static libc_test<libc_sqrt_test> test6 ("libc_sqrt");
static libc_test<libc_exp_test>  test7 ("libc_exp");
static libc_test<libc_log_test>  test8 ("libc_log");

static libc_test<libc_cos_test>  testA ("libc_cos");
static libc_test<libc_sin_test>  testB ("libc_sin");
static libc_test<libc_tan_test>  testC ("libc_tan");
static libc_test<libc_acos_test> testD ("libc_acos");
static libc_test<libc_asin_test> testE ("libc_asin");
static libc_test<libc_atan_test> testF ("libc_atan");

static libc_test<libc_cosh_test>  testAh ("libc_cosh");
static libc_test<libc_sinh_test>  testBh ("libc_sinh");
static libc_test<libc_tanh_test>  testCh ("libc_tanh");
static libc_test<libc_acosh_test> testDh ("libc_acosh");
static libc_test<libc_asinh_test> testEh ("libc_asinh");
static libc_test<libc_atanh_test> testFh ("libc_atanh");

/* Do the test */
template <class T>
bool libc_test<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;
  volatile double a, b;

  /* Init and set tables if first call */
  if (size == 0) {
    mpfr_t x;
    size = base.size ();
    tim = new timming (size);
    table = new double[size];
    mpfr_init2 (x, 530);
    for (i = 0 ; i < size ; i++) {
      mpfr_set_str (x, base[i].c_str(), 10, MPFR_RNDN);
      table[i] = mpfr_get_d (x, MPFR_RNDN);
    }
    mpfr_clear (x);
  }

  /* Do Measure */
  for(i = 0 ; i < (size-1) ; i++) {
    b = table[i];
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE(a = f.func (b) );
    b = a;
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}
