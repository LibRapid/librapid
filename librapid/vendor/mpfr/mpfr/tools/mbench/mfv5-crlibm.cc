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
#include "timp.h"

#include <math.h>
#include <float.h>
#include "crlibm.h"
#include "mpfr.h"

using namespace std;

/* Register New Test */
template <class T>
class crlibm_test : public registered_test {
private:
  unsigned long size;
  double *table;
  timming *tim;
public:
  crlibm_test (const char *n) : registered_test (n), size (0) {}
  ~crlibm_test () {
    if (size != 0) {
      delete tim;
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class crlibm_exp_test {
public:
  double func (double a) {
    return exp_rn (a);
  }
};
class crlibm_log_test {
public:
  double func (double a) {
    return log_rn (a);
  }
};
class crlibm_cos_test {
public:
  double func (double a) {
    return cos_rn (a);
  }
};
class crlibm_sin_test {
public:
  double func (double a) {
    return sin_rn (a);
  }
};
class crlibm_tan_test {
public:
  double func (double a) {
    return tan_rn (a);
  }
};
class crlibm_atan_test {
public:
  double func (double a) {
    return atan_rn (a);
  }
};
class crlibm_cosh_test {
public:
  double func (double a) {
    return cosh_rn (a);
  }
};
class crlibm_sinh_test {
public:
  double func (double a) {
    return sinh_rn (a);
  }
};
static crlibm_test<crlibm_exp_test>  test7 ("crlibm_exp");
static crlibm_test<crlibm_log_test>  test8 ("crlibm_log");

static crlibm_test<crlibm_cos_test>  testA ("crlibm_cos");
static crlibm_test<crlibm_sin_test>  testB ("crlibm_sin");
static crlibm_test<crlibm_tan_test>  testC ("crlibm_tan");
static crlibm_test<crlibm_atan_test> testF ("crlibm_atan");
static crlibm_test<crlibm_cosh_test>  testAh ("crlibm_cosh");
static crlibm_test<crlibm_sinh_test>  testBh ("crlibm_sinh");

/* Do the test */
template <class T>
bool crlibm_test<T>::test (const vector<string> &base, const option_test &opt) {
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
  TIMP_OVERHEAD ();
  for(i = 0 ; i < (size-1) ; i++) {
    b = table[i];
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE (a = f.func (b) );
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}
