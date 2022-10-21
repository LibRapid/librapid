/*
Copyright 2005-2022 Free Software Foundation, Inc.
Contributed by Patrick Pelissier, INRIA.
Small changes by Paul Zimmermann.

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
#include "mpfr.h"
#include "timp.h"

using namespace std;

/* Register New Test */
template <class T>
class mpfr_test : public registered_test {
private:
  unsigned long size;
  mpfr_t *table;
  mpfr_t a, b, c;
  timming *tim;
public:
  mpfr_test (const char *n) : registered_test (n), size (0) {}
  ~mpfr_test () {
    if (size != 0) {
      unsigned long i;
      delete tim;
      mpfr_clears (a, b, c, NULL);
      for (i = 0 ; i < size ; i++)
	mpfr_clear (table[i]);
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

template <class T>
class mpfr_test3 : public registered_test {
private:
  unsigned long size;
  mpfr_t *table;
  mpfr_t a, b, c, d;
  timming *tim;
public:
  mpfr_test3 (const char *n) : registered_test (n), size (0) {}
  ~mpfr_test3 () {
    if (size != 0) {
      unsigned long i;
      delete tim;
      mpfr_clears (a, b, c, d, NULL);
      for (i = 0 ; i < size ; i++)
	mpfr_clear (table[i]);
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

template <class T>
class mpfr_test4 : public registered_test {
private:
  unsigned long size;
  mpfr_t *table;
  mpfr_t a, b, c, d, e;
  timming *tim;
public:
  mpfr_test4 (const char *n) : registered_test (n), size (0) {}
  ~mpfr_test4 () {
    if (size != 0) {
      unsigned long i;
      delete tim;
      mpfr_clears (a, b, c, d, e, NULL);
      for (i = 0 ; i < size ; i++)
	mpfr_clear (table[i]);
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class mpfr_add_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_add (a,b,c,r);
  }
};

class mpfr_sub_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_sub (a,b,c,r);
  }
};

class mpfr_mul_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_mul (a,b,c,r);
  }
};

class mpfr_sqr_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_sqr (a,b,r);
  }
};

class mpfr_fma_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_srcptr d, mp_rnd_t r) {
    return mpfr_fma (a,b,c,d,r);
  }
};

class mpfr_fms_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_srcptr d, mp_rnd_t r) {
    return mpfr_fms (a,b,c,d,r);
  }
};

#if MPFR_VERSION_MAJOR >= 4
class mpfr_fmma_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_srcptr d, mpfr_srcptr e, mp_rnd_t r) {
    return mpfr_fmma (a,b,c,d,e,r);
  }
};

class mpfr_fmms_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_srcptr d, mpfr_srcptr e, mp_rnd_t r) {
    return mpfr_fmms (a,b,c,d,e,r);
  }
};
#endif

class mpfr_div_test {
public:
  int func(mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_div (a,b,c,r);
  }
};
class mpfr_set_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_set (a,b,r);
  }
};
class mpfr_sqrt_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_sqrt (a,b,r);
  }
};
class mpfr_exp_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_exp (a,b,r);
  }
};
class mpfr_expm1_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_expm1 (a,b,r);
  }
};
class mpfr_log_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_log (a,b,r);
  }
};
class mpfr_log1p_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_log1p (a,b,r);
  }
};
class mpfr_erf_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_erf (a,b,r);
  }
};
class mpfr_cos_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_cos (a,b,r);
  }
};
class mpfr_sin_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_sin (a,b,r);
  }
};
class mpfr_tan_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_tan (a,b,r);
  }
};
class mpfr_acos_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_acos (a,b,r);
  }
};
class mpfr_asin_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_asin (a,b,r);
  }
};
class mpfr_atan_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_atan (a,b,r);
  }
};
class mpfr_cosh_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_cosh (a,b,r);
  }
};
class mpfr_sinh_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_sinh (a,b,r);
  }
};
class mpfr_tanh_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_tanh (a,b,r);
  }
};
class mpfr_acosh_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_acosh (a,b,r);
  }
};
class mpfr_asinh_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_asinh (a,b,r);
  }
};
class mpfr_atanh_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_atanh (a,b,r);
  }
};
class mpfr_pow_test {
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    return mpfr_pow (a,b,c,r);
  }
};
class mpfr_get_ld_test {
  long double ld;
public:
  int func (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mp_rnd_t r) {
    ld = mpfr_get_ld (c, r);
    return 0;
  }
};

static mpfr_test<mpfr_add_test> test1 ("mpfr_add");
static mpfr_test<mpfr_sub_test> test2 ("mpfr_sub");
static mpfr_test<mpfr_mul_test> test3 ("mpfr_mul");
static mpfr_test3<mpfr_fma_test> test10 ("mpfr_fma");
static mpfr_test3<mpfr_fms_test> test11 ("mpfr_fms");
#if MPFR_VERSION_MAJOR >= 4
static mpfr_test4<mpfr_fmma_test> test12 ("mpfr_fmma");
static mpfr_test4<mpfr_fmms_test> test13 ("mpfr_fmms");
#endif
static mpfr_test<mpfr_sqr_test> test14 ("mpfr_sqr");
static mpfr_test<mpfr_div_test> test4 ("mpfr_div");
static mpfr_test<mpfr_set_test> test5 ("mpfr_set");

static mpfr_test<mpfr_sqrt_test> test6 ("mpfr_sqrt");
static mpfr_test<mpfr_exp_test>  test7 ("mpfr_exp");
static mpfr_test<mpfr_log_test>  test8 ("mpfr_log");
static mpfr_test<mpfr_log_test>  test9 ("mpfr_erf");

static mpfr_test<mpfr_cos_test>  testA ("mpfr_cos");
static mpfr_test<mpfr_sin_test>  testB ("mpfr_sin");
static mpfr_test<mpfr_tan_test>  testC ("mpfr_tan");
static mpfr_test<mpfr_acos_test> testD ("mpfr_acos");
static mpfr_test<mpfr_asin_test> testE ("mpfr_asin");
static mpfr_test<mpfr_atan_test> testF ("mpfr_atan");
static mpfr_test<mpfr_log1p_test> testG ("mpfr_log1p");
static mpfr_test<mpfr_expm1_test> testH ("mpfr_expm1");

static mpfr_test<mpfr_cosh_test>  testAh ("mpfr_cosh");
static mpfr_test<mpfr_sinh_test>  testBh ("mpfr_sinh");
static mpfr_test<mpfr_tanh_test>  testCh ("mpfr_tanh");
static mpfr_test<mpfr_acosh_test> testDh ("mpfr_acosh");
static mpfr_test<mpfr_asinh_test> testEh ("mpfr_asinh");
static mpfr_test<mpfr_atanh_test> testFh ("mpfr_atanh");
static mpfr_test<mpfr_pow_test>   testGh ("mpfr_pow");

static mpfr_test<mpfr_get_ld_test> testFj ("mpfr_get_ld");

/* Do the test */
template <class T>
bool mpfr_test<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;

  /* Init and set tables if first call */
  if (size == 0) {
    size = base.size ();
    tim = new timming (size);
    table = new mpfr_t[size];
    for (i = 0 ; i < size ; i++) {
      mpfr_init2 (table[i], opt.prec);
      mpfr_set_str (table[i], base[i].c_str(), 10, opt.rnd);
    }
    mpfr_inits2 (opt.prec, a, b, c, NULL);
  }

  /* Do Measure */
  for(i = 0 ; i < (size-1) ; i++) {
    mpfr_set (b, table[i], opt.rnd);
    mpfr_set (c, table[i+1], opt.rnd);
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE (f.func (a, b, c, opt.rnd) );
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}

/* Do the test */
template <class T>
bool mpfr_test3<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;

  /* Init and set tables if first call */
  if (size == 0) {
    size = base.size ();
    tim = new timming (size-2);
    table = new mpfr_t[size];
    for (i = 0 ; i < size ; i++) {
      mpfr_init2 (table[i], opt.prec);
      mpfr_set_str (table[i], base[i].c_str(), 10, opt.rnd);
    }
    mpfr_inits2 (opt.prec, a, b, c, d, NULL);
  }

  /* Do Measure */
  for(i = 0 ; i < (size-2) ; i++) {
    mpfr_set (b, table[i], opt.rnd);
    mpfr_set (c, table[i+1], opt.rnd);
    mpfr_set (d, table[i+2], opt.rnd);
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE (f.func (a, b, c, d, opt.rnd) );
    //cout << "m = " << m << endl;
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}

/* Do the test */
template <class T>
bool mpfr_test4<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;

  /* Init and set tables if first call */
  if (size == 0) {
    size = base.size ();
    tim = new timming (size-2);
    table = new mpfr_t[size];
    for (i = 0 ; i < size ; i++) {
      mpfr_init2 (table[i], opt.prec);
      mpfr_set_str (table[i], base[i].c_str(), 10, opt.rnd);
    }
    mpfr_inits2 (opt.prec, a, b, c, d, e, NULL);
  }

  /* Do Measure */
  for(i = 0 ; i < (size-3) ; i++) {
    mpfr_set (b, table[i], opt.rnd);
    mpfr_set (c, table[i+1], opt.rnd);
    mpfr_set (d, table[i+2], opt.rnd);
    mpfr_set (e, table[i+3], opt.rnd);
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE (f.func (a, b, c, d, e, opt.rnd) );
    //cout << "m = " << m << endl;
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}
