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

#include "cln/cln.h"

#include "timp.h"

using namespace std;
using namespace cln;

/* Register New Test */
template <class T>
class cln_test : public registered_test {
private:
  unsigned long size;
  cl_F **table;
  timming *tim;
public:
  cln_test (const char *n) : registered_test (n), size (0) {}
  ~cln_test () {
    if (size != 0) {
      delete tim;
      for (unsigned long i = 0 ; i < size ; i++) {
	delete table[i];
      }
      delete[] table;
    }
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class cln_add_test {
public:
  void func (cl_F &a, const cl_F &b, const cl_F &c) {
    a = b + c;
  }
};
class cln_sub_test {
public:
  void func (cl_F &a, const cl_F &b, const cl_F &c) {
    a = b - c;
  }
};
class cln_mul_test {
public:
  void func (cl_F &a, const cl_F &b, const cl_F &c) {
    a = b * c;
  }
};
class cln_div_test {
public:
  void func (cl_F &a, const cl_F &b, const cl_F &c) {
    a = b / c;
  }
};

static cln_test<cln_add_test> test1 ("cln_add");
static cln_test<cln_sub_test> test2 ("cln_sub");
static cln_test<cln_mul_test> test3 ("cln_mul");
static cln_test<cln_div_test> test4 ("cln_div");

static bool prec_print = false;

/* Do the test */
template <class T>
bool cln_test<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;

  /* Init and set tables if first call */
  if (size == 0) {
    unsigned long prec=(unsigned long)(((double)opt.prec)*0.3010299956639811);
    if (opt.verbose && !prec_print) {
	cout << " Decimal Prec[CLN]=" << prec << endl;
	prec_print = true;
    }
    size = base.size ();
    tim = new timming (size);
    table = new cl_F *[size];
    // (cl_float (0.0, float_format_t (opt.prec)));
    for (i = 0 ; i < size ; i++) {
      char * Buffer = new char[base[i].size () + 100];
      sprintf (Buffer, "%s_%lu", base[i].c_str (), prec);
      table[i] = new cl_F (Buffer);
      delete[] Buffer;
    }
  }

  cl_F a = cl_float(0.0, float_format_t(opt.prec));
  cl_F b = cl_float(0.0, float_format_t(opt.prec));
  cl_F c = cl_float(0.0, float_format_t(opt.prec));

  /* Do Measure */
  for(i = 0 ; i < (size-1) ; i++) {
    b = *table[i];
    c = *table[i+1];
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE (f.func (a, b, c) );
    cont = tim->update (i, m) || cont;
  }

  tim->print (get_name(), opt);
  return cont;
}
