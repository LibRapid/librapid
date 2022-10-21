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

#include "pari/pari.h"

using namespace std;

static int pari_init_cpt = 0;

/* Register New Test */
template <class T>
class pari_test : public registered_test {
private:
  unsigned long size;
  GEN *table;
  GEN a, b, c;
  timming *tim;
public:
  pari_test (const char *n) : registered_test (n), size (0) {
    if (pari_init_cpt == 0)
      pari_init (40000000, 10000);
    pari_init_cpt ++;
  }
  ~pari_test () {
    if (size != 0)
      delete[] table;
    if (-- pari_init_cpt == 0)
      (void) 0; // pari_clear ();
  }
  bool test (const vector<string> &base, const option_test &opt);
};

class pari_add_test {
public:
  void func (GEN a, GEN b, GEN c) {
    mpaddz (b, c, a);
  }
};

class pari_sub_test {
public:
  void func (GEN a, GEN b, GEN c) {
    mpsubz (b, c, a);
  }
};

class pari_mul_test {
public:
  void func (GEN a, GEN b, GEN c) {
    mpmulz (b, c, a);
  }
};

class pari_div_test {
public:
  void func (GEN a, GEN b, GEN c) {
    mpdivz (b, c, a);
  }
};

static pari_test<pari_add_test> test1 ("pari_add");
static pari_test<pari_sub_test> test2 ("pari_sub");
static pari_test<pari_mul_test> test3 ("pari_mul");
static pari_test<pari_div_test> test4 ("pari_div");


/* Do the test */
template <class T>
bool pari_test<T>::test (const vector<string> &base, const option_test &opt) {
  unsigned long i;
  unsigned long long m;
  T f;
  bool cont = false;
  GEN stck;

  /* Init and set tables if first call */
  if (size == 0) {
    size = base.size ();
    tim = new timming (size);
    table = new GEN[size];
    /* FIXME: How to really fix the size of table[i]? */
    for (i = 0 ; i < size ; i++)
      table[i] = flisexpr((char *) base[i].c_str());
    a = gsqrt(stoi(3), (opt.prec - 1)/BITS_IN_LONG + 1 + 2);
    b = gsqrt(stoi(5), (opt.prec - 1)/BITS_IN_LONG + 1 + 2);
    c = gsqrt(stoi(7), (opt.prec - 1)/BITS_IN_LONG + 1 + 2);
  }

  /* Do Measure */
  stck = (GEN) avma;
  for(i = 0 ; i < (size-1) ; i++) {
    mpaff (table[i], b);
    mpaff (table[i+1], c);
    TIMP_OVERHEAD ();
    m = TIMP_MEASURE(f.func (a, b, c) );
    cont = tim->update (i, m) || cont;
  }
  avma = (ulong) stck;

  tim->print (get_name(), opt);
  return cont;
}
