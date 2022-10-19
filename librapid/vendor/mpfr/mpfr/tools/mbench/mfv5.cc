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

#include <mpfr.h>

#include "timp.h"

#include "mfv5.h"

#define USAGE                                                           \
 "Bench functions for Pentium (V5++).\n"                                \
 __FILE__ " " __DATE__ " " __TIME__ " GCC " __VERSION__ "\n"		\
 "Usage: mfv5 [-pPREC] [-sSEED] [-mSIZE] [-iPRIO] [-lLIST] [-xEXPORT_BASE] [-XIMPORT_BASE] [-rROUNDING_MODE] [-eEXP] [-dDIFF] tests ...\n"

using namespace std;

registered_test *first_registered_test = 0;

bool
do_test (const char *n, const vector<string> &base, const option_test &opt) {
  registered_test *p;
  int all = strcmp (n, "all") == 0;
  bool ret = false;

  TIMP_OVERHEAD ();

  p = first_registered_test;
  while (p) {
    if (all || p->is (n))
      ret = p->test (base, opt) || ret;
    p = p->next ();
  }
  return ret;
}

void
list_test (void) {
  registered_test *p;

  p = first_registered_test;
  while (p) {
    cout << p->get_name () << endl;
    p = p->next ();
  }
  cout << "all\n";
  return;
}

void
build_base (vector<string> &base, const option_test &opt)
{
  unsigned long i, n = opt.stat;
  mpfr_t x;
  gmp_randstate_t state;
  const char *str;
  mpfr_exp_t e;
  char *buffer;
  FILE *f = fopen (opt.export_base.c_str(), "w");

  mpfr_init2 (x, opt.prec);
  gmp_randinit_lc_2exp_size (state, 128);
  gmp_randseed_ui (state, opt.seed);

  for (i = 0 ; i < n ; i++) {
    mpfr_urandomb (x, state);
    if (opt.exp_diff == -1)
      mpfr_mul_2si  (x, x, (rand() % opt.max_exp) - (opt.max_exp / 2), MPFR_RNDN);
    else if (opt.exp_diff == -2)
      mpfr_set_exp (x, opt.max_exp);
    else /* set the exponent to -i*exp_diff */
      mpfr_set_exp (x, -(long) i * opt.exp_diff);
    str = mpfr_get_str (NULL, &e, 10, 0, x, MPFR_RNDN);
    if (str == 0)
      abort ();
    buffer = (char *) malloc (strlen(str)+50);
    if (buffer == 0)
      abort ();
    sprintf (buffer, "%sE%ld", str, (long) e - (long) strlen(str));
    if (f)
      fprintf (f, "%s\n", buffer);
    base.push_back (buffer);
    if (opt.verbose)
      mpfr_printf ("[%lu] = %Re\n", i, x);
    free (buffer);
    mpfr_free_str ((char*)str);
  }
  if (f)
    fclose (f);

  gmp_randclear(state);
  mpfr_clear (x);
}

void
read_base (vector<string> &base, const option_test &opt)
{
  unsigned long i, n = opt.stat;
  std::string x;
  std::ifstream f(opt.import_base.c_str());
  std::cout << "Read data from " << opt.import_base << std::endl;

  for (i = 0 ; i < n ; i++) {
    getline(f, x);
    base.push_back (x.c_str());
  }
}


int main (int argc, const char *argv[])
{
  option_test options;
  vector<string> base;
  int i, j, cont, prio;

  /* Parse option */
  prio = 19;
  for(i = 1 ; i < argc ; i++)
    {
      if (argv[i][0] == '-')
	{
	  switch (argv[i][1])
	    {
	    case 'h':
	      cout << USAGE;
	      exit (0);
	      break;
	    case 'p':
	      options.prec = atol (argv[i]+2);
	      break;
	    case 's':
	      options.seed = atol (argv[i]+2);
	      break;
 	    case 'm':
	      options.stat = atol (argv[i]+2);
	      break;
	    case 'v':
	      options.verbose = true;
	      break;
	    case 'i':
	      prio = atol (argv[i]+2);
	      break;
	    case 'e':
              options.max_exp = atol (argv[i]+2);
	      assert (options.max_exp > 0);
              break;
	    case 'd':
              options.exp_diff = atol (argv[i]+2);
	      assert (options.exp_diff >= -2);
	      /* exp_diff = -1 (default): exponent is chosen in [-e/2,e/2]
		 exp_dif >=0: the exponent of the ith value is -i*exp_diff
		 exp_dif = -2: the exponent is exactly e */
              break;
            case 'r':
              {
                switch (argv[i][2])
		  {
		  case 'n':
		  case 'N':
		    options.rnd = MPFR_RNDN;
		    break;
		  case 'z':
		  case 'Z':
		    options.rnd = MPFR_RNDZ;
		    break;
		  case 'u':
		  case 'U':
		    options.rnd = MPFR_RNDU;
		    break;
		  case 'd':
		  case 'D':
		    options.rnd = MPFR_RNDD;
		    break;
		  case 'f':
		  case 'F':
		    options.rnd = MPFR_RNDF;
		    break;
		  case 'a':
		  case 'A':
		    options.rnd = MPFR_RNDA;
		    break;
		  default:
		    cerr << "Unknown rounding mode." << endl;
		    exit(1);
		    break;
		  }
              }
              break;
	    case 'l':
	      list_test ();
	      exit (0);
	      break;
	    case 'x':
	      options.export_base = (argv[i]+2);
	      break;
	    case 'X':
	      options.import_base = (argv[i]+2);
	      break;
	    default:
	      cerr <<  "Unkwown option:" << argv[i] << endl;
	      exit (1);
	      break;
	    }
	}
    }

  if (options.verbose)
    cout << USAGE;

  /* Set low priority */
  setpriority(PRIO_PROCESS, 0, prio);

  /* Build Used Base */
  if (options.verbose)
    cout << "Building DATA Base\n";
  mp_set_memory_functions (NULL, NULL, NULL);


  cout << "GMP VERSION HEADER= " << __GNU_MP_VERSION  << "." << __GNU_MP_VERSION_MINOR << "." << __GNU_MP_VERSION_PATCHLEVEL << " LIB=" << gmp_version << endl;
  cout << "MPFR VERSION HEADER= " << MPFR_VERSION_STRING << " LIB=" << mpfr_get_version() << endl;

  if (options.import_base != "")
    read_base (base, options);
  else
    build_base (base, options);

  /* Do test */
  for (j = 1, cont = 5 ; cont ; j++, cont--) {
    cout << "Pass " << j << endl;
    for(i = 1 ; i < argc ; i++)
      if (argv[i][0] != '-')
	if (do_test (argv[i], base, options))
	  cont = 5;
  }

  /* Final */
  if (options.verbose)
    cout << "Used precision: " << options.prec << endl;

  return 0;
}
