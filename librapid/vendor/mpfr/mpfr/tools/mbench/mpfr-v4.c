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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "timp.h"

/* To avoid __gmpf_cmp to be declared as pure */
#define __GMP_NO_ATTRIBUTE_CONST_PURE
#include "gmp.h"
#include "mpfr.h"

/* Theses macros help the compiler to determine if a test is likely*/
/* or unlikely. */
#if __GNUC__ >= 3
# define LIKELY(x)   (__builtin_expect(!!(x),1))
# define UNLIKELY(x) (__builtin_expect((x),0))
#else
# define LIKELY(x)   (x)
# define UNLIKELY(x) (x)
#endif

/*
 * List of all the tests to do.
 * Macro "Bench" is defined furthermore
 */
#define TEST_LIST \
  BENCH("MPFR::::::::::", ; ); \
  BENCH("add", mpfr_add(a,b,c,MPFR_RNDN)); \
  BENCH("sub", mpfr_sub(a,b,c,MPFR_RNDN)); \
  BENCH("mul", mpfr_mul(a,b,c,MPFR_RNDN)); \
  BENCH("div", mpfr_div(a,b,c,MPFR_RNDN)); \
  BENCH("sqrt", mpfr_sqrt(a,b,MPFR_RNDN)); \
  BENCH("cmp", mpfr_cmp(b,c)); \
  BENCH("set", mpfr_set(a,b, MPFR_RNDN)); \
  BENCH("set0", mpfr_set_ui(a,0,MPFR_RNDN)); \
  BENCH("set1", mpfr_set_ui(a,1,MPFR_RNDN)); \
  BENCH("setz", mpfr_set_z(a,zz,MPFR_RNDN)); \
  BENCH("swap", mpfr_swap(b,c)); \
  BENCH("MPF:::::::::::", ; ); \
  BENCH("add", mpf_add(x,y,z)); \
  BENCH("sub", mpf_sub(x,y,z)); \
  BENCH("mul", mpf_mul(x,y,z)); \
  BENCH("div", mpf_div(x,y,z)); \
  BENCH("sqrt", mpf_sqrt(x,y)); \
  BENCH("cmp", mpf_cmp(y,z)); \
  BENCH("set", mpf_set(x,y)); \
  BENCH("set0", mpf_set_ui(x,0)); \
  BENCH("set1", mpf_set_ui(x,1)); \
  BENCH("swap", mpf_swap(y,z));


#define USAGE                                                                \
 "Bench the low-level functions of Mpfr (V4).\n"                             \
  __FILE__" " __DATE__" " __TIME__" GCC "__VERSION__ "\n"                    \
 "Usage: mpfr_bench [-pPRECISION] [-sRANDSEED] [-mSTAT_SIZE] [-v]\n"         \
 " [-paPREC_RESULT] [-pbPREC_OP1] [-pcPREC_OP2] [-bOP1_VALUE] [-cOP2_VALUE]\n"

int verbose = 0;

void mpfr_bench(mpfr_prec_t prec_a, mpfr_prec_t prec_b, mpfr_prec_t prec_c,
		const char *b_str, const char *c_str, unsigned long seed)
{
  mpfr_t a,b,c;
  mpf_t x,y,z;
  mpz_t zz;
  gmp_randstate_t state;

  gmp_randinit_lc_2exp_size (state, 128);
  gmp_randseed_ui (state, seed);

  mpfr_init2(a, prec_a);
  mpfr_init2(b, prec_b);
  mpfr_init2(c, prec_c);

  mpf_init2(x,  prec_a);
  mpf_init2(y,  prec_b);
  mpf_init2(z,  prec_c);

  if (b_str)
    mpf_set_str(y, b_str, 10);
  else
    mpf_urandomb(y, state, prec_b);
  if (c_str)
    mpf_set_str(z, c_str, 10);
  else
    mpf_urandomb(z, state, prec_c);
  mpfr_set_f(b, y, MPFR_RNDN);
  mpfr_set_f(c, z, MPFR_RNDN);
  mpz_init (zz);
  mpz_urandomb (zz, state, 2*prec_b);

  if (verbose)
    {
      printf("B="); mpfr_out_str(stdout, 10, 0, b, MPFR_RNDD);
      printf("\nC="); mpfr_out_str(stdout, 10, 0, c, MPFR_RNDD);
      putchar('\n');
    }
  TIMP_OVERHEAD ();
#undef BENCH
#define BENCH(TEST_STR, TEST) printf(" "TEST_STR": %Lu\n", TIMP_MEASURE(TEST))
  TEST_LIST;

  mpz_clear (zz);
  mpfr_clear (a);
  mpfr_clear (b);
  mpfr_clear (c);
  mpf_clear (x);
  mpf_clear (y);
  mpf_clear (z);
  gmp_randclear (state);
}

#define MAX_OP 40
void mpfr_stats (unsigned long num, mpfr_prec_t prec_a, mpfr_prec_t prec_b,
		 mpfr_prec_t prec_c, unsigned long seed)
{
  mpf_t xt[num],yt[num],zt[num];
  unsigned long long mc[num][MAX_OP], m;
  mpfr_t a, b, c;
  mpf_t x, y, z;
  mpz_t zz;
  unsigned long long min,max,moy;
  gmp_randstate_t state;
  int i,j=0, op, cont;
  int imin=0, imax=0;

  mpf_init2(x, prec_a);
  mpf_init2(y, prec_b);
  mpf_init2(z, prec_c);

  mpfr_init2(a, prec_a);
  mpfr_init2(b, prec_b);
  mpfr_init2(c, prec_c);

  gmp_randinit_lc_2exp_size (state, 128);
  gmp_randseed_ui (state, seed);

  mpz_init (zz);
  mpz_urandomb (zz, state, 2*prec_b);

  TIMP_OVERHEAD ();

  for(i = 0 ; i < num ; i++)
    {
      mpf_init2(xt[i],  prec_a);
      mpf_init2(yt[i],  prec_b);
      mpf_init2(zt[i],  prec_c);
      mpf_urandomb(yt[i], state, prec_b);
      yt[i][0]._mp_exp += (rand() % prec_b) / GMP_NUMB_BITS;
      mpf_urandomb(zt[i], state, prec_c);
      /* zt[i][0]._mp_exp += (rand() % prec_c) / GMP_NUMB_BITS; */
      for(op = 0 ; op < MAX_OP ; op++)
	mc[i][op] = 0xFFFFFFFFFFFFFFFLL;
    }

  for(j = 0, cont = 5 ; cont ; j++, cont--)
    {
      printf("Pass %d...\n", j+1);
      for(i = 0 ; i < num ; i++)
	{
	  op = 0;
	  mpf_set(y,yt[i]);
	  mpf_set(z,zt[i]);
	  mpfr_set_f(b, yt[i], MPFR_RNDN);
	  mpfr_set_f(c, zt[i], MPFR_RNDN);
#undef BENCH
#define BENCH(TEST_STR, TEST)                                           \
 m = TIMP_MEASURE(TEST); if (m < mc[i][op]) {mc[i][op] = m; cont = 4;} op++;
	  TEST_LIST;
	}

#undef BENCH
#define BENCH(TEST_STR, TEST)                                       \
  min = 0xFFFFFFFFFFFFFFFLL; max = 0LL; moy = 0LL;                  \
  for(i = 0 ; i < num ; i++) {                                      \
      if (mc[i][op] < min) imin = i, min = mc[i][op];               \
      if (mc[i][op] > max) imax = i, max = mc[i][op];               \
      moy += mc[i][op];                                             \
    }                                                               \
  printf(" %s: %Lu / %Lu.%02Lu / %Lu", TEST_STR, min,               \
	 (unsigned long long) moy/num, (moy*100LL/num)%100LL, max); \
  if (verbose) printf ("\tMIN:%e,%e\tMAX:%e,%e", mpf_get_d(yt[imin]),\
		       mpf_get_d(zt[imin]), mpf_get_d(yt[imax]),    \
		       mpf_get_d(zt[imax]));                        \
  putchar ('\n');                                                   \
  op++;

      op =0;
      TEST_LIST;
     }

  printf("End\n");
  mpz_clear (zz);
  mpfr_clear(a);
  mpfr_clear(b);
  mpfr_clear(c);
  mpf_clear(x);
  mpf_clear(y);
  mpf_clear(z);
  for(i = 0 ; i < num ; i++)
    {
      mpf_clear(xt[i]);
      mpf_clear(yt[i]);
      mpf_clear(zt[i]);
    }
  gmp_randclear(state);
}

int main(int argc, const char *argv[])
{
  mpfr_prec_t prec_a, prec_b, prec_c;
  unsigned long seed, stat;
  int i;
  const char *b_strptr, *c_strptr;

  printf(USAGE);

  prec_a = prec_b = prec_c = 53;
  b_strptr = c_strptr = NULL;
  seed = 14528596;
  stat = 0;

  for(i = 1 ; i < argc ; i++)
    {
      if (argv[i][0] == '-')
	{
	  switch (argv[i][1])
	    {
	    case 'b':
	      b_strptr = (const char *) (argv[i]+2);
	      break;
	    case 'c':
	      c_strptr = (const char *) (argv[i]+2);
	      break;
	    case 'p':
	      switch (argv[i][2])
		{
		case 'a':
		  prec_a = atol(argv[i]+3);
		  break;
		case 'b':
		  prec_b = atol(argv[i]+3);
		  break;
		case 'c':
		  prec_c = atol(argv[i]+3);
		  break;
		default:
		  prec_a = prec_b = prec_c = atol(argv[i]+2);
		  break;
		}
	      break;
	    case 's':
	      seed = atol(argv[i]+2);
	      break;
 	    case 'm':
	      stat = atol(argv[i]+2);
	      break;
	    case 'v':
	      verbose = 1;
	      break;
	    default:
	      fprintf(stderr, "Unkwown option: %s\n", argv[i]);
	      break;
	    }
	}
    }
  /* Set low priority */
  setpriority(PRIO_PROCESS,0,15);

  if (stat)
    mpfr_stats(stat, prec_a, prec_b, prec_c, seed);
  else
    mpfr_bench(prec_a, prec_b, prec_c, b_strptr, c_strptr, seed);

  return 0;
}
