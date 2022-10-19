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
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "gmp.h"
#include "mpfr.h"

#ifndef mpfr_version
# define mpfr_version "< 2.1.0"
#endif

#include "timp.h"
#include "generate.c"

#ifdef SCS_SUPPORT
# define SCS(x) x
# include "scs.h"
# define EXTRA_TEST_LIST \
 BENCH("SCSLIB( Compiled var ):::::::::::", ; ); \
 BENCH("add", scs_add(sc1,sc2,sc3)); \
 BENCH("sub", scs_sub(sc1,sc2,sc3)); \
 BENCH("mul", scs_mul(sc1,sc2,sc3)); \
 BENCH("div", scs_div(sc1,sc2,sc3)); \
 BENCH("set", scs_set(sc1,sc2)); \
 BENCH("set0", scs_set_si(sc1,0)); \
 BENCH("set1", scs_set_si(sc1,1));
#else
# define SCS(x) (void) 0
# define EXTRA_TEST_LIST (void)0;
#endif

/*
 * List of all the tests to do.
 * Macro "Bench" is defined furthermore
 */
#define TEST_LIST1 \
  BENCH("MPFR::::::::::", ; ); \
  BENCH("add", mpfr_add(a,b,c,MPFR_RNDN)); \
  BENCH("sub", mpfr_sub(a,b,c,MPFR_RNDN)); \
  BENCH("mul", mpfr_mul(a,b,c,MPFR_RNDN)); \
  BENCH("div", mpfr_div(a,b,c,MPFR_RNDN)); \
  BENCH("sqrt", mpfr_sqrt(a,b,MPFR_RNDN)); \
  BENCH("cmp", mpfr_cmp(b,c)); \
  BENCH("sgn", (mpfr_sgn)(b)); \
  BENCH("set", mpfr_set(a,b, MPFR_RNDN)); \
  BENCH("set0", mpfr_set_si(a,0,MPFR_RNDN)); \
  BENCH("set1", mpfr_set_si(a,1,MPFR_RNDN)); \
  BENCH("swap", mpfr_swap(b,c)); \
  BENCH("MPF:::::::::::", ; ); \
  BENCH("add", mpf_add(x,y,z)); \
  BENCH("sub", mpf_sub(x,y,z)); \
  BENCH("mul", mpf_mul(x,y,z)); \
  BENCH("div", mpf_div(x,y,z)); \
  BENCH("sqrt", mpf_sqrt(x,y)); \
  BENCH("cmp", (mpf_cmp)(y,z)); \
  BENCH("set", mpf_set(x,y)); \
  BENCH("set0", mpf_set_si(x,0)); \
  BENCH("set1", mpf_set_si(x,1)); \
  BENCH("swap", mpf_swap(y,z)); \
  EXTRA_TEST_LIST

#define TEST_LIST2 \
  BENCH("mpfr_exp", mpfr_exp(a,b,MPFR_RNDN)); \
  BENCH("mpfr_log", mpfr_log(a,b,MPFR_RNDN)); \
  BENCH("mpfr_sin", mpfr_sin(a,b,MPFR_RNDN)); \
  BENCH("mpfr_cos", mpfr_cos(a,b,MPFR_RNDN)); \
  BENCH("mpfr_tan", mpfr_tan(a,b,MPFR_RNDN)); \
  BENCH("mpfr_asin", mpfr_asin(a,b,MPFR_RNDN)); \
  BENCH("mpfr_acos", mpfr_acos(a,b,MPFR_RNDN)); \
  BENCH("mpfr_atan", mpfr_atan(a,b,MPFR_RNDN)); \
  BENCH("mpfr_agm", mpfr_agm(a,b,c,MPFR_RNDN)); \
  BENCH("mpfr_const_log2", (mpfr_const_log2) (a, MPFR_RNDN)); \
  BENCH("mpfr_const_pi", (mpfr_const_pi)(a, MPFR_RNDN)); \
  BENCH("mpfr_sinh", mpfr_sinh(a,b,MPFR_RNDN)); \
  BENCH("mpfr_cosh", mpfr_cosh(a,b,MPFR_RNDN)); \
  BENCH("mpfr_tanh", mpfr_tanh(a,b,MPFR_RNDN)); \
  BENCH("mpfr_asinh", mpfr_asinh(a,b,MPFR_RNDN)); \
  BENCH("mpfr_acosh", mpfr_acosh(a,b,MPFR_RNDN)); \
  BENCH("mpfr_atanh", mpfr_atanh(a,b,MPFR_RNDN)); \
  BENCH("exp", d1 = exp(d2)); \
  BENCH("log", d1 = log(d2)); \
  BENCH("sin", d1 = sin(d2)); \
  BENCH("cos", d1 = cos(d2)); \
  BENCH("tan", d1 = tan(d2)); \
  BENCH("asin", d1 = asin(d2)); \
  BENCH("acos", d1 = acos(d2)); \
  BENCH("atan", d1 = atan(d2));

#define TEST_LIST3 \
  BENCH("mpfr_cos", mpfr_cos(a,b,MPFR_RNDN));

#define TEST_LIST4 \
  BENCH("get_d", d1 = mpfr_get_d (b, MPFR_RNDN)); \
  BENCH("set_d", mpfr_set_d (b, d2, MPFR_RNDN));  \
  BENCH("mul_ui", mpfr_mul_si (b, b, 123, MPFR_RNDN));

#ifndef TEST_LIST
# define TEST_LIST TEST_LIST2
#endif

#define USAGE \
 "Bench the low-level functions of Mpfr (V6).\n" \
  __FILE__" " __DATE__" " __TIME__" GCC "__VERSION__ "\n"\
 "Usage: mpfr_v6 [-pPRECISION] [-mSIZE] [-v] [-ffilename]\n" \
 " [-oFUNC] [-l] [-goutgfxname] [-sSMOOTH] [-rGRANULARITY]\n"

int verbose = 0;

void mpf_set_fr (mpf_t dest, mpfr_t src, mp_rnd_t rnd)
{
  mpfr_exp_t exp;
  char *tmp, *tmp2;
  long len;

  tmp = mpfr_get_str(NULL, &exp, 10, 0, src, rnd);
  len = strlen(tmp);
  tmp2 = (char *) malloc(len+30);
  if (tmp2 == NULL)
    {
      fprintf(stderr, "mpf_set_mpfr: error memory\n");
      exit (1);
    }
  sprintf(tmp2, "%s@%ld", tmp, exp-len);
  mpf_set_str(dest, tmp2, -10);
  free(tmp);
  free(tmp2);
}

#define MAX_OP 40
void make_stats(const char *filename, int num, mpfr_prec_t prec, int select_op,
		const char *outputname, int smooth, int granularity)
{
  mpfr_t tab[num+1];
  unsigned long long mc[num][MAX_OP], m;
  mpfr_t a, b, c;
  mpf_t x, y, z;
  double d1, d2, d3;
  unsigned long long min, max, moy;
  int i, j, op, cont/*, min_i, max_i*/;
  SCS(( scs_t sc1, sc2, sc3 ));

  /* INIT */
  mpf_init2 (x, prec);
  mpf_init2 (y, prec);
  mpf_init2 (z, prec);
  mpfr_init2 (a, prec);
  mpfr_init2 (b, prec);
  mpfr_init2 (c, prec);
  for(i = 0 ; i < num ; i++)
    {
      mpfr_init2 (tab[i], prec);
      for(op = 0 ; op < MAX_OP ; op++)
	mc[i][op] = 0xFFFFFFFFFFFFFFFLL;
    }
  mpfr_init2 (tab[i], prec);

  /* SET */
  gnumb_read (filename, tab, num+1);

  TIMP_OVERHEAD ();

  for(j = 0, cont = smooth ; cont ; j++, cont--)
    {
      printf("Pass %d...\n", j+1);
      for(i = 0 ; i < num ; i++)
	{
	  mpfr_set (b, tab[i+0], MPFR_RNDN);
	  mpfr_set (c, tab[i+1], MPFR_RNDN);
	  mpf_set_fr (y, b, MPFR_RNDN);
	  mpf_set_fr (z, c, MPFR_RNDN);
	  SCS(( scs_set_mpfr(sc2, b), scs_set_mpfr(sc3, c) ));
	  d1 = d2 = mpfr_get_d1 (b);
	  d3 = mpfr_get_d1 (c);
#undef BENCH
#define BENCH(TEST_STR, TEST)                              \
   if (op==select_op || select_op<0)                       \
      {m = TIMP_MEASURE(TEST);                             \
       if (m < mc[i][op]) {mc[i][op] = m; cont = smooth;}} \
   op++;
	  op = 0;
	  TEST_LIST;
	}

#undef BENCH
#define BENCH(TEST_STR, TEST)                      \
 if (op==select_op || select_op<0)  {              \
  min = 0xFFFFFFFFFFFFFFFLL; max = 0LL; moy = 0LL; \
  for(i = 0 ; i < num ; i++) {                     \
      if (mc[i][op] < min) min = mc[i][op];        \
      if (mc[i][op] > max) max = mc[i][op];        \
      moy += mc[i][op];                            \
  }                                                \
  printf(" %s: %Lu / %Lu.%02Lu / %Lu\n", TEST_STR,min,moy/num,(moy*100LL/num)%100LL, max);\
 }                                                 \
  op++;
      op =0;
      TEST_LIST;
     }
  printf("End\n");

  if (verbose && select_op != 0) {
    for (i = 0 ; i < num ; i++) {
      printf ("Tab[%02d]=", i); mpfr_out_str (stdout, 2, 10, tab[i], MPFR_RNDN);
      printf ("\tt=%Lu\n", mc[i][select_op]);
    }
  }

  /* Output GNUPLOT data ? */
  if (outputname != NULL)
    {
      unsigned long count[granularity][MAX_OP];
      FILE *out;
      char filename[100];

      // Get min and max of cycle for all ops
#undef BENCH
#define BENCH(TEST_STR, TEST) \
  if (op==select_op || select_op<0) \
     {for(i = 0 ; i < num ; i++) {\
      if (mc[i][op] < min) min = mc[i][op];\
      if (mc[i][op] > max) max = mc[i][op];}} op++;
      min = 0xFFFFFFFFFFFFFFFLL; max = 0LL; moy = 0LL; op = 0;
      TEST_LIST;
      // Count it
#undef BENCH
#define BENCH(TEST_STR, TEST) \
  if (op==select_op || select_op<0) \
   {for(i = 0 ; i < num ; i++) count[(mc[i][op]-min)*granularity/max][op]++;} op++;
      memset (count, 0, sizeof(count));
      max -= min-1; op = 0;
      TEST_LIST;

      // Output data
      sprintf(filename, "%s.data", outputname);
      out = fopen(filename, "w");
      if (out == NULL)
	{
	  fprintf(stderr, "ERROR: Can't open %s\n", filename);
	  exit (-2);
	}
      for(i = 0 ; i < granularity ; i++)
	{
	  fprintf (out, "%Lu\t", (min + max * i / granularity));
#undef BENCH
#define BENCH(TEST_STR, TEST) \
  if (op==select_op || select_op<0) \
   fprintf(out, "%lu\t", count[i][op]); op++;
	  op = 0;
	  TEST_LIST;
	  fprintf(out, "\n");
	}
      fclose (out);

      // Output GNUPLOT Info
      sprintf(filename, "%s.gnuplot", outputname);
      out = fopen(filename, "w");
      if (out == NULL)
        {
          fprintf(stderr, "ERROR: Can't open %s\n", filename);
          exit (-2);
        }
      fprintf (out, "set key left\n"
	       "set data style linespoints\n"
	       "plot ");

      // "toto.data" using 1:2 title "mpfr_log",
      // "toto.data" using 1:3 title "mpfr_exp"
#undef BENCH
#define BENCH(TEST_STR, TEST) \
  if (op==select_op || select_op<0) \
   fprintf(out, "%c \"%s.data\" using 1:%d title \"%s\" ", ((i==2) ? ' ' : ','), outputname, i, TEST_STR), i++; op++;
      op = 0; i = 2;
      TEST_LIST;
      fprintf(out, "\nload \"-\"\n");
      fclose (out);
    }

  mpfr_clear(a);
  mpfr_clear(b);
  mpfr_clear(c);
  mpf_clear(x);
  mpf_clear(y);
  mpf_clear(z);
  for(i = 0 ; i < num+1 ; i++)
    mpfr_clear(tab[i]);
}

int main(int argc, const char *argv[])
{
  mpfr_prec_t prec;
  unsigned long stat;
  int i, select_op = -1, smooth = 3, granularity = 10, op;
  const char *filename = "float.data";
  const char *output   = NULL;

  printf(USAGE);

  prec = 53;
  stat = 100;

  for(i = 1 ; i < argc ; i++)
    {
      if (argv[i][0] == '-')
	{
	  switch (argv[i][1])
	    {
	    case 'l':
#undef BENCH
#define BENCH(STR,TEST) printf("%d: %s\n", op, STR); op++;
	      op = 0;
	      TEST_LIST;
	      exit (0);
	      break;
	    case 'p':
	      prec = atol(argv[i]+2);
	      break;
 	    case 'm':
	      stat = atol(argv[i]+2);
	      break;
	    case 'v':
	      verbose = 1;
	      break;
	    case 'f':
	      filename = &argv[i][2];
	      break;
	    case 'o':
	      select_op = atoi(argv[i]+2);
	      break;
	    case 'g':
	      output  = &argv[i][2];
	      break;
	    case 's':
	      smooth  = atoi(argv[i]+2);
	      break;
	    case 'r':
	      granularity = atoi(argv[i]+2);
	      break;
	    default:
	      fprintf(stderr, "Unkwown option: %s\n", argv[i]);
	      exit (1);
	      break;
	    }
	}
    }
  /* Set low priority */
  setpriority(PRIO_PROCESS,0,15);
  printf("GMP: %s\tMPFR: %s\t DATA: %s\n",
	 gmp_version, mpfr_version, filename);
  make_stats (filename, stat, prec, select_op, output, smooth, granularity);

  return 0;
}
