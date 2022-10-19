/*
Copyright 2005-2022 Free Software Foundation, Inc.
Contributed by Patrick Pelissier and Paul Zimmermann, INRIA.

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
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "timp.h"

/* To avoid __gmpf_cmp to be declared as pure */
#define __GMP_NO_ATTRIBUTE_CONST_PURE
#include "gmp.h"
#include "mpfr.h"

#ifdef SCS_SUPPORT
# define SCS(x) x
# include "scs.h"
# define EXTRA_TEST_LIST \
 BENCH("SCSLIB( Compiled var ):::::::::::", ; ); \
 BENCH("scs_add", scs_add(sc1,sc2,sc3)); \
 BENCH("scs_sub", scs_sub(sc1,sc2,sc3)); \
 BENCH("scs_mul", scs_mul(sc1,sc2,sc3)); \
 BENCH("scs_div", scs_div(sc1,sc2,sc3)); \
 BENCH("scs_set", scs_set(sc1,sc2)); \
 BENCH("scs_set0", scs_set_si(sc1,0)); \
 BENCH("scs_set1", scs_set_si(sc1,1));
#else
# define SCS(x) ((void) 0)
# define EXTRA_TEST_LIST ((void)0);
#endif

#undef EXTRA_TEST_LIST
# define EXTRA_TEST_LIST \
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
  BENCH("mpfr_atanh", mpfr_atanh(a,b,MPFR_RNDN));



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
 * Macro "BENCH" is defined below.
 */
#define TEST_LIST \
  BENCH("MPFR::::::::::", ; ); \
  BENCH("mpfr_add", mpfr_add(a,b,c,MPFR_RNDN)); \
  BENCH("mpfr_sub", mpfr_sub(a,b,c,MPFR_RNDN)); \
  BENCH("mpfr_mul", mpfr_mul(a,b,c,MPFR_RNDN)); \
  BENCH("mpfr_div", mpfr_div(a,b,c,MPFR_RNDN)); \
  BENCH("mpfr_sqrt", mpfr_sqrt(a,b,MPFR_RNDN)); \
  BENCH("mpfr_cmp", mpfr_cmp(b,c)); \
  BENCH("mpfr_sgn", mpfr_sgn(b)); \
  BENCH("mpfr_set", mpfr_set(a,b, MPFR_RNDN)); \
  BENCH("mpfr_set0", mpfr_set_si(a,0,MPFR_RNDN)); \
  BENCH("mpfr_set1", mpfr_set_si(a,1,MPFR_RNDN)); \
  BENCH("mpfr_swap", mpfr_swap(b,c)); \
  BENCH("MPF:::::::::::", ; ); \
  BENCH("mpf_add", mpf_add(x,y,z)); \
  BENCH("mpf_sub", mpf_sub(x,y,z)); \
  BENCH("mpf_mul", mpf_mul(x,y,z)); \
  BENCH("mpf_div", mpf_div(x,y,z)); \
  BENCH("mpf_sqrt", mpf_sqrt(x,y)); \
  BENCH("mpf_cmp", mpf_cmp(y,z)); \
  BENCH("mpf_set", mpf_set(x,y)); \
  BENCH("mpf_set0", mpf_set_si(x,0)); \
  BENCH("mpf_set1", mpf_set_si(x,1)); \
  BENCH("mpf_swap", mpf_swap(y,z)); \
  EXTRA_TEST_LIST

#define USAGE                                                           \
  "Get the graph of the low-level functions of Mpfr (gnuplot).\n"       \
          __FILE__" " __DATE__" " __TIME__" GCC "__VERSION__ "\n"       \
  "Usage: mpfr-gfx [-bPREC_BEGIN] [-ePREC_END] [-sPREC_STEP] [-rPREC_RATIO]\n" \
  "       [-mSTAT_SIZE] [-oFILENAME] [-xFUNCTION_NUM] [-yFUNCTION_NUM] [-c]\n" \
  "       [-fSMOOTH] [-p]\n"

unsigned long num;
mpf_t *xt, *yt, *zt;
int smooth = 3; /* (default) minimal number of routine calls for each number */

void lets_start(unsigned long n, mpfr_prec_t p)
{
  unsigned long i;
  gmp_randstate_t state;

  num = n;
  xt = malloc(sizeof(mpf_t) * num);
  yt = malloc(sizeof(mpf_t) * num);
  zt = malloc(sizeof(mpf_t) * num);
  if (xt==NULL || yt==NULL || zt==NULL)
    {
      fprintf(stderr, "Can't allocate tables!\n");
      abort();
    }

  gmp_randinit_lc_2exp_size (state, 128);
  gmp_randseed_ui (state, 1452369);
  for(i = 0 ; i < num ; i++)
    {
      mpf_init2(xt[i], p);
      mpf_init2(yt[i], p);
      mpf_init2(zt[i], p);
      mpf_urandomb(yt[i], state, p);
      mpf_urandomb(zt[i], state, p);
    }
  gmp_randclear(state);
}

void lets_end(void)
{
  unsigned long i;

  for(i = 0 ; i < num ; i++)
    {
      mpf_clear(xt[i]);
      mpf_clear(yt[i]);
      mpf_clear(zt[i]);
    }
  free (xt);
  free (yt);
  free (zt);
}

double get_speed(mpfr_prec_t p, int select)
{
  unsigned long long mc[num], m;
  mpfr_t a,b,c;
  mpf_t x,y,z;
  unsigned long long moy;
  int i,j=0, op, cont, print_done = 0;
  const char *str = "void";
  SCS(( scs_t sc1, sc2, sc3 ));

  mpf_init2(x, p);  mpf_init2(y, p);  mpf_init2(z, p);
  mpfr_init2(a, p); mpfr_init2(b, p); mpfr_init2(c, p);

  for(i = 0 ; i < num ; i++)
    {
      //      yt[i][0]._mp_exp = (rand() % p) / GMP_NUMB_BITS;
      //zt[i][0]._mp_exp = (rand() % p) / GMP_NUMB_BITS;
      mc[i] = 0xFFFFFFFFFFFFFFFLL;
    }

  TIMP_OVERHEAD ();

  /* we perform at least smooth loops */
  for(j = 0, cont = smooth ; cont ; j++, cont--)
    {
      /* we loop over each of the num random numbers */
      for(i = 0 ; i < num ; i++)
	{
	  /* Set var for tests */
	  mpf_set(y, yt[i]);
	  mpf_set(z, zt[i]);
	  mpfr_set_f(b, yt[i], MPFR_RNDN);
	  mpfr_set_f(c, zt[i], MPFR_RNDN);
	  SCS(( scs_set_mpfr(sc2, b), scs_set_mpfr(sc3, c) ));
	  /* if the measured time m is smaller than the smallest one
	     observed so far mc[i] for the i-th random number, we start
	     again the smooth loops */
#undef BENCH
#define BENCH(TEST_STR, TEST)                           \
	  if (op++ == select) {                         \
	      m = TIMP_MEASURE(TEST);                   \
              str = TEST_STR;                           \
	      if (m < mc[i]) {mc[i] = m; cont = smooth;}\
	    }
	  op = 0;
	  TEST_LIST;
	  if (print_done == 0 && strcmp (str, "void") != 0 )
	    {
	      printf("Prec=%4.4lu Func=%20.20s", p, str);
	      fflush (stdout);
	      print_done = 1;
	    }
	}
    }
  mpfr_clear(a);  mpfr_clear(b);  mpfr_clear(c);
  mpf_clear(x);   mpf_clear(y);   mpf_clear(z);
  /* End */
  /* Save result */
  moy = mc[0];
  for(i = 1 ; i < num ; i++) moy += mc[i];
  printf(" Pass=%4.4d..................%Lu.%Lu\n",
	 j+1, moy/num, (moy*100LL/num)%100LL);
  return (double) (moy) / (double) num;
}

/* compares two functions given by indices select1 and select2
   (by default select1 refers to mpfr and select2 to mpf).

   If postscript=0, output is plain gnuplot;
   If postscript=1, output is postscript.
*/
int
write_data (const char *filename,
	    unsigned long num,
	    mpfr_prec_t p1, mpfr_prec_t p2, mpfr_prec_t ps, float pr,
	    int select1, int select2, int postscript)
{
  char strf[256], strg[256];
  FILE *f, *g;
  mpfr_prec_t p, step;
  int op = 0;

  lets_start (num, p2);
  strcpy (strf, filename);
  strcat (strf, ".data");
  f = fopen (strf, "w");
  if (f == NULL)
    {
      fprintf (stderr, "Can't open %s!\n", strf);
      lets_end ();
      abort ();
    }
  strcpy (strg, filename);
  strcat (strg, ".gnuplot");
  g = fopen (strg, "w");
  if (g == NULL)
    {
      fprintf (stderr, "Can't open %s!\n", strg);
      lets_end ();
      abort ();
    }
  fprintf (g, "set style data lines\n");
  if (postscript)
    fprintf (g, "set terminal postscript\n");
#undef BENCH
#define BENCH(TEST_STR, TEST)						\
  if (++op == select1)							\
    fprintf (g, "plot  \"%s\" using 1:2 title \"%s\", \\\n", strf,	\
	     TEST_STR);							\
  else if (op == select2)						\
    fprintf (g, "      \"%s\" using 1:3 title \"%s\"\n", strf, TEST_STR);
  op = -1;
  TEST_LIST;

  step = ps;
  for (p = p1 ; p < p2 ; p+=step)
    {
      fprintf(f, "%lu\t%1.20e\t%1.20e\n", p,
              get_speed(p, select1),
              get_speed(p, select2));
      if (pr != 0.0)
        {
          step = (mpfr_prec_t) (p * pr - p);
          if (step < 1)
            step = 1;
        }
    }

  fclose (f);
  fclose (g);
  lets_end ();
  if (postscript == 0)
    fprintf (stderr, "Now type: gnuplot -persist %s.gnuplot\n", filename);
  else
    fprintf (stderr, "Now type: gnuplot %s.gnuplot > %s.ps\n", filename,
	     filename);
  return 0;
}

/* this function considers all functions from s_begin to s_end */
int
write_data2 (const char *filename,
	     unsigned long num,
	     mpfr_prec_t p_begin, mpfr_prec_t p_end, mpfr_prec_t p_step, float p_r,
	     int s_begin, int s_end)
{
  FILE *f;
  mpfr_prec_t p, step;
  int s;

  lets_start (num, p_end);
  f = fopen (filename, "w");
  if (f == NULL)
    {
      fprintf (stderr, "Can't open %s!\n", filename);
      lets_end ();
      exit (1);
    }

  step = p_step;
  for (p = p_begin ; p < p_end ; p += step)
    {
      fprintf (f, "%lu", p);
      for (s = s_begin ; s <= s_end ; s++)
	fprintf (f, "\t%1.20e", get_speed (p, s));
      fprintf (f, "\n");
      if (p_r != 0.0)
        {
          step = (mpfr_prec_t) (p * p_r - p);
          if (step < 1)
            step = 1;
        }
    }
  fclose (f);
  lets_end ();
  return 0;
}

int op_num (void)
{
  int op;
#undef BENCH
#define BENCH(TEST_STR, TEST) op++;
  op = 0;
  TEST_LIST;
  return op;
}

int main(int argc, const char *argv[])
{
  mpfr_prec_t p1, p2, ps;
  float pr;
  int i;
  unsigned long stat;
  const char *filename = "plot";
  int select1, select2, max_op, conti;
  int postscript = 0;

  printf (USAGE);

  max_op = op_num ();
  select1 = 1; select2 = 13;
  p1 = 2; p2 = 500; ps = 4; pr = 0.0;
  stat = 500; /* number of different random numbers */
  conti = 0;

  for(i = 1 ; i < argc ; i++)
    {
      if (argv[i][0] == '-')
	{
	  switch (argv[i][1])
	    {
	    case 'b':
	      p1 = atol(argv[i]+2);
	      break;
	    case 'e':
	      p2 = atol(argv[i]+2);
	      break;
            case 's':
              ps = atol(argv[i]+2);
              break;
            case 'r':
              pr = atof (argv[i]+2);
              if (pr <= 1.0)
                {
                  fprintf (stderr, "-rPREC_RATIO must be > 1.0\n");
                  exit (1);
                }
              break;
 	    case 'm':
	      stat = atol(argv[i]+2);
	      break;
	    case 'x':
	      select1 = atoi (argv[i]+2);
	      select2 = select1 + 12;
	      break;
            case 'y':
              select2 = atoi (argv[i]+2);
              break;
	    case 'o':
	      filename = argv[i]+2;
	      break;
	    case 'c':
	      conti = 1;
	      break;
	    case 'p':
	      postscript = 1;
	      break;
	    case 'f':
	      smooth = atoi  (argv[i]+2);
	      break;
	    default:
	      fprintf(stderr, "Unkwown option: %s\n", argv[i]);
	      abort ();
	    }
	}
    }
  /* Set low priority */
  setpriority(PRIO_PROCESS,0,14);
  if (pr == 0.0)
    printf("GMP:%s MPFR:%s From p=%lu to %lu by %lu Output: %s N=%ld\n",
           gmp_version, mpfr_get_version(), p1,p2,ps, filename, stat);
  else
    printf("GMP:%s MPFR:%s From p=%lu to %lu by %f Output: %s N=%ld\n",
           gmp_version, mpfr_get_version(), p1, p2, pr, filename, stat);

  if (select2 >= max_op)
    select2 = max_op-1;
  if (select1 >= max_op)
    select1 = max_op-1;

  if (conti == 0)
    write_data (filename, stat, p1, p2, ps, pr, select1, select2, postscript);
  else
    write_data2 (filename, stat, p1, p2, ps, pr, select1, select2);

  return 0;
}
