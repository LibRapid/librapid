/* Check the values of some constants.

Copyright 2000, 2001, 2002, 2003 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MP Library; see the file COPYING.LIB.  If not, write to
the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
MA 02110-1301, USA. */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include "mpir.h"
#include "tests.h"

#define printf gmp_printf

#ifdef ULLONG_MAX
char *ullong_max_def = "defined";
#else
char *ullong_max_def = "not defined";
#endif
#ifdef LLONG_MAX
char *llong_max_def = "defined";
#else
char *llong_max_def = "not defined";
#endif

#ifdef ULONG_MAX
char *ulong_max_def = "defined";
#else
char *ulong_max_def = "not defined";
#endif
#ifdef LONG_MAX
char *long_max_def = "defined";
#else
char *long_max_def = "not defined";
#endif

#ifdef UINT_MAX
char *uint_max_def = "defined";
#else
char *uint_max_def = "not defined";
#endif
#ifdef INT_MAX
char *int_max_def = "defined";
#else
char *int_max_def = "not defined";
#endif

#ifdef USHRT_MAX
char *ushrt_max_def = "defined";
#else
char *ushrt_max_def = "not defined";
#endif
#ifdef SHRT_MAX
char *shrt_max_def = "defined";
#else
char *shrt_max_def = "not defined";
#endif

#include "gmp-impl.h"
#include "longlong.h"


#ifdef _LONG_LONG_LIMB
#define LL(l,ll)  ll
#else
#define LL(l,ll)  l
#endif

#if __GMP_MP_SIZE_T_INT
#define SS(i,l)   i
#else
#define SS(i,l)   l
#endif


#define CHECK_LIMB_S(x, xname, y, yname)                \
  do {                                                  \
    if ((x) != (y))                                     \
      {                                                 \
        printf (LL("%s == %lx, but %s == %lx\n",        \
                   "%s == %llx, but %s == %llx\n"),     \
                xname, x, yname, y);                    \
        error = 1;                                      \
      }                                                 \
  } while (0)

#define CHECK_INT_S(x, xname, y, yname)                                 \
  do {                                                                  \
    if ((x) != (y))                                                     \
      {                                                                 \
        printf ("%s == %d, but %s == %d\n", xname, x, yname, y);        \
        error = 1;                                                      \
      }                                                                 \
  } while (0)



#define CHECK_CONDITION_S(x, xname)             \
  do {                                          \
    if (!(x))                                   \
      {                                         \
        printf ("%s is false\n", xname);        \
        error = 1;                              \
      }                                         \
  } while (0)


/* How many bits seem to work in the given type. */
#define CALC_BITS(result, type) \
  do {                          \
    type  n = 1;                \
    result = 0;                 \
    while (n != 0)              \
      {                         \
        n <<= 1;                \
        result++;               \
      }                         \
  } while (0)

#define CHECK_BITS_S(constant, constant_name, type)     \
  do {                                                  \
    int   calculated;                                   \
    CALC_BITS (calculated, type);                       \
    if (calculated != constant)                         \
      {                                                 \
        printf ("%s == %d, but calculated %d\n",        \
                constant_name, constant, calculated);   \
        error = 1;                                      \
      }                                                 \
  } while (0)


#define CHECK_HIGHBIT_S(value, value_name, type, format)        \
  do {                                                          \
    type  n = value;                                            \
    if (n == 0)                                                 \
      {                                                         \
        printf ("%s == 0\n", value_name);                       \
        error = 1;                                              \
      }                                                         \
    n <<= 1;                                                    \
    if (n != 0)                                                 \
      {                                                         \
        printf ("%s << 1 = ", value_name);                      \
        printf (format, n);                                     \
        printf (" != 0\n");                                     \
        error = 1;                                              \
      }                                                         \
  } while (0)


#define CHECK_MAX_S(max_val, max_name, min_val, min_name, type, format) \
  do {                                                                  \
    type  maxval = max_val;                                             \
    type  minval = min_val;                                             \
    type  n = maxval;                                                   \
    n++;                                                                \
    if (n != minval)                                                    \
      {                                                                 \
        printf ("%s + 1 = ", max_name);                                 \
        printf (format, n);                                             \
        printf (" != %s = ", min_name);                                 \
        printf (format, minval);                                        \
        printf ("\n");                                                  \
        error = 1;                                                      \
      }                                                                 \
    if (maxval <= minval)                                               \
      {                                                                 \
        printf ("%s = ", max_name);                                     \
        printf (format, maxval);                                        \
        printf (" <= %s = ", min_name);                                 \
        printf (format, minval);                                        \
        printf ("\n");                                                  \
        error = 1;                                                      \
      }                                                                 \
  } while (0)


#if HAVE_STRINGIZE
#define CHECK_LIMB(x,y)      CHECK_LIMB_S (x, #x, y, #y)
#define CHECK_INT(x,y)       CHECK_INT_S (x, #x, y, #y)
#define CHECK_CONDITION(x)   CHECK_CONDITION_S (x, #x)
#define CHECK_BITS(c,t)      CHECK_BITS_S (c, #c, t)
#define CHECK_MAX(m,n,t,f)   CHECK_MAX_S (m, #m, n, #n, t, f)
#define CHECK_HIGHBIT(n,t,f) CHECK_HIGHBIT_S (n, #n, t, f)
#else
#define CHECK_LIMB(x,y)      CHECK_LIMB_S (x, "x", y, "y")
#define CHECK_INT(x,y)       CHECK_INT_S (x, "x", y, "y")
#define CHECK_CONDITION(x)   CHECK_CONDITION_S (x, "x")
#define CHECK_BITS(c,t)      CHECK_BITS_S (c, "c", t)
#define CHECK_MAX(m,n,t,f)   CHECK_MAX_S (m, "m", n, "n", t, f)
#define CHECK_HIGHBIT(n,t,f) CHECK_HIGHBIT_S (n, "n", t, f)
#endif


/* The tests below marked "Bad!" fail on Cray T90 systems, where int, short
   and mp_size_t are 48 bits or some such but don't wraparound in a plain
   twos complement fashion.  In particular,

       INT_HIGHBIT << 1 = 0xFFFFC00000000000 != 0
       INT_MAX + 1 = 35184372088832 != INT_MIN = -35184372088832

   This is a bit bizarre, but doesn't matter because GMP doesn't rely on any
   particular overflow behaviour for int or short, only for mp_limb_t.  */

int
main (int argc, char *argv[])
{
  int  error = 0;

  CHECK_INT (BYTES_PER_MP_LIMB, sizeof(mp_limb_t));
  CHECK_INT (mp_bits_per_limb, BITS_PER_MP_LIMB);
  CHECK_INT (GMP_LIMB_BITS, BITS_PER_MP_LIMB);

  CHECK_BITS (BITS_PER_MP_LIMB, mp_limb_t);
  CHECK_BITS (BITS_PER_ULONG, unsigned long);

  CHECK_HIGHBIT (GMP_LIMB_HIGHBIT, mp_limb_t,      LL("0x%lX","0x%llX"));
  CHECK_HIGHBIT (ULONG_HIGHBIT,     unsigned long,  "0x%lX");
  CHECK_HIGHBIT (UINT_HIGHBIT,      unsigned int,   "0x%X");
  CHECK_HIGHBIT (USHRT_HIGHBIT,     unsigned short, "0x%hX");
  CHECK_HIGHBIT (LONG_HIGHBIT,      long,           "0x%lX");
  CHECK_HIGHBIT (GMP_UI_HIBIT,      mpir_ui,         "%#Mx");

#if 0 /* Bad! */
  CHECK_HIGHBIT (INT_HIGHBIT,       int,            "0x%X");
  CHECK_HIGHBIT (SHRT_HIGHBIT,      short,          "0x%hX");
#endif

#if 0 /* Bad! */
  CHECK_MAX (LONG_MAX,      LONG_MIN,      long,           "%ld");
  CHECK_MAX (INT_MAX,       INT_MIN,       int,            "%d");
  CHECK_MAX (SHRT_MAX,      SHRT_MIN,      short,          "%hd");
#endif
  CHECK_MAX (ULONG_MAX,     0,             unsigned long,  "%lu");
  CHECK_MAX (UINT_MAX,      0,             unsigned int,    "%u");
  CHECK_MAX (GMP_UI_MAX,    0,             mpir_ui,          "%M");
  CHECK_MAX (USHRT_MAX,     0,             unsigned short, "%hu");
#if 0 /* Bad! */
  CHECK_MAX (MP_SIZE_T_MAX, MP_SIZE_T_MIN, mp_size_t,      SS("%d","%ld"));
#endif

  /* UHWtype should have at least enough bits for half a UWtype */
  {
    int  bits_per_UWtype, bits_per_UHWtype;
    CALC_BITS (bits_per_UWtype,  UWtype);
    CALC_BITS (bits_per_UHWtype, UHWtype);
    CHECK_CONDITION (2*bits_per_UHWtype >= bits_per_UWtype);
  }

  ASSERT_ALWAYS_LIMB (MODLIMB_INVERSE_3);
  {
    mp_limb_t  modlimb_inverse_3_calc;
    modlimb_invert (modlimb_inverse_3_calc, CNST_LIMB(3));
    ASSERT_ALWAYS_LIMB (modlimb_inverse_3_calc);
    CHECK_LIMB (MODLIMB_INVERSE_3, modlimb_inverse_3_calc);
  }
  {
    mp_limb_t  MODLIMB_INVERSE_3_times_3
      = (MODLIMB_INVERSE_3 * CNST_LIMB(3)) & GMP_NUMB_MASK;
    CHECK_LIMB (MODLIMB_INVERSE_3_times_3, CNST_LIMB(1));
  }

  {
    mp_limb_t  hi, lo;
    hi = refmpn_umul_ppmm (&lo, GMP_NUMB_CEIL_MAX_DIV3-1,
                           CNST_LIMB(3) << GMP_NAIL_BITS);
    if (! (hi < 1))
      {
        printf ("GMP_NUMB_CEIL_MAX_DIV3 too big\n");
        error = 1;
      }
    hi = refmpn_umul_ppmm (&lo, GMP_NUMB_CEIL_MAX_DIV3,
                           CNST_LIMB(3) << GMP_NAIL_BITS);
    if (! (hi >= 1))
      {
        printf ("GMP_NUMB_CEIL_MAX_DIV3 too small\n");
        error = 1;
      }
  }

  {
    mp_limb_t  hi, lo;
    hi = refmpn_umul_ppmm (&lo, GMP_NUMB_CEIL_2MAX_DIV3-1,
                           CNST_LIMB(3) << GMP_NAIL_BITS);
    if (! (hi < 2))
      {
        printf ("GMP_NUMB_CEIL_2MAX_DIV3 too big\n");
        error = 1;
      }
    hi = refmpn_umul_ppmm (&lo, GMP_NUMB_CEIL_2MAX_DIV3,
                           CNST_LIMB(3) << GMP_NAIL_BITS);
    if (! (hi >= 2))
      {
        printf ("GMP_NUMB_CEIL_2MAX_DIV3 too small\n");
        error = 1;
      }
  }

#ifdef PP_INVERTED
  {
    mp_limb_t  pp_inverted_calc;
    invert_limb (pp_inverted_calc, PP);
    CHECK_LIMB (PP_INVERTED, pp_inverted_calc);
  }
#endif

  if (argc >= 2 || error)
    {
      int  bits;

      printf ("\n");
      printf ("After mpir.h,\n");
      printf ("  ULLONG_MAX  %s\n", ullong_max_def);
      printf ("  LLONG_MAX   %s\n", llong_max_def);
      printf ("  ULONG_MAX  %s\n", ulong_max_def);
      printf ("  LONG_MAX   %s\n", long_max_def);
      printf ("  UINT_MAX   %s\n", uint_max_def);
      printf ("  INT_MAX    %s\n", int_max_def);
      printf ("  USHRT_MAX  %s\n", ushrt_max_def);
      printf ("  SHRT_MAX   %s\n", shrt_max_def);
      printf ("\n");

#if HAVE_LONG_LONG && defined(ULLONG_MAX)
      printf ("ULLONG_MAX     %llX\n", ULLONG_MAX);
      printf ("LLONG_MAX      %llX\n", LLONG_MAX);
      printf ("LLONG_MIN      %llX\n", LLONG_MIN);
#endif
#if defined(ULLONG_MAX)
      printf ("GMP_UI_MAX     %Mx\n", ULLONG_MAX);
      printf ("GMP_SI_MAX     %Mx\n", LLONG_MAX);
      printf ("GMP_SI_MIN     %Mx\n", LLONG_MIN);
#endif

      printf ("ULONG_MAX      %lX\n", ULONG_MAX);
      printf ("ULONG_HIGHBIT  %lX\n", ULONG_HIGHBIT);
      printf ("LONG_MAX       %lX\n", LONG_MAX);
      printf ("LONG_MIN       %lX\n", LONG_MIN);

      printf ("UINT_MAX       %#X\n", UINT_MAX);
      printf ("UINT_HIGHBIT   %X\n", UINT_HIGHBIT);
      printf ("INT_MAX        %X\n", INT_MAX);
      printf ("INT_MIN        %X\n", INT_MIN);

      printf ("USHRT_MAX      %hX\n", USHRT_MAX);
      printf ("USHRT_HIGHBIT  %hX\n", USHRT_HIGHBIT);
      printf ("SHRT_MAX       %hX\n", SHRT_MAX);
      printf ("SHRT_MIN       %hX\n", SHRT_MIN);

      printf ("\n");
      printf ("Bits\n");
      CALC_BITS (bits, long long);           printf ("  long long           %d\n", bits);
      CALC_BITS (bits, long);                printf ("  long                %d\n", bits);
      CALC_BITS (bits, int);                 printf ("  int                 %d\n", bits);
      CALC_BITS (bits, short);               printf ("  short               %d\n", bits);
      CALC_BITS (bits, mpir_si);              printf ("  GMP signed int      %d\n", bits);
      CALC_BITS (bits, unsigned long long);  printf ("  unsigned long long  %d\n", bits);
      CALC_BITS (bits, unsigned long);       printf ("  unsigned long       %d\n", bits);
      CALC_BITS (bits, unsigned int);        printf ("  unsigned int        %d\n", bits);
      CALC_BITS (bits, unsigned short);      printf ("  unsigned short      %d\n", bits);
      CALC_BITS (bits, mp_size_t);           printf ("  mp_size_t           %d\n", bits);
      CALC_BITS (bits, mpir_ui);              printf ("  GMP unsigned int    %d\n", bits);
    }

  if (error)
    abort ();

  exit (0);
}
