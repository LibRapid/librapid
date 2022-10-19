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

#ifndef __TIMP__H__
#define __TIMP__H__

/* Usage:
 *  Before doing the measure, call TIMP_OVERHEAD ();
 *  Then unsigned long long t = TIMP_MEASURE (f(x));
 *  to measure the # of cycles taken by the call to f(x).
 */

#define TIMP_VERSION 1*100+1*10+0

#ifndef __GNUC__
# error  CC != GCC
#endif

/* High accuracy timing */
#if defined (USE_CLOCK_MONOTONIC)

/* Needs to include -lrt in the library section */
#include <time.h>

#define timp_rdtsc()                                           \
  ({ unsigned long long int x;                                 \
    struct timespec ts;                                        \
    clock_gettime(CLOCK_MONOTONIC, &ts);                       \
    x = ts.tv_sec * 1000000000ULL + ts.tv_nsec;                \
    x; })
#define timp_rdtsc_before(time) (time = timp_rdtsc())
#define timp_rdtsc_after(time)  (time = timp_rdtsc())

#elif defined (__i386__) || defined(__amd64__)

#if !defined(corei7) && !defined(__core_avx2__)

/* The following implements Section 3.2.3 of the article cited below. */
#define timp_rdtsc_before(time)           \
        __asm__ __volatile__(             \
                ".p2align 6\n\t"          \
                "xorl %%eax,%%eax\n\t"    \
                "cpuid\n\t"               \
                "rdtsc\n\t"               \
                "movl %%eax,(%0)\n\t"     \
                "movl %%edx,4(%0)\n\t"    \
                "xorl %%eax,%%eax\n\t"    \
                "cpuid\n\t"               \
                : /* no output */         \
                : "S"(&time)              \
                : "eax", "ebx", "ecx", "edx", "memory")

#define timp_rdtsc_after(time)            \
        __asm__ __volatile__(             \
                "xorl %%eax,%%eax\n\t"    \
                "cpuid\n\t"               \
                "rdtsc\n\t"               \
                "movl %%eax,(%0)\n\t"     \
                "movl %%edx,4(%0)\n\t"    \
                "xorl %%eax,%%eax\n\t"    \
                "cpuid\n\t"               \
                : /* no output */         \
                : "S"(&time)              \
                : "eax", "ebx", "ecx", "edx", "memory")
#else

/* corei7 and corei5 offer newer instruction rdtscp, which should be better,
   see https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/ia-32-ia-64-benchmark-code-execution-paper.pdf */
#define timp_rdtsc_before(time)           \
        __asm__ __volatile__(             \
                ".p2align 6\n\t"          \
                "xorl %%eax,%%eax\n\t"    \
                "cpuid\n\t"               \
                "rdtsc\n\t"               \
                "movl %%eax,(%0)\n\t"     \
                "movl %%edx,4(%0)\n\t"    \
                : /* no output */         \
                : "S"(&time)              \
                : "eax", "ebx", "ecx", "edx", "memory")

#define timp_rdtsc_after(time)            \
        __asm__ __volatile__(             \
                "rdtscp\n\t"               \
                "movl %%eax,(%0)\n\t"     \
                "movl %%edx,4(%0)\n\t"    \
                "xorl %%eax,%%eax\n\t"    \
                "cpuid\n\t"               \
                : /* no output */         \
                : "S"(&time)              \
                : "eax", "ebx", "ecx", "edx", "memory")

#endif

#elif defined (__ia64)

#define timp_rdtsc()                                           \
({ unsigned long long int x;                                   \
  __asm__ __volatile__("mov %0=ar.itc" : "=r"(x) :: "memory"); \
  x; })
#define timp_rdtsc_before(time) (time = timp_rdtsc())
#define timp_rdtsc_after(time)  (time = timp_rdtsc())

#elif defined (__alpha)

#define timp_rdtsc()                              \
({ unsigned long long int x;                      \
   __asm__ volatile ("rpcc %0\n\t" : "=r" (x));   \
   x; })
#define timp_rdtsc_before(time) (time = timp_rdtsc())
#define timp_rdtsc_after(time)  (time = timp_rdtsc())

#else
# error Unsupported CPU
#endif

/* We do several measures and keep the minimum to avoid counting
 * hardware interrupt cycles.
 * The filling of the CPU cache is done because we do several loops,
 * and get the minimum.
 * Declaring num_cycle as "volatile" is to avoid optimization when it is
 * possible (to properly compute overhead).
 * overhead is calculated outside by a call to:
 *   overhead = MEASURE("overhead", ;)
 * Use a lot the preprocessor.
 * It is a macro to be very flexible.
 */
static unsigned long long int timp_overhead = 0;

#define TIMP_NUM_TRY  4327
#define TIMP_MAX_WAIT_FOR_MEASURE 10000000ULL

#define TIMP_MEASURE_AUX(CODE)                                        \
  ({                                                                  \
  volatile unsigned long long int num_cycle, num_cycle2;              \
  unsigned long long int min_num_cycle, start_num_cycle;              \
  int _i;                                                             \
  timp_rdtsc_before (start_num_cycle);                                \
  min_num_cycle = 0xFFFFFFFFFFFFFFFFLL;                               \
  for(_i = 0 ; _i < TIMP_NUM_TRY ; _i++) {                            \
    timp_rdtsc_before(num_cycle);                                     \
    CODE;                                                             \
    timp_rdtsc_after(num_cycle2);                                     \
    num_cycle = num_cycle2 < num_cycle ? 0 /* shouldn't happen */     \
      : num_cycle2 - num_cycle;                                       \
    if (num_cycle < min_num_cycle)                                    \
      min_num_cycle = num_cycle;                                      \
    if (num_cycle2 - start_num_cycle > TIMP_MAX_WAIT_FOR_MEASURE)     \
      break;                                                          \
  }                                                                   \
  min_num_cycle < timp_overhead ? 0 : min_num_cycle - timp_overhead; })

/* If the return value of TIMP_MEASURE_AUX() is 0, this probably means
   that timp_overhead was too large and incorrect; this can occur just
   after starting the process. In this case, TIMP_OVERHEAD() is called
   again to recompute timp_overhead and the timing is redone. */
#define TIMP_MEASURE(CODE)                                            \
  ({                                                                  \
    unsigned long long int _m;                                        \
    while ((_m = TIMP_MEASURE_AUX(CODE)) == 0)                        \
      TIMP_OVERHEAD();                                                \
    _m; })

#define TIMP_OVERHEAD()                                               \
  (timp_overhead = 0, timp_overhead = TIMP_MEASURE_AUX((void) 0) )

#endif /* __TIMP__H__ */
