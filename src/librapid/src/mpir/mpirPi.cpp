#include <cmath>
#include <thread>
#include <librapid/internal/config.hpp>
#include <librapid/utils/time.hpp>
#include <librapid/math/mpir.hpp>

namespace librapid {
	Chudnovsky::Chudnovsky(int64_t dig10) {
		DIGITS			= dig10;
		A				= 13591409;
		B				= 545140134;
		C				= 640320;
		D				= 426880;
		E				= 10005;
		DIGITS_PER_TERM = 14.1816474627254776555; // = log(53360^3) / log(10)
		C3_24			= C * C * C / 24;
		N				= (int64_t)((double)DIGITS / DIGITS_PER_TERM);
		PREC			= (int64_t)((double)DIGITS * log2(10));
	}

	detail::PQT Chudnovsky::compPQT(int32_t n1, int32_t n2) const {
		int32_t m;
		detail::PQT res;

		if (n1 + 1 == n2) {
			res.P = mpz(2 * n2 - 1);
			res.P *= (6 * n2 - 1);
			res.P *= (6 * n2 - 5);
			res.Q = C3_24 * n2 * n2 * n2;
			res.T = (A + B * n2) * res.P;
			if ((n2 & 1) == 1) res.T = -res.T;
		} else {
			m				 = (n1 + n2) / 2;
			detail::PQT res1 = compPQT(n1, m);
			detail::PQT res2 = compPQT(m, n2);
			res.P			 = res1.P * res2.P;
			res.Q			 = res1.Q * res2.Q;
			res.T			 = res1.T * res2.Q + res1.P * res2.T;
		}

		return res;
	}

	mpf Chudnovsky::pi() const {
		// Compute Pi
		if ((double)DIGITS < DIGITS_PER_TERM) return {3.1415926535897932385};
		if (DIGITS > 500) return piMultiThread();
		detail::PQT pqt = compPQT(0, N);
		mpf pi(0, PREC);
		pi = D * sqrt((mpf_class)E) * pqt.Q;
		pi /= (A * pqt.Q + pqt.T);
		return pi;
	}

	mpf Chudnovsky::piMultiThread() const {
		// Compute Pi
		if ((double)DIGITS < DIGITS_PER_TERM) return {3.1415926535897932385};
		detail::PQT pqt = compPQT2(*this, 0, N);
		mpf pi(0, PREC);
		pi = D * sqrt((mpf_class)E) * pqt.Q;
		pi /= (A * pqt.Q + pqt.T);
		return pi;
	}

	/* Pi computation using Chudnovsky's algortithm.

	* Copyright 2002, 2005 Hanhong Xue (macroxue at yahoo dot com)

						 * Slightly modified 2005 by Torbjorn Granlund (tege at swox dot com) to
	allow more than 2G digits to be computed.

							 * Modifed 2008 by David Carver (dcarver at tacc dot utexas dot edu) to
	enable multi-threading using the algorithm from "Computation of High-Precision Mathematical
	Constants in a Combined Cluster and Grid Environment" by Daisuke Takahashi, Mitsuhisa Sato, and
	Taisuke Boku.

						  For gcc 4.3
						  gcc -fopenmp -Wall -O2 -o pgmp-chudnovsky pgmp-chudnovsky.c -lgmp -lm

						For Intel 10.1 compiler
						icc -openmp  -O2 -o pgmp-chudnovsky pgmp-chudnovsky.c -lgmp -lm

						For AIX xlc
						xlc_r -qsmp=omp -O2 -o pgmp-chudnovsky pgmp-chudnovsky.c  -lgmp -lm

								   Note: add -DNO_FACTOR to disable factorization performance
	enhancement and use less memory.

								   * Redistribution and use in source and binary forms, with or
	without
									* modification, are permitted provided that the following
	conditions are met:
			* 1. Redistributions of source code must retain the above copyright notice,
						  * this list of conditions and the following disclaimer.
							* 2. Redistributions in binary form must reproduce the above copyright
	notice,
						  * this list of conditions and the following disclaimer in the
	documentation
								* and/or other materials provided with the distribution.
								 *
								   * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY
	EXPRESS OR
									 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF
							* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
	IN NO
							  * EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT,
	INCIDENTAL,
						  * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
	TO,
												   * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS;
												   * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	ANY THEORY OF LIABILITY,
						  * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
									* OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
	EVEN IF
							  * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
								*/

#define A 13591409
#define B 545140134
#define C 640320
#define D 12

#define BITS_PER_DIGIT	3.32192809488736234787
#define DIGITS_PER_ITER 14.1816474627254776555
#define DOUBLE_PREC		53

#ifndef NO_FACTOR

#	define min(x, y) ((x) < (y) ? (x) : (y))
#	define max(x, y) ((x) > (y) ? (x) : (y))

	typedef struct {
		unsigned long max_facs;
		unsigned long num_facs;
		unsigned long *fac;
		unsigned long *pow;
	} fac_t[1];

	typedef struct {
		long int fac;
		long int pow;
		long int nxt;
	} sieve_t;

	fac_t **fpstack, **fgstack;

	sieve_t *sieve;
	long int sieve_size;
	fac_t *ftmp, *fmul;
	double gcd_time = 0;

#	define INIT_FACS 32

	inline void fac_reset(fac_t f) { f[0].num_facs = 0; }

	inline void fac_init_size(fac_t f, long int s) {
		if (s < INIT_FACS) s = INIT_FACS;

		f[0].fac	  = (unsigned long *)malloc(s * sizeof(unsigned long) * 2);
		f[0].pow	  = f[0].fac + s;
		f[0].max_facs = s;

		fac_reset(f);
	}

	inline void fac_init(fac_t f) { fac_init_size(f, INIT_FACS); }

	inline void fac_clear(fac_t f) { free(f[0].fac); }

	inline void fac_resize(fac_t f, long int s) {
		if (f[0].max_facs < s) {
			fac_clear(f);
			fac_init_size(f, s);
		}
	}

	// f = base^pow
	inline void fac_set_bp(fac_t f, unsigned long base, long int pow) {
		long int i;
		LR_ASSERT(base < sieve_size, "Base must be less than sieve_size");
		for (i = 0, base /= 2; base > 0; i++, base = sieve[base].nxt) {
			f[0].fac[i] = sieve[base].fac;
			f[0].pow[i] = sieve[base].pow * pow;
		}
		f[0].num_facs = i;
		LR_ASSERT(i <= f[0].max_facs, "Unknown Error");
	}

	// r = f*g
	inline void fac_mul2(fac_t r, fac_t f, fac_t g) {
		long int i, j, k;

		for (i = j = k = 0; i < f[0].num_facs && j < g[0].num_facs; k++) {
			if (f[0].fac[i] == g[0].fac[j]) {
				r[0].fac[k] = f[0].fac[i];
				r[0].pow[k] = f[0].pow[i] + g[0].pow[j];
				i++;
				j++;
			} else if (f[0].fac[i] < g[0].fac[j]) {
				r[0].fac[k] = f[0].fac[i];
				r[0].pow[k] = f[0].pow[i];
				i++;
			} else {
				r[0].fac[k] = g[0].fac[j];
				r[0].pow[k] = g[0].pow[j];
				j++;
			}
		}
		for (; i < f[0].num_facs; i++, k++) {
			r[0].fac[k] = f[0].fac[i];
			r[0].pow[k] = f[0].pow[i];
		}
		for (; j < g[0].num_facs; j++, k++) {
			r[0].fac[k] = g[0].fac[j];
			r[0].pow[k] = g[0].pow[j];
		}
		r[0].num_facs = k;
		assert(k <= r[0].max_facs);
	}

	// f *= g
	inline void fac_mul(fac_t f, fac_t g, unsigned long index) {
		fac_t tmp;
		fac_resize(fmul[index], f[0].num_facs + g[0].num_facs);
		fac_mul2(fmul[index], f, g);
		tmp[0]		   = f[0];
		f[0]		   = fmul[index][0];
		fmul[index][0] = tmp[0];
	}

	// f *= base^pow
	inline void fac_mul_bp(fac_t f, unsigned long base, unsigned long pow, unsigned long index) {
		fac_set_bp(ftmp[index], base, pow);
		fac_mul(f, ftmp[index], index);
	}

	// remove factors of power 0
	inline void fac_compact(fac_t f) {
		long int i, j;
		for (i = 0, j = 0; i < f[0].num_facs; i++) {
			if (f[0].pow[i] > 0) {
				if (j < i) {
					f[0].fac[j] = f[0].fac[i];
					f[0].pow[j] = f[0].pow[i];
				}
				j++;
			}
		}
		f[0].num_facs = j;
	}

	// convert factorized form to number
	void bs_mul(mpz_t r, long int a, long int b, unsigned long index) {
		long int i, j;
		if (b - a <= 32) {
			mpz_set_ui(r, 1);
			for (i = a; i < b; i++)
				for (j = 0; j < fmul[index][0].pow[i]; j++) mpz_mul_ui(r, r, fmul[index][0].fac[i]);
		} else {
			mpz_t r2;
			mpz_init(r2);
			bs_mul(r2, a, (a + b) / 2, index);
			bs_mul(r, (a + b) / 2, b, index);
			mpz_mul(r, r, r2);
			mpz_clear(r2);
		}
	}

	mpz_t *gcd;

	// f /= gcd(f,g), g /= gcd(f,g)
	void fac_remove_gcd(mpz_t p, fac_t fp, mpz_t g, fac_t fg, unsigned long index) {
		long int i, j, k, c;
		fac_resize(fmul[index], min(fp->num_facs, fg->num_facs));
		for (i = j = k = 0; i < fp->num_facs && j < fg->num_facs;) {
			if (fp->fac[i] == fg->fac[j]) {
				c = min(fp->pow[i], fg->pow[j]);
				fp->pow[i] -= c;
				fg->pow[j] -= c;
				fmul[index]->fac[k] = fp->fac[i];
				fmul[index]->pow[k] = c;
				i++;
				j++;
				k++;
			} else if (fp->fac[i] < fg->fac[j]) {
				i++;
			} else {
				j++;
			}
		}
		fmul[index]->num_facs = k;
		assert(k <= fmul[index]->max_facs);

		if (fmul[index]->num_facs) {
			bs_mul(gcd[index], 0, fmul[index]->num_facs, index);
			mpz_tdiv_q(p, p, gcd[index]);
			mpz_tdiv_q(g, g, gcd[index]);
			fac_compact(fp);
			fac_compact(fg);
		}
	}

	void build_sieve(long int n, sieve_t *s) {
		long int m, i, j, k, id2, jd2;

		sieve_size = n;
		m		   = (long int)sqrt(n);
		memset(s, 0, sizeof(sieve_t) * n / 2);

		s[1 / 2].fac = 1;
		s[1 / 2].pow = 1;

		for (i = 3; i <= n; i += 2) {
			id2 = i >> 1;
			if (s[id2].fac == 0) {
				s[id2].fac = i;
				s[id2].pow = 1;
				if (i <= m) {
					for (j = i * i, k = id2; j <= n; j += i + i, k++) {
						jd2 = j >> 1;
						if (s[jd2].fac == 0) {
							s[jd2].fac = i;
							if (s[k].fac == i) {
								s[jd2].pow = s[k].pow + 1;
								s[jd2].nxt = s[k].nxt;
							} else {
								s[jd2].pow = 1;
								s[jd2].nxt = k;
							}
						}
					}
				}
			}
		}
	}

#endif /* NO_FACTOR */

	////////////////////////////////////////////////////////////////////////////

	int out = 0;
	mpz_t **pstack, **qstack, **gstack;
	long int cores	= 1, depth, cores_depth;
	double progress = 0, percent;

	// binary splitting
	void sum(unsigned long i, unsigned long j, unsigned long gflag) {
		mpz_mul(pstack[i][0], pstack[i][0], pstack[j][0]);
		mpz_mul(qstack[i][0], qstack[i][0], pstack[j][0]);
		mpz_mul(qstack[j][0], qstack[j][0], gstack[i][0]);
		mpz_add(qstack[i][0], qstack[i][0], qstack[j][0]);
		if (gflag) { mpz_mul(gstack[i][0], gstack[i][0], gstack[j][0]); }
	}
	void bs(unsigned long a, unsigned long b, unsigned long gflag, unsigned long level,
			unsigned long index, unsigned long top) {
#ifndef NO_FACTOR
		unsigned long i, mid;
#else
		unsigned long mid;
#endif
		int ccc;

		if (out & 2) {
			fprintf(stderr,
					"bs: a = %ld b = %ld gflag = %ld index = %ld level = %ld top = %ld \n",
					a,
					b,
					gflag,
					index,
					level,
					top);
			fflush(stderr);
		}

		if ((b > a) && (b - a == 1)) {
			/*
			  g(b-1,b) = (6b-5)(2b-1)(6b-1)
			  p(b-1,b) = b^3 * C^3 / 24
			  q(b-1,b) = (-1)^b*g(b-1,b)*(A+Bb).
			*/
			mpz_set_ui(pstack[index][top], b);
			mpz_mul_ui(pstack[index][top], pstack[index][top], b);
			mpz_mul_ui(pstack[index][top], pstack[index][top], b);
			mpz_mul_ui(pstack[index][top], pstack[index][top], (C / 24) * (C / 24));
			mpz_mul_ui(pstack[index][top], pstack[index][top], C * 24);

			mpz_set_ui(gstack[index][top], 2 * b - 1);
			mpz_mul_ui(gstack[index][top], gstack[index][top], 6 * b - 1);
			mpz_mul_ui(gstack[index][top], gstack[index][top], 6 * b - 5);

			mpz_set_ui(qstack[index][top], b);
			mpz_mul_ui(qstack[index][top], qstack[index][top], B);
			mpz_add_ui(qstack[index][top], qstack[index][top], A);
			mpz_mul(qstack[index][top], qstack[index][top], gstack[index][top]);
			if (b % 2) mpz_neg(qstack[index][top], qstack[index][top]);

#ifndef NO_FACTOR
			i = b;
			while ((i & 1) == 0) i >>= 1;
			fac_set_bp(fpstack[index][top], i, 3); // b^3
			fac_mul_bp(fpstack[index][top], 3 * 5 * 23 * 29, 3, index);
			fpstack[index][top][0].pow[0]--;

			fac_set_bp(fgstack[index][top], 2 * b - 1, 1);		  // 2b-1
			fac_mul_bp(fgstack[index][top], 6 * b - 1, 1, index); // 6b-1
			fac_mul_bp(fgstack[index][top], 6 * b - 5, 1, index); // 6b-5
#endif															  /* NO_FACTOR */

			if (b > (int)(progress)) {
				// fprintf(stderr, ".");
				fmt::print("[ PROGRESS ] {:>6.3f}\n", progress / percent);
				// fflush(stderr);
				progress += percent * 2;
			}

		} else {
			/*
			  p(a,b) = p(a,m) * p(m,b)
			  g(a,b) = g(a,m) * g(m,b)
			  q(a,b) = q(a,m) * p(m,b) + q(m,b) * g(a,m)
			*/
			mid = a + (b - a) * 0.5224; // tuning parameter
			bs(a, mid, 1, level + 1, index, top);

			bs(mid, b, gflag, level + 1, index, top + 1);

			ccc = level == 0;

#ifndef NO_FACTOR
			if (level >= 4) { // tuning parameter
				double t = librapid::now();
				fac_remove_gcd(pstack[index][top + 1],
							   fpstack[index][top + 1],
							   gstack[index][top],
							   fgstack[index][top],
							   index);
				gcd_time += librapid::now() - t;
			}
#endif /* NO_FACTOR */

			mpz_mul(pstack[index][top], pstack[index][top], pstack[index][top + 1]);
			mpz_mul(qstack[index][top], qstack[index][top], pstack[index][top + 1]);
			mpz_mul(qstack[index][top + 1], qstack[index][top + 1], gstack[index][top]);
			mpz_add(qstack[index][top], qstack[index][top], qstack[index][top + 1]);

#ifndef NO_FACTOR
			fac_mul(fpstack[index][top], fpstack[index][top + 1], index);
#endif /* NO_FACTOR */

			if (gflag) {
				mpz_mul(gstack[index][top], gstack[index][top], gstack[index][top + 1]);
#ifndef NO_FACTOR
				fac_mul(fgstack[index][top], fgstack[index][top + 1], index);
#endif /* NO_FACTOR */
			}
		}

#ifndef NO_FACTOR
		if (out & 2) {
			fprintf(stderr, "p(%ld,%ld)=", a, b);
			if (gflag) fprintf(stderr, "g(%ld,%ld)=", a, b);
		}
#endif /* NO_FACTOR */
	}

	mpf pi(int64_t digits, long threads, int outMode) {
		mpf_t pi, qi;
		long int d = 100, terms, i, j, k, cores_size;
		unsigned long psize, qsize, mid;
		double begin, mid0, mid1, mid2, mid3, mid4, end;

		d	  = digits;
		out	  = outMode;
		cores = threads;

		terms = d / DIGITS_PER_ITER;
		depth = 0;
		while ((1L << depth) < terms) depth++;
		depth++;

		if (cores < 1) {
			fprintf(stderr, "Number of cores reset from %ld to 1\n", cores);
			fflush(stderr);
			cores = 1;
		}
		if ((terms > 0) && (terms < cores)) {
			fprintf(stderr, "Number of cores reset from %ld to %ld\n", cores, terms);
			fflush(stderr);
			cores = terms;
		}
		cores_depth = 0;
		while ((1L << cores_depth) < cores) cores_depth++;
		cores_size = std::pow(2, cores_depth);

		percent = terms / 100.0;

		fprintf(stderr, "#terms=%ld, depth=%ld, cores=%ld\n", terms, depth, cores);

		begin = librapid::now();

#ifndef NO_FACTOR
		sieve_size = max(3 * 5 * 23 * 29 + 1, terms * 6);
		sieve	   = (sieve_t *)malloc(sizeof(sieve_t) * sieve_size / 2);
		build_sieve(sieve_size, sieve);
#endif /* NO_FACTOR */

		mid0 = librapid::now();

#ifndef NO_FACTOR
		fprintf(stderr, "sieve   cputime = %6.3f\n", (double)(mid0 - begin));
#endif /* NO_FACTOR */

		/* allocate stacks */
		pstack = (mpz_t **)malloc(sizeof(mpz_t *) * cores);
		qstack = (mpz_t **)malloc(sizeof(mpz_t *) * cores);
		gstack = (mpz_t **)malloc(sizeof(mpz_t *) * cores);
		for (j = 0; j < cores; j++) {
			pstack[j] = (mpz_t *)malloc(sizeof(mpz_t) * depth);
			qstack[j] = (mpz_t *)malloc(sizeof(mpz_t) * depth);
			gstack[j] = (mpz_t *)malloc(sizeof(mpz_t) * depth);
			for (i = 0; i < depth; i++) {
				mpz_init(pstack[j][i]);
				mpz_init(qstack[j][i]);
				mpz_init(gstack[j][i]);
			}
		}

		/* begin binary splitting process */
		if (terms <= 0) {
			mpz_set_ui(pstack[0][0], 1);
			mpz_set_ui(qstack[0][0], 0);
			mpz_set_ui(gstack[0][0], 1);
			for (i = 1; i < cores; i++) {
				mpz_clear(pstack[i][0]);
				mpz_clear(qstack[i][0]);
				mpz_clear(gstack[i][0]);
				free(pstack[i]);
				free(qstack[i]);
				free(gstack[i]);
			}
		} else {
#ifndef NO_FACTOR
			gcd		= (mpz_t *)malloc(sizeof(mpz_t) * cores);
			ftmp	= (fac_t *)malloc(sizeof(fac_t) * cores);
			fmul	= (fac_t *)malloc(sizeof(fac_t) * cores);
			fpstack = (fac_t **)malloc(sizeof(fac_t *) * cores);
			fgstack = (fac_t **)malloc(sizeof(fac_t *) * cores);
			for (j = 0; j < cores; j++) {
				fpstack[j] = (fac_t *)malloc(sizeof(fac_t) * depth);
				fgstack[j] = (fac_t *)malloc(sizeof(fac_t) * depth);
				mpz_init(gcd[j]);
				fac_init(ftmp[j]);
				fac_init(fmul[j]);
				for (i = 0; i < depth; i++) {
					fac_init(fpstack[j][i]);
					fac_init(fgstack[j][i]);
				}
			}
#endif /* NO_FACTOR */

			mid0 = librapid::now();

			mid = terms / cores;

#ifdef _OPENMP
#	ifndef NO_FACTOR
#		pragma omp parallel for default(shared) private(i) reduction(+ : gcd_time)
#	else
#		pragma omp parallel for default(shared) private(i)
#	endif
#endif
			for (i = 0; i < cores; i++) {
				if (i < (cores - 1))
					bs(i * mid, (i + 1) * mid, 1, cores_depth, i, 0);
				else
					bs(i * mid, terms, 1, cores_depth, i, 0);
			}
			for (j = 0; j < cores; j++) {
				for (i = 1; i < depth; i++) {
					mpz_clear(pstack[j][i]);
					mpz_clear(qstack[j][i]);
					mpz_clear(gstack[j][i]);
				}
			}
#ifndef NO_FACTOR
			for (j = 0; j < cores; j++) {
				mpz_clear(gcd[j]);
				fac_clear(ftmp[j]);
				fac_clear(fmul[j]);
				for (i = 0; i < depth; i++) {
					fac_clear(fpstack[j][i]);
					fac_clear(fgstack[j][i]);
				}
				free(fpstack[j]);
				free(fgstack[j]);
			}
			free(gcd);
			free(ftmp);
			free(fmul);
			free(fpstack);
			free(fgstack);
#endif /* NO_FACTOR */

			for (k = 1; k < cores_size; k *= 2) {
#ifdef _OPENMP
#	pragma omp parallel for default(shared) private(i)
#endif
				for (i = 0; i < cores; i = i + 2 * k) {
					if (i + k < cores) {
						sum(i, i + k, 1);
						mpz_clear(pstack[i + k][0]);
						mpz_clear(qstack[i + k][0]);
						mpz_clear(gstack[i + k][0]);
						free(pstack[i + k]);
						free(qstack[i + k]);
						free(gstack[i + k]);
					}
				}
			}
		}
		mpz_clear(gstack[0][0]);
		free(gstack[0]);
		free(gstack);

		mid1 = librapid::now();
		fprintf(stderr, "\nbs      cputime = %6.3f\n", (double)(mid1 - mid0));

#ifndef NO_FACTOR
		fprintf(stderr, "gcd     cputime = %6.3f\n", (double)(gcd_time));

		// fprintf(stderr,"misc    "); fflush(stderr);

		/* free some resources */
		free(sieve);
#endif /* NO_FACTOR */

		/* prepare to convert integers to floats */
		mpf_set_default_prec((long int)(d * BITS_PER_DIGIT + 16));

		/*
			p*(C/D)*sqrt(C)
		  pi = -----------------
			   (q+A*p)
		*/

		psize = mpz_sizeinbase(pstack[0][0], 10);
		qsize = mpz_sizeinbase(qstack[0][0], 10);

		mpz_addmul_ui(qstack[0][0], pstack[0][0], A);
		mpz_mul_ui(pstack[0][0], pstack[0][0], C / D);

		mpf_init(pi);
		mpf_set_z(pi, pstack[0][0]);
		mpz_clear(pstack[0][0]);

		mpf_init(qi);
		mpf_set_z(qi, qstack[0][0]);
		mpz_clear(qstack[0][0]);

		free(pstack[0]);
		free(qstack[0]);
		free(pstack);
		free(qstack);

		mid2 = librapid::now();

		/* final step */
		fprintf(stderr, "div     ");
		fflush(stderr);
		mpf_div(qi, pi, qi);
		mid3 = librapid::now();
		fprintf(stderr, "cputime = %6.3f\n", (double)(mid3 - mid2));

		fprintf(stderr, "sqrt    ");
		fflush(stderr);
		mpf_sqrt_ui(pi, C);
		mid4 = librapid::now();
		fprintf(stderr, "cputime = %6.3f\n", (double)(mid4 - mid3));

		fprintf(stderr, "mul     ");
		fflush(stderr);
		mpf_mul(qi, qi, pi);
		end = librapid::now();
		fprintf(stderr, "cputime = %6.3f\n", (double)(end - mid4));

		fprintf(stderr, "total   cputime = %6.3f\n", (double)(end - begin));
		fflush(stderr);

		fprintf(stderr,
				"   P size=%ld digits (%f)\n"
				"   Q size=%ld digits (%f)\n",
				psize,
				(double)psize / d,
				qsize,
				(double)qsize / d);

		/* output Pi and timing statistics */
		if (out & 1) {
			fprintf(stdout, "pi(0,%ld)=\n", terms);
			mpf_out_str(stdout, 10, d + 2, qi);
			fprintf(stdout, "\n");
		}

		mpf piResult(qi);

		/* free float resources */
		mpf_clear(pi);
		// mpf_clear(qi);

		return piResult;
	}
} // namespace librapid