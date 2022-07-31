#pragma once

#include "../../internal/config.hpp"
#include "../../math/zTheory.hpp"
#include "reducedMath.hpp"
#include "index.hpp"

namespace librapid::detail {
	struct Prerotator {
		ReducedDivisor m, b;

		Prerotator() : m(1), b(1) {}
		Prerotator(int _m, int _b) : m(_m), b(_b) {}

		int x {};
		void setJ(const int &j) { x = b.div(j); }
		int operator()(const int &i) const { return m.mod(i + x); }
	};

	struct Postpermuter {
		ReducedDivisor m;
		int n;
		ReducedDivisor a;
		int j;

		Postpermuter() : m(1), a(1) {}
		Postpermuter(int _m, int _n, int _a) : m(_m), n(_n), a(_a) {}

		void setJ(const int &_j) { j = _j; }

		int operator()(const int &i) const { return m.mod((i * n + j - a.div(i))); }
	};

	struct Shuffle {
		int m, n, k, i;
		ReducedDivisor b;
		ReducedDivisor c;

		Shuffle() : b(1), c(1) {}
		Shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k), b(_n / _c), c(_c) {}

		void setI(const int &_i) { i = _i; }

		LR_NODISCARD("") int f(const int &j) const {
			int r = j + i * (n - 1);
			// The (int) casts here prevent unsigned promotion
			// and the subsequent underflow: c implicitly casts
			// int - unsigned int to
			// unsigned int - unsigned int
			// rather than to
			// int - int
			// Which leads to underflow if the result is negative.
			if (i - (int)c.mod(j) <= m - (int)c.get()) {
				return r;
			} else {
				return r + m;
			}
		}

		LR_NODISCARD("") int operator()(const int &j) {
			int fij = f(j);
			unsigned int fijdivc, fijmodc;
			c.divMod(fij, fijdivc, fijmodc);
			// The extra mod in here prevents overflowing 32-bit int
			int term1 = b.mod(k * b.mod(fijdivc));
			int term2 = ((int)fijmodc) * (int)b.get();
			return term1 + term2;
		}
	};

	template<typename T, typename F>
	void colShuffle(int m, int n, T *d, T *tmp, F fn) {
		using Packet						 = typename internal::traits<T>::Packet;
		static constexpr int64_t packetWidth = internal::traits<T>::PacketWidth;

		RowMajorIndex rm(m, n);

		T *privTmp;
		F privFn;
		int tid;
		int i;

#pragma omp parallel private(tid, privTmp, privFn, i) num_threads(matrixThreads)
		{
#if defined(LIBRAPID_HAS_OMP)
			tid = omp_get_thread_num();
#else
			tid = 0;
#endif
			privFn	= fn;
			privTmp = tmp + m * tid;
#pragma omp for
			for (int j = 0; j < n; j++) {
				privFn.setJ(j);

				for (i = 0; i < m; i++) { privTmp[i] = d[rm(privFn(i), j)]; }
				for (i = 0; i < m; i++) { d[rm(i, j)] = privTmp[i]; }
			}
		}
	}

	template<typename T, typename F>
	void rowShuffle(int m, int n, T *d, T *tmp, F fn) {
		RowMajorIndex rm(m, n);
		T *privTmp;
		F privFn;
		int tid;
		int j;

#pragma omp parallel private(tid, privTmp, privFn, j) num_threads(matrixThreads)
		{
#if defined(LIBRAPID_HAS_OMP)
			tid = omp_get_thread_num();
#else
			tid = 0;
#endif
			privFn	= fn;
			privTmp = tmp + n * tid;
#pragma omp for
			for (int i = 0; i < m; i++) {
				privFn.setI(i);
				for (j = 0; j < n; j++) { privTmp[j] = d[rm(i, privFn(j))]; }
				for (j = 0; j < n; j++) { d[rm(i, j)] = privTmp[j]; }
			}
		}
	}

	template<typename T>
	void transpose(bool rowMajor, T *data, int m, int n, T *tmp) {
		if (!rowMajor) { std::swap(m, n); }

		int c = 0, t = 0, k = 0;
		extendedGCD(m, n, c, t);
		if (c > 1) {
			extendedGCD(m / c, n / c, t, k);
		} else {
			k = t;
		}

		if (c > 1) { colShuffle(m, n, data, tmp, Prerotator(m, n / c)); }
		rowShuffle(m, n, data, tmp, Shuffle(m, n, c, k));
		colShuffle(m, n, data, tmp, Postpermuter(m, n, m / c));
	}
} // namespace librapid::detail
