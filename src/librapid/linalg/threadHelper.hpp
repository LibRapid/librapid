#pragma once

namespace librapid {
	LR_INLINE void setBlasThreads(int64_t n) {
		LR_ASSERT(n >= 1, "Number of threads must be greater than or equal to 1");
#if defined(LIBRAPID_HAS_OPENBLAS)
		openblas_set_num_threads((int)n);
		goto_set_num_threads((int)n);
#endif
#if defined(LIBRAPID_HAS_OMP)
		omp_set_num_threads((int)n);
#endif
	}
} // namespace librapid