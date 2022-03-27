#include <librapid/config.hpp>
#include <librapid/array/multiarray.hpp>
#include <librapid/array/multiarray_operations.hpp>

#include <thread>

namespace librapid {
	void Array::fill(double val) {
		Array::applyBinaryOp(*this, *this, Array(val), ops::Fill());
	}

	RawArray Array::createRaw() const {
		return {m_dataStart, m_dtype, m_location};
	}

	Array Array::copy(const Datatype &dtype, const Accelerator &locn) {
		Datatype resDtype	= (dtype == Datatype::NONE) ? m_dtype : dtype;
		Accelerator resLocn = (locn == Accelerator::NONE) ? m_location : locn;

		Array res(m_extent, resDtype, resLocn);
		auto ptrDst	   = res.createRaw();
		res.m_isScalar = m_isScalar;

		if (m_stride.isTrivial() && m_stride.isContiguous()) {
			// Trivial stride, so just memcpy
			rawArrayMemcpy(ptrDst, createRaw(), m_extent.size());
		} else if (m_location == Accelerator::CPU && locn == Accelerator::GPU) {
#ifdef LIBRAPID_HAS_CUDA
			// Non-trivial stride, so apply more complicated algorithm
			applyUnaryOp(*this, res, ops::Copy());

#else
			throw std::invalid_argument(
			  "CUDA support was not enabled, so cannot"
			  " copy array to GPU");
#endif
		}

		return res;
	}

	void optimiseThreads(double timePerThread, bool verbose) {
		fmt::print(
		  "Optimising LibRapid's thread count. This should take around {} "
		  "seconds per thread\n",
		  timePerThread);

		int threadCount = (int)std::thread::hardware_concurrency();

#ifdef LIBRAPID_HAS_OMP
		fmt::print("Optimising OpenMP Threads:{}", verbose ? "\n" : "");

		int optimalOmpThreads = 1;
		double fastestOmpTime = 1E10;

		for (int i = 1; i <= threadCount; ++i) {
			if (!verbose) fmt::print("#");

			omp_set_num_threads(i);

			// Create an array
			Array testArray = Array(Extent({1000, 1000}), "f32", "cpu");
			Array testRes	= testArray.clone();
			testArray.fill(1);

			int64_t iters = 0;
			double start  = seconds();
			while (seconds() - start < timePerThread * 0.5) {
				add(testArray, testArray, testRes);
				++iters;
			}
			double end = seconds();

			if (((end - start) / (double)iters) < fastestOmpTime) {
				fastestOmpTime	  = ((end - start) / (double)iters);
				optimalOmpThreads = i;
			}

			if (verbose) {
				fmt::print(
				  "{} threads: {} seconds total | {} iters | {} ms average",
				  i,
				  (end - start),
				  iters,
				  (end - start) / (double)iters * 1000);
			}
		}

		fmt::print("\nOptimal number of threads is {}, at {} ms per operation\n",
				   optimalOmpThreads, fastestOmpTime * 1000);
		omp_set_num_threads(optimalOmpThreads);
#else
		std::cout
		  << "OpenMP was not found when compiling LibRapid, so all code will "
			 "run serially\n";
#endif

		// TODO: Make this work for more than just OpenBLAS
#ifdef LIBRAPID_HAS_OPENBLAS
		std::cout << "Optimising OpenBLAS Threads: ";
		if (verbose) std::cout << "\n";

		int optimalBlasThreads = 1;
		double fastestBlasTime = 1E10;

		for (int i = 1; i <= threadCount; ++i) {
			if (!verbose) std::cout << "#";

			openblas_set_num_threads(i);

			// Create an array
			Array testArray = Array(Extent({1000, 1000}), "f32", "cpu");
			Array testRes	= testArray.clone();
			testArray.fill(1);
			int64_t iters = 0;
			double start  = seconds();
			while (seconds() - start < timePerThread * 0.5) {
				dot(testArray, testArray, testRes);
				++iters;
			}
			double end = seconds();

			if (((end - start) / (double)iters) < fastestBlasTime) {
				fastestBlasTime	   = ((end - start) / (double)iters);
				optimalBlasThreads = i;
			}

			if (verbose) {
				std::cout << i << " threads: " << (end - start)
						  << " seconds total | " << iters << " iters | "
						  << ((end - start) / (double)iters) * 1000
						  << " ms average\n";
			}
		}

		std::cout << "\nOptimal number of threads is " << optimalBlasThreads
				  << ", at " << fastestBlasTime * 1000 << " ms per operation\n";
		openblas_set_num_threads(optimalBlasThreads);
#else
		std::cout
		  << "OpenBLAS was not found when compiling LibRapid, so all code "
			 "will use custom algorithms\n";
#endif
	}
} // namespace librapid