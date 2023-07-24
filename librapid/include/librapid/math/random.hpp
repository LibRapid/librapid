#ifndef LIBRAPID_MATH_RANDOM_HPP
#define LIBRAPID_MATH_RANDOM_HPP

namespace librapid {
	template<typename T = double>
	LIBRAPID_NODISCARD LIBRAPID_INLINE T random(T lower = 0, T upper = 1) {
		// Random floating point value in range [lower, upper)

		static std::uniform_real_distribution<double> distribution(0., 1.);
		static std::mt19937 generator((uint32_t) global::randomSeed);

		if (global::reseed) {
			generator.seed((uint32_t) global::randomSeed);
			global::reseed = false;
		}

		return (T)(lower + (upper - lower) * distribution(generator));
	}

	LIBRAPID_NODISCARD LIBRAPID_INLINE int64_t randint(int64_t lower, int64_t upper) {
		// Random integral value in range [lower, upper]
		return (int64_t)random((double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1);
	}

	LIBRAPID_NODISCARD LIBRAPID_INLINE double trueRandomEntropy() {
		static std::random_device rd;
		return rd.entropy();
	}

	template<typename T = double>
	LIBRAPID_NODISCARD LIBRAPID_INLINE double trueRandom(T lower = 0, T upper = 1) {
		// Truly random value in range [lower, upper)
		static std::random_device rd;
		std::uniform_real_distribution<double> dist((double)lower, (double)upper);
		return dist(rd);
	}

	LIBRAPID_NODISCARD LIBRAPID_INLINE int64_t trueRandint(int64_t lower, int64_t upper) {
		// Truly random value in range [lower, upper)
		return (int64_t)trueRandom((double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1);
	}

	/**
	 * Adapted from
	 * https://docs.oracle.com/javase/6/docs/api/java/util/Random.html#nextGaussian()
	 */
	template<typename T = double>
	LIBRAPID_NODISCARD LIBRAPID_INLINE double randomGaussian() {
		static double nextGaussian;
		static bool hasNextGaussian = false;

		double res;
		if (hasNextGaussian) {
			hasNextGaussian = false;
			res				= nextGaussian;
		} else {
			double v1;
			double v2;
			double s;
			do {
				v1 = random<double>(-1, 1); // between -1.0 and 1.0
				v2 = random<double>(-1, 1); // between -1.0 and 1.0
				s  = v1 * v1 + v2 * v2;
			} while (s >= 1 || s == 0);
			double multiplier = sqrt(-2 * ::librapid::log(s) / s);
			nextGaussian	  = v2 * multiplier;
			hasNextGaussian	  = true;
			res				  = v1 * multiplier;
		}

		return static_cast<T>(res);
	}
} // namespace librapid

#endif // LIBRAPID_MATH_RANDOM_HPP