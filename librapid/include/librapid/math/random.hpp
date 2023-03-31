#ifndef LIBRAPID_MATH_RANDOM_HPP
#define LIBRAPID_MATH_RANDOM_HPP

namespace librapid {
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_INLINE T random(T lower = 0, T upper = 1, uint64_t seed = -1) {
		// Random floating point value in range [lower, upper)

		static std::uniform_real_distribution<double> distribution(0., 1.);
		static std::mt19937 generator(
		  seed == (uint64_t)-1 ? (unsigned int)(now() * 10) : seed);
		return (T)(lower + (upper - lower) * distribution(generator));
	}

	LIBRAPID_NODISCARD LIBRAPID_INLINE int64_t randint(int64_t lower, int64_t upper,
													   uint64_t seed = -1) {
		// Random integral value in range [lower, upper]
		return (int64_t)random((double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1, seed);
	}

	inline double trueRandomEntropy() {
		static std::random_device rd;
		return rd.entropy();
	}

	template<typename T = double>
	inline double trueRandom(T lower = 0, T upper = 1) {
		// Truly random value in range [lower, upper)
		static std::random_device rd;
		std::uniform_real_distribution<double> dist((double)lower, (double)upper);
		return dist(rd);
	}

	inline int64_t trueRandint(int64_t lower, int64_t upper) {
		// Truly random value in range [lower, upper)
		return (int64_t)trueRandom((double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1);
	}

	/**
	 * Adapted from
	 * https://docs.oracle.com/javase/6/docs/api/java/util/Random.html#nextGaussian()
	 */
	inline double randomGaussian() {
		static double nextGaussian;
		static bool hasNextGaussian = false;

		double res;
		if (hasNextGaussian) {
			hasNextGaussian = false;
			res				= nextGaussian;
		} else {
			double v1, v2, s;
			do {
				v1 = 2 * random() - 1; // between -1.0 and 1.0
				v2 = 2 * random() - 1; // between -1.0 and 1.0
				s  = v1 * v1 + v2 * v2;
			} while (s >= 1 || s == 0);
			double multiplier = sqrt(-2 * ::librapid::log(s) / s);
			nextGaussian	  = v2 * multiplier;
			hasNextGaussian	  = true;
			res				  = v1 * multiplier;
		}

		return res;
	}
} // namespace librapid

#endif // LIBRAPID_MATH_RANDOM_HPP