#include <librapid/utils/time_utils.hpp>
#include <librapid/math/core_math.hpp>

#include <string>
#include <vector>
#include <random>
#include <chrono>

namespace librapid
{
	lr_int product(const std::vector<lr_int> &vals)
	{
		lr_int res = 1;
		for (const auto &val : vals)
			res *= val;
		return res;
	}

	lr_int product(const lr_int *vals, lr_int num)
	{
		lr_int res = 1;
		for (lr_int i = 0; i < num; i++)
			res *= vals[i];
		return res;
	}

	double product(const std::vector<double> &vals)
	{
		double res = 1;
		for (const auto &val : vals)
			res *= val;
		return res;
	}

	double product(const double *vals, lr_int num)
	{
		double res = 1;
		for (lr_int i = 0; i < num; i++)
			res *= vals[i];
		return res;
	}

	bool anyBelow(const std::vector<lr_int> &vals, lr_int bound)
	{
		for (const auto &val : vals)
			if (val < bound)
				return true;
		return false;
	}

	bool anyBelow(const lr_int *vals, lr_int dims, lr_int bound)
	{
		for (lr_int i = 0; i < dims; i++)
			if (vals[i] < bound)
				return true;
		return false;
	}

	double map(double val,
			   double start1, double stop1,
			   double start2, double stop2)
	{
		return start2 + (stop2 - start2) *
			((val - start1) / (stop1 - start1));
	}

	double random(double lower, double upper)
	{
		// Random floating point value in range [lower, upper)

		static std::uniform_real_distribution<double> distribution(0., 1.);
		static std::mt19937 generator((unsigned int) (seconds() * 10));
		return lower + (upper - lower) * distribution(generator);
	}

	lr_int randint(lr_int lower, lr_int upper)
	{
		// Random integral value in range [lower, upper]
		return (lr_int) random((double) lower, (double) upper + 1);
	}

	double pow10(lr_int exponent)
	{
		const static double pows[] = {0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};
		if (exponent >= -5 && exponent <= 5)
			return pows[exponent + 5];

		double res = 1;

		if (exponent > 0)
			for (lr_int i = 0; i < exponent; i++)
				res *= 10;
		else
			for (lr_int i = 0; i > exponent; i--)
				res *= 0.1;

		return res;
	}

	double round(const double num, lr_int dp)
	{
		double alpha = pow10(dp);
		double beta = pow10(-dp);

		double absx = abs(num * alpha);
		double y = floor(absx);

		if (absx - y >= 0.5) y += 1;

		return (num >= 0 ? y : -y) / alpha;
	}

	double roundSigFig(const double num, lr_int figs)
	{
		if (figs <= 0)
			throw std::invalid_argument("Cannot round to "
										+ std::to_string(figs)
										+ " significant figures. Must be greater than 0");

		double tmp = num > 0 ? num : -num;
		lr_int n = 0;

		while (tmp > 10)
		{
			tmp /= 10;
			++n;
		}

		while (tmp < 1)
		{
			tmp *= 10;
			--n;
		}

		return (tmp > 0 ? 1 : -1) * (round(tmp, figs - 1) * pow10(n));
	}
}