#include <iostream>
#include <chrono>

#define ND_MAX_DIMS 10
#define ND_NUM_THREADS 32
#define LIBRAPID_CBLAS

#include <librapid/ndarray/ndarray.hpp>

/// <summary>
/// A little function to add two numbers
/// </summary>
/// <param name="a">First number</param>
/// <param name="b">Second number</param>

/// <summary>
/// 
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="a"></param>
/// <param name="b"></param>
/// <returns></returns>
template<typename T>
ND_INLINE T test_function(T a, T b)
{
	return a + b;
}

int main()
{
	int res = test_function(123, 456);

	auto lhs = librapid::ndarray::basic_ndarray<double>(librapid::ndarray::extent({2, 3}));
	auto rhs = librapid::ndarray::basic_ndarray<double>(librapid::ndarray::extent({3}));

	for (nd_int i = 0; i < librapid::ndarray::math::product(lhs.get_extent().get_extent(), lhs.get_extent().ndim()); i++)
		lhs.set_value(i, i + 1);

	for (nd_int i = 0; i < librapid::ndarray::math::product(rhs.get_extent().get_extent(), rhs.get_extent().ndim()); i++)
		rhs.set_value(i, i + 1);

	std::cout << "LHS:\n" << lhs.str() << "\n\n";
	std::cout << "RHS:\n" << rhs.str() << "\n\n";

	std::cout << librapid::ndarray::reshape(lhs, librapid::ndarray::extent{6}).str() << "\n";

	for (const auto &val : lhs.get_stride())
		std::cout << "Value: " << val << "\n";

	std::cout << "\n\nResult:\n";
	std::cout << librapid::ndarray::add((int) 123, (int) 456) << "\n";

	return 0;
}
