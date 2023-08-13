#include <librapid>

namespace lrc = librapid;

auto main() -> int {
	fmt::print("LibRapid Example -- Vector 1\n");

#if 0 // Currently broken -- switching SIMD backend
	// Create a 3 dimensional vector
	lrc::Vec3d myVector(2, 3, 4);
	lrc::Vec3d myOtherVector(10, 5, 8);
	fmt::print("My Vector = {}\n", myVector);
	fmt::print("My Other Vector = {}\n", myOtherVector);

	// Simple operations
	fmt::print("Vec * Scalar: {}\n", myVector * 2);
	fmt::print("Vec * Vec: {}\n", myVector * myOtherVector);
	fmt::print("Vector dot product: {}\n", lrc::dot(myVector, myOtherVector));
	fmt::print("Vector cross product: {}\n", lrc::cross(myVector, myOtherVector));

	fmt::print("\nTrigonometry with Vectors:\n");
	auto cross	  = lrc::cross(myVector, myOtherVector);
	double theta1 = (lrc::dot(myVector, myOtherVector)) / (lrc::mag(myVector) * lrc::mag(myOtherVector));
	double theta2 = (lrc::dot(myVector, cross)) / (lrc::mag(myVector) * lrc::mag(cross));
	fmt::print("A cross B = {}\n", cross);
	fmt::print("Angle between A and B = {:.3f}\n", theta1);
	fmt::print("Angle between A and (A cross B) = {}pi\n", lrc::acos(theta2) / lrc::PI);

	// Functions operate on each element of a vector
	fmt::print("sin(Vec(pi/4, pi/3, pi/2, pi)) = {:.3f}\n",
			   lrc::sin(lrc::Vec4d(lrc::PI / 4, lrc::PI / 3, lrc::PI / 2, lrc::PI)));

	fmt::print("\n");

	// Other functions also work with vectors (you can also mix data types)
	lrc::Vec3d value(0.5, 5, 100);
	lrc::Vec3d start1(0, 0, 0);
	lrc::Vec3d end1(1, 10, 200);
	lrc::Vec3d start2(0, 0, 0);
	lrc::Vec3d end2(100, 100, 100);
	fmt::print("Mapping: {}\n", lrc::map(value, start1, end1, start2, end2));

	// Polar coordinates
	auto polarVec = lrc::Vector<float, 2>::fromPolar(1, lrc::PI / 4);
	fmt::print("Polar vector: {:.3f}\n", polarVec);

	// Filled vectors
	auto zero = lrc::Vec3d::zero();
	auto one = lrc::Vec3d::one();
	auto full = lrc::Vec3d::full(49);
	auto random = lrc::Vec3d::random(-5, 5);
	fmt::print("Zero vector: {}\n", zero);
	fmt::print("One vector: {}\n", one);
	fmt::print("Full vector: {}\n", full);
	fmt::print("Random vector: {:.3f}\n", random);
#endif

	return 0;
}