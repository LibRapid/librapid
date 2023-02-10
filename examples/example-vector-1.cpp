#include <librapid>

namespace lrc = librapid;

int main() {
	fmt::print("LibRapid Example -- Vector 1\n");

	// Create a 3 dimensional vector
	lrc::Vec3d myVector(2, 3, 4);
	lrc::Vec3d myOtherVector(10, 5, 8);
	fmt::print("My Vector = {}\n", myVector);
	fmt::print("My Other Vector = {}\n", myOtherVector);

	// Simple operations
	fmt::print("Vec * Scalar: {}\n", myVector * 2);
	fmt::print("Vec * Vec: {}\n", myVector * myOtherVector);
	fmt::print("Vector dot product: {}\n", myVector.dot(myOtherVector));
	fmt::print("Vector cross product: {}\n", myVector.cross(myOtherVector));

	fmt::print("\nTrigonometry with Vectors:\n");
	auto cross	  = myVector.cross(myOtherVector);
	double theta1 = (myVector.dot(myOtherVector)) / (myVector.mag() * myOtherVector.mag());
	double theta2 = (myVector.dot(cross)) / (myVector.mag() * cross.mag());
	fmt::print("A cross B = {}\n", cross);
	fmt::print("Angle between A and B = {}\n", theta1);
	fmt::print("Angle between A and (A cross B) = {}pi\n", lrc::acos(theta2) / lrc::PI);

	// Functions operate on each element of a vector
	fmt::print("sin(Vec(pi/4, pi/3, pi/2, pi)) = {}\n",
			   lrc::sin(lrc::Vec4d(lrc::PI / 4, lrc::PI / 3, lrc::PI / 2, lrc::PI)));

	fmt::print("\n");

	// Other functions also work with vectors (you can also mix data types)
	lrc::Vec3d value(0.5, 5, 100);
	lrc::Vec3d start1(0, 0, 0);
	lrc::Vec3d end1(1, 10, 200);
	lrc::Vec3d start2(0, 0, 0);
	lrc::Vec3d end2(100, 100, 100);
	fmt::print("Mapping: {}\n", lrc::map(value, start1, end1, start2, end2));

	return 0;
}