#include <librapid/utils/time_utils.hpp>

namespace librapid {
	double seconds() {
		return (double) std::chrono::high_resolution_clock().now().time_since_epoch().count() / 1000000000;
	}

	void sleep(double s) {
		auto start = seconds();

		while (seconds() - start < s) {
		}
	}
}