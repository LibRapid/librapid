#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <librapid>

namespace lrc = librapid;
using namespace ankerl;

int main() {
	nanobench::Bench benchmark;

	benchmark.minEpochTime(
	  std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)));

	benchmark.run("Allocate and deallocate", [] {
		lrc::Storage<int> bigStorage(1000000);
		nanobench::doNotOptimizeAway(bigStorage[0]);
	});
}
