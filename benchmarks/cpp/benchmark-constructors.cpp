// Benchmark array constructor -- a fairly useless benchmark but why not :)

#include <librapid>
#include "helpers.hpp"

namespace lrc = librapid;

template<typename Scalar, typename Device>
void benchmarkConstructor(double benchTime) {
	std::vector<lrc::Extent> sizes = {lrc::Extent(10),
									  lrc::Extent(100),
									  lrc::Extent(1000),
									  lrc::Extent(10000),
									  lrc::Extent(100000),
									  lrc::Extent(1000000),
									  lrc::Extent(10000000),
									  lrc::Extent(100000000),
									  lrc::Extent(1000000000)};

	std::vector<std::string> headings = {"Extent", "Bytes", "Elapsed", "Average", "StdDev"};
	std::vector<std::vector<std::string>> rows;

	fmt::print("Benchmarking Array Constructors for {} on {}\n",
			   lrc::internal::traits<Scalar>::Name,
			   lrc::device::toString<Device>());

	for (lrc::i64 i = 0; i < sizes.size(); ++i) {
		auto time = lrc::timeFunction(
		  [&]() { auto benchy = lrc::Array<Scalar>(sizes[i]); }, -1, -1, benchTime);
		rows.emplace_back(std::vector<std::string> {
		  sizes[i].str(),
		  fmt::format("{}", sizeof(Scalar) * sizes[i].size()),
		  lrc::formatTime<lrc::time::nanosecond>(time.elapsed),
		  lrc::formatTime<lrc::time::nanosecond>(time.avg),
		  lrc::formatTime<lrc::time::nanosecond>(time.stddev),
		});

		// You can comment this out for a slightly nicer table printing, assuming it'll fit
		// FULLY in the console without scrolling.
		// if (i > 0) lrc::moveCursor(0, -(lrc::i32(rows.size()) + 1));
		// printTable(headings, rows);
	}

	// If you print the table above, comment this out
	printTable(headings, rows);
	fmt::print("\n\n");
}

int main() {
	fmt::print("Hello, World\n");

	benchmarkConstructor<bool, lrc::device::CPU>(1);
	benchmarkConstructor<lrc::i16, lrc::device::CPU>(1);
	benchmarkConstructor<float, lrc::device::CPU>(1);
	benchmarkConstructor<lrc::i64, lrc::device::CPU>(1);

#if defined(LIBRAPID_HAS_CUDA)
	fmt::print("\n\n");

	benchmarkConstructor<bool, lrc::device::GPU>(1);
	benchmarkConstructor<lrc::i16, lrc::device::GPU>(1);
	benchmarkConstructor<float, lrc::device::GPU>(1);
	benchmarkConstructor<lrc::i64, lrc::device::GPU>(1);
#endif // LIBRAPID_HAS_CUDA

	return 0;
}
