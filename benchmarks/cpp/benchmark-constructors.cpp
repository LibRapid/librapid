#include <librapid>
#include "helpers.hpp"

namespace lrc = librapid;

template<typename T>
void timeArrayConstructor(double timePerRow = 1) {
	std::vector<lrc::Extent> sizes = {lrc::Extent(10),
									  lrc::Extent(100),
									  lrc::Extent(1000),
									  lrc::Extent(10000),
									  lrc::Extent(100000),
									  lrc::Extent(1000000),
									  lrc::Extent(10, 10),
									  lrc::Extent(100, 100),
									  lrc::Extent(1000, 1000),
									  lrc::Extent(10000, 10000),
									  lrc::Extent(10, 10, 10),
									  lrc::Extent(100, 100, 100),
									  lrc::Extent(1000, 1000, 1000)};

	std::vector<std::string> headings = {"Extent", "Bytes", "Elapsed", "Average", "StdDev"};
	std::vector<std::vector<std::string>> rows;

	for (const auto &size : sizes) {
		auto time = lrc::timeFunction([&]() { auto benchy = lrc::Array<T>(size);}, -1, -1, timePerRow);
		rows.emplace_back(std::vector<std::string>{
		  size.str(),
		  fmt::format("{}", sizeof(T) * size.size()),
		  lrc::formatTime<lrc::time::nanosecond>(time.elapsed),
		  lrc::formatTime<lrc::time::nanosecond>(time.avg),
		  lrc::formatTime<lrc::time::nanosecond>(time.stddev),
		});
		fmt::print("\r{}", size.str());
	}
	fmt::print("\n");

	printTable(headings, rows);
}

int main() {
	fmt::print("Hello, World\n");

	// Benchmark Development

	// Benchmark 1: Matrix Construction
	// Benchmark 2: Matrix Addition
	// Benchmark 3: Matrix Multiplication
	// Benchmark 4: Matrix Transpose

	std::vector<std::string> headings = {"Heading 1", "ABCD", "Heading 3", "Time"};

	std::vector<std::vector<std::string>> rows = {{"abc", "123", "hello", "REALLY LONG THING"},
												  {"def", "456", "world", "REALLY LONG THING"},
												  {"ghi", "789", "!", "REALLY LONG THING"}};

	printTable(headings, rows);

	timeArrayConstructor<float>(1);

	return 0;
}
