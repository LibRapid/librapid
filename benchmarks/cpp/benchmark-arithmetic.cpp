// Benchmark array arithmetic

#include <librapid>
#include "helpers.hpp"

namespace lrc = librapid;

template<typename Scalar, typename Device>
void benchmarkElementwise(double benchTime) {
	std::string filename = "benchmark-arithmetic-" + lrc::device::toString<Device>() + "-" +
						   lrc::internal::traits<Scalar>::Name + ".csv";

	std::vector<lrc::Extent> sizes = {lrc::Extent(10, 10),
									  lrc::Extent(25, 25),
									  lrc::Extent(50, 50),
									  lrc::Extent(75, 75),
									  lrc::Extent(100, 100),
									  lrc::Extent(250, 250),
									  lrc::Extent(500, 500),
									  lrc::Extent(750, 750),
									  lrc::Extent(1000, 1000),
									  lrc::Extent(1250, 1250),
									  lrc::Extent(5000, 5000),
									  lrc::Extent(5750, 5750),
									  lrc::Extent(10000, 10000)};

	std::vector<std::string> headings = {
	  "Extent", "Assign Type", "Bytes", "Elapsed", "Average", "StdDev"};
	std::vector<std::vector<std::string>> rows;

	fmt::print("Benchmarking Array-Array Arithmetic for {} on {}\n",
			   lrc::internal::traits<Scalar>::Name,
			   lrc::device::toString<Device>());

	// Write benchmark to file
	auto file = std::fstream(filename, std::ios::out);
	for (const auto &heading : headings) { file << heading << ","; }
	file << "\n";

	for (lrc::i64 i = 0; i < sizes.size(); ++i) {
		auto left  = lrc::Array<Scalar>(sizes[i]);
		auto right = lrc::Array<Scalar>(sizes[i]);

		auto benchmark =
		  lrc::timeFunction([&]() { auto benchy = left + right; }, -1, -1, benchTime);
		rows.emplace_back(std::vector<std::string> {
		  sizes[i].str(),
		  "Lazy-Eval",
		  fmt::format("{}", sizeof(Scalar) * sizes[i].size()),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.elapsed),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.avg),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.stddev),
		});

		file << fmt::format("{},{},{},{},{},{}\n",
							sizes[i].str(),
							"Lazy-Eval",
							sizeof(Scalar) * sizes[i].size(),
							benchmark.elapsed,
							benchmark.avg,
							benchmark.stddev);
	}

	rows.emplace_back(std::vector<std::string>(headings.size(), ""));

	for (lrc::i64 i = 0; i < sizes.size(); ++i) {
		auto left  = lrc::Array<Scalar>(sizes[i]);
		auto right = lrc::Array<Scalar>(sizes[i]);

		auto benchmark =
		  lrc::timeFunction([&]() { auto benchy = (left + right).eval(); }, -1, -1, benchTime);
		rows.emplace_back(std::vector<std::string> {
		  sizes[i].str(),
		  "Construct",
		  fmt::format("{}", sizeof(Scalar) * sizes[i].size()),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.elapsed),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.avg),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.stddev),
		});

		file << fmt::format("{},{},{},{},{},{}\n",
							sizes[i].str(),
							"Construct",
							sizeof(Scalar) * sizes[i].size(),
							benchmark.elapsed,
							benchmark.avg,
							benchmark.stddev);
	}

	rows.emplace_back(std::vector<std::string>(headings.size(), ""));

	for (lrc::i64 i = 0; i < sizes.size(); ++i) {
		auto left  = lrc::Array<Scalar>(sizes[i]);
		auto right = lrc::Array<Scalar>(sizes[i]);

		lrc::Array<Scalar> benchy(sizes[i]);
		auto benchmark = lrc::timeFunction([&]() { benchy = left + right; }, -1, -1, benchTime);
		rows.emplace_back(std::vector<std::string> {
		  sizes[i].str(),
		  "Assign",
		  fmt::format("{}", sizeof(Scalar) * sizes[i].size()),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.elapsed),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.avg),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.stddev),
		});

		file << fmt::format("{},{},{},{},{},{}\n",
							sizes[i].str(),
							"Assign",
							sizeof(Scalar) * sizes[i].size(),
							benchmark.elapsed,
							benchmark.avg,
							benchmark.stddev);
	}

	file.close();
	printTable(headings, rows);
	fmt::print("\n\n");
}

template<typename Scalar, typename Device>
void benchmarkScalar(double benchTime) {
	std::vector<lrc::Extent> sizes = {lrc::Extent(10, 10),
									  lrc::Extent(25, 25),
									  lrc::Extent(50, 50),
									  lrc::Extent(75, 75),
									  lrc::Extent(100, 100),
									  lrc::Extent(250, 250),
									  lrc::Extent(500, 500),
									  lrc::Extent(750, 750),
									  lrc::Extent(1000, 1000),
									  lrc::Extent(1250, 1250),
									  lrc::Extent(5000, 5000),
									  lrc::Extent(5750, 5750),
									  lrc::Extent(10000, 10000)};

	std::vector<std::string> headings = {
	  "Extent", "Assign Type", "Bytes", "Elapsed", "Average", "StdDev"};
	std::vector<std::vector<std::string>> rows;

	fmt::print("Benchmarking Array-Array Arithmetic for {} on {}\n",
			   lrc::internal::traits<Scalar>::Name,
			   lrc::device::toString<Device>());

	for (lrc::i64 i = 0; i < sizes.size(); ++i) {
		auto left  = lrc::Array<Scalar>(sizes[i]);
		auto right = Scalar(123);

		auto benchmark =
		  lrc::timeFunction([&]() { auto benchy = left + right; }, -1, -1, benchTime);
		rows.emplace_back(std::vector<std::string> {
		  sizes[i].str(),
		  "Lazy-Eval",
		  fmt::format("{}", sizeof(Scalar) * sizes[i].size()),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.elapsed),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.avg),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.stddev),
		});
	}

	rows.emplace_back(std::vector<std::string>(headings.size(), ""));

	for (lrc::i64 i = 0; i < sizes.size(); ++i) {
		auto left  = lrc::Array<Scalar>(sizes[i]);
		auto right = lrc::Array<Scalar>(sizes[i]);

		auto benchmark =
		  lrc::timeFunction([&]() { auto benchy = (left + right).eval(); }, -1, -1, benchTime);
		rows.emplace_back(std::vector<std::string> {
		  sizes[i].str(),
		  "Construct",
		  fmt::format("{}", sizeof(Scalar) * sizes[i].size()),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.elapsed),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.avg),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.stddev),
		});
	}

	rows.emplace_back(std::vector<std::string>(headings.size(), ""));

	for (lrc::i64 i = 0; i < sizes.size(); ++i) {
		auto left  = lrc::Array<Scalar>(sizes[i]);
		auto right = lrc::Array<Scalar>(sizes[i]);

		lrc::Array<Scalar> benchy(sizes[i]);
		auto benchmark = lrc::timeFunction([&]() { benchy = left + right; }, -1, -1, benchTime);
		rows.emplace_back(std::vector<std::string> {
		  sizes[i].str(),
		  "Assign",
		  fmt::format("{}", sizeof(Scalar) * sizes[i].size()),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.elapsed),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.avg),
		  lrc::formatTime<lrc::time::nanosecond>(benchmark.stddev),
		});
	}

	printTable(headings, rows);
	fmt::print("\n\n");
}

int main() {
	fmt::print("Hello, World\n");

	double time = 0.1;

	benchmarkElementwise<lrc::i16, lrc::device::CPU>(time);
	benchmarkElementwise<float, lrc::device::CPU>(time);
	benchmarkElementwise<lrc::i64, lrc::device::CPU>(time);

#if defined(LIBRAPID_HAS_CUDA)
	fmt::print("\n\n");

	benchmarkElementwise<lrc::i16, lrc::device::GPU>(time);
	benchmarkElementwise<float, lrc::device::GPU>(time);
	benchmarkElementwise<lrc::i64, lrc::device::GPU>(time);
#endif // LIBRAPID_HAS_CUDA

	benchmarkScalar<lrc::i16, lrc::device::CPU>(time);
	benchmarkScalar<float, lrc::device::CPU>(time);
	benchmarkScalar<lrc::i64, lrc::device::CPU>(time);

#if defined(LIBRAPID_HAS_CUDA)
	fmt::print("\n\n");

	benchmarkScalar<lrc::i16, lrc::device::GPU>(time);
	benchmarkScalar<float, lrc::device::GPU>(time);
	benchmarkScalar<lrc::i64, lrc::device::GPU>(time);
#endif // LIBRAPID_HAS_CUDA

	return 0;
}
