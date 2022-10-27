#include <librapid>
#include <celero/Celero.h>

std::random_device RandomDevice;
std::uniform_int_distribution<int> UniformDistribution(0, 1024);

CELERO_MAIN

BASELINE(DemoSimple, Baseline, 10, 1000000) {
	celero::DoNotOptimizeAway(static_cast<float>(sin(UniformDistribution(RandomDevice))));
}

BENCHMARK(DemoSimple, Complex1, 10, 1000000) {
	celero::DoNotOptimizeAway(
	  static_cast<float>(sin(fmod(UniformDistribution(RandomDevice), 3.14159265))));
}

BENCHMARK(DemoSimple, Complex2, 10, 1000000) {
	celero::DoNotOptimizeAway(
	  static_cast<float>(sin(fmod(UniformDistribution(RandomDevice), 3.14159265))));
}

BENCHMARK(DemoSimple, Complex3, 10, 1000000) {
	celero::DoNotOptimizeAway(
	  static_cast<float>(sin(fmod(UniformDistribution(RandomDevice), 3.14159265))));
}
