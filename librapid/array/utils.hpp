#ifndef LIBRAPID_ARRAY_UTILS
#define LIBRAPID_ARRAY_UTILS

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/autocast/autocast.hpp>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include <map>
#include <set>

namespace librapid
{
	namespace arrayUtils
	{
		// Map of device version to device number
		std::multimap<std::pair<int, int>, int> getIdenticalGPUs()
		{
			int numGpus = 0;
			checkCudaErrors(cudaGetDeviceCount(&numGpus));

			std::multimap<std::pair<int, int>, int> identicalGpus;

			for (int i = 0; i < numGpus; i++)
			{
				int isMemPoolSupported = 0;
				checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
								// cudaDevAttrMemoryPoolsSupported, i));
								cudaDevAttrMemoryPoolsSupported, i));

				// Filter unsupported devices
				if (isMemPoolSupported)
				{
					int major = 0, minor = 0;
					checkCudaErrors(
						cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i));
					checkCudaErrors(
						cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, i));
					identicalGpus.emplace(std::make_pair(major, minor), i);
				}
			}

			return identicalGpus;
		}

		std::pair<int, int> getP2PCapableGpuPair()
		{
			constexpr size_t kNumGpusRequired = 2;

			auto gpusByArch = getIdenticalGPUs();

			auto it = gpusByArch.begin();
			auto end = gpusByArch.end();

			auto bestFit = std::make_pair(it, it);
			// use std::distance to find the largest number of GPUs amongst architectures
			auto distance = [](decltype(bestFit) p)
			{
				return std::distance(p.first, p.second);
			};

			// Read each unique key/pair element in order
			for (; it != end; it = gpusByArch.upper_bound(it->first))
			{
				// first and second are iterators bounded within the architecture group
				auto testFit = gpusByArch.equal_range(it->first);
				// Always use devices with highest architecture version or whichever has the
				// most devices available
				if (distance(bestFit) <= distance(testFit)) bestFit = testFit;
			}

			if (distance(bestFit) < kNumGpusRequired)
			{
				printf(
					"No Two or more GPUs with same architecture capable of cuda Memory "
					"Pools found."
					"\nWaiving the sample\n");
				exit(EXIT_WAIVED);
			}

			std::set<int> bestFitDeviceIds;

			// check & select peer-to-peer access capable GPU devices.
			int devIds[2];
			for (auto itr = bestFit.first; itr != bestFit.second; itr++)
			{
				int deviceId = itr->second;
				checkCudaErrors(cudaSetDevice(deviceId));

				std::for_each(itr, bestFit.second, [&deviceId, &bestFitDeviceIds,
							  &kNumGpusRequired](
							  decltype(*itr) mapPair)
				{
					if (deviceId != mapPair.second)
					{
						int access = 0;
						checkCudaErrors(
							cudaDeviceCanAccessPeer(&access, deviceId, mapPair.second));
						printf("Device=%d %s Access Peer Device=%d\n", deviceId,
							   access ? "CAN" : "CANNOT", mapPair.second);
						if (access && bestFitDeviceIds.size() < kNumGpusRequired)
						{
							bestFitDeviceIds.emplace(deviceId);
							bestFitDeviceIds.emplace(mapPair.second);
						}
						else
						{
							printf("Ignoring device %i (max devices exceeded)\n", mapPair.second);
						}
					}
				});

				if (bestFitDeviceIds.size() >= kNumGpusRequired)
				{
					printf("Selected p2p capable devices - ");
					int i = 0;
					for (auto devicesItr = bestFitDeviceIds.begin();
						 devicesItr != bestFitDeviceIds.end(); devicesItr++)
					{
						devIds[i++] = *devicesItr;
						printf("deviceId = %d  ", *devicesItr);
					}
					printf("\n");
					break;
				}
			}

			// if bestFitDeviceIds.size() == 0 it means the GPUs in system are not p2p
			// capable, hence we add it without p2p capability check.
			if (!bestFitDeviceIds.size())
			{
				printf("No Two or more Devices p2p capable found.. exiting..\n");
				exit(EXIT_WAIVED);
			}

			auto p2pGpuPair = std::make_pair(devIds[0], devIds[1]);

			return p2pGpuPair;
		}
	}
}

#endif // LIBRAPID_ARRAY_UTILS
