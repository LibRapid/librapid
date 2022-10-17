#pragma once

#include "../../internal/config.hpp"
#include "../../internal/forward.hpp"

namespace librapid::functors {
	template<typename Derived, typename OtherDerived, bool evalBeforeAssign>
	struct AssignSelector;

	template<typename Derived, typename OtherDerived>
	struct AssignSelector<Derived, OtherDerived, false> {
		LR_FORCE_INLINE static Derived &run(Derived &left, const OtherDerived &right) {
			return left.assignLazy(right);
		}
	};

	template<typename Derived, typename OtherDerived>
	struct AssignOp {
		LR_FORCE_INLINE static void run(Derived &dst, const OtherDerived &src) {
			constexpr bool dstIsHost =
			  is_same_v<typename internal::traits<Derived>::Device, device::CPU>;
			constexpr bool srcIsHost =
			  is_same_v<typename internal::traits<OtherDerived>::Device, device::CPU>;
			using Scalar					 = typename internal::traits<Derived>::Scalar;
			using BaseScalar				 = typename internal::traits<Derived>::BaseScalar;
			using Packet					 = typename internal::traits<Scalar>::Packet;
			static constexpr i64 Flags		 = internal::traits<OtherDerived>::Flags;
			constexpr bool isMatrixOp		 = (bool)(Flags & internal::flags::Matrix);
			constexpr bool hasCustomFunction = (bool)(Flags & internal::flags::CustomFunctionGen);
			constexpr bool allowPacket		 = !((bool)(Flags & internal::flags::NoPacketOp));

#if !defined(LIBRAPID_HAS_CUDA)
			static_assert(dstIsHost && srcIsHost, "CUDA support was not enabled");
#endif

			if constexpr (dstIsHost && srcIsHost) {
				i64 packetWidth = internal::traits<Scalar>::PacketWidth;
				i64 len			= dst.extent().sizeAdjusted();
				i64 alignedLen	= len - (len % packetWidth);
				if (alignedLen < 0) alignedLen = 0;
				i64 processThreads = isMatrixOp ? matrixThreads : numThreads;

#if defined(LIBRAPID_HAS_OMP)
				bool multiThread = true;
				if (processThreads < 2) multiThread = false;
				if (len < threadThreshold) multiThread = false;
#else
				bool multiThread = false;
#endif

				// Only use a Packet type if possible
				if constexpr (!is_same_v<Packet, std::false_type> &&
							  !(Flags & internal::flags::NoPacketOp)) {
					// Use the entire packet width where possible
					if (!multiThread) {
						for (i64 i = 0; i < alignedLen; i += packetWidth) { dst.loadFrom(i, src); }
					}
#if defined(LIBRAPID_HAS_OMP)
					else {
						// Multithreaded approach
#	pragma omp parallel for shared(dst, src, alignedLen, packetWidth) default(none)               \
	  num_threads(processThreads)
						for (i64 i = 0; i < alignedLen; i += packetWidth) { dst.loadFrom(i, src); }
					}
#endif
				} else {
					alignedLen = 0;
				}

				// Ensure the remaining values are filled
				i64 start = alignedLen * allowPacket;
				if (!multiThread) {
					for (i64 i = start < 0 ? 0 : start; i < len; ++i) {
						dst.loadFromScalar(i, src);
					}
				}
#if defined(LIBRAPID_HAS_OMP)
				else {
#	pragma omp parallel for shared(start, len, dst, src) default(none) num_threads(processThreads)
					for (i64 i = start < 0 ? 0 : start; i < len; ++i) {
						dst.loadFromScalar(i, src);
					}
				}
#endif
			} else {
#if defined(LIBRAPID_HAS_CUDA)
				// LR_LOG_STATUS("Size of Type: {}", sizeof(OtherDerived));
				static_assert(sizeof(OtherDerived) < (1 << 15), // Defines the max op size
							  "Calculation is too large to be run in a single call. Please call "
							  "eval() somewhere earlier");

				i64 elems = src.extent().sizeAdjusted();

				if constexpr (is_same_v<Scalar, bool>) {
					elems += sizeof(typename internal::traits<Scalar>::BaseScalar) * 8;
					elems /= sizeof(typename internal::traits<Scalar>::BaseScalar) * 8;
				}

				std::vector<BaseScalar *> arrays = {dst.storage().heap()};
				std::string scalarName			 = internal::traits<BaseScalar>::Name;
				i64 index						 = 0;
				std::string microKernel			 = src.genKernel(arrays, index);

				std::string mainArgs;
				for (i64 i = 0; i < index; ++i) {
					mainArgs += fmt::format("{} *{}{}", scalarName, "arg", i);
					if (i + 1 < index) mainArgs += ", ";
				}

				std::string functionArgs;
				for (i64 i = 0; i < index; ++i) {
					functionArgs += fmt::format("{} arg{}", scalarName, i);
					if (i + 1 < index) functionArgs += ", ";
				}

				std::string indArgs;
				for (i64 i = 0; i < index; ++i) {
					indArgs += fmt::format("arg{}[kernelIndex]", i);
					if (i + 1 < index) indArgs += ", ";
				}

				std::string varExtractor;
				for (i64 i = 0; i < index; ++i)
					varExtractor +=
					  fmt::format("{0} *arg{1} = pointers[{2}];\n\t", scalarName, i, i + 1);

				std::string varArgs;
				for (i64 i = 0; i < index; ++i) {
					varArgs += fmt::format("src{}", i);
					if (i + 1 < index) varArgs += ", ";
				}

				std::string customFunctionDefinition;

				if constexpr (hasCustomFunction) {
					customFunctionDefinition = src.genCustomFunction();
				}

				std::string opKernel = fmt::format(R"V0G0N(
{6}

__device__
{0} function({1}) {{
	return {2};
}}

__global__
void applyOp({0} **pointers, i64 size) {{
	const i64 kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	{0} *dst = pointers[0];
	{4}

	if (kernelIndex < size) {{
		dst[kernelIndex] = function({5});
	}}
}}
				)V0G0N",
												   scalarName,				  // 0
												   functionArgs,			  // 1
												   microKernel,				  // 2
												   mainArgs,				  // 3
												   varExtractor,			  // 4
												   indArgs,					  // 5
												   customFunctionDefinition); // 6

				std::string kernel = detail::kernelGenerator(opKernel, cudaHeaders);

#	if defined(LIBRAPID_PYTHON)
				// This fixes a bug in Python that means GPU handles aren't initialized
				for (i64 i = 0; !memory::streamCreated && i < memory::handleSize; ++i) {
					checkCudaErrors(cublasCreate_v2(&(memory::cublasHandles[i])));
					checkCudaErrors(
					  cublasSetStream_v2(memory::cublasHandles[i], memory::cudaStream));
				}

				memory::streamCreated = true;
#	endif

				// fmt::print("Kernel: {}\n", kernel);

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, nvccOptions);

				i64 threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (elems < 512) {
					threadsPerBlock = elems;
					blocksPerGrid	= 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid	= static_cast<i64>(
						ceil(static_cast<double>(elems) / static_cast<double>(threadsPerBlock)));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				// Copy the pointer array to the device
				BaseScalar **devicePointers = memory::malloc<BaseScalar *, device::GPU>(index + 1);
				memory::memcpy<BaseScalar *, device::GPU, BaseScalar *, device::CPU>(
				  devicePointers, &arrays[0], index + 1);

				jitifyCall(program.kernel("applyOp")
							 .instantiate()
							 .configure(grid, block, 0, memory::cudaStream)
							 .launch(devicePointers, elems));

				// Free device pointers
				memory::free<BaseScalar *, device::GPU>(devicePointers);
#endif
			}
		}
	};
} // namespace librapid::functors
