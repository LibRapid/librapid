#pragma once

#include "config.hpp"

// Memory alignment adapted from
// https://gist.github.com/dblalock/255e76195676daa5cbc57b9b36d1c99a

namespace librapid::memory {
	constexpr uint64_t memAlign = 32;

	template<typename T = char, typename d = device::CPU,
			 typename std::enable_if_t<std::is_same_v<d, device::CPU>, int> = 0>
	LR_NODISCARD("Do not leave a dangling pointer")
	LR_FORCE_INLINE T *malloc(size_t num, size_t alignment = memAlign, bool zero = false) {
		if constexpr (!internal::traits<T>::CanAlign) {
			// Value cannot be aligned to a boundary, so use a simpler allocation technique
			auto buf = new T[num];

			LR_ASSERT(buf != nullptr,
					  "Memory allocation failed. Cannot allocate {} items of size "
					  "{} ({} bytes total)!",
					  num,
					  sizeof(T),
					  sizeof(T) * num);

			// Slightly altered traceback call to log unsigned chars being allocated
#ifdef LIBRAPID_TRACEBACK
			LR_STATUS(
			  "LIBRAPID TRACEBACK -- MALLOC {} unsigned chars -> {}", sizeof(T) * num, (void *)buf);
#endif

			if (zero) std::memset(buf, 0, sizeof(T) * num);
			return buf;
		}

		size_t size		   = sizeof(T) * num;
		size_t requestSize = size + alignment;

		auto *buf = new unsigned char[requestSize];
		if (zero) std::memset(buf, 0, requestSize);

		LR_ASSERT(buf != nullptr,
				  "Memory allocation failed. Cannot allocate {} items of size "
				  "{} ({} bytes total)!",
				  num,
				  sizeof(T),
				  requestSize);

		size_t remainder   = ((size_t)buf) % alignment;
		size_t offset	   = alignment - remainder;
		unsigned char *ret = buf + (unsigned char)offset;

		// store how many extra unsigned chars we allocated in the unsigned char just before
		// the pointer we return
		*(unsigned char *)(ret - 1) = (unsigned char)offset;

// Slightly altered traceback call to log unsigned chars being allocated
#ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- MALLOC {} unsigned chars -> {}", size, (void *)buf);
#endif

		return (T *)ret;
	}

	template<typename T = char, typename d = device::CPU,
			 typename std::enable_if_t<std::is_same_v<d, device::CPU>, int> = 0>
	LR_FORCE_INLINE void free(T *alignedPtr) {
#ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- FREE {}", (void *)alignedPtr);
#endif
		if constexpr (!internal::traits<T>::CanAlign) {
			// Value cannot be aligned to a boundary, so use a simpler freeing technique
			delete[] alignedPtr;
			return;
		}

		int offset = *(((unsigned char *)alignedPtr) - 1);
		delete[] ((unsigned char *) alignedPtr - offset);
	}

	// Only supports copying between host pointers
	template<typename T, typename d, typename T_, typename d_,
			 typename std::enable_if_t<
			   std::is_same_v<d, device::CPU> && std::is_same_v<d_, device::CPU>, int> = 0>
	LR_FORCE_INLINE void memcpy(T *dst, T_ *src, int64_t size) {
		if constexpr (std::is_same_v<T, T_> && internal::traits<T>::CanMemcpy) {
			std::copy(src, src + size, dst);
		} else {
			// TODO: Optimise this?
			for (int64_t i = 0; i < size; ++i) { dst[i] = src[i]; }
		}
	}

	template<typename T, typename d,
			 typename std::enable_if_t<std::is_same_v<d, device::CPU>, int> = 0>
	LR_FORCE_INLINE void memset(T *dst, int val, int64_t size) {
		std::memset((void *)dst, val, sizeof(T) * size);
	}

	template<typename A, typename B>
	struct PromoteDevice {};

	template<>
	struct PromoteDevice<device::CPU, device::CPU> {
		using type = device::CPU;
	};

	template<>
	struct PromoteDevice<device::CPU, device::GPU> {
#if defined(LIBRAPID_PREFER_GPU)
		using type = device::GPU;
#else
		using type = device::CPU;
#endif
	};

	template<>
	struct PromoteDevice<device::GPU, device::CPU> {
#if defined(LIBRAPID_PREFER_GPU)
		using type = device::GPU;
#else
		using type = device::CPU;
#endif
	};

	template<>
	struct PromoteDevice<device::GPU, device::GPU> {
		using type = device::GPU;
	};
} // namespace librapid::memory