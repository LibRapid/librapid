#pragma once

#include "config.hpp"

// Memory alignment adapted from
// https://gist.github.com/dblalock/255e76195676daa5cbc57b9b36d1c99a

namespace librapid::memory {
	constexpr uint64_t memAlign = 128;

	template<typename T, typename d,
			 typename std::enable_if_t<std::is_same_v<d, device::CPU>, int> = 0>
	LR_NODISCARD("Do not leave a dangling pointer")
	LR_FORCE_INLINE T *malloc(size_t num, size_t alignment = memAlign,
							  bool zero = false) {
		size_t size		   = sizeof(T) * num;
		size_t requestSize = size + alignment;
		auto *buf =
		  (u_char *)(zero ? calloc(1, requestSize) : std::malloc(requestSize));

		LR_ASSERT(buf != nullptr,
				  "Memory allocation failed. Cannot allocate {} items of size "
				  "{} ({} bytes total)!",
				  num,
				  sizeof(T),
				  requestSize);

		size_t remainder = ((size_t)buf) % alignment;
		size_t offset	 = alignment - remainder;
		u_char *ret		 = buf + (u_char)offset;

		// store how many extra u_chars we allocated in the u_char just before
		// the pointer we return
		*(u_char *)(ret - 1) = (u_char)offset;

// Slightly altered traceback call to log u_chars being allocated
#ifdef LIBRAPID_TRACEBACK
		LR_STATUS(
		  "LIBRAPID TRACEBACK -- MALLOC {} u_charS -> {}", size, (void *)buf);
#endif

		return (T *)ret;
	}

	template<typename T, typename d,
			 typename std::enable_if_t<std::is_same_v<d, device::CPU>, int> = 0>
	LR_FORCE_INLINE void free(T *alignedPtr) {
#ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- FREE {}", (void *)alignedPtr);
#endif

		int offset = *(((u_char *)alignedPtr) - 1);
		std::free(((u_char *)alignedPtr) - offset);
	}

	// Only supports copying between host pointers
	template<typename T, typename d, typename T_, typename d_,
			 typename std::enable_if_t<std::is_same_v<d, device::CPU> &&
										 std::is_same_v<d_, device::CPU>,
									   int> = 0>
	LR_FORCE_INLINE void memcpy(T *dst, T_ *src, int64_t size) {
		if constexpr (std::is_same_v<T, T_>) {
			std::copy(src, src + size, dst);
		} else {
			// TODO: Optimise this?
			for (int64_t i = 0; i < size; ++i) { dst[i] = src[i]; }
		}
	}
} // namespace librapid::memory