#pragma once

#include "../internal/config.hpp"
#include "../internal/memUtils.hpp"
#include "../cuda/memUtils.hpp"

namespace librapid::memory {
	template<typename T, typename d>
	class DenseStorage {
		using Type = T;

	public:
		DenseStorage() = default;

		explicit DenseStorage(int64_t size) : m_size(size), m_heap(memory::malloc<T, d>(size)) {
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (std::is_same_v<d, device::GPU>) initializeCudaStream();
#endif
		}

		DenseStorage(const DenseStorage<T, d> &other) {
			m_size = other.m_size;
			m_heap = memory::malloc<T, d>(m_size);
			memory::memcpy<T, d>(m_heap, other.m_heap, m_size);
		}

		template<typename T_, typename d_>
		explicit DenseStorage(const DenseStorage<T_, d_> &other) {
			m_size = other.m_size;
			m_heap = memory::malloc<T, d>(m_size);
			memory::memcpy<T, T_, d, d_>(m_heap, other.m_heap, m_size);
		}

		~DenseStorage() { memory::free<T, d>(m_heap); }

		LR_NODISCARD("") T *heap() const { return m_heap; }

		LR_NODISCARD("") int64_t size() const { return m_size; }

		LR_NODISCARD("") int64_t bytes() const { return sizeof(T) * m_size; }

	private:
		int64_t m_size = 0;
		T *m_heap	   = nullptr;
	};

	template<typename T, typename d, typename T_, typename d_>
	LR_INLINE void memcpy(DenseStorage<T, d> &dst, const DenseStorage<T_, d_> &src) {
		LR_ASSERT(dst.size() == src.size(),
				  "Cannot copy data between DenseStorage objects with different sizes");
		memcpy<T, d, T_, d_>(dst.heap(), src.heap(), dst.size());
	}
} // namespace librapid::memory
