#pragma once

#include "../internal/config.hpp"
#include "../internal/memUtils.hpp"
#include "../cuda/memUtils.hpp"

namespace librapid::memory {
	template<typename T, typename d>
	class DenseStorage {
	public:
		using Type = T;

		DenseStorage() = default;

		explicit DenseStorage(int64_t size) : m_size(size), m_heap(memory::malloc<T, d>(size)) {
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (std::is_same_v<d, device::GPU>) initializeCudaStream();
#endif
		}

		template<typename T_, typename d_>
		explicit DenseStorage(const DenseStorage<T_, d_> &other) {
			m_size = other.m_size;
			m_heap = memory::malloc<T, d>(m_size);
			LR_ASSERT(m_heap, "Memory Error -- malloc failed");
			memory::memcpy<T, T_, d, d_>(m_heap, other.m_heap, m_size);
		}

		template<typename T_, typename d_>
		DenseStorage &operator=(const DenseStorage<T_, d_> &other) {
			if (this == &other) return *this;

			m_size = other.m_size;
			m_heap = memory::malloc<T, d>(m_size);
			LR_ASSERT(m_heap, "Memory Error -- malloc failed");
			memory::memcpy<T, T_, d, d_>(m_heap, other.m_heap, m_size);

			return *this;
		}

		~DenseStorage() {
			if (!m_heap) return;
			memory::free<T, d>(m_heap);
		}

		T &operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return m_heap[index];
		}

		T &operator[](int64_t index) {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return m_heap[index];
		}

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
