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

		explicit DenseStorage(int64_t size) :
				m_size(size), m_heap(memory::malloc<T, d>(size)),
				m_refCount(new std::atomic<int64_t>(1)) {
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (std::is_same_v<d, device::GPU>) initializeCudaStream();
#endif
		}

		explicit DenseStorage(const DenseStorage<T, d> &other) {
			m_refCount = other.m_refCount;
			m_size	   = other.m_size;
			m_heap	   = other.m_heap;
			increment();
		}

		DenseStorage &operator=(const DenseStorage<T, d> &other) {
			if (this == &other) return *this;

			decrement();
			m_size	   = other.m_size;
			m_heap	   = other.m_heap;
			m_refCount = other.m_refCount;
			increment();

			return *this;
		}

		~DenseStorage() { decrement(); }

		operator bool() const { return (bool)m_refCount; }

		LR_NODISCARD("") const T &get(int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);

			// Host data
			if constexpr (std::is_same_v<d, device::CPU>) return m_heap[index];

#if defined(LIBRAPID_HAS_CUDA)
			// Device data
			T tmp;
			memory::memcpy<T, device::CPU, T, device::GPU>(&tmp, m_heap + index, 1);
			return tmp;
#endif
		}

		LR_NODISCARD("") T &get(int64_t index) {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);

			// Host data
			if constexpr (std::is_same_v<d, device::CPU>) return m_heap[index];
			LR_ASSERT(false, "Non-const access is not valid on Device Array");
		}

		void set(int64_t index, const T &value) const {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);

			// Host data
			if constexpr (std::is_same_v<d, device::CPU>) {
				m_heap[index] = value;
				return;
			}

#if defined(LIBRAPID_HAS_CUDA)
			// Device data
			T tmp = value;
			memory::memcpy<T, device::GPU, T, device::CPU>(m_heap + index, &tmp, 1);
#endif
		}

		// WARNING: ONLY WORKS FOR HOST ACCESSES
		T &operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return m_heap[index];
		}

		// WARNING: ONLY WORKS FOR HOST ACCESSES
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

		void increment() const {
			// LR_LOG_STATUS("Incrementing");
			(*m_refCount)++;
		}

		void decrement() {
			// LR_LOG_STATUS("Decrementing");
			if (!m_refCount) return;
			(*m_refCount)--;
			if (*m_refCount == 0) {
				// LR_LOG_WARN("Freeing Memory At: {}", (void *) m_heap);
				delete m_refCount;
				memory::free<T, d>(m_heap);
			}
		}

	private:
		int64_t m_size					 = 0;
		T *m_heap						 = nullptr;
		std::atomic<int64_t> *m_refCount = nullptr;
	};

	template<typename T, typename d, typename T_, typename d_>
	LR_INLINE void memcpy(DenseStorage<T, d> &dst, const DenseStorage<T_, d_> &src) {
		LR_ASSERT(dst.size() == src.size(),
				  "Cannot copy data between DenseStorage objects with different sizes");
		memcpy<T, d, T_, d_>(dst.heap(), src.heap(), dst.size());
	}
} // namespace librapid::memory
