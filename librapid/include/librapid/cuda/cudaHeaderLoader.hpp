#ifndef LIBRAPID_CUDA_HEADER_LOADER_HPP
#define LIBRAPID_CUDA_HEADER_LOADER_HPP

namespace librapid::detail {
	/// Implement a simple cache for storing CUDA kernels
	class KernelCache {
	public:
		/// Default constructor
		KernelCache() = default;

		/// Add a kernel to the cache, loading
		/// \param name
		/// \param kernel
		void addKernel(const std::string &name, const std::string &kernel);

		/// Check if a kernel exists in the cache
		/// \param name Name of the cached kernel
		/// \return True if the cache exists
		bool hasKernel(const std::string &name) const;

		/// Get a kernel from the cache
		/// \param name Name of the kernel function
		/// \return The kernel from the cache
		std::string getKernel(const std::string &name) const;

		/// Clear the cache
		void clear();

	private:
		/// Map of kernel names to kernel code
		std::unordered_map<std::string, std::string> m_cache;
	};

	std::string parseKernel(const std::string &filename,
							const std::map<std::string, std::string> &replacements);
} // namespace librapid::detail

#endif // LIBRAPID_CUDA_HEADER_LOADER_HPP