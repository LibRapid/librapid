#include <librapid/librapid.hpp>

namespace librapid::detail {
	void KernelCache::addKernel(
	  const std::string &name,
	  const std::pair<std::unordered_map<std::string, std::string>, std::string> &kernel) {
		m_cache.insert(std::make_pair(name, kernel));
	}

	bool KernelCache::hasKernel(const std::string &name) const {
		return m_cache.find(name) != m_cache.end();
	}

	std::string KernelCache::getKernel(const std::string &name) const {
		auto it = m_cache.find(name);
		if (it != m_cache.end()) return it->second.second;
		LIBRAPID_ASSERT(false, "Kernel named '{}' was not found in cache", name);
	}

	void KernelCache::clear() { m_cache.clear(); }

	std::string parseKernel(const std::string &filename, const std::map<std::string, std::string>) {
		static KernelCache cache;
		// If the file exists in cache, return it to avoid parsing again
		if (cache.hasKernel(filename)) return cache.getKernel(filename);

		std::ifstream file(filename);
		if (!file.is_open()) { throw std::runtime_error("Could not open file " + filename); }

		file.close();
	}
} // namespace librapid::detail
