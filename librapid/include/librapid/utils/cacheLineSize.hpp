#ifndef LIBRAPID_UTILS_CACHE_LINE_SIZE_HPP
#define LIBRAPID_UTILS_CACHE_LINE_SIZE_HPP

namespace librapid {
    /// Returns the cache line size of the processor, in bytes. If the cache size cannot be
    /// determined, the return value is 64.
    /// \return Cache line size in bytes
    size_t cacheLineSize();
} // namespace librapid

#endif // LIBRAPID_UTILS_CACHE_LINE_SIZE_HPP