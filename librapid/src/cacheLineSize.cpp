/*
 * I do not take credit for this code. It is taken from the following repository, and all
 * credit goes to the original author(s):
 *
 * https://github.com/NickStrupat/CacheLineSize
 *
 * Many thanks for making this code available and open source!
 *
 * The code has been modified slightly to better fit the needs of LibRapid, however, so the code
 * below is not identical to the original.
 */

#include <librapid/librapid.hpp>

#if defined(LIBRAPID_APPLE)

#    include <sys/sysctl.h>

namespace librapid {
    size_t cacheLineSize() {
        size_t lineSize       = 64;
        size_t sizeOfLineSize = sizeof(lineSize);
        sysctlbyname("hw.cachelinesize", &lineSize, &sizeOfLineSize, 0, 0);
        return lineSize;
    }
} // namespace librapid

#elif defined(LIBRAPID_WINDOWS) && !defined(LIBRAPID_NO_WINDOWS_H)

namespace librapid {
    size_t cacheLineSize() {
        size_t lineSize                              = 64;
        DWORD bufferSize                             = 0;
        DWORD i                                      = 0;
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buffer = 0;

        GetLogicalProcessorInformation(0, &bufferSize);
        buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(bufferSize);
        GetLogicalProcessorInformation(&buffer[0], &bufferSize);

        for (i = 0; i != bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i) {
            if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == 1) {
                lineSize = buffer[i].Cache.LineSize;
                break;
            }
        }

        free(buffer);
        return lineSize;
    }
} // namespace librapid

#elif defined(LIBRAPID_LINUX)

namespace librapid {
    size_t cacheLineSize() {
        FILE *p = 0;
        p       = fopen("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
        unsigned int lineSize = 64;
        if (p) {
            fscanf(p, "%d", &lineSize);
            fclose(p);
        }
        return lineSize;
    }
} // namespace librapid

#else

namespace librapid {
    size_t cacheLineSize() {
        // On unknown platforms, return 64
        return 64;
    }
} // namespace librapid

#endif