# From https://jugit.fz-juelich.de/mlz/kww/blob/a9694baf275d81d7024feb96609e858f7c88d7e1/cmake/FindMPIR.cmake
# Thank you!

# Find mpir with API version ?.?
#
# Usage:
#   find_package(mpir [REQUIRED] [QUIET])
#
# Sets the following variables:
#   - MPIR_FOUND          .. true if library is found
#   - MPIR_LIBRARY      .. full path to library
#   - MPIR_INCLUDE_DIR    .. full path to include directory
#
# Honors the following optional variables:
#   - MPIR_INCLUDE_LOC    .. include directory path, to be searched before defaults
#   - MPIR_LIBRARY_LOC    .. the library's directory path, to be searched before defaults
#   - MPIR_STATIC_LIBRARY .. if true, find the static library version
#
# Copyright 2015 Joachim Coenen, Forschungszentrum Jülich.
# Redistribution permitted.

# find the mpir include directory
find_path(MPIR_INCLUDE_DIR mpir.h
        PATH_SUFFIXES include mpir/include mpir
        PATHS
        ${MPIR_INCLUDE_LOC}
        "C:/Program Files/mpir/"
        ~/Library/Frameworks/
        /Library/Frameworks/
        /usr/local/
        /usr/
        /sw/ # Fink
        /opt/local/ # DarwinPorts
        /opt/csw/ # Blastwave
        /opt/
        )


set(CMAKE_REQUIRED_INCLUDES ${MPIR_INCLUDE_DIR})
set(CMAKE_REQUIRED_QUIET False)

# attempt to find static library first if this is set
if(MPIR_STATIC_LIBRARY)
    set(MPIR_STATIC mpir.a)
    #set(MPIR_STATIC mpir.lib)
endif()

# find the mpir library
find_library(MPIR_LIBRARY
        NAMES
        #${MPIR_STATIC}
        mpir
        PATH_SUFFIXES lib64 lib
        PATHS
        ${MPIR_LIBRARY_LOC}
        "C:/Program Files/mpir/"
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local
        /usr
        /sw
        /opt/local
        /opt/csw
        /opt
        )

message(STATUS
        "mpir: FOUND=${MPIR_FOUND}, VERSION=${MPIR_VERSION}, LIB=${MPIR_LIBRARY}")

message(STATUS "Found FindMPIR: ${MPIR_LIBRARY}")
mark_as_advanced(MPIR_INCLUDE_DIR MPIR_LIBRARY)
