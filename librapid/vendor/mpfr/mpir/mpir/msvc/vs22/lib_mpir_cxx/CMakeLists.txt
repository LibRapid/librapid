set(PROJECT_NAME lib_mpir_cxx)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
    "../../../gmp-h.in"
)
source_group("" FILES ${no_group_source_files})

set(Header_Files
    "../../../config.h"
    "../../../gmp-impl.h"
    "../../../gmp-mparam.h"
    "../../../longlong.h"
    "../../../mpir.h"
    "../../../mpirxx.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files__cxx
    "../../../cxx/dummy.cc"
    "../../../cxx/isfuns.cc"
    "../../../cxx/ismpf.cc"
    "../../../cxx/ismpq.cc"
    "../../../cxx/ismpz.cc"
    "../../../cxx/ismpznw.cc"
    "../../../cxx/osdoprnti.cc"
    "../../../cxx/osfuns.cc"
    "../../../cxx/osmpf.cc"
    "../../../cxx/osmpq.cc"
    "../../../cxx/osmpz.cc"
)
source_group("Source Files\\cxx" FILES ${Source_Files__cxx})

set(ALL_FILES
    ${no_group_source_files}
    ${Header_Files}
    ${Source_Files__cxx}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "$ENV{MPIR_Props_External}")
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    use_props(${PROJECT_NAME} Debug   "../../mpir_debug_lib.cmake")
    use_props(${PROJECT_NAME} Release "../../mpir_release_lib.cmake")
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    use_props(${PROJECT_NAME} Debug   "../../mpir_debug_lib.cmake")
    use_props(${PROJECT_NAME} Release "../../mpir_release_lib.cmake")
endif()

set(ROOT_NAMESPACE mpirxx)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
################################################################################
# Target name
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        TARGET_NAME_DEBUG   "mpirxx"
        TARGET_NAME_RELEASE "mpirxx"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        TARGET_NAME_DEBUG   "mpirxx"
        TARGET_NAME_RELEASE "mpirxx"
    )
endif()
################################################################################
# Include directories
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    )
endif()

################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_LIB;"
        "HAVE_CONFIG_H"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_LIB;"
        "HAVE_CONFIG_H;"
        "_WIN64"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_compile_options(${PROJECT_NAME} PRIVATE
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    endif()
endif()

################################################################################
# Post build events
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> cd ..\\..\\
        COMMAND   $<CONFIG:Debug> postbuild "$<SHELL_PATH:$<TARGET_FILE:${PROJECT_NAME}>>" 22
        COMMAND $<CONFIG:Release> cd ..\\..\\
        COMMAND $<CONFIG:Release> postbuild "$<SHELL_PATH:$<TARGET_FILE:${PROJECT_NAME}>>" 22
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    add_custom_command_if(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMANDS
        COMMAND   $<CONFIG:Debug> cd ..\\..\\
        COMMAND   $<CONFIG:Debug> postbuild "$<SHELL_PATH:$<TARGET_FILE:${PROJECT_NAME}>>" 22
        COMMAND $<CONFIG:Release> cd ..\\..\\
        COMMAND $<CONFIG:Release> postbuild "$<SHELL_PATH:$<TARGET_FILE:${PROJECT_NAME}>>" 22
    )
endif()

