install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flens
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.tcc"
        PATTERN "*.cxx"
        PATTERN "netlib" EXCLUDE
        )