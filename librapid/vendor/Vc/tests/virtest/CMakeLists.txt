if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "You don't want to configure in the source directory!")
endif()

project(virtest)
cmake_minimum_required(VERSION 2.6)
install(DIRECTORY vir DESTINATION include/vir)

enable_testing()
add_subdirectory(tests)
