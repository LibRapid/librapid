function(make_benchmark name)
    add_executable(benchmark-${name} benchmark-${name}.cpp)
    target_link_libraries(benchmark-${name} PRIVATE librapid)

    # Add as a cmake test so it can be run with ctest
    add_test(NAME ${name}
            COMMAND "${CMAKE_CURRENT_BINARY_DIR}/benchmark-${name}")
endfunction()

make_benchmark(storage)
# make_benchmark(sizetype)

enable_testing()
