function(make_test name)
    add_executable(test-${name} test-${name}.cpp)
    target_link_libraries(test-${name} PRIVATE librapid)

    add_test(NAME ${name}
            COMMAND "${CMAKE_CURRENT_BINARY_DIR}/test-${name}")
endfunction()

make_test(storage)
make_test(sizetype)

enable_testing()
