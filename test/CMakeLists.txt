function(make_test name)
    add_executable(test-${name} test-${name}.cpp)
    target_link_libraries(test-${name} PRIVATE librapid)

    message(STATUS "[ LIBRAPID ] Adding test ${name}")
    add_test(NAME ${name}
            COMMAND test-${name} -s --skip-benchmarks)
endfunction()

make_test(storage)
make_test(fixedStorage)
make_test(sizetype)
make_test(multiprecision)
make_test(vector)
make_test(array)
make_test(arrayView)
make_test(mathUtilities)
