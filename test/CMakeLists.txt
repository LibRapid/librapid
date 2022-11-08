function(make_test name)
    add_executable(test-${name} test-${name}.cpp)
    target_link_libraries(test-${name} PRIVATE librapid)

    add_test(NAME ${name}
            COMMAND test-${name} --skip-benchmarks)
endfunction()

make_test(storage)
make_test(sizetype)
make_test(multiprecision)
