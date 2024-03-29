include("warnings")

function(make_test name)
    add_executable(test-${name} test-${name}.cpp)
    target_link_libraries(test-${name} PRIVATE librapid)
    disable_all_warnings(test-${name}) # Disable warnings for tests since they test unrealistic scenarios

    message(STATUS "[ LIBRAPID ] Adding test ${name}")
    add_test(NAME ${name}
            COMMAND test-${name} -s --skip-benchmarks)
endfunction()

make_test(sizetype)
make_test(storage)
make_test(cudaStorage)
make_test(openCLStorage)
make_test(fixedStorage)
make_test(arrayConstructors)
make_test(arrayIndexing)
make_test(arrayArithmetic)
make_test(arrayComparisons)
make_test(generalArrayView)
make_test(pseudoConstructors)
make_test(arrayOps)

make_test(multiprecision)
make_test(vector)
make_test(complex)
make_test(mathUtilities)
make_test(set)

make_test(sigmoid)
