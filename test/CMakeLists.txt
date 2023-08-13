include("warnings")

function(make_test name)
    add_executable(test-${name} test-${name}.cpp)
    target_link_libraries(test-${name} PRIVATE librapid)
    disable_all_warnings(test-${name}) # Disable warnings for tests since they test unrealistic scenarios

    target_compile_features(test-${name} PRIVATE cxx_std_23)
    target_compile_options(librapid PUBLIC cxx_std_23)

    message(STATUS "[ LIBRAPID ] Adding test ${name}")
    add_test(NAME ${name}
            COMMAND test-${name} -s --skip-benchmarks)
endfunction()

make_test(sizetype)
make_test(storage)
make_test(cudaStorage)
make_test(openCLStorage)
make_test(fixedStorage)
make_test(array)
make_test(arrayView)
make_test(pseudoConstructors)
make_test(arrayOps)

make_test(multiprecision)
make_test(vector)
make_test(complex)
make_test(mathUtilities)

make_test(sigmoid)
