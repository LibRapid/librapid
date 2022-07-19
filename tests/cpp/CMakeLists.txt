function(make_test name)
    add_executable(test-${name} test-${name}.cpp)
    target_link_libraries(test-${name} PRIVATE librapid)
endfunction()

make_test(fundamental)
make_test(arithmetic)