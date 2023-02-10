function(make_example name)
    add_executable(example-${name} example-${name}.cpp)
    target_link_libraries(example-${name} PRIVATE librapid)

    # Ensure all examples finish correctly
    message(STATUS "[ LIBRAPID ] Adding example ${name}")
    add_test(NAME ${name}
            COMMAND example-${name})
endfunction()

make_example(array-1)
make_example(vector-1)
make_example(complex-1)
