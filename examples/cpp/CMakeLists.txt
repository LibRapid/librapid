function(make_example name)
    add_executable(example-${name} example-${name}.cpp)
    target_link_libraries(example-${name} PRIVATE librapid)
endfunction()

make_example(1)
make_example(half)
make_example(map)
make_example(pi)
