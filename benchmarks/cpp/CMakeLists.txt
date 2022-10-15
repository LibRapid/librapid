function(make_benchmark name)
    add_executable(benchmark-${name} benchmark-${name}.cpp)
    target_link_libraries(benchmark-${name} PRIVATE librapid)
endfunction()

make_benchmark(constructors)
make_benchmark(arithmetic)
