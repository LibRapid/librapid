execute_process(
   COMMAND ./plotdist -v --plotdist plotdist.dat
   RESULT_VARIABLE ok)

if(NOT ok EQUAL 0)
   message(FATAL_ERROR "running plotdist failed")
endif()

file(READ plotdist.dat data)
set(expected "# reference\tdistance\n1\t0\t2\n2\t0\n3\t0\t1\t2\t3\n")
if(NOT "${data}" STREQUAL "${expected}")
   message(FATAL_ERROR "plotdist output broken:\nexpected output:\n${expected}\nactual output:\n${data}")
endif()

message(" PASS: plotdist output matches the expectation")
