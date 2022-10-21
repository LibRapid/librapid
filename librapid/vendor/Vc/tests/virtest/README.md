# Vir's Unit Test Framework

[![license](https://img.shields.io/github/license/mattkretz/virtest.svg)](https://github.com/mattkretz/virtest/blob/master/LICENSE)
[![language](https://img.shields.io/badge/language-C%2B%2B11-blue.svg)](https://isocpp.org/)

[![Build Status](https://travis-ci.org/mattkretz/virtest.svg)](https://travis-ci.org/mattkretz/virtest)
[![Build status](https://ci.appveyor.com/api/projects/status/lxqk5tqs4og6dr3e?svg=true)](https://ci.appveyor.com/project/mattkretz/virtest)

## Why another test framework?

The test framework was developed inside the [Vc](https://github.com/VcDevel/Vc) repository.
The goal was to build a test framework supporting:

* Minimal / no test registration or setup. Just write the test function and you're good.
* Simple way to disable compilation of tests, without having to comment out sections of the source
  file.
* Simple instantiation of test code with types of a given list.
* Support for fuzzy compares of floating point results with fine control over the ULP specification.
* Assertion testing (i.e. verify that assertions fail on violated preconditions).
* Simple but effective output (no XML, JSON, whatever; outputs a recognizable source location for more
  effective test driven development)

All the test frameworks I looked at in 2009 (and 2010) did not even come close to supporting the
above requirements.

Since I'm now very familiar with this test framework I want to use it in my other projects. The only
sensible choice is to release the test framework on its own.

## Usage

### Creating an executable
To write a test executable all you need is to include the test header:
```cpp
#include <vir/test.h>
```

This defines a main function, but at this point there are no tests, so it'll pass with the following
output:
```
 Testing done. 0 tests passed. 0 tests failed. 0 tests skipped.
```

### Creating a test function
Simple test functions are created with the `TEST` macro. Checks inside the test are done with
macros. The need for macros is due to the requirement to output the source location on failure. (The
macros `__FILE__` and `__LINE__` only yield the right value when expanded at the location of the
test.) You can use the following macros:

* `COMPARE(value, reference)`
  Compares `value` against `reference`, requiring the two to be equal. The comparison is done via
  equality operator, if it is usable. If no equality operator is defined for the type a fallback to
  memcmp is done. Note that this may yield incorrect failures if the types contain uninitialized
  padding bits. Also, if the equality operator does not return a boolean, the implementation will
  try to reduce the result to a boolean via calling `all_of(value == reference)`.

* `FUZZY_COMPARE(value, reference)`
  As above, but without memcmp fallback and allowance for a maximum difference measured in ULP.

* `COMPARE_ABSOLUTE_ERROR(value, reference, error)`
  As above, but allowing an absoluted difference between `value` and `reference`.

* `COMPARE_RELATIVE_ERROR(value, reference, error)`
  As above, but allowing a relative difference between `value` and `reference`.

* `COMPARE_NOEQ(value, reference)`
  Compares `value` against `reference` and fails if the values are equal. In the case of a
  non-boolean return type from `operator!=`, the test only passes if `all_of(value != reference)`
  passes.

* `MEMCOMPARE(value, reference)`
  Executes a memcp over the storage bytes of `value` and `reference`. The number of bytes compared
  is determined via `sizeof`.

* `VERIFY(boolean)`
  Passes if the argument converted to `bool` is `true`. Fails otherwise.

Example:
```cpp
TEST(test_name) {
  VERIFY(1 > 0);
  COMPARE(1, 1);

  struct A { int x; };
  COMPARE(A(), A());     // implicitly does memcmp
  MEMCOMPARE(A(), A());  // explicitly does memcmp
}
```

### Creating a test function instantiated from a typelist
```cpp
TEST_TYPES(T, test_name, (int, float, short)) {
  COMPARE(T(), T(0));
}
```

### Creating a test function that expects an exception
```cpp
TEST_CATCH(test_name, std::exception) {
  throw std::exception();
}
```

### Output additional information on failure
Every compare/verify macro acts as an output stream, usable for printing more information in the failure case.
Example:
```cpp
TEST(test_name) {
  int test = 3;
  COMPARE(test, 2) << "more details";
  VERIFY(1 > 0);
}
```
Prints:
```
 FAIL: ┍ at tests/testfile.cpp:5 (0x40451f):
 FAIL: │ test (3) == 2 (2) -> false more details
 FAIL: ┕ test_name

 Testing done. 0 tests passed. 1 tests failed.
```


