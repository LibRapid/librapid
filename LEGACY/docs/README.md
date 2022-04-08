# Documentation Standards

This document shows the general format for documentation in LibRapid. This format should be followed as closely as
possible, unless it would detract from the overall quality or readability of the documentation. The same style should
also be used between languages as far as possible to increase the consistency in the documentation.

This is a guide mostly for developers, though can also be used by users of LibRapid to better understand the format of
the documentation to write better code.

## Documentation Formatting

When writing the documentaiton (or reading it). Some points are not required, as some functions may be very simple and
not need much explaining

1. A general overview of what the function or class does
2. More specific information about the function, such different modes based on the input
3. Any warnings or hints, such as known bugs or things to watch out for
4. Input values and their data types if possible
5. The return value of the function and it's data type
6. Any additional information that may be relevant to the function

## Developer Information

Documentation should be written as a multi-line comment above the corresponding target (i.e. above the function name or
class). In order for the docs to be built succesfully, they should be formatted as in the example below:

**_Please note that the type of multi-line comment used must be one that is recognized by Doxygen._**

``` text
/**
 * \rst
 *
 * << Documentation Contents >>
 *
 * \endrst
 */
```

The multi-line comment should follow the JavaDoc comment format, shown below:

``` text
/**
 * ... text ...
 */
 ```

### More information about reStructuredText documentation coming soon
