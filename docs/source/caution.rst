Caution
#######

.. warning::

    LibRapid developers had to make certain decisions regarding the underlying data layout used by the library. We made
    these decisions with the best interests of the library in mind, and while they may improve performance or usability,
    they may also incur adverse side effects.

    While the developers of LibRapid may not be aware of all the side effects of their design choices, we have done our
    best to identify and justify those we know of.

Array Referencing Issues
------------------------

LibRapid uses lazy evaluation to reduce the number of intermediate variables and copies required for any given
operation, significantly improving performance. A side effect of this is that combined operations store references to
Array objects.

As a result, if any of the referenced Array instances go out of scope before the lazy object is evaluated, an invalid
memory location will be accessed, incurring a segmentation fault.

The easiest fix for this is to make sure you evaluate temporary results in time, though this is easier said than done.
LibRapid aims to identify when a lazy object is using an invalid value and notify the user, but this will not work in
all cases.

The code below will cause a segmentation fault since ``testArray`` will go out of scope upon returning from the function
while the returned object contains two references to the array.

.. code-block:: cpp
    :linenos:

    /* References invalid memory
    vvvv */
    auto doesThisBreak() {
        lrc::Array<float> testArray(lrc::Shape({3, 3}));
        testArray << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        return testArray + testArray;
    }

.. code-block:: cpp
    :linenos:

    /*   Changed
    -------vvv------- */
    lrc::Array<float> doesThisBreak() {
        lrc::Array<float> testArray(lrc::Shape({3, 3}));
        testArray << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        return testArray + testArray;
    }
