===============
LibRapid Stride
===============

A stride (basic_stride) stores the distance in memory one must move to
increment by one value in a given axis inside an NDarray. It determines
the order in which data is accessed inside an NDarray as well, and
simply relocating values in the stride is often used instead of
rearranging the data itself in order to improve performance.
