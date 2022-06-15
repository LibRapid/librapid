{
    quom src/librapid/librapid.hpp godbolt/singleHeader.hpp
} || {
    pip install quom
    quom src/librapid/librapid.hpp godbolt/singleHeader.hpp
}
