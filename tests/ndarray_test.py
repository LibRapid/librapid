import librapid as libr

# Test the extent object
def test_extent():
    from_list = libr.extent([2, 3, 4])
    from_extent = libr.extent(from_list)
    from_args = libr.extent(2, 3, 4)

    assert from_list[0] == 2
    assert from_list[1] == 3
    assert from_list[2] == 4

    assert from_extent[0] == 2
    assert from_extent[1] == 3
    assert from_extent[2] == 4

    assert from_args[0] == 2
    assert from_args[1] == 3
    assert from_args[2] == 4

    to_compress = libr.extent(1, 1, 2, 3, 4, 1, 1)
    compressed = to_compress.compressed()

    assert compressed[0] == 2
    assert compressed[1] == 3
    assert compressed[2] == 4

    assert from_list.ndim == 3

    valid = libr.extent(5)
    not_valid = libr.extent()
    assert valid.is_valid == True
    assert not_valid.is_valid == False

    to_reshape = libr.extent(2, 3, 4)
    to_reshape.reshape([2, 1, 0])

    assert to_reshape[0] == 4
    assert to_reshape[1] == 3
    assert to_reshape[2] == 2

    
