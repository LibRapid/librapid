import librapid as lrp

def testArrayPrint():
    """
    Simple array formatting. This tests indexing as well,
    and if this succeeds, it allows us to test other features
    more easily as we know certain fundamental operations are
    working as expected
    """
    
    x = lrp.Array(lrp.Extent(5), "f32")
    for i in range(5):
        x[i] = i + 1
    assert(str(x) == "[1 2 3 4 5]")
