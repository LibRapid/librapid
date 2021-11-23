import librapid


def test_librapid():
    assert (librapid.test.testLibrapid(5) == 25)
