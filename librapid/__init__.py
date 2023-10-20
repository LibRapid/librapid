import _librapid


class Shape(_librapid.Shape):
    """
    Stores the dimensions of an N-dimensional Array.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       