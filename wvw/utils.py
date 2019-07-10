def boundaries_of(x):
    """
    Given a sequence of central values, calculate its boundaries.
    For an input of length N, the returned sequence is of length N+1.

    A sorted copy of the input parameter is used.
    """
    try:
        if len(x) < 2:
            raise Exception(
                    "Boundaries can only be calculated"
                    " for sequences with at least two elements!")
        x = sorted(x)
        centers = tuple((m + n) / 2 for m, n in zip(x[:-1], x[1:]))
        return tuple([2 * x[0] - centers[0]]) + centers + tuple([2 * x[-1] - centers[-1]])
    except TypeError as te:  # raised when the input is not a sequence
        raise Exception(
                "Boundaries can only be calculated for a sequence type"
                " that minimally should support sorting, length, and indexing."
                "  Received " + repr(type(x)) + " instead!")
