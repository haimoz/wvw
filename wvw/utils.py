import serial
from serial.tools.list_ports import comports


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


def get_serial(port, *args, **kwargs):
    """
    A wrapper method for opening serial port with automatic port listing and
    automatic/interactive selection.
    """
    if port is not None:
        return serial.Serial(port, *args, **kwargs)
    else:
        ports = tuple(comports())  # list detected serial ports
    if len(ports) == 0:
        raise Exception("No serial ports found!")
    elif len(ports) == 1:
        port = ports[0]
    else:
        port = None
        while port is None:
            sel = int(input(
                "\nAvailable ports:\n{}\n\nSelect port to use: ".format(
                    '\n'.join([
                        str(i) + ": " + x.device
                        for i, x in enumerate(ports)]))))
            if sel >= 0 and sel < len(ports):
                port = ports[sel]
    print("\nUsing port: {}".format(port.device))
    return serial.Serial(port, *args, **kwargs)
