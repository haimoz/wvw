from wvw.devices import Spectrometer
from random import randint
import math


if __name__ == '__main__':

    # mock data
    def r(n):
        while 1:
            yield tuple(randint(0, 1000) / 1000 for x in range(n))
    def r2(n):
        t = 0
        while 1:
            t += 0.01
            f = math.sin(t)
            yield tuple(math.sin(f*t) for x in range(n))

    # number of sensor input
    NUM_CH = 1

    # initialization of a spectrometer
    s = Spectrometer(
            number_of_channels=NUM_CH,
            spectrum_init_params={
                'window_size'     : 500,
                'window_stride'   : 500,
                'history_size'    : 1000,
                'fixed_data_rate' : None,
            },
            fps=5
        )

    # sending data to the spectrometer for FFT and visualization
    # the input data needs to be a tuple of length NUM_CH
    for data in r2(NUM_CH):
        if s.update(data):
            s.draw()
