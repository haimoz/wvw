from wvw.devices import Spectrometer
from random import randint


if __name__ == '__main__':

    def r(n):
        while 1:
            yield tuple(randint(0, 1000) / 1000 for x in range(n))

    NUM_CH = 2

    s = Spectrometer(
            number_of_channels=NUM_CH,
            spectrum_init_params={
                'window_size'     : 32,
                'window_stride'   : 32,
                'history_size'    : 50,
                'fixed_data_rate' : None,
            }
        )

    for data in r(NUM_CH):
        if s.update(data):
            s.draw()
