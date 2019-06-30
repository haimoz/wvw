from wvw.devices import Spectrometer
from random import randint


if __name__ == '__main__':

    def r(n):
        while 1:
            yield tuple(randint(0, 1000) / 1000 for x in range(n))

    NUM_CH = 2

    s = Spectrometer(NUM_CH)  # a single-chanel spectrometer

    for data in r(NUM_CH):
        if s.update(data):
            s.draw()
