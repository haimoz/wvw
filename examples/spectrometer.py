from wvw.devices import Spectrometer
from random import randint


if __name__ == '__main__':

    def r(n):
        while 1:
            yield tuple(randint(0, 1000) / 1000 for x in range(n))

    s = Spectrometer(1)  # a single-chanel spectrometer
    s.start()

    #for data in r(1):
    #    s.update(data)
