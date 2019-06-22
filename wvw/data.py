class IMUReading(object):

    def __init__(self):
        self.a = [0.0] * 3;  # accelerometer
        self.g = [0.0] * 3;  # gyroscope
        self.m = [0.0] * 3;  # magnetometer

    @staticmethod
    def parse(data):
        x = IMUReading()
        x.a = data[0:3]
        x.g = data[3:6]
        x.m = data[6:]
        return x

    def __str__(self):
        return "< {:4.3f}, {:4.3f}, {:4.3f} | {:3.0f}, {:3.0f}, {:3.0f} | {:4.0f}, {:4.0f}, {:4.0f} >".format(
                self.a[0], self.a[1], self.a[2],
                self.g[0], self.g[1], self.g[2],
                self.m[0], self.m[1], self.m[2])


class Frame(object):
    """
    A data frame consists of six groups of IMU readings, each consisting of 3
    linear accelerations in the unit of g's, 3 angular velocities in the unit
    of degrees per second, and 3 magnitometer readings in the unit of milli
    Gauss.
    """
    def __init__(self):
        self.readings = None
        self.timestamp = None

    @staticmethod
    def parse(data):
        x = Frame()
        x.readings = [IMUReading.parse(data[i:i+9]) for i in range(0,54,9)]
        x.timestamp = data[54]
        return x

    def __str__(self):
        return '\n'.join([str(x) for x in self.readings]) + '\n@ {}'.format(self.timestamp)
