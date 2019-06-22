import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime as dt
from .utils import boundaries_of


class Spectrogram(object):
    """
    A spectrogram consists of two functional parts: FFT and visualization.
    The user could define a window size to calculate the spectrum for,
    and a fixed sampling frequency to use.

    When a fixed sampling frequency is not provided, sampling rate will be
    calculated dynamically based on the timestamps of the data in the current
    window.  The timestamps for each data then need to be either provided
    from an external source, or, when not provided, is considered to be the
    time when the `update` method is called.

    A spectrogram corresponds to a single 1D scalar time series that is plotted
    on a matplotlib axes.
    """
    def __init__(self, window_size=32, window_stride=None, history_size=10, fixed_frequency=None, ax=None, mode='magnitude'):
        """
        Create a spectrometer that performs FFT on continual data in a given
        window size, with a given memory size (i.e., number of windows to
        remember), and with optional fixed frequency.

        When the `frequency` parameter is provided, the frequency bins are
        calculated based on it.  Otherwise, either the user needs to provide
        an external timestamp for each new data, or, with absence of external
        timestamp, the timestamp is taken at the time of the call to the
        `update` method.

        By default, the windows do not overlap.

        `mode` parameter is similar to the `mode` parameter in
        matplotlib.axes.Axes.specgram.
        See:
        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.specgram.html#matplotlib-axes-axes-specgram
        Possible modes:
            'psd' -- power
            'magnitude' -- a.k.a. amplitude
            'angle' -- the unwrapped phase (i.e., [0, 360) instead of [-180, 180))
        """
        plt.ion()

        # parameters for FFT calculation

        if window_size > 0 and int(window_size) == window_size:
            self.window_size = window_size
            self.window_data = tuple()
        else:
            raise Exception(
                    "`window_size` must be a positive whole number!"
                    "  Got " + repr(window_size) + " instead.")

        if window_stride is None:
            self.window_stride = self.window_size  # default is non-overlapping windows
        elif window_stride > 0 and int(window_stride) == window_stride:
            self.window_stride = window_stride
        else:
            raise Exception(
                    "`window_stride` must be a positive whole number or None!"
                    "  Got " + repr(window_stride) + " instead.")

        if fixed_frequency is None:
            self.timestamp_mode = None
        else:
            self.timestamp_mode = 'fixed'
            self.sample_spacing = 1.0 / fixed_frequency
        self.window_timestamps = tuple()

        # history data

        if history_size > 0 and int(history_size) == history_size:
            self.history_size = history_size
        else:
            raise Exception(
                    "`history_size` must be a positive whole number!"
                    "  Got " + repr(history_size) + " instead.")
        self.coefficients = tuple()  # complex FFT coefficients
        self.frequencies = tuple()
        self.timestamps = tuple()
        self.ts_offset = None  # offset for timestmap, so that the data starts from time point 0

        # for visualization

        if mode == 'psd':
            self.vfunc = lambda x: np.abs(x)**2
        elif mode == 'magnitude':
            self.vfunc = np.abs
        elif mode == 'angle':
            self.vfunc = np.angle
        else:
            raise Exception(
                    "`mode` is expected to be one of"
                    " { 'psd', 'magnitude', 'angle' },"
                    " but received " + repr(mode) + " instead!")

        num_bins = len(np.fft.rfftfreq(window_size))
        self.mesh_timestamps = np.zeros([num_bins, history_size + 1])
        self.mesh_frequencies = np.zeros([num_bins, history_size + 1])
        self.mesh_values = np.zeros([num_bins - 1, history_size])  # the value to be visualized, could be amplitude, power, or phase
        if ax is not None:
            self.axes = ax
            self.bind_to_axes(self.axes)
        else:
            self.axes = None

    def update(self, value, timestamp=None):
        """
        Update the spectrometer with data value and timestamp.

        The timestamp should be in the unit of seconds.
        """
        # check and update timestamp mode
        if self.timestamp_mode == 'fixed':
            # do not expect an external timestamp for fixed-frequency
            if timestamp is not None:
                warnings.warn(
                        "A timestamp is not needed "
                        "to update a fixed-frequency spectrometer.  "
                        "The user-provided timestamp will be ignored.")
        elif self.timestamp_mode == 'external':
            # expect a user-supplied timestamp in external timestamp mode
            if timestamp is None:
                raise Exception("User-supplied timestamp is expected!")
        elif self.timestamp_mode == 'internal':
            # do not expect a user-supplied timestamp in internal timestamp mode
            if timestamp is not None:
                warnings.warn(
                        "A user-supplied timestamp is not need "
                        "in an internally-timestamped spectrometer.  "
                        "The user-provided timestamp will be ignored.")
        elif self.timestamp_mode is None:
            if timestamp is None:
                self.timestamp_mode = 'internal'
            else:
                self.timestamp_mode = 'external'
        else:
            raise Exception(
                    "Unknown timestamp_mode: " + repr(self.timestamp_mode))

        # figure out the timestamp of the current data
        if self.timestamp_mode == 'internal':
            timestamp = dt.now().timestamp()
        elif self.timestamp_mode == 'fixed':
            if self.ts_offset is None:
                # NOTE: DO NOT change the initial timestamp in fixed timestamp
                # mode, otherwise it will break the offset adjustment!
                timestamp = 0
            else:
                timestamp = self.window_timestamps[-1] + self.sample_spacing
        else:
            assert(self.timestamp_mode == 'external')
        if self.ts_offset is None:
            self.ts_offset = timestamp
        # Adjust the timestamp so that data starts from 0, instead of an
        # arbitrary device-specific or runtime-specific time point.
        timestamp -= self.ts_offset

        # update data
        self.window_data += tuple([value])
        self.window_timestamps += tuple([timestamp])

        # peform FFT on the window
        # assumes uniform sampling
        # TODO future implementation could use:
        # https://scicomp.stackexchange.com/questions/593/how-do-i-take-the-fft-of-unevenly-spaced-data
        # and
        # https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform
        if len(self.window_data) == self.window_size + self.window_stride:
            self.window_data = self.window_data[-self.window_size:]
            self.window_timestamps = self.window_timestamps[-self.window_size:]
            if self.timestamp_mode in ('internal', 'external'):
                self.sample_spacing = (max(self.window_timestamps) - min(self.window_timestamps)) / len(self.window_timestamps)
            else:
                assert(self.timestamp_mode == 'fixed')
            coefficients = np.fft.rfft(self.window_data)
            frequencies = np.fft.rfftfreq(len(self.window_data), self.sample_spacing)

            # update history
            self.coefficients = (self.coefficients + tuple([coefficients]))[-self.history_size:]
            self.frequencies = (self.frequencies + tuple([frequencies]))[-self.history_size:]
            self.timestamps = (self.timestamps + tuple([timestamp]))[-self.history_size:]

            # interpolate frequency and timestamp boundaries for the pseudo color mesh
            self.mesh_values = np.column_stack([self.mesh_values[:,1:], self.vfunc(coefficients[1:])])
            self.mesh_timestamps = np.column_stack([self.mesh_timestamps[:,1:], np.full(len(coefficients), timestamp)])
            self.mesh_frequencies = np.column_stack([self.mesh_frequencies[:,1:], boundaries_of(frequencies[1:])])

            # update plot
            if self.axes is not None:
                self.render_on_axes(self.axes)

    def bind_to_axes(self, ax):
        """
        Bind the spectrometer to a plot output.
        """
        if self.axes is not None and ax is not None:
            if self.axes is ax:
                warnings.warn("The spectrometer is already bound to this axes!")
            else:
                warnings.warn("Rebinding the spectrometer to a different axes!")
        self.axes = ax
        if self.axes is not None:
            self.render_on_axes(self.axes)

    def render_on_axes(self, ax):
        """
        Draw the data on the given axes.
        """
        ax.clear()
        ax.pcolormesh(self.mesh_timestamps, self.mesh_frequencies, self.mesh_values)
        plt.pause(0.01)

class Spectrometer(object):
    """
    A spectrometer is a collection of spectrograms.  It corresponds to a
    matplotlib figure.
    """
    def __init__(self, number_of_channels, **spectrogram_kwargs):
        plt.ion()
        self.figure, self.axes = plt.subplots(number_of_channels, 1, sharex=True, sharey=True)
        if spectrogram_kwargs.pop('ax', None) is not None:
            warnings.warn("`ax` parameter is not allowed in initializing a spectrometer!")
        self.spectrograms = tuple(Spectrogram(**spectrogram_kwargs) for x in range(number_of_channels))
        for sp, ax in zip(self.spectrograms, self.axes):
            sp.bind_to_axes(ax)

    def update(self, data, timestamp):
        for sp, d in zip(self.spectrograms, data):
            sp.update(d, timestamp)
