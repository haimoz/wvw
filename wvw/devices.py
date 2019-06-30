import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import warnings
from datetime import datetime as dt
from .utils import boundaries_of
import abc


class Spectrum(object):
    """
    A spectrum represents a 1D time series in frequency domain.  The user could
    define a window size for FFT, how many FFT results to keep in the history,
    and optionally a fixed sampling frequency to use.

    When a fixed sampling frequency is not provided, sampling rate will be
    calculated as the overall sampling frequency of the data in the current
    window.  The timestamps for each data then need to be either provided
    from an external source, or, when not provided, is considered to be the
    time when the `update` method is called.

    TODO:
    Users have the options to choose the specific FFT implementation between
    `scipy.fftpack` and `numpy.fft`.
    """
    def __init__(self,
            window_size=32,
            window_stride=None,
            history_size=50,
            fixed_frequency=None,
            fft_backend='numpy',
            ):
        """
        Create a spectrum that performs FFT on continual data in a given
        window size, with a given history size (i.e., number of FFT results to
        remember), and with optional fixed frequency.

        When the `fixed_frequency` parameter is provided, the frequency bins
        are calculated based on it.  Otherwise, either the user needs to
        provide an external timestamp for each new data, or, with absence of
        external timestamp, the timestamp is taken at the time of the call to
        the `update` method.

        By default, the windows do not overlap.  The `window_stride` parameter
        defines how many samples are between the start of consecutive windows.
        """

        # check parameters for FFT calculation

        if window_size > 0 and int(window_size) == window_size:
            self.window_size = window_size
            self.window_data = tuple()
        else:
            raise Exception(
                    "`window_size` must be a positive whole number!"
                    "  Got " + repr(window_size) + " instead.")
        self.num_bins = len(np.fft.rfftfreq(self.window_size))

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
        self.window_count = 0

    def update(self, value, timestamp=None):
        """
        Update the spectrum with data value and timestamp.

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
            self.window_count += 1
            print("windows {}".format(self.window_count))
            return True
        return False


class Spectrogram(metaclass=abc.ABCMeta):
    """
    A spectrogram defines a specific way to visualize a spectrum.

    This means that a spectrum could be visualized with multiple spectrograms.
    A spectrogram plots onto a matplotlib.Axes, which can be combined into
    different figures.
    """
    def __init__(self, spectrum, mode='magnitude'):
        """
        The `mode` parameter is similar to the `mode` parameter in
        matplotlib.axes.Axes.specgram.
        See:
        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.specgram.html#matplotlib-axes-axes-specgram
        Possible modes:
            'psd' -- power
            'magnitude' -- a.k.a. amplitude
            'angle' -- the unwrapped phase (i.e., [0, 360) instead of [-180, 180))
        """

        self.spectrum = spectrum

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

    @abc.abstractmethod
    def render(self, ax):
        raise NotImplementedError(
                "The `render` method in the `Spectrogram` base class"
                " should be overridden by derived classes"
                " and is not meant to be called!")


class MeshSpectrogram(Spectrogram):

    def render(self, ax):
        """
        Draw the data on the given axes.
        """
        # interpolate frequency and timestamp boundaries for the pseudo color mesh
        # mesh dimension: history_size x num_bins
        self.mesh_values = self.vfunc(np.array(self.spectrum.coefficients))
        self.mesh_timestamps = np.broadcast_to(self.spectrum.timestamps, [self.spectrum.num_bins, self.spectrum.history_size]).T
        self.mesh_frequencies = np.array([boundaries_of(x) for x in self.spectrum.frequencies])

        ax.clear()
        return ax.pcolormesh(
                self.mesh_timestamps.T,
                self.mesh_frequencies.T,
                self.mesh_values.T,
                cmap='magma',
                norm=colors.LogNorm())


class ImageSpectrogram(Spectrogram):

    def render(self, ax):
        if len(self.spectrum.coefficients) == 0:
            return
        self.values = self.vfunc(np.array(self.spectrum.coefficients))
        ax.clear()
        return ax.imshow(self.values, cmap='magma', norm=colors.LogNorm())


class Display(metaclass=abc.ABCMeta):
    """
    A display object takes care of rendering the visualizations at a target
    frame rate.
    """
    def __init__(self, fps=60):
        if fps > 0:
            self.fps = fps
            self.figure = plt.figure()
        else:
            raise Exception(
                    "`fps` must be a positive number"
                    " to indicate intended frame rate!"
                    "  Received " + repr(fps) + " instead!")
        self.last_draw = None
        self.draw_count = 0

    def draw(self):
        ts = dt.now().timestamp()
        if self.last_draw is None or ts - self.last_draw >= 1 / self.fps:
            self.render()
            self.last_draw = ts
            self.draw_count += 1
            print("drawn {}".format(self.draw_count))
            plt.ion()
            plt.pause(0.000001)


    @abc.abstractmethod
    def render(self):
        raise NotImplementedError(
                "The `render` method in the view base class"
                " is not implemented and is not expected to be called.")


class Spectrometer(Display):
    """
    A spectrometer is a combination of spetrum (i.e., data processing) and
    spectrogram (i.e., visualization) functionalities.
    """
    def __init__(self, number_of_channels, spectrum_kwargs=None, **kwargs):
        Display.__init__(self, **kwargs)
        self.axes = self.figure.subplots(number_of_channels, 1, sharex=True, sharey=True, squeeze=False)
        if spectrum_kwargs is None:
            self.spectra = tuple(Spectrum() for x in range(number_of_channels))
        else:
            self.spectra = tuple(Spectrum(**spectrum_kwargs) for x in range(number_of_channels))
        self.spectrograms = tuple(ImageSpectrogram(x) for x in self.spectra)

    def update(self, data, timestamp=None):
        return any([sp.update(v, timestamp) for sp, v in zip(self.spectra, data)])

    def render(self):
        return [x.render(ax) for x, ax in zip(self.spectrograms, self.axes.flat)]
