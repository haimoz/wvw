import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
from time import time  # access time, way faster than datetime.datetime.now().timestamp()
from .utils import boundaries_of
import abc


class Processor(metaclass=abc.ABCMeta):
    """
    A processor is an abstract representation of a data processing unit in a
    workbench device.

    A processor performs computation on windows, i.e., a subset of contiguous
    data, transforming them into possibly other representations.  It also keeps
    the history of processing results from previous windows.

    A processor takes care of timestamping of the data.  When the
    `fixed_data_rate` parameter is provided, the data rate is assumed to be at
    that fixed value.  Otherwise, either the user needs to provide an external
    timestamp for each new data, or, when external timestamp is absent, the
    timestamp is taken at the time of the call to the `update` method.
    """
    def __init__(self,
            window_size,
            window_stride,
            history_size,
            fixed_data_rate=None,
            ):
        """
        """

        # processing window
        #======================================================================
        # 1. validate window size
        if window_size > 0 and int(window_size) == window_size:
            self.window_size = window_size
        else:
            raise Exception(
                    "`window_size` must be a positive whole number!"
                    "  Got " + repr(window_size) + " instead.")
        # 2. validate window stride
        if window_stride is None:
            self.window_stride = self.window_size  # default is non-overlapping windows
        elif window_stride > 0 and int(window_stride) == window_stride:
            self.window_stride = window_stride
        else:
            raise Exception(
                    "`window_stride` must be a positive whole number or None!"
                    "  Got " + repr(window_stride) + " instead.")
        # 3. prepare window data storage
        self.window_data = tuple()
        self.window_timestamps = tuple()
        self.window_next_start = 0  # countdown to next start of window
        self.window_next_end = self.window_size - 1  # countdown to next end of window
        self.window_trim_amount = min(self.window_stride, self.window_size)  # how many data to remove from the window after it has been processed
        self.window_is_contiguous = self.window_stride <= self.window_size
        self.window_is_growing = True  # whether the window is growing, i.e., accepting new data

        # history
        #======================================================================
        # 1. validate history size
        if history_size > 0 and int(history_size) == history_size:
            self.history_size = history_size
        else:
            raise Exception(
                    "`history_size` must be a positive whole number!"
                    "  Got " + repr(history_size) + " instead.")
        # history data storage
        self.history_data = tuple()
        self.history_timestamps = tuple()

        # data rate & timestamping
        #======================================================================
        # 1. validate data rate
        if fixed_data_rate is None:
            self.timestamp_mode = None
        elif fixed_data_rate > 0:
            self.timestamp_mode = 'fixed'
            self.sample_spacing = 1.0 / fixed_data_rate
        else:
            raise Exception(
                    "`fixed_data_rate` must be a positive number or None!"
                    "  Got " + repr(fixed_data_rate) + " instead.")
        # 2. timestamp alignment (so that data starts from time point 0)
        self.ts_offset = None

        # performance logging
        #======================================================================
        self.window_count = 0
        self.data_count = 0

    def update(self, value, timestamp=None):
        """
        Update the processor with a single new data value and timestamp.

        Returns True if a new window is successfully processed (and thus should
        update the visualization as well).
        """

        if self.window_is_growing:

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
                self.timestamp_mode = 'internal' if timestamp is None else 'external'
            else:
                raise Exception(
                        "Unknown timestamp_mode: " + repr(self.timestamp_mode))

            # figure out the timestamp of the current data
            if self.timestamp_mode == 'internal':
                timestamp = time()
            elif self.timestamp_mode == 'fixed':
                if self.ts_offset is None:
                    # NOTE: DO NOT change the initial timestamp in fixed timestamp
                    # mode, otherwise it will break the offset adjustment!
                    timestamp = 0
                else:
                    timestamp = self.window_timestamps[-1] + self.sample_spacing
            else:
                assert(self.timestamp_mode == 'external' and timestamp is not None)
            if self.ts_offset is None:
                self.ts_offset = timestamp
            # Adjust the timestamp so that data starts from 0, instead of an
            # arbitrary device-specific or runtime-specific time point.
            timestamp -= self.ts_offset

            # update data
            self.window_data += tuple([value])
            self.window_timestamps += tuple([timestamp])

        # check whether there is enough data in the window to be processed
        results = None
        if self.window_next_end == 0:
            self.window_next_end = self.window_stride
            self.window_is_growing = self.window_is_contiguous
            self.window_count += 1
            if self.timestamp_mode in ('internal', 'external'):
                self.sample_spacing = (max(self.window_timestamps) - min(self.window_timestamps)) / len(self.window_timestamps)
            else:
                assert(self.timestamp_mode == 'fixed')
            # if trimmed correctly, the windows would naturally be in the expected length
            self.window_data = self.window_data#[-self.window_size:]
            self.window_timestamps = self.window_timestamps#[-self.window_size:]
            results = self.process()
            self.window_data = self.window_data[self.window_trim_amount:]
            self.window_timestamps = self.window_timestamps[self.window_trim_amount:]
        if self.window_next_start == 0:
            self.window_next_start = self.window_stride
            self.window_is_growing = True

        # update counters
        self.window_next_start -= 1
        self.window_next_end -= 1
        self.data_count += 1

        # return whether there are any processing result and update history
        if results is None:
            return False
        else:
            # update history when the processing of the window returns some result
            self.history_data = (self.history_data + tuple([results]))[-self.history_size:]
            self.history_timestamps = (self.history_timestamps + tuple([timestamp]))[-self.history_size:]
            return True

    @abc.abstractmethod
    def process(self):
        """
        The process method processes a window.  Derived classes are expected
        not to change any of its members inherited from the Processor base
        class.
        """
        raise NotImplementedError(
                "The `process` method in the `Processor` base class"
                " is not implemented.  Deriving classes are expected to"
                " override this method.")


class Spectrum(Processor):
    """
    A spectrum is the frequency domain representation of a 1D time series.

    TODO:
    Users have the options to choose the specific FFT implementation between
    `scipy` (using `scipy.fftpack`) or `numpy` (using `numpy.fft`).
    """
    def __init__(self,
            fft_backend='numpy',
            **kwargs
            ):
        """
        TODO:
        - Support selection of FFT backends
        """
        if kwargs.get('window_size') == 1:
            raise Exception(
                    "A spectrum needs at least two samples per window!")
        Processor.__init__(self, **kwargs)

    def process(self):
        """
        Process the current window using FFT.
        """
        # peform FFT on the window
        # assumes uniform sampling
        # TODO future implementation could use:
        # https://scicomp.stackexchange.com/questions/593/how-do-i-take-the-fft-of-unevenly-spaced-data
        # and
        # https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform
        return (
                np.fft.rfft(self.window_data),
                np.fft.rfftfreq(len(self.window_data), self.sample_spacing)
        )


class Spectrogram(metaclass=abc.ABCMeta):
    """
    A spectrogram is a specific way to visualize a spectrum.

    This means that a spectrum could be visualized with multiple spectrograms.
    A spectrogram plots onto a matplotlib.Axes, which can be combined into
    different figures.
    """
    def __init__(self, spectrum, freq_dir, time_dir, mode='magnitude'):
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

        # bind to a specific spectrum processor
        self.spectrum = spectrum

        # handle mode
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
        """
        The name of the `render` method is intentionally different from similar
        names in matplotlib (`show` & `draw`).
        """
        raise NotImplementedError(
                "The `render` method in the `Spectrogram` base class"
                " should be overridden by derived classes"
                " and is not meant to be called!")


class ImageSpectrogram(Spectrogram):
    """
    Renders a spectrogram as a square image.  Since actual frequency bins can
    have slight differences between windows, this is an approximate
    visualization which should be faster than `pcolormesh`-based methods.
    The frequency bins shown are those from the last processed window.
    """
    def __init__(self, freq_dir='up', time_dir='right', **kwargs):
        # set default parameter for parent class initialization
        kwargs.update(freq_dir=freq_dir, time_dir=time_dir)
        Spectrogram.__init__(self, **kwargs)

        # define the orientation of the plot
        horizontal = {'left', 'right'}
        vertical = {'up', 'down'}
        if freq_dir in horizontal and time_dir in vertical:
            self.flip = lambda img: img
            self.origin = 'upper'
            self.extent = lambda f, t: (min(f), max(f), max(t), min(t))
            self.xlim = (lambda ext: (ext[0], ext[1])) if freq_dir == 'right' else (lambda ext: (ext[1], ext[0]))
            self.ylim = (lambda ext: (ext[2], ext[3])) if time_dir == 'down' else (lambda ext: (ext[3], ext[2]))
            self.set_ticks = lambda ax, f, t: (ax.set_xticks(f), ax.set_yticks(t))
        elif freq_dir in vertical and time_dir in horizontal:
            self.flip = lambda img: img.T
            self.origin = 'upper'
            self.extent = lambda f, t: (min(t), max(t), max(f), min(f))
            self.xlim = (lambda ext: (ext[0], ext[1])) if time_dir == 'right' else (lambda ext: (ext[1], ext[0]))
            self.ylim = (lambda ext: (ext[2], ext[3])) if freq_dir == 'down' else (lambda ext: (ext[3], ext[2]))
            self.set_ticks = lambda ax, f, t: (ax.set_xticks(t), ax.set_yticks(f))
        else:
            raise Exception(
                    "Invalid combination of"
                    " frequency direction " + repr(freq_dir) +
                    " and timestamp direction " + repr(time_dir))

    def render(self, ax):
        if len(self.spectrum.history_data) == 0:
            return
        # prepare image data
        img = self.vfunc(np.array(tuple(x[0] for x in self.spectrum.history_data)))
        # prepare axis
        frequencies = self.spectrum.history_data[-1][1]
        frequency_edges = boundaries_of(frequencies)
        timestamps = tuple([0]) + self.spectrum.history_timestamps
        ax.clear()
        ext = self.extent(frequency_edges, timestamps)
        img = ax.imshow(self.flip(img), origin=self.origin, extent=None, cmap='magma', norm=colors.LogNorm())
        #ax.set_xlim(*self.xlim(ext))
        #ax.set_ylim(*self.ylim(ext))
        #self.set_ticks(ax, frequencies, timestamps)
        return img


class BarSpectrogram(Spectrogram):
    """
    A bar spectrogram plots FFT components using bar plot, which is accurate.
    """
    def render(self, ax):
        #TODO
        pass


class Display(metaclass=abc.ABCMeta):
    """
    A display object takes care of rendering the visualizations at a target
    frame rate.
    """
    def __init__(self, fps=5):
        if fps > 0:
            self.fps = fps
            self.figure = plt.figure()
        else:
            raise Exception(
                    "`fps` must be a positive number"
                    " to indicate intended frame rate!"
                    "  Received " + repr(fps) + " instead!")
        self.next_draw = None
        self.draw_count = 0
        plt.ion()  # turn on interactive mode so drawing does not block updating

    def draw(self):
        ts = time()
        if self.next_draw is None or ts >= self.next_draw:
            self.render()
            self.next_draw = ts + 1.0 / self.fps
            self.draw_count += 1
            #plt.draw()  # won't need to explicitly request for drawing in interactive mode
            plt.pause(10e-6)

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError(
                "The `render` method in the display base class"
                " is not implemented and is not expected to be called.")


class Spectrometer(Display):
    """
    A spectrometer is a combination of spetrum (i.e., data processing) and
    spectrogram (i.e., visualization) functionalities.
    """
    def __init__(self, number_of_channels,
            spectrum_init_params=None,
            spectrogram_init_params=None,
            **kwargs):
        Display.__init__(self, **kwargs)
        # prepare figures to be plotted on
        self.axes = self.figure.subplots(number_of_channels, 1, sharex=True, sharey=True, squeeze=False)
        # initialize spectra
        if spectrum_init_params is None:
            spectrum_init_params = {}
        self.spectra = tuple(Spectrum(**spectrum_init_params) for x in range(number_of_channels))
        # initialize spectrograms
        if spectrogram_init_params is None:
            spectrogram_init_params = {}
        self.spectrograms = tuple(ImageSpectrogram(spectrum=x, **spectrogram_init_params) for x in self.spectra)

    def update(self, data, timestamp=None):
        return any([sp.update(v, timestamp) for sp, v in zip(self.spectra, data)])

    def render(self):
        return [x.render(ax) for x, ax in zip(self.spectrograms, self.axes.flat)]
