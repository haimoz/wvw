import abc
import numpy as np
import matplotlib.colors as colors
from ..utils import boundaries_of
from ..core import Processor, Display


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
        #img = ax.imshow(self.flip(img), origin=self.origin, extent=None, cmap='magma', norm=colors.LogNorm())
        img = ax.imshow(self.flip(img), origin=self.origin, extent=None, cmap='magma', vmin=0, vmax=100)
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
