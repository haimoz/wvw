"""
Core (abstract) componets for any device.
"""
import abc
from time import time  # access time, way faster than datetime.datetime.now().timestamp()
import warnings
import matplotlib.pyplot as plt


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
    timestamp for each new data, or, when an external timestamp is absent, the
    timestamp is taken at the time of the call to the `update` method.
    """
    def __init__(self,
            window_size,
            window_stride,
            history_size,
            fixed_data_rate=None,
            ):
        """
        parameters:

            window_size:

                A positive integer indicating the number of data in a window.

            window_stride:

                A positive integer indicating how far two windows are apart.

            history_size:

                A positive integer indicating how many results for past windows
                are to be retained.

            fixed_data_rate:

                A positive number or None.

                If a positive number is provided, then it is considered to be
                the number of incoming data per second, assuming a constant
                data rate.  In this case, the timestamp is not affected by when
                a data is supplied, and any timestamps supplied with the data
                updates will be ignored.

                If None is provided, the timestamp mode will be either
                "internal" or "external", depending on whether a timestamp
                is supplied at the first update of data.  If a timestamp is
                supplied, then the timestamp mode is "external", and each data
                update must supply an externally determined timestamp, such as
                from a microcontroller.  Otherwise, the timestamp mode is
                "internal", and the timestamps for each data update is the
                time of invocation of the `Processor.update` method, ignoring
                the timestamps supplied to this method.
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
            # default is non-overlapping windows
            self.window_stride = self.window_size
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

        Returns True if a new window is successfully processed (and thus might
        need to update the visualization as well).
        """

        if self.window_is_growing:

            # check and update timestamp mode
            if self.timestamp_mode == 'fixed':
                # do not expect an external timestamp for fixed-frequency
                if timestamp is not None:
                    warnings.warn(
                            "A timestamp is not needed "
                            "to update a fixed-frequency samples processor.  "
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
                            "in an internally-timestamped samples processor.  "
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
            # trim data that are not needed for the next window
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

        DEVELOPER NOTE:
        The processing of windows could have been implemented as a callback
        function (taking two sequences of corresponding data values and
        timestamps), rather than as an abstract method.  The reason to
        implement it as an abstract method is to account for situations where
        subclasses need to process a window based on other information such as
        those in the history (e.g., results from previous windows).
        """
        raise NotImplementedError(
                "The `process` method in the `Processor` base class"
                " is not implemented.  Deriving classes are expected to"
                " override this method.")


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
