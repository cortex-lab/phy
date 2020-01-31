"""Show how to add a custom raw data filter for the TraceView and Waveform View

Use Alt+R in the GUI to toggle the filter.

"""

from scipy.signal import butter, filtfilt

from phy import IPlugin


class ExampleRawDataFilterPlugin(IPlugin):
    def attach_to_controller(self, controller):
        b, a = butter(3, 150.0 / controller.model.sample_rate * 2.0, 'high')

        @controller.raw_data_filter.add_filter
        def high_pass(arr, axis=0):
            return filtfilt(b, a, arr, axis=axis)
