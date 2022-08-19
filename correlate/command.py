import pandas as pd
from scipy.signal import correlate, correlation_lags
from otlang.sdk.syntax import Positional, OTLType
from pp_exec_env.base_command import BaseCommand, Syntax


class CorrelateCommand(BaseCommand):
    """
    Compute cross-correlate two N-dimensional arrays

    | correlate first_array second_array
    """
    syntax = Syntax(
        [
            Positional("first_signal", required=True, otl_type=OTLType.TEXT),
            Positional("second_signal", required=True, otl_type=OTLType.TEXT),
        ],
    )
    use_timewindow = False  # Does not require time window arguments
    idempotent = True  # Does not invalidate cache

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_progress('Start correlate command')

        first_signal_column_name = self.get_arg("first_signal").value
        second_signal_column_name = self.get_arg("second_signal").value

        signal_1 = df[first_signal_column_name].values
        signal_2 = df[second_signal_column_name].values
        correlation = correlate(signal_1, signal_2)
        lags = correlation_lags(len(signal_1), len(signal_2))

        df = pd.DataFrame({"correlation": correlation,
                           "lags": lags})

        self.log_progress('First part is complete.', stage=1, total_stages=1)
        return df
