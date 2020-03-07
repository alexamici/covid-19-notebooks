
import numpy as np


REFERENCE_DATETIME = '2020-02-18T16:00'


def to_days(date, reference_datetime=REFERENCE_DATETIME, hour_of_day=16):
    if isinstance(date, str):
        date = np.datetime64(date)
    date_adjusted = date + np.timedelta64(hour_of_day, 'h')
    return (date_adjusted - np.datetime64(reference_datetime)) / np.timedelta64(1, 'D')
