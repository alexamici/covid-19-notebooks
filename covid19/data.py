import pathlib

import numpy as np
import requests


REFERENCE_DATETIME = "2020-02-18T16:00"


def to_days(date, reference_datetime=REFERENCE_DATETIME, hour_of_day=None):
    if isinstance(date, str):
        date = np.datetime64(date)
    if hour_of_day:
        date = date + np.timedelta64(hour_of_day, "h")
    return (date - np.datetime64(reference_datetime)) / np.timedelta64(1, "D")


def download(url, path='.'):
    root_path = pathlib.Path(path)
    download_path = root_path / url.rpartition('/')[2]
    with requests.get(url) as resp:
        with open(download_path, 'wb') as fp:
            fp.write(resp.content)
    return str(download_path)
