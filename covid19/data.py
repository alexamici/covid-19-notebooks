import pathlib

import numpy as np
import pandas as pd
import requests


DATA_REPOS = {
    "world": {
        "url": "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master",
        "streams": {
            "deaths": "{url}/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
        },
    },
    "italy": {
        "url": "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master",
        "streams": {
            "andamento-nazionale": "{url}/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv",
            "regioni": "{url}/dati-regioni/dpc-covid19-ita-regioni.csv",
            "province": "{url}/dati-province/dpc-covid19-ita-province.csv",
        },
    },
}


def download(url, path=".", repo="italy"):
    repo = DATA_REPOS[repo]
    base_url = repo["url"]
    stream_url = repo["streams"].get(url, url).format(url=base_url)
    root_path = pathlib.Path(path)
    download_path = root_path / stream_url.rpartition("/")[2]
    with requests.get(stream_url) as resp:
        with open(download_path, "wb") as fp:
            fp.write(resp.content)
    return str(download_path)


def reformat(path):
    raw_data = pd.read_csv(path)
    lines = []
    dates = [np.datetime64('20{2}-{0:02d}-{1:02d}'.format(*map(int, d.split('/')))) for d in raw_data.columns[4:]]
    for i, record in raw_data.iterrows():
        for i, d in enumerate(record[4:]):
            location = record['Country/Region'].strip()
            if isinstance(record['Province/State'], str):
                location += ' - ' + record['Province/State'].strip()
            if d > 0:
                lines.append({
                    'location': location,
                    'country': record['Country/Region'],
                    'deaths': d,
                    'date': dates[i]
                })

    return pd.DataFrame(lines).set_index('date')
