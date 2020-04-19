import pathlib

import numpy as np
import pandas as pd
import requests


DATA_REPOS = {
    "world": {
        "url": "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master",
        "streams": {
            "deaths": "{url}/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
            "cases": "{url}/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
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
    "usa": {
        "url": "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master",
        "streams": {
            "deaths": "{url}/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv",
            "cases": "{url}/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv",
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


REFORMAT = {
    "world": {
        "date_start": 4,
        "country": 'Country/Region',
        "state": 'Province/State',
    },
    "usa": {
        "date_start": 16,
        "country": 'Country_Region',
        "state": 'Province_State',
    },
}


def reformat(path, kind='world'):
    raw_data = pd.read_csv(path)
    date_start = REFORMAT[kind]["date_start"]
    country = REFORMAT[kind]["country"]
    state = REFORMAT[kind]["state"]
    dates = [np.datetime64('20{2}-{0:02d}-{1:02d}'.format(*map(int, d.split('/')))) for d in raw_data.columns[date_start:]]
    lines = {}
    for i, record in raw_data.iterrows():
        for i, d in enumerate(record[date_start:]):
            location = record[country].strip()
            if isinstance(record[state], str):
                location += ' - ' + record[state].strip()
            line = lines.setdefault((location, dates[i]), {
                'location': location,
                'country': record[country],
                'deaths': 0,
                'population': 0,
                'date': dates[i]
            })
            line['population'] += record.get('Population', 0)
            line['deaths'] += d

    return pd.DataFrame(lines.values()).set_index('date')


def istat_to_pandas(path, drop=True):
    istat = pd.read_csv(path, encoding='8859', na_values=9999)

    # make a date index from GE
    def ge2dayofyear(x):
        return x - 101 if x < 132 else x - 170 if x < 230 else x - 241 if x < 400 else x - 310

    daysofyear = istat['GE'].map(ge2dayofyear).values
    istat['time'] = np.datetime64('2020-01-01', 'D') + np.timedelta64(1, 'D') * daysofyear

    def cl_eta2age(x):
        return '0-49' if x <= 10 else f'{(x - 1) // 2 * 10}-{(x - 1) // 2 * 10 + 9}' if x < 19 else '90+'

    istat['age_class'] = istat['CL_ETA'].apply(cl_eta2age)

    if drop:
        istat = istat[np.isfinite(istat['TOTALE_20'])]

    return istat


def istat_to_xarray(path):
    istat = istat_to_pandas(path)
    tmp = istat.groupby(['time', 'age_class', 'NOME_PROVINCIA']).agg(
        deaths=('TOTALE_20', sum),
    )
    data = tmp.to_xarray().rename({'NOME_PROVINCIA': 'location'}).fillna(0)
    return data.assign_coords({'region': ('location', istat.groupby(['NOME_PROVINCIA'])['NOME_REGIONE'].first())})
