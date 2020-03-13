import pathlib

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
            "regioni": "{url}/dati-regioni/dpc-covid19-ita-regioni.csv",
            "andamento-nazionale": "{url}/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv",
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
