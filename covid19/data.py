import pathlib
import warnings

import numpy as np
import pandas as pd
import requests
import xarray as xr


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


CDM_DIMS = {"location", "time", "age_class", "dayofyear", "month", "year"}
CDM_COORDS = CDM_DIMS | {"country", "state", "region", "province", "lat", "lon"}


def cdm_check(da):
    dim_names = set(da.dims)
    if dim_names.difference(CDM_DIMS):
        raise ValueError(f"Unsupported dims: {dim_names.difference(CDM_DIMS)}")
    coord_names = set(da.coords)
    if coord_names.difference(CDM_COORDS):
        raise ValueError(f"Unsupported coords: {coord_names.difference(CDM_COORDS)}")


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
    "world": {"date_start": 4, "country": "Country/Region", "state": "Province/State",},
    "usa": {"date_start": 16, "country": "Country_Region", "state": "Province_State",},
}


def reformat(path, kind="world"):
    warnings.warn("deprecatred")
    raw_data = pd.read_csv(path)
    date_start = REFORMAT[kind]["date_start"]
    country = REFORMAT[kind]["country"]
    state = REFORMAT[kind]["state"]
    dates = [
        np.datetime64("20{2}-{0:02d}-{1:02d}".format(*map(int, d.split("/"))))
        for d in raw_data.columns[date_start:]
    ]
    lines = {}
    for i, record in raw_data.iterrows():
        for i, d in enumerate(record[date_start:]):
            location = record[country].strip()
            if isinstance(record[state], str):
                location += " - " + record[state].strip()
            line = lines.setdefault(
                (location, dates[i]),
                {
                    "location": location,
                    "country": record[country],
                    "deaths": 0,
                    "population": 0,
                    "date": dates[i],
                },
            )
            line["population"] += record.get("Population", 0)
            line["deaths"] += d

    return pd.DataFrame(lines.values()).set_index("date")


def read_jhu_global(path):
    df = pd.read_csv(path, keep_default_na=False)
    ds = df.to_xarray()
    ds = ds.rename(
        {
            "Country/Region": "country",
            "Province/State": "state",
            "Lat": "lat",
            "Long": "lon",
        }
    )
    ds = ds.set_coords(["country", "state", "lat", "lon"])
    da = ds.to_array("date")
    time = [
        "2020-%02d-%02d" % tuple(map(int, d.split("/")[:2])) for d in da.date.values
    ]
    location = [
        " / ".join(i for i in items if i)
        for items in zip(da.country.values, da.state.values)
    ]
    da = da.assign_coords(
        {
            "country": ("index", ds.country.astype(str)),
            "time": ("date", np.array(time, "datetime64")),
            "location": ("index", location),
        }
    )
    da = da.swap_dims({"date": "time", "index": "location"})
    da = da.drop(["index", "date", "state"])
    return da.to_dataset(name="deaths")


def read_jhu_usa(path):
    df = pd.read_csv(path, keep_default_na=False)
    ds = df.to_xarray()
    ds = ds.rename(
        {
            "Country_Region": "country",
            "Province_State": "state",
            "Admin2": "county",
            "Lat": "lat",
            "Long_": "lon",
            "Population": "population",
        }
    )
    ds = ds.assign_coords({"state": ("index", "US / " + ds.state)})
    ds = ds.set_coords(["country", "state", "county", "lat", "lon", "population"])
    ds = ds.drop(["UID", "iso2", "iso3", "code3", "FIPS", "Combined_Key"])
    da = ds.to_array("date")
    time = [
        "2020-%02d-%02d" % tuple(map(int, d.split("/")[:2])) for d in da.date.values
    ]
    da = da.assign_coords(
        {
            "country": ("index", da.country.astype(str)),
            "state": ("index", da.state.astype(str)),
            "time": ("date", np.array(time, "datetime64")),
            "location": ("index", (ds.state + " / " + ds.county).astype(str)),
        }
    )
    da = da.swap_dims({"date": "time", "index": "location"})
    da = da.drop(["index", "date", "county"])
    ds = da.to_dataset(name="deaths")
    return ds.reset_coords("population")


def istat_to_pandas(path, drop=True):
    istat = pd.read_csv(path, encoding="8859", na_values=9999)

    # make a date index from GE
    def ge2dayofyear(x):
        return (
            x - 101
            if x < 132
            else x - 170
            if x < 230
            else x - 241
            if x < 400
            else x - 310
        )

    daysofyear = istat["GE"].map(ge2dayofyear).values
    istat["time"] = (
        np.datetime64("2020-01-01", "D") + np.timedelta64(1, "D") * daysofyear
    )

    def cl_eta2age(x):
        return (
            "0-49"
            if x <= 10
            else f"{(x - 1) // 2 * 10}-{(x - 1) // 2 * 10 + 9}"
            if x < 19
            else "90+"
        )

    istat["age_class"] = istat["CL_ETA"].apply(cl_eta2age)

    if drop:
        istat = istat[np.isfinite(istat["TOTALE_20"])]

    return istat


def read_istat(path, **kwargs):
    istat = istat_to_pandas(path, **kwargs)
    tmp = istat.groupby(["time", "age_class", "NOME_COMUNE"]).agg(
        **{
            "2015": ("TOTALE_15", sum),
            "2016": ("TOTALE_16", sum),
            "2017": ("TOTALE_17", sum),
            "2018": ("TOTALE_18", sum),
            "2019": ("TOTALE_19", sum),
            "2020": ("TOTALE_20", sum),
        }
    )
    data = tmp.to_xarray().rename({"NOME_COMUNE": "location"}).fillna(0)
    data = data.to_array("year")

    coords = {
        "region": (
            "location",
            "Italy / " + istat.groupby(["NOME_COMUNE"])["NOME_REGIONE"].first(),
        ),
        "province": (
            "location",
            istat.groupby(["NOME_COMUNE"])["NOME_PROVINCIA"].first(),
        ),
        "year": ("year", data.coords["year"].astype(int)),
    }
    data = data.assign_coords(coords)
    return istat, data


def read_dpc(path):
    df = pd.read_csv(path, parse_dates=["data"], index_col=["data"])
    df.index = df.index.normalize().rename("time")
    df["location"] = "Italy / " + df["denominazione_regione"]
    df = df.set_index("location", append=True)
    ds = df[["ricoverati_con_sintomi", "terapia_intensiva", "deceduti"]].to_xarray()
    ds = ds.assign_coords(
        {
            "lat": ("location", df.groupby("location")["lat"].first()),
            "lon": ("location", df.groupby("location")["long"].first()),
            "country": ("location", ["Italy"] * ds.location.size),
            "location": ("location", [str(l) for l in ds.location.values]),
        }
    )
    ds = ds.rename(
        {
            "ricoverati_con_sintomi": "current_severe",
            "terapia_intensiva": "current_critical",
            "deceduti": "deaths",
        }
    )
    population = {
        "Lombardia": 10018806,
        "Lazio": 5898124,
        "Campania": 5839084,
        "Sicilia": 5056641,
        "Veneto": 4907529,
        "Emilia-Romagna": 4448841,
        "Piemonte": 4392526,
        "Puglia": 4063888,
        "Toscana": 3742437,
        "Calabria": 1965128,
        "Sardegna": 1653135,
        "Liguria": 1565307,
        "Marche": 1538055,
        "Abruzzo": 1322247,
        "Friuli Venezia Giulia": 1217872,
        "Umbria": 888908,
        "Basilicata": 570365,
        "Molise": 310449,
        "Valle d'Aosta": 126883,
        "P.A. Bolzano": 524256,
        "P.A. Trento": 538604,
    }
    ds = ds.assign(
        {
            "population": (
                "location",
                [population[l.partition(" / ")[2]] for l in ds.location.values],
            )
        }
    )
    return ds


def interp_on_observations(gridded, observed, index="location"):
    if isinstance(observed, xr.DataArray):
        observed = {name: values.values for name, values in observed.coords.items()}
    interp_dims = set(gridded.dims) & set(observed)
    interpolated = []
    coords = {
        dim: observed[dim] % 360.0 if dim == "lon" else observed[dim]
        for dim in interp_dims
    }
    for coord_values in zip(*coords.values()):
        selection = {dim: coord_values[i] for i, dim in enumerate(interp_dims)}
        interpolated.append(gridded.sel(**selection, method="nearest"))
    data = xr.concat(interpolated, dim=index)
    return data.assign_coords({index: (index, observed[index])})


def read_outbreaks_metadata(path):
    return pd.read_csv(path).to_dict(orient="records")
