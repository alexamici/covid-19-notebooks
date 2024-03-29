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
            "confirmed": "{url}/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
            "LUT": "{url}/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv",
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
            "confirmed": "{url}/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv",
        },
    },
}


CDM_DIMS = {"location", "time", "age_class", "dayofyear", "month", "year"}
CDM_COORDS = CDM_DIMS | {"country", "state_region", "lat", "lon"}

POPULATION_BY_REGION = {
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
    "Italia": 60360000,
}
POPULATION_70_BY_REGION = {
    "Lombardia": 1733849,
    "Lazio": 952612,
    "Campania": 788697,
    "Sicilia": 772880,
    "Veneto": 852983,
    "Emilia-Romagna": 826482,
    "Piemonte": 848816,
    "Puglia": 658660,
    "Toscana": 730740,
    "Calabria": 305986,
    "Sardegna": 287140,
    "Liguria": 342003,
    "Marche": 289306,
    "Abruzzo": 234575,
    "Friuli Venezia Giulia": 244902,
    "Umbria": 172921,
    "Basilicata": 95735,
    "Molise": 56548,
    "Valle d'Aosta": 22653,
    "P.A. Bolzano": 79886,
    "P.A. Trento": 90702,
    "Italia": 10388076,
}
POPULATION_80_BY_REGION = {
    "Lombardia": 737640,
    "Lazio": 400605,
    "Campania": 304317,
    "Sicilia": 315915,
    "Veneto": 358540,
    "Emilia-Romagna": 369353,
    "Piemonte": 371400,
    "Puglia": 268126,
    "Toscana": 320589,
    "Calabria": 130778,
    "Sardegna": 116283,
    "Liguria": 155969,
    "Marche": 133365,
    "Abruzzo": 104003,
    "Friuli Venezia Giulia": 103493,
    "Umbria": 77917,
    "Basilicata": 43930,
    "Molise": 26257,
    "Valle d'Aosta": 9564,
    "P.A. Bolzano": 33273,
    "P.A. Trento": 38386,
    "Italia": 4419703,
}


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


def read_jhu_global(path, lut_path=None):
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
        "20{2:02d}-{0:02d}-{1:02d}".format(*map(int, d.split("/")))
        for d in da.date.values
    ]
    location = [
        " / ".join(i for i in items if i)
        for items in zip(da.country.values, da.state.values)
    ]
    state_region = da.country + " / " + da.state
    da = da.assign_coords(
        {
            "country": ("index", ds.country.astype(str)),
            "time": ("date", np.array(time, "datetime64")),
            "location": ("index", location),
            "state_region": ("index", state_region.astype(str)),
        }
    )
    da = da.swap_dims({"date": "time", "index": "location"})
    if lut_path is not None:
        lut = pd.read_csv(lut_path)
        da = da.assign_coords(population=("location", [np.nan] * da.location.size))
        for country, state, county, population in lut[
            ["Country_Region", "Province_State", "Admin2", "Population"]
        ].values:
            if county is not np.nan:
                continue
            try:
                da.population[
                    da.location == country
                    if state is np.nan
                    else f"{country} / {state}"
                ] = population
            except KeyError:
                pass
    da = da.drop(["index", "date", "state"])
    return da.to_dataset(name="deaths")


def read_jhu_usa(deaths_path):
    df = pd.read_csv(deaths_path, keep_default_na=False)
    ds = df.to_xarray()
    ds = ds.rename(
        {
            "Country_Region": "country",
            "Province_State": "state_region",
            "Admin2": "county",
            "Lat": "lat",
            "Long_": "lon",
        }
    )
    ds = ds.assign_coords({"state_region": ("index", "US / " + ds.state_region.data)})
    ds = ds.set_coords(["country", "state_region", "county", "lat", "lon"])
    ds = ds.drop(["UID", "iso2", "iso3", "code3", "FIPS", "Combined_Key"])
    if "Population" in ds:
        # confirmed dataset has no population
        ds = ds.rename({"Population": "population"})
        ds = ds.set_coords(["population"])
    da = ds.to_array("date")
    time = [
        "20%02d-%02d-%02d" % tuple(map(int, d.split("/")[2:3] + d.split("/")[:2])) for d in da.date.values
    ]
    da = da.assign_coords(
        {
            "country": ("index", da.country.astype(str).data),
            "state_region": ("index", da.state_region.astype(str).data),
            "time": ("date", np.array(time, "datetime64")),
            "location": ("index", (ds.state_region + " / " + ds.county).astype(str).data),
        }
    )
    da = da.swap_dims({"date": "time", "index": "location"})
    da = da.drop(["index", "date", "county"])
    if "population" in ds:
        ds = da.to_dataset(name="deaths").reset_coords("population")
    else:
        ds = da.to_dataset(name="confirmed")
    return ds


def istat_to_pandas(path):
    istat = pd.read_csv(path, encoding="8859", na_values="n.d.")

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

    return istat


def read_istat(path, **kwargs):
    istat = istat_to_pandas(path, **kwargs)
    tmp = istat.groupby(["time", "age_class", "NOME_COMUNE"]).agg(
        **{
            "2015": ("T_15", sum),
            "2016": ("T_16", sum),
            "2017": ("T_17", sum),
            "2018": ("T_18", sum),
            "2019": ("T_19", sum),
        }
    )
    tmp["2020"] = istat.groupby(["time", "age_class", "NOME_COMUNE"])["T_20"].sum(
        min_count=1
    )

    data = tmp.to_xarray().rename({"NOME_COMUNE": "location"})  # .fillna(0)
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


def read_dpc(path, location_prefix="Italy / "):
    df = pd.read_csv(path, parse_dates=["data"], index_col=["data"])
    df.index = df.index.normalize().rename("time")
    df["location"] = df["denominazione_regione"]
    df = df.set_index("location", append=True)
    ds = df[
        [
            "ricoverati_con_sintomi",
            "terapia_intensiva",
            "deceduti",
            "nuovi_positivi",
            "tamponi",
            "ingressi_terapia_intensiva",
        ]
    ].to_xarray()
    ds = ds.assign_coords(
        {
            "lat": ("location", df.groupby("location")["lat"].first()),
            "lon": ("location", df.groupby("location")["long"].first()),
            "country": ("location", ["Italy"] * ds.location.size),
            "location": ("location", [location_prefix + str(l) for l in ds.location.values]),
            "state_region": ("location", [str(l) for l in ds.location.values]),
            "population": (
                "location",
                [POPULATION_BY_REGION[l] for l in ds.location.values],
            ),
        }
    )
    ds = ds.rename(
        {
            "ricoverati_con_sintomi": "current_severe",
            "terapia_intensiva": "current_critical",
            "deceduti": "deaths",
            "nuovi_positivi": "daily_confirmed",
            "tamponi": "tests",
            "ingressi_terapia_intensiva": "daily_critical",
        }
    )
    return ds.fillna(0)


FIX_LABELS = {
    "Friuli-Venezia Giulia": "Friuli Venezia Giulia",
    "Provincia Autonoma Bolzano / Bozen": "P.A. Bolzano",
    "Provincia Autonoma Trento": "P.A. Trento",
    "Valle d'Aosta / Vallée d'Aoste": "Valle d'Aosta",
}


def read_vaccini(path):
    df = pd.read_csv(
        path, parse_dates=["data_somministrazione"], index_col=["data_somministrazione"]
    )
    for label, new_label in FIX_LABELS.items():
        df.loc[df.nome_area == label, "nome_area"] = new_label
    df.index = df.index.normalize().rename("time")
    df = df.set_index(["nome_area", "fornitore", "fascia_anagrafica"], append=True)
    df = df.rename(columns={"prima_dose": "first", "seconda_dose": "second", "dose_addizionale_booster": "booster"})
    ds = df[["first", "second", "booster"]].to_xarray().fillna(0)
    ds = ds.to_array("dose_type", name="doses").to_dataset()
    ds = ds.rename(
        {
            "nome_area": "location",
            "fornitore": "provider",
            "fascia_anagrafica": "age_class",
        }
    )
    ds = xr.merge([ds, ds.sum("location").expand_dims(location=["Italia"])])
    ds = ds.assign_coords(
        {
            "population": (
                "location",
                [POPULATION_BY_REGION[l] for l in ds.location.values],
            ),
            "population_70": (
                "location",
                [POPULATION_70_BY_REGION[l] for l in ds.location.values],
            ),
            "population_80": (
                "location",
                [POPULATION_80_BY_REGION[l] for l in ds.location.values],
            ),
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


def shift_and_scale(data, delay, ratio, drop_negative=False, x="time"):
    if delay is not None:
        if isinstance(delay, (int, float)):
            delay = delay * np.timedelta64(24 * 3600, "s")
        data = data.assign_coords({x: (x, data.coords[x] + delay)})
    if ratio is not None:
        data = data / ratio
    if drop_negative:
        data = data[data >= 0]
    return data
