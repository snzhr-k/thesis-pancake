from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

USER_AGENT = "Mozilla/5.0 thesis-dataset-builder/1.0"

ZILLOW_DATASETS = {
    "zori_rent": [
        "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv",
        "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv",
    ],
    "zhvi_home_value": [
        "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    ],
}

FRED_SERIES = {
    "mortgage_30yr_rate": "MORTGAGE30US",
    "unemployment_rate": "UNRATE",
    "cpi": "CPIAUCSL",
    "fed_funds_rate": "FEDFUNDS",
    "building_permits": "PERMIT",
    "housing_starts": "HOUST",
}


def download(url: str, path: Path, refresh: bool) -> None:
    if path.exists() and not refresh:
        return

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def download_first_available(urls: Iterable[str], path: Path, refresh: bool) -> str:
    last_error: Exception | None = None
    for url in urls:
        try:
            download(url, path, refresh)
            return url
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download {path.name}: {last_error}")


def zillow_wide_to_long(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_columns = [column for column in df.columns if column[:4].isdigit()]
    id_columns = [column for column in df.columns if column not in date_columns]

    long = df.melt(
        id_vars=id_columns,
        value_vars=date_columns,
        var_name="date",
        value_name=value_name,
    )
    long["date"] = pd.to_datetime(long["date"]).dt.to_period("M").dt.to_timestamp()
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")

    wanted = [
        "RegionID",
        "SizeRank",
        "RegionName",
        "RegionType",
        "StateName",
        "date",
        value_name,
    ]
    available = [column for column in wanted if column in long.columns]
    return long[available]


def fred_series(series_id: str, raw_dir: Path, refresh: bool) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    path = raw_dir / f"fred_{series_id}.csv"
    download(url, path, refresh)

    df = pd.read_csv(path)
    value_column = series_id
    df = df.rename(columns={"observation_date": "date", value_column: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date").sort_index()

    monthly = df.resample("MS").mean()
    return monthly.rename(columns={"value": series_id})


def build_dataset(refresh: bool) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    source_rows = []
    zillow_frames = []
    for value_name, urls in ZILLOW_DATASETS.items():
        path = RAW_DIR / f"{value_name}.csv"
        used_url = download_first_available(urls, path, refresh)
        source_rows.append({"dataset": value_name, "source": used_url})
        zillow_frames.append(zillow_wide_to_long(path, value_name))

    housing = zillow_frames[0]
    for frame in zillow_frames[1:]:
        join_keys = [
            column
            for column in ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName", "date"]
            if column in housing.columns and column in frame.columns
        ]
        housing = housing.merge(frame, on=join_keys, how="outer")

    econ_frames = []
    for output_name, series_id in FRED_SERIES.items():
        frame = fred_series(series_id, RAW_DIR, refresh).rename(columns={series_id: output_name})
        econ_frames.append(frame)
        source_rows.append(
            {
                "dataset": output_name,
                "source": f"https://fred.stlouisfed.org/series/{series_id}",
            }
        )

    econ = pd.concat(econ_frames, axis=1).sort_index()
    econ["cpi_yoy_pct"] = econ["cpi"].pct_change(12, fill_method=None) * 100
    econ["building_permits_yoy_pct"] = econ["building_permits"].pct_change(12, fill_method=None) * 100
    econ = econ.reset_index().rename(columns={"index": "date"})

    dataset = housing.merge(econ, on="date", how="left")
    dataset = dataset.sort_values(["RegionID", "date"]).reset_index(drop=True)

    grouped = dataset.groupby("RegionID", group_keys=False)
    dataset["rent_lag_1m"] = grouped["zori_rent"].shift(1)
    dataset["rent_lag_12m"] = grouped["zori_rent"].shift(12)
    dataset["rent_yoy_pct"] = (dataset["zori_rent"] / dataset["rent_lag_12m"] - 1) * 100
    dataset["home_value_lag_12m"] = grouped["zhvi_home_value"].shift(12)
    dataset["home_value_yoy_pct"] = (dataset["zhvi_home_value"] / dataset["home_value_lag_12m"] - 1) * 100
    dataset["price_to_annual_rent"] = dataset["zhvi_home_value"] / (dataset["zori_rent"] * 12)

    dataset = dataset.dropna(subset=["zori_rent", "zhvi_home_value"])

    output_path = PROCESSED_DIR / "metro_housing_economic_panel.csv"
    dataset.to_csv(output_path, index=False)

    complete_cases_path = PROCESSED_DIR / "metro_housing_economic_panel_complete_cases.csv"
    dataset.dropna().to_csv(complete_cases_path, index=False)

    sources_path = PROCESSED_DIR / "sources.csv"
    pd.DataFrame(source_rows).to_csv(sources_path, index=False)

    dictionary = pd.DataFrame(
        [
            ("RegionID", "Zillow metro identifier."),
            ("SizeRank", "Zillow metro size rank."),
            ("RegionName", "Metro area name."),
            ("RegionType", "Zillow geography type."),
            ("StateName", "State abbreviation/name where Zillow provides it."),
            ("date", "Monthly observation date."),
            ("zori_rent", "Zillow Observed Rent Index, typical rent in dollars."),
            ("zhvi_home_value", "Zillow Home Value Index, typical home value in dollars."),
            ("mortgage_30yr_rate", "30-year fixed mortgage average rate from FRED."),
            ("unemployment_rate", "US unemployment rate from FRED."),
            ("cpi", "Consumer Price Index for all urban consumers from FRED."),
            ("fed_funds_rate", "Effective federal funds rate from FRED."),
            ("building_permits", "New privately-owned housing units authorized from FRED."),
            ("housing_starts", "New privately-owned housing units started from FRED."),
            ("cpi_yoy_pct", "Year-over-year CPI percentage change."),
            ("building_permits_yoy_pct", "Year-over-year building permits percentage change."),
            ("rent_lag_1m", "Prior-month ZORI rent for the same metro."),
            ("rent_lag_12m", "Rent 12 months earlier for the same metro."),
            ("rent_yoy_pct", "Year-over-year rent percentage change."),
            ("home_value_lag_12m", "Home value 12 months earlier for the same metro."),
            ("home_value_yoy_pct", "Year-over-year home value percentage change."),
            ("price_to_annual_rent", "Home value divided by annualized rent."),
        ],
        columns=["column", "description"],
    )
    dictionary.to_csv(PROCESSED_DIR / "data_dictionary.csv", index=False)

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a joined Zillow + FRED housing economics dataset.")
    parser.add_argument("--refresh", action="store_true", help="Redownload raw source files.")
    args = parser.parse_args()

    dataset = build_dataset(refresh=args.refresh)
    print(f"rows={len(dataset):,}")
    print(f"metros={dataset['RegionID'].nunique():,}")
    print(f"date_min={dataset['date'].min().date()}")
    print(f"date_max={dataset['date'].max().date()}")
    print(f"output={PROCESSED_DIR / 'metro_housing_economic_panel.csv'}")
    print(f"complete_cases_output={PROCESSED_DIR / 'metro_housing_economic_panel_complete_cases.csv'}")


if __name__ == "__main__":
    main()
