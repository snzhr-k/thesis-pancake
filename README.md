# Housing and Economic Factors Dataset

This workspace contains a reproducible dataset builder for a bachelor thesis on
predicting rental or housing prices from housing-market and economic factors.

The current project scope is **regional prediction**. Each row represents one
US region in one month, not an individual apartment or house listing.

Example prediction task:

```text
Use Austin, TX housing and economic conditions in 2023-06 to predict
Austin, TX typical rent in 2023-07.
```

Current modeling objective:

```text
Compare whether economic change features improve next-month regional rent
prediction compared with a simple neural network using only current levels and
housing-history features.
```

## Outputs

- `data/processed/metro_housing_economic_panel.csv`  
  Main joined metro-month panel. Keeps rows with partial missingness, which is
  useful for inspection and custom imputation.
- `data/processed/metro_housing_economic_panel_complete_cases.csv`  
  ML-ready complete-case subset with rows containing no missing values.
- `data/processed/data_dictionary.csv`  
  Column descriptions.
- `data/processed/sources.csv`  
  Source URL used for each downloaded dataset.

Current dataset size:

```text
Main dataset:          52,577 rows, 22 columns, 734 regions
Complete-case dataset: 44,016 rows, 22 columns, 593 regions
Time period:           2015-01 to 2026-08
Frequency:             Monthly
```

## Data Sources

- Zillow Research public CSVs:
  - Zillow Observed Rent Index (ZORI)
  - Zillow Home Value Index (ZHVI)
- FRED public CSVs:
  - 30-year mortgage rate
  - unemployment rate
  - CPI
  - federal funds rate
  - building permits
  - housing starts

## Columns

| Column | Meaning |
|---|---|
| `RegionID` | Zillow's numeric identifier for the region. Useful for grouping rows belonging to the same region. |
| `SizeRank` | Zillow's size ranking for the region. Lower values represent larger/more important regions. |
| `RegionName` | Human-readable region name, for example `Austin, TX` or `United States`. |
| `RegionType` | Zillow geography type, such as metro area or country. |
| `StateName` | US state abbreviation/name where Zillow provides it. Can be missing for national rows. |
| `date` | Monthly observation date. Used for sorting, train/test splitting, and creating time features such as year or month. |
| `zori_rent` | Zillow Observed Rent Index. Represents typical regional rent in dollars. This can be used as the current rent feature or shifted forward to create a future rent target. |
| `zhvi_home_value` | Zillow Home Value Index. Represents typical regional home value in dollars. |
| `mortgage_30yr_rate` | Average 30-year fixed mortgage rate from FRED. |
| `unemployment_rate` | US unemployment rate from FRED. |
| `cpi` | Consumer Price Index from FRED. Measures the general price level. |
| `fed_funds_rate` | Effective federal funds rate from FRED. |
| `building_permits` | Number of new privately-owned housing units authorized from FRED. |
| `housing_starts` | Number of new privately-owned housing units started from FRED. |
| `cpi_yoy_pct` | Year-over-year percentage change in CPI. This is an inflation-like feature. |
| `building_permits_yoy_pct` | Year-over-year percentage change in building permits. |
| `rent_lag_1m` | Rent value from one month earlier for the same region. |
| `rent_lag_12m` | Rent value from twelve months earlier for the same region. |
| `rent_yoy_pct` | Year-over-year percentage change in rent for the same region. |
| `home_value_lag_12m` | Home value from twelve months earlier for the same region. |
| `home_value_yoy_pct` | Year-over-year percentage change in home value for the same region. |
| `price_to_annual_rent` | Home value divided by annualized rent, calculated as `zhvi_home_value / (zori_rent * 12)`. |

## Modeling Notes

Do not use all columns blindly as model inputs.

For regional rent prediction, a clean setup is:

```text
inputs at month t -> rent at month t + 1
```

For example:

```text
inputs: Austin, TX features in 2023-06
target: Austin, TX zori_rent in 2023-07
```

Recommended first target:

```text
target_rent_next_1m
```

This target should be created by shifting `zori_rent` one month forward within
each `RegionID`.

Recommended input features:

```text
SizeRank
year
month
zori_rent
zhvi_home_value
mortgage_30yr_rate
unemployment_rate
cpi
fed_funds_rate
building_permits
housing_starts
cpi_yoy_pct
building_permits_yoy_pct
rent_lag_1m
rent_lag_12m
rent_yoy_pct
home_value_lag_12m
home_value_yoy_pct
price_to_annual_rent
```

Current neural-network experiments:

```text
1. Naive baseline
   Prediction: next month's rent equals current month's rent.

2. Simple network
   Inputs: base housing and economic level features.
   Target: next-month rent change.

3. Delta-feature network
   Inputs: base features plus economic change features.
   Target: next-month rent change.
```

Economic change features are created from the existing economic columns. For
example:

```text
unemployment_rate_change_1m = current unemployment - previous month's unemployment
mortgage_30yr_rate_change_3m = current mortgage rate - mortgage rate 3 months ago
building_permits_pct_change_1m = percentage change in building permits from previous month
```

Metadata columns such as `RegionName`, `StateName`, and `date` are useful for
interpretation, plotting, and splitting, but should not be used as raw numeric
inputs.

## Rebuild

```bash
python3 scripts/build_housing_econ_dataset.py --refresh
```

Use without `--refresh` to rebuild from the already downloaded raw CSV files:

```bash
python3 scripts/build_housing_econ_dataset.py
```

## Notes

The current dataset is a US metro-level monthly panel, not an individual
property-listing dataset. It includes housing-market features such as typical
rent, typical home value, rent lag, annual rent growth, home-value growth, and
price-to-rent ratio. Individual property features such as square meters,
bedrooms, floor, balcony, or building condition require a separate listing or
transaction dataset and can be joined later by location and date.
