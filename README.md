# lbs-data


## Data Pipeline

- (before) shapefile_shapers : make necessary shapefiles
- preprocess_filtering : prune the dataset
- attach_areas : attach census statistical areas to data points with (lat, lon) coordinates

Then things like...

- home_work : to compare dataset to population estimates
- trajectory synthesis



## Data Source

### ACS data

Demographic data is sourced from the 2018 American Community Survey 5-year estimates

To download the data:
- go to https://data.census.gov/
- Select a geographic type: county subdivision, census tract, census block group, etc (need to download separately)
- Select state --> Select all within state
- Select desired tables from the ACS 5-year estimates and download CSVs
  - using: total population (ID: B01003)
  - others of  interest: race as "RACE" (ID:B02001) - contains population total estimates, median income as "MEDIAN INCOME IN THE PAST 12 MONTHS (IN <year> INFLATION-ADJUSTED DOLLARS)" (ID:S1903)
  - download and save to data/ACS/[state]_acs_5_year_census_[geographic type: block_group|tract|etc]_[year]
