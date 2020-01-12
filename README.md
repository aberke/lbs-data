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

Demographic data is sourced from the ACS 5-year estimates.
- Year: working with 2017, TODO: work with 2018 (2018 data to be released in december.  Refer to https://www.census.gov/programs-surveys/acs/news/data-releases/2018/release-schedule.html)

To download the data:
- go to https://factfinder.census.gov/faces/nav/jsf/pages/download_center.xhtml
- deal with the 2000's era graphics
- Choose "I know the dataset or table(s) that I want to download."
- Select "American Community Survey"
- Select a geographic type: county subdivision, census tract, census block group, etc (need to download separately)
- Select state --> Select all within state --> Add to selection --> Next
- Select tables (can use search) and download CSVs
  - race as "RACE" (ID:B02001) -- contains population total estimates
  - median income as "MEDIAN INCOME IN THE PAST 12 MONTHS (IN <year> INFLATION-ADJUSTED DOLLARS)" (ID:S1903)
  - download and save to data/ACS/[state]_acs_5_year_census_[geographic type: block_group|tract|etc]_[year]
