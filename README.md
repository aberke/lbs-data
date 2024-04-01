# lbs-data

## Publication

An overview of this work is described in the following paper. Please cite appropriately.

Berke, A., Doorley, R., Larson, K. and Moro, E., 2022. Generating synthetic mobility data for a realistic population with RNNs to improve utility and privacy. In Proceedings of The 37th ACM/SIGAPP Symposium on Applied Computing (SAC ’22). 955-959. 
https://doi.org/10.1145/3477314.3507230

Also available at [https://github.com/aberke/lbs-data/blob/master/paper.pdf](https://github.com/aberke/lbs-data/blob/master/paper.pdf)

## Motivation

Location data is collected from the public by private firms via mobile devices. 
Can this data also be used to serve the public good while preserving privacy? 
Can we realize this goal by generating synthetic data for use instead of the real data? The synthetic data would need to balance utility and privacy.


## Overview


What:


This project uses location based services (LBS) data provided by a location intelligence company in order to train a RNN model to generate synthetic location data.
The goal is for the synthetic data to maintain the properties of the real data, at the individual and aggregate levels, in order to retain its utility. At the same time, the synthetic data should sufficiently differ from the real data at the individual level, in order to preserve user privacy.

Furthermore, the system uses home and work areas as labels and inputs in order to generate location data  for synthetic users  with the  given home and work areas.  
This addresses the issue of limited sample sizes. Population data, such as census data, can be used to create the input necessary to output a synthetic location dataset that represents the true population in size and distribution.


## Data

`/data/`

### ACS data

`data/ACS/ma_acs_5_year_census_tract_2018/`

Population data is sourced from the 2018 American Community Survey 5-year estimates.

### LBS data

`/data/mount/`

Privately stored on a remote server.


## Geography and time period

- Geography: The region of study is limited to 3 counties surrounding Boston, MA.
- Time period: The training and output data is for the first 5-day workweek of May 2018.


## Data representation

The LBS data are provided as rows.

```
device ID, latitude, longitude, timestamp, dwelltime
```

The data are transformed into "stay trajectories", which are sequences where each index of a sequence represents a 1-hour time interval. 
Each stay trajectory represents the data for one user (device ID).
The value at that index represents the location/area (census tract) where the user spent the most time during that 1-hour interval.

e.g.
```
[A,B,D,C,A,A,A,NULL,B...]
```

Where each letter represents a location. There are null values when no location data is reported in the time interval.

home and work locations are inferred for each user stay trajectory. stay trajectories are prefixed with the home and work locations. This home, work prefixes then serve as labels.

```
[home,work,A,B,D,C,A,A,A,NULL,B...]
```

Where home,work values are also elements (frequently) occuring in their associated stay trajectory (e.g. home=A).

These sequences are used to train the model and are also output by the model.


## RNN

The RNN model developed in this work is meant to be simple and replicable. It was implemented via the open source textgenrnn library. https://github.com/minimaxir/textgenrnn.

Many models (>70) are trained with a variety of hyper parameter values. The models are each trained on the same training data and then use the same input (home, work labels) to generate output synthetic data. The output is evalued via a variety of utility and privacy metrics in order to determine the best model/parameters.


## Pipeline

### Preprocessing

#### Define geography / shapefiles

`./shapefile_shaper.ipynb`

Our study uses 3 counties surrounding Boston, MA: Middlesex, Norfolk, Suffolk counties.

shapefile_shaper prunes MA shapefiles for this geography.

Output files are in `./shapefiles/ma/`

Census tracts are used as "areas"/locations in stay trajectories.

#### Data filtering

`./preprocess_filtering.ipynb`

The LBS data is sparse. Some users report just a few datapoints, while other users report many. In order to confidently infer home and work locations, and learn patterns, we only include data from devices with sufficient reporting.

`./preprocess_filtering.ipynb` filters the data accordingly. It pokes the data to try to determine what the right level of filtering is. It outputs saved files with filtered data. Namely, it saves a datafile with LBS data from devices that reported at least 3 days and 3 nights of data during the 1 workweek of the study period. This is the pruned dataset used in the following work.


#### Attach areas

`/attach_areas.ipynb`

Census areas are attached to LBS data rows. 


### Home, work inference

`./infer_home_work.ipynb`

Defines functions to infer home and work locations (census tracts ) for each device user, based on their LBS data. The home location is where the user spends most time in nighttime hours. The "work" location is where the user spends the most time in workday hours. These locations can be the same. 

This file helps determine good hours to use for nighttime hours.
Once the functions are defined, they are used to evaluate the data representativeness by comparing the inferred population statistics to ACS 2018 census data.

Saves a mapping of LBS user IDS to the inferred home,work locations.


### Stay trajectories setup

`./trajectory_synthesis/trajectory_synthesis_notebook.ipynb`

Transforms preprocessed LBS data into prefixed stay trajectories.

And outputs files for model training, data generation, and comparison.

Note: for the purposes of model training and data generation, the area tokens within stay trajectories can be arbitrary. What is important for the model’s success is the relationship between them. In order to save the stay trajectories in this repository yet keep real data private, we do the following. We map real census areas to integers, and map areas in stay trajectories to the integers representing the areas. We use the transformed stay trajectories for model training and data generation. The mapping between real census areas and their integer representations is kept private. We can then map the integers in stay trajectories back to the real areas they represent when needed (such as when evaluating trip distance metrics).


Output files:

`./data/relabeled_trajectories_1_workweek.txt`: 
D: Full training set of 22704 trajectories


`./data/relabeled_trajectories_1_workweek_prefixes_to_counts.json`: 
Maps D home,work label prefixes to counts

`./data/relabeled_trajectories_1_workweek_sample_2000.txt`: 
S: Random sample of 2000 trajectories from D.

`./data/relabeled_trajectories_1_workweek_prefixes_to_counts_sample_2000.json`:
Maps S home,work label prefixes to counts
- This is used as the input for data generation so that the output sythetic sample, S', has a home,work label pair distribution that matches S.


### Model training and data generation

`./trajectory_synthesis/textgenrnn_generator/`

Models with a variety of hyperparameter combinations were trained and then used to generate a synthetic sample.

The files `model_trainer.py` and `generator.py` are the templates for the scripts used to train and generate.


The model (hyper)parameter combinations were tracked in a spreadsheet.
`./trajectory_synthesis/textgenrnn_generator/textgenrnn_model_parameters_.csv`


## Evaluation

`./trajectory_synthesis/evaluation/evaluate_rnn.ipynb`

A variety of utility and privacy evaluation tools and metrics were developed. Models were evaluated by their synthetic data outputs (`S'`). This was done in `./trajectory_synthesis/evaluation/evaluate_rnn.ipynb`. The best model (i.e. best parameters) was determined by these evaluations. The results for this model are captured in `trajectory_synthesis/evaluation/final_eval_plots.ipynb`.

