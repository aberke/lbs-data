"""
Utilities for evaluation.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



#########################################
# To Understand and evaluate trajectories with visual tools.
#########################################


DAYS = 5
HOURS_PER_DAY = 24

PLACE_LABEL = 'place'
PLACE_LABEL_BY_FREQUENCY = 'place by frequency'
PLACE_LABEL_BY_FREQUENCY_ORDER = 'place by frequency order'

def break_prefix(v, days=DAYS, hours=HOURS_PER_DAY):
    prefix = []
    n_prefix_labels = len(v) - days*hours
    if n_prefix_labels > 0:
        prefix = v[:n_prefix_labels]
        v = v[-(days*hours):]
    return prefix, v


def print_dwell_vector_by_days(v):
    print('----- vector ----')
    prefix, v = break_prefix(v)
    if len(prefix) > 0:
        print('prefix labels: ', prefix)
    day_chunks = [v[i:i+HOURS_PER_DAY] for i in range(0, len(v), HOURS_PER_DAY)]
    for day_chunk in day_chunks:
        print(day_chunk)


def plot_trajectory_vector_frequencies(trajectory_vector, by_frequency_order=False):
    """Plots the places the the person with the given trajectory has gone, by the frequency which
        the person goes to them.
        Each index is an hour. The value at that index represents where the user spent the most
        time during that hour.
        
        args: (bool) when by_frequency_order is true, the frequency shown is relative
            to the other places the user spent time.
        
        The null values that represent when the user location was
            not recorded are set to frequency=0 in the plot.
            
        Returns dataframe representing trajectory frequencies.
    """
    _prefix, trajectory_vector = break_prefix(trajectory_vector)
    df = pd.DataFrame({PLACE_LABEL:trajectory_vector})
    labels = list(set(trajectory_vector))
    labels_to_frequency = {label: df[df[PLACE_LABEL] == label].shape[0] for label in labels}
    # Set null location labels to frequency=0
    labels_to_frequency[0] = 0

    sorted_frequencies = sorted(labels_to_frequency.values())
    frequency_to_frequency_order = {sorted_frequencies[f]: f for f in range(len(sorted_frequencies))}
    df[PLACE_LABEL_BY_FREQUENCY] = df[PLACE_LABEL].map(labels_to_frequency)
    df[PLACE_LABEL_BY_FREQUENCY_ORDER] = df[PLACE_LABEL_BY_FREQUENCY].map(frequency_to_frequency_order)
    
    y = PLACE_LABEL_BY_FREQUENCY_ORDER if by_frequency_order else PLACE_LABEL_BY_FREQUENCY
    title = 'Places visited per hour, over %s days, by frequency.  Total places visited: %s' % (DAYS, len(labels))
    ax = df.plot(
        y=y,
        style='o',
        figsize=(18, 4),
        title=title,
    )
    plt.xticks([24*i for i in range(DAYS)])
    plt.show()
    return df



