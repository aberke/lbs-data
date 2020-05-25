"""
Utilities for evaluation.
"""

import os

from matplotlib.lines import Line2D
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


def plot_trajectory_vector_frequencies(trajectory_vector, by_frequency_order=False, show_null=False):
    [home_label, work_label], unprefixed_t = break_prefix(trajectory_vector)
    df = pd.DataFrame({PLACE_LABEL:unprefixed_t})
    labels = list(set(trajectory_vector))
    labels_to_frequency = df[PLACE_LABEL].value_counts().to_dict()

    if show_null:
        # Set null location labels to frequency=0
        labels_to_frequency[0] = 0
    elif (0 in labels_to_frequency):
        del labels_to_frequency[0]

    sorted_frequencies = sorted(labels_to_frequency.values())
    frequency_to_frequency_order = {sorted_frequencies[f]: f for f in range(len(sorted_frequencies))}

    df[PLACE_LABEL_BY_FREQUENCY] = df[PLACE_LABEL].map(labels_to_frequency)
    df[PLACE_LABEL_BY_FREQUENCY_ORDER] = df[PLACE_LABEL_BY_FREQUENCY].map(frequency_to_frequency_order)

    y = df[PLACE_LABEL_BY_FREQUENCY_ORDER] if by_frequency_order else df[PLACE_LABEL_BY_FREQUENCY]

    places_count = len(set(labels) - set([0]))
    title = 'Places visited per hour, over %s days, by frequency. %s different places visited.' % (DAYS, places_count)
    fig, ax = plt.subplots(1, figsize=(16, 4))
    colors = get_plot_colors(home_label, unprefixed_t)
    marker = 'o'
    marker_size = 35
    plt.rc('font', **{'family':'normal','weight':'regular','size': 12})
    plt.scatter(
        range(len(df)), # x
        y,
        c=colors,
        s=marker_size, # marker size
        marker=marker,
    )
    ax.set_title(title)
    legend_elements = [
        Line2D([0], [0], marker=marker, color=HOME_COLOR, label='Home', markerfacecolor=HOME_COLOR),
        Line2D([0], [0], marker=marker, color=OTHER_PLACE_COLOR, label='Other place', markerfacecolor=OTHER_PLACE_COLOR),
    ]
    ax.legend(
        handles=legend_elements, loc='upper right',
        labelspacing=1,
        markerscale=1.5,
        # bbox (x, y, width, height)
        bbox_to_anchor=(1.15, 1) #, 0, 0.1))
    )
    plt.xticks([24*i for i in range(DAYS)])
    plt.grid(b=True,which='major',axis='x',color='gray',linestyle='--')
    # ax.get_yaxis().set_ticks([]) # to turn off the numbers for frequency
    plt.grid(False,axis='y')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Hour')
    plt.show()

    return df


HOME_COLOR = 'purple'
OTHER_PLACE_COLOR  = 'coral'

def get_plot_colors(h_label, labels, h_color=HOME_COLOR, o_color=OTHER_PLACE_COLOR):
    """
    Helper function to plot_trajectory_vector_frequencies.
    Returns list of colors corrsponding to list of labels.
    """
    # cast to same type, just in case
    h_label, labels = int(h_label), [int(l) for l in labels]
    colors = [h_color if int(l) == h_label  else o_color for l in labels]
    return colors

