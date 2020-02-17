"""
python get_minimum_edit_distances.py

Run remote as:

nohup python3 get_minimum_edit_distances.py > get_minimum_edit_distances_5.out 2> get_minimum_edit_distances_5.err < /dev/null &
nohup python3 get_minimum_edit_distances.py > get_minimum_edit_distances_8.out 2> get_minimum_edit_distances_8.err < /dev/null &
nohup python3 get_minimum_edit_distances.py > get_minimum_edit_distances_8.out 2> get_minimum_edit_distances_8.err < /dev/null &
nohup python3 get_minimum_edit_distances.py > get_minimum_edit_distances_10.out 2> get_minimum_edit_distances_10.err < /dev/null &
nohup python3 get_minimum_edit_distances.py > get_minimum_edit_distances_13.out 2> get_minimum_edit_distances_13.err < /dev/null &
nohup python3 get_minimum_edit_distances.py > get_minimum_edit_distances_15.out 2> get_minimum_edit_distances_15.err < /dev/null &
nohup python3 get_minimum_edit_distances.py > get_minimum_edit_distances_18.out 2> get_minimum_edit_distances_18.err < /dev/null &


Compare trajectories using edit distance as a metric,
where the edit distance is the Levenshtein between the two vectors.
"""

from datetime import datetime
import json
import random
import os


USE_GPU = False
if not USE_GPU:
    # Prevent the environment from seeing the available GPUs (to avoid error on matlaber cluster)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# The file reading and writing utilities:
def get_generated_trajectories_filename(sample_name):
    return '../textgenrnn_generator/output/{}.txt'.format(sample_name)

def read_trajectories_from_file(filename):
    """
    Returns a list of lists, where each list represents a trajectory written to file.
    Expects file format where each line is one trajectory, and places in the trajectory are delimited by spaces.
    """
    trajectories = []
    with open(filename, 'r') as f:
        trajectories = [[int(x) for x in line.strip().split()] for line in f]
    return trajectories

def get_min_edit_distances_filename(sample_name):
    return '../textgenrnn_generator/output/min-edit-distances-to-real-pop-from-{}.txt'.format(sample_name)

def write_min_edit_distances_to_file(min_edit_distances_list, filename):
    with open(filename, 'w') as f:
        f.write(' '.join([str(i) for i in min_edit_distances_list]))

# The trajectory vector utilities:
HOURS_PER_DAY = 24
DAYS = 5

def break_prefix(v, days=DAYS, hours=HOURS_PER_DAY):
    prefix = []
    n_prefix_labels = len(v) - days*hours
    if n_prefix_labels > 0:
        prefix = v[:n_prefix_labels]
        v = v[-(days*hours):]
    return prefix, v

def get_unprefixed_vectors(prefixed_vectors):
    unprefixed_vectors = []
    for pv in prefixed_vectors:
        _p, v = break_prefix(pv, days=DAYS, hours=HOURS_PER_DAY)
        unprefixed_vectors.append(v)
    return unprefixed_vectors


# Implementation of levenshtein distance:
def levenshtein_distance(s1, s2):
    """
    Returns (int) the levenshtein edit distance between two lists, where those lists can be arbitrary integers.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Quick demonstration that this distance metric works as desired
assert(levenshtein_distance([], []) == 0)
assert(levenshtein_distance([234, 1], [1]) == 1)

# Take reference vector
v1 = [random.randint(0, 652) for iter in range(120)]
# The edit distance from the vector to itself is 0
assert(levenshtein_distance(v1, v1) == 0)

# Substitute 1 item in the vector to create new vector -- expect levenshtein edit distance of 1
v2 = v1[:]
v2[0] = 1 if (not v1[0] is 1) else 100
assert(len(v1) == len(v2) and levenshtein_distance(v1, v2) == 1)


# Substitute 3 items in the vector to create new vector -- expect levenshtein edit distance of 3
v2 = v1[:]
v2[0] = 1 if (not v1[0] is 1) else 100
v2[1] = 1 if (not v1[0] is 1) else 100
v2[-1] = 1 if (not v1[0] is 1) else 100
assert(len(v1) == len(v2) and levenshtein_distance(v1, v2) ==3)

# Shift vector by 1 item to create new vector -- expect levenshtein edit distance of 2
v2 = v1[1:] + [v1[0]]
assert(len(v1) == len(v2) and levenshtein_distance(v1, v2) == 2)


def get_min_edit_distance(vector, comparison_vectors):
    return min([levenshtein_distance(vector, comparison_vector) for comparison_vector in comparison_vectors])

def get_min_edit_distances(vectors, comparison_vectors, truncate=None):
	"""
	Returns a list of integers representing minimum edit distances from comparison vectors.
	There is one integer in the list corresponding to each vector in the passed in vectors,
	where this integer represents the minimum edit distance this vector is from any of the
	comparison vectors.
	The optional truncate (int) parameter can be used to compare vectors up to the given truncation length.
	"""
	use_vectors = vectors[:truncate]
	use_comparison_vectors = comparison_vectors[:truncate]
	min_edit_distances_list = []
	for i, v in enumerate(use_vectors):
		d1 = datetime.now()
		min_dist = get_min_edit_distance(v, use_comparison_vectors)
		d2 = datetime.now()
		if (i % 10) == 0:
			print('%s : time to check minimum edit distance for vector: %s' % (i, d2-d1))
		min_edit_distances_list.append(min_dist)
	return min_edit_distances_list

# Sanity check:
test_vectors = [[random.randint(0, 652) for iter in range(120)] for i in range(22000)]
assert(sum(get_min_edit_distances(test_vectors[:2], test_vectors)) == 0)


LIMIT_SAMPLE_SIZE = 200
TRUNCATE_LEN = None


def trajectory_has_allowed_prefix(prefixed_trajectory, allowed_prefixes_set):
    # The prefixed trajectory is a list of integers.
    # The items in the allowed prefixes set are strings representing the first 2 items of such a list
    return ' '.join([str(t) for t in prefixed_trajectory[:2]]) in allowed_prefixes_set

def filter_to_trajectories_with_prefixes(prefixed_trajectories_list, allowed_prefixes):
    return [t for t in prefixed_trajectories_list if trajectory_has_allowed_prefix(t, allowed_prefixes)]

# Quick test:
allowed_prefixes = {'1 20', '201', '1201', '1 2011', '1 2'}
allowed_t = [1, 20, 5]
not_allowed_t  = [1, 201, 5]
assert([allowed_t] == filter_to_trajectories_with_prefixes([allowed_t, not_allowed_t], allowed_prefixes))


def get_min_edit_distances_list(prefixed_sample_vectors, allowed_prefixes, comparison_vectors,
                                is_real_sample=False, limit_sample_size=LIMIT_SAMPLE_SIZE):
    # Filter the trajectory vectors to those with allowed prefixes
    filtered_prefixed_vectors = filter_to_trajectories_with_prefixes(prefixed_sample_vectors, allowed_prefixes)
   	# limit the sample size to maximum value, and take their random sample with that limit
    random.shuffle(filtered_prefixed_vectors)
    filtered_prefixed_vectors = filtered_prefixed_vectors[:limit_sample_size]
    # get the unprefixed version of the vector
    filtered_unprefixed_sample_vectors = get_unprefixed_vectors(filtered_prefixed_vectors)
    comparison_vectors = get_unprefixed_vectors(comparison_vectors)
    if is_real_sample:
        # Each vector in the real sample occurs in the real full population but should not
        # be compared against itself.
        # Remove each vector from the comparison list, but only once (if duplicates occur, only one should be removed)
        comparison_vectors_less_real_sample = comparison_vectors[:] # Make a copy
        for v in filtered_unprefixed_sample_vectors:
            for _i, cv in enumerate(comparison_vectors):
                if v == cv:
                    # remove cv from the list less real sample; only do this once
                    comparison_vectors_less_real_sample.remove(v)
                    break
        assert(len(comparison_vectors_less_real_sample) == (len(real_trajectories) - len(filtered_unprefixed_sample_vectors)))
        comparison_vectors = comparison_vectors_less_real_sample

    return get_min_edit_distances(filtered_unprefixed_sample_vectors,
                                  comparison_vectors=comparison_vectors,
                                  truncate=TRUNCATE_LEN)


# The functions have been defined!  Now read in the data and do the work

# Get the set of (home, work) pairs in the sample populations that are unique.
# They were generated by using a {'home work': count} mapping file.  We use this.
trajectories_prefixes_to_counts_sample_2000_filename = './../data/relabeled_trajectories_1_workweek_prefixes_to_counts_sample_2000.json'
sample_prefixes_to_counts_dict = None

with open(trajectories_prefixes_to_counts_sample_2000_filename) as json_file:
    sample_prefixes_to_counts_dict = json.load(json_file)
non_unique_prefix_set = {prefix for (prefix, count) in sample_prefixes_to_counts_dict.items() if count > 1}


#  Read in the real trajectories data...
relabeled_trajectories_filename = '../data/relabeled_trajectories_1_workweek.txt'
real_trajectories = read_trajectories_from_file(relabeled_trajectories_filename)



# Get the set of sampled real trajectories and compute min distances, write to file
# real_sample_trajectories_filename = '../data/relabeled_trajectories_1_workweek_sample_2000.txt'
# real_trajectories_sample = read_trajectories_from_file(real_sample_trajectories_filename)
# print('getting the min edit distances for %s' % 'real_sample_2000')
# # Get the min edit distances lists and write them to file
# real_sample_min_edit_distances_list = get_min_edit_distances_list(real_trajectories_sample,
#                                                                   non_unique_prefix_set,
#                                                                   real_trajectories, is_real_sample=True)
# real_sample_min_edit_distances_filename = get_min_edit_distances_filename('real_sample_2000')
# print('writing min edit distances to file %s...' % real_sample_min_edit_distances_filename)
# write_min_edit_distances_to_file(real_sample_min_edit_distances_list, real_sample_min_edit_distances_filename)
# print('...wrote min edit distances to file %s' % real_sample_min_edit_distances_filename)


# Do the same for the generated samples

generated_sample_names = [
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:70-rnn_layers:3-rnn_size:128-dropout:0.1-dim_embeddings:128-temperature:0.9',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:70-rnn_layers:3-rnn_size:128-dropout:0.1-dim_embeddings:128-temperature:1.0',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:70-rnn_layers:3-rnn_size:128-dropout:0.1-dim_embeddings:128-temperature:0.8',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:60-rnn_layers:3-rnn_size:128-dropout:0.1-dim_embeddings:128-temperature:0.9',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:60-rnn_layers:3-rnn_size:128-dropout:0.1-dim_embeddings:128-temperature:1.1',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:60-rnn_layers:3-rnn_size:128-dropout:0.1-dim_embeddings:128-temperature:1.0',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:70-rnn_layers:3-rnn_size:128-dropout:0.1-dim_embeddings:128-temperature:1.1',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:60-rnn_layers:3-rnn_size:128-dropout:0.2-dim_embeddings:128-temperature:1.1',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:60-rnn_layers:3-rnn_size:128-dropout:0.2-dim_embeddings:128-temperature:0.9',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:60-rnn_layers:3-rnn_size:128-dropout:0.2-dim_embeddings:128-temperature:1.0',
	# 10
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:72-rnn_layers:2-rnn_size:256-dropout:0.3-dim_embeddings:100-temperature:0.8',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:72-rnn_layers:2-rnn_size:256-dropout:0.3-dim_embeddings:100-temperature:0.9',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:72-rnn_layers:2-rnn_size:256-dropout:0.3-dim_embeddings:100-temperature:1.0',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:72-rnn_layers:2-rnn_size:256-dropout:0.3-dim_embeddings:100-temperature:1.1',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:60-rnn_layers:3-rnn_size:128-dropout:0.1-dim_embeddings:128-temperature:0.8',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:60-rnn_layers:3-rnn_size:128-dropout:0.2-dim_embeddings:128-temperature:0.8',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:50-rnn_layers:3-rnn_size:128-dropout:0.2-dim_embeddings:128-temperature:1.0',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:50-rnn_layers:3-rnn_size:128-dropout:0.2-dim_embeddings:128-temperature:1.1',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:50-rnn_layers:2-rnn_size:256-dropout:0.3-dim_embeddings:100-temperature:0.9',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:50-rnn_layers:3-rnn_size:128-dropout:0.2-dim_embeddings:128-temperature:0.9',
	'generated-sample-trajectories-rnn_bidirectional:True-max_len:72-rnn_layers:2-rnn_size:256-dropout:0.3-dim_embeddings:100-temperature:1.2',
	# 21
]


for i, generated_sample_name in enumerate(generated_sample_names):
	if i < 18: # 18, 15, 13, 10, 8, 5:
		continue
	print('%s : getting the min edit distances for %s' % (i, generated_sample_name))
	generated_sample_filename = get_generated_trajectories_filename(generated_sample_name)
	generated_trajectories = read_trajectories_from_file(generated_sample_filename)
	print('read %s trajectories from filename %s' % (len(generated_trajectories), generated_sample_filename))
	# Get the min edit distances lists and write them to file
	generated_sample_min_edit_distances_list = get_min_edit_distances_list(generated_trajectories, non_unique_prefix_set, real_trajectories)
	generated_sample_min_edit_distances_filename = get_min_edit_distances_filename(generated_sample_name)
	print('writing min edit distances to file %s...' % generated_sample_min_edit_distances_filename)
	write_min_edit_distances_to_file(generated_sample_min_edit_distances_list, generated_sample_min_edit_distances_filename)
	print('...wrote min edit distances to file %s' % generated_sample_min_edit_distances_filename)

print('...and done')

