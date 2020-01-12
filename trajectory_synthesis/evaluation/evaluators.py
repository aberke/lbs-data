"""
evaluators.py

Run the unittests:
$ python evaluators.py
"""

import unittest


# check that fake users do not travel impossibly far:
# We check that between two consecutive time intervals,
# synthetic users don’t travel impossibly far; that is, they don’t suddenly go 
# from one census area to another that is too far away to have been traveled within 
# that time interval. We can do this by looking at the geographic locations of the 
# centroids or the boundaries of census areas. We consider two consecutive stay locations 
# in a stay vector sequence as a bigram, and must verify that each bigram is sensible. 
# It is possible for sensible bigrams in the generated data to occur that have not been seen in 
# the training data. However, instead of verifying that all such bigrams are sensible, 
# we have a shortcut due to how well our RNN models are fit to our training data. 
# We are able to collect the set of bigrams in the real sequences, and the set of bigrams 
# in the synthetic sequences, and verify that only bigrams from the real sequences occur in 
# the synthetic sequences. If this is true, all the bigrams in our generated data must be sensible.

def get_bigrams_for_trajectory_vector(tv, with_skip=False):
    """
    Returns set of tuples {(a, b) for each sequential a, b found in tv}
    
    If with_skip is True, the set of tuples also include those where one element is skipped.
    i.e. A sequence of "A B C D" will then produce bigrams (A, B), (A, C), (B, C), (B, D), (C, D).
    """
    t_bigrams = set()
    for i in range(1, len(tv)):
        t_bigrams.add((tv[i-1], tv[i]))
        if (with_skip and (i >= 2)):
            t_bigrams.add((tv[i-2], tv[i]))
    return t_bigrams


def get_bigrams_for_trajectory_vectors(tvs, with_skip=False):
    """
    Returns set of tuples: {(a, b) for each sequential a, b found in any of the vectors tv, in list of vectors tvs}
    
    If with_skip is True, the set of tuples also include those where one element is skipped.
    i.e. A sequence of "A B C D" will then produce bigrams (A, B), (A, C), (B, C), (B, D), (C, D).
    """
    bigrams = set()
    for tv in tvs:
        tv_bigrams = get_bigrams_for_trajectory_vector(tv, with_skip=with_skip)
        bigrams.update(tv_bigrams)
    return bigrams


def get_unseen_bigrams(bigram_set, trajectories):
	"""
	Finds bigrams in the trajectories that are not in the bigram_set.
	Returns a dictionary mapping these bigrams to the number of times they occured.
		i.e. {unseen bigram --> count} where each unseen bigram is a tuple.
	"""
	unseen_bigram_to_count = {}
	for trajectory in trajectories:
		for t in range(1, len(trajectory)):
			bigram = (trajectory[t-1], trajectory[t])
			if bigram in bigram_set:
				continue
			# otherwise this is an 'unseen bigram'
			if bigram not in unseen_bigram_to_count:
				unseen_bigram_to_count[bigram] = 0
			unseen_bigram_to_count[bigram] += 1
	return unseen_bigram_to_count


class TestBigramsForTrajectoryVectors(unittest.TestCase):

	def test_empty(self):
		empty_bgs0 = get_bigrams_for_trajectory_vectors([[]])
		assert(len(get_unseen_bigrams(empty_bgs0, [[]])) == 0)
		empty1 = [[i for i in range(10)]]
		empty_bgs1 = get_bigrams_for_trajectory_vectors(empty1)
		assert(len(get_unseen_bigrams(empty_bgs1, [[]])) == 0)
		assert(len(get_unseen_bigrams(empty_bgs0, empty1)) > 0)

	def test_nonempty(self):
		t2 = [[i for i in range(2)]]
		t3 = [[i for i in range(3)]]
		bgs2 = get_bigrams_for_trajectory_vectors(t2)
		bgs3 = get_bigrams_for_trajectory_vectors(t3)
		assert(len(get_unseen_bigrams(bgs2, t3)) > 0)
		assert(len(get_unseen_bigrams(bgs3, t2)) == 0)
		t2_5 = [[i for i in range(x, x+2)] for x in range(5)]
		bgs2_5 = get_bigrams_for_trajectory_vectors(t2_5)
		assert(len(get_unseen_bigrams(bgs2, t2_5)) > 0)
		assert(len(get_unseen_bigrams(bgs2_5, bgs2)) == 0)



if __name__ == '__main__':
	unittest.main()
