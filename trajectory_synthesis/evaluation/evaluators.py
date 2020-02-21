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
#
# To do this, we use a shortcut that exploits how well our RNN models fit 
# the training data: 
# We assume that any bigram that occurs in a real trajectory is sensible.  
# Clearly if all the bigrams in the generated data have been seen in the real data,
# then all of the bigrams are sensible.
# We also note that if the distance from area A to area B is sensible, 
# then so must be the distance from area B to area A.  
# So if bigram (A, B) is sensible, then so must be its reverse (B, A).
# We collect the set of all of the bigrams "seen" in the real trajectories. 
# We then also collect all of the "unseen" bigrams from the generated trajectories, where these "unseen" bigrams and their reverses are not in the "seen" set.
# TODO: handle reverses

def get_bigrams_for_trajectory_vector(tv, with_skip=False):
    """
    Returns set of tuples
    {(a, b) for each sequential a, b or b, a found in tv, s.t. a < b}
    i.e. (a, b) and (b, a) are considered the same.
    Elements in the tuples are sorted to ensure duplicates are avoided.
    
    If with_skip is True, the set of tuples also include those where one element is skipped.
    i.e. A sequence of "A B C D" will then produce bigrams (A, B), (A, C), (B, C), (B, D), (C, D).
    """
    t_bigrams = set()
    for i in range(1, len(tv)):
        tup = tuple(sorted((tv[i-1], tv[i])))
        t_bigrams.add(tup)
        if (with_skip and (i >= 2)):
            tup = tuple(sorted((tv[i-2], tv[i])))
            t_bigrams.add(tup)
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
			bigram = tuple(sorted((trajectory[t-1], trajectory[t])))
			if bigram in bigram_set:
				continue
			# otherwise this is an 'unseen bigram'
			if bigram not in unseen_bigram_to_count:
				unseen_bigram_to_count[bigram] = 0
			unseen_bigram_to_count[bigram] += 1
	return unseen_bigram_to_count


class TestBigramsForTrajectoryVectors(unittest.TestCase):

	def test_get_unseen_bigrams(self):
		bigrams = {(1,2), (2, 30)}
		ts = [[1, 30, 2, 1, 30], [2, 1], [4, 1]]
		unseen_bigrams = get_unseen_bigrams(bigrams, ts)
		assert({(1,30):2, (1,4):1} == unseen_bigrams)

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

	def test_reverse_bigrams(self):
		"""
		Bigrams (a, b) and (b, a) are the same.
		Only (a, b) should be in the resulting set, s.t. a < b
		"""
		t = [4, 3, 4, 2]
		bgs = get_bigrams_for_trajectory_vectors([t])
		assert(bgs == {(3, 4), (2, 4)})



if __name__ == '__main__':
	unittest.main()
