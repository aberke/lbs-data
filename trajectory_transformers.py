"""Trajectory Transformers (and makers, etc).


Run the unittests:
$ python trajectory_transformers.py
"""
from datetime import datetime, timedelta
from dateutil import tz, parser


TIMESTAMP = 'timestamp'
DWELLTIME = 'DwellTime'
TRACT = 'TRACT'


STARTDAY = '2018-05-07'

NIGHTTIME_ENDHOUR = 9  # non inclusive
NIGHTTIME_STARTHOUR = 20  # inclusive


def dwells_to_trajectory_vector(dwells, time_period_days=5, startday=STARTDAY):
	"""
	Transforms input dwells data into trajectory data vector.

	input data is an array of dwells for one user.
	These dwells should not overlap.
		[{timestamp: (str) 'TSTMP', DwellTime: (float) minutes, TRACT: (string) tract}]
		e.g.
		[{timestamp: '2018-05-04T06:36:08-04:00', DwellTime: 6.54, TRACT: 'A'}]

	Time unit: minutes

	Returns a list of strings where the indices of the list are time intervals for each
		time interval in the total time period.
		The string values at each time interval are the areas where the user spent the
		most time during that time interval.

	"""
	# There is an index in the output vector for each hour of the day.
	total_hours = 24*time_period_days
	# As an intermediary step, a trajectory vector as 
	# [(area, dwelltime) for each hour in total_hours] is generated
	# Initialize this intermediary trajectory vector to null areas with zeros as dwelltimes
	dwells_vector = [(None, 0) for hour in range(total_hours)]
	# sort the input dwells so that the earliest is first
	dwells = sorted(dwells, key=lambda k: k[TIMESTAMP])
	for dwell in dwells:
		tstmp, dwelltime, tract = dwell[TIMESTAMP], dwell[DWELLTIME], dwell[TRACT]
		# validate tstmp
		tstmp_dt = parser.parse(str(tstmp))
		startday_dt = parser.parse(startday).replace(tzinfo=tstmp_dt.tzinfo)
		# get hour index for timestamp
		day = (tstmp_dt - startday_dt).days
		hr = 24*day + tstmp_dt.hour
		d = False
		while dwelltime > 0:
			if hr >= len(dwells_vector):
				break
			hr_dwelltime = min(dwelltime, 60)
			if (hr >= 0) and (hr_dwelltime > dwells_vector[hr][1]):
				dwells_vector[hr] = (tract, hr_dwelltime)
			subtract_mins = (60 - tstmp_dt.minute) if not d else 60
			dwelltime -= subtract_mins
			hr += 1
			d = True
	return [area for (area, time) in dwells_vector]


# Trajectory vectors have sensitive information in that the labels at each index are meaningful places.
# We relabel vectors by replacing meaningful labels with ints.
# There is a dictionary (not committed publicly) that maps labels to ints.
# Trajectory ectors can then be recreated by mapping from int labels back to original labels.

def to_int_vocab(vectors):
	"""Relabels set of vectors with int vocab.
	Returns relabeled vectors and {label -> int} mapping.
	"""
	# make the word to int dict
	# None must be zero
	label_to_int_dict = {None: 0}
	for vector in vectors:
		for label in vector:
			if label not in label_to_int_dict:
				label_to_int_dict[label] = len(label_to_int_dict)
	# Translate the words in the vectors to ints
	int_vectors = []
	for vector in vectors:
		int_vectors.append([label_to_int_dict[label] for label in vector])
	return int_vectors, label_to_int_dict


def vectors_from_int_vocab(vectors, label_to_int_dict):
	"""Restores original labels to vectors by using {label -> int} mapping.
	Returns set of relabeled vectors.
	"""
	int_to_label = list(0 for i in range(len(label_to_int_dict)))
	for label, int_ in label_to_int_dict.items():
		int_to_label[int_] = label

	relabeled_vectors = []
	for vector in vectors:
		relabeled_vector = [int_to_label[i] for i in vector]
		relabeled_vectors.append(relabeled_vector)
	return relabeled_vectors


# The following functions are to infer the most likely home / work
# places from the trajectory vectors.
# They assume the indices of the vectors represent hours,
# and that the 0th index is the 0th hour (12am).
def get_trajectory_home_label(trajectory_vector):
	"""
	Infers and returns the most likely home location as the label
	that occurs most frequently during the nighttime hours.

	Args: trajectory_vector as a list of labels for each hour.


	Returns the most likely home location as the 
		label that occurs most frequently during nighttime hours.
	"""
	nighttime_hours = get_nighttime_hours(len(trajectory_vector))
	(place, _dwell_hours) = get_max_dwell_tract_for_hours(trajectory_vector, nighttime_hours)
	return place


def get_trajectory_work_label(trajectory_vector):
	"""
	Infers and returns the most likely work location as the label
	that occurs most frequently during the workday hours.

	Args: trajectory_vector as a list of place labels for each hour.

	Returns None if there were not enough place labels occuring during workday hours.
		Otherwise returns label occuring most often during workday hours.
	"""
	work_label_threshold_hours = round(len(trajectory_vector)/24)
	workday_hours = get_workday_hours(len(trajectory_vector))
	(place, dwell_hours) = get_max_dwell_tract_for_hours(trajectory_vector, workday_hours)
	if dwell_hours >= work_label_threshold_hours:
		return place
	return 0
	


def get_max_dwell_tract_for_hours(trajectory_vector, hour_indicies):
	"""
	Returns (place, hours) for the place that occurs most often
		in trajectory vector for given hours indices.

	The hours_indicies is a list of the indicies of interest within the trajectory vector.
	"""
	assert(len(hour_indicies) < len(trajectory_vector))
	# build dictionary of aggregate dwell hours for each place
	dwells_dict = dict()  # place -> hours
	for hr in hour_indicies:
		place = trajectory_vector[hr]
		if (not place):
			continue
		if place not in dwells_dict:
			dwells_dict[place] = 0
		dwells_dict[place] += 1
	# get place, hours with max hours
	max_hours = 0
	max_hours_place = 0
	for place, hours in dwells_dict.items():
		if hours > max_hours:
			max_hours = hours
			max_hours_place = place
	return (max_hours_place, max_hours)


def get_nighttime_hours(total_hours):
	night_hours = []
	# assumes index 0 is hour 0
	hour = 0
	daytime_length = NIGHTTIME_STARTHOUR - NIGHTTIME_ENDHOUR
	while hour < total_hours:
		night_hours += [h for h in range(hour, min((hour + NIGHTTIME_ENDHOUR), total_hours))]
		hour += NIGHTTIME_ENDHOUR
		# skip the day time
		hour += daytime_length
		night_hours += [h for h in range(hour, min((hour + (24 - NIGHTTIME_STARTHOUR)), total_hours))]
		hour += (24 - NIGHTTIME_STARTHOUR) 
	return night_hours


def get_workday_hours(total_hours,  startday=STARTDAY):
	"""
	Returns workday hours as (all possible hours) -  (nighttime hours) - (weekend hours)
	"""
	weekend_hours = []
	hr = 0
	day_of_week = parser.parse(startday).weekday()
	while hr < total_hours:
		while day_of_week < 5: # skip the weekdays
			day_of_week = ((day_of_week + 1) % 7)
			hr += 24
		if hr >= total_hours:
			break
		weekend_hours += [hr]
		hr += 1
		if (hr % 24) == 0:
			day_of_week = ((day_of_week + 1) % 7)
	all_hours = [h for h in range(total_hours)]
	nighttime_hours = get_nighttime_hours(total_hours)
	workday_hours_set = set(all_hours).difference(nighttime_hours).difference(weekend_hours)
	workday_hours = sorted(list(workday_hours_set))
	return workday_hours



# *************************
# Tests
# *************************


import unittest


class TestTrajectoryLabelers(unittest.TestCase):
	
	def test_home_label(self):
		trajectory_vector = [None]*4 + ['A'] + [None]*5 + ['B']*3
		self.assertEqual('A', get_trajectory_home_label(trajectory_vector))
		
	def test_home_label_nonezero(self):
		# From a real bug!
		trajectory_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 0, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 0, 0, 0, 0, 69, 69, 69, 69, 69, 0, 0, 0, 0, 70, 71, 72, 68, 68, 0, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73, 73, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 0, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 69, 69, 69, 68, 68, 0, 0, 0, 0, 0, 0, 0, 0, 69, 69, 69, 69, 69, 69, 74, 69, 69, 69, 69, 69, 69, 69, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 0, 0, 69, 69, 69, 69, 69, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 75, 68, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 76, 0, 0, 0, 0, 0, 68, 68, 68, 68, 68, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 77, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 0, 68, 68, 68, 72, 68, 68, 68, 0, 0, 0, 0, 68, 0, 0, 0, 0, 0]
		self.assertEqual(68, get_trajectory_home_label(trajectory_vector))

	def test_work_label(self):
		sparse_trajectory_vector = ['A']*5 + [None]*5 + ['B']*1 + [None]*48
		self.assertEqual(0, get_trajectory_work_label(sparse_trajectory_vector))
		dense_trajectory_vector = [None]*4 + ['A'] + [None]*5 + ['B']*3
		self.assertEqual('B', get_trajectory_work_label(dense_trajectory_vector))


class TestHourGetters(unittest.TestCase):

	daytime_length = NIGHTTIME_STARTHOUR - NIGHTTIME_ENDHOUR

	def test_get_workday_hours(self):
		# Test that it can avoid weekends
		startday = '2018-05-04'  # A Friday
		workday_hours = get_workday_hours(72,  startday=startday)
		self.assertEqual(len(workday_hours), self.daytime_length)
		startday = '2018-05-01'  # A Tuesday
		workday_hours = get_workday_hours(24 + 10,  startday=startday)
		self.assertEqual(len(workday_hours), (self.daytime_length + 10 - NIGHTTIME_ENDHOUR))


class TestDwellsToTrajectoryVector(unittest.TestCase):

	def test_out_of_range_hour(self):
		"""Test that when timestamp & dwelltime out of range,
			not included in trajectory vector.
		"""
		dwells = [{TIMESTAMP: '2018-04-30T20:03:31-04:00', DWELLTIME: 60, TRACT: 'A'}]
		time_period_days = 1
		startday='2018-05-01'
		output = dwells_to_trajectory_vector(dwells, time_period_days=time_period_days, startday=startday)
		self.assertEqual(output, [None]*24)

	def test_start_before_startday(self):
		"""Test """
		dwells = [{TIMESTAMP: '2018-04-30T23:03:31-04:00', DWELLTIME: 60, TRACT: 'A'}]
		time_period_days = 1
		startday='2018-05-01'
		output = dwells_to_trajectory_vector(dwells, time_period_days=time_period_days, startday=startday)
		self.assertEqual(output, ['A'] + [None]*23)

	def test_handle_final_hour(self):
		dwells = [{TIMESTAMP: '2018-05-01T20:03:31-04:00', DWELLTIME: 774.27, TRACT: 'A'}]
		time_period_days = 1
		startday='2018-05-01'
		output = dwells_to_trajectory_vector(dwells, time_period_days=time_period_days, startday=startday)
		self.assertEqual(output, [None]*20 + ['A']*4)

	def test_empty(self):
		"""Test the trivial case of no dwells"""
		dwells = []
		time_period_days = 14
		startday='2018-05-01'
		total_intervals = time_period_days*24
		expected_output = [None for i in range(total_intervals)]
		output = dwells_to_trajectory_vector(dwells, time_period_days=time_period_days, startday=startday)
		self.assertEqual(output, expected_output)

	def test_zero_hour(self):
		time_period_days = 1
		startday='2018-05-01'
		dwells = [{TIMESTAMP: '2018-05-01T00:36:08-04:00', DWELLTIME: 6.54, TRACT: 'A'}]
		expected_output = ['A'] + [None]*23
		self.assertEqual(expected_output, dwells_to_trajectory_vector(dwells, 
			time_period_days=time_period_days, startday=startday))

	def test_sparse(self):
		time_period_days = 3
		startday='2018-05-01'
		dwells = [
			{TIMESTAMP:'2018-05-02T06:36:08-04:00', DWELLTIME: 5.0, TRACT: 'A'},
			{TIMESTAMP:'2018-05-02T07:06:08-04:00', DWELLTIME: 35.0, TRACT: 'A'},
			{TIMESTAMP:'2018-05-03T07:36:08-04:00', DWELLTIME: 65.0, TRACT: 'B'},
		]
		expected_output = [None]*(24+6) + ['A']*2 + [None]*23 + ['B']*2 + [None]*15
		self.assertEqual(expected_output, dwells_to_trajectory_vector(dwells, 
			time_period_days=time_period_days, startday=startday))

		def test_dense(self):
			time_period_days = 1
			startday='2018-05-01'
			dwells = [
				{TIMESTAMP:'2018-05-01T01:36:08-04:00', DWELLTIME: 5.0, TRACT: 'A'},
				{TIMESTAMP:'2018-05-02T01:46:08-04:00', DWELLTIME: 45.0, TRACT: 'B'},
				{TIMESTAMP:'2018-05-03T02:36:08-04:00', DWELLTIME: 5.5, TRACT: 'C'},
				{TIMESTAMP:'2018-05-03T02:56:08-04:00', DWELLTIME: 15.5, TRACT: 'D'},
				{TIMESTAMP:'2018-05-03T08:46:08-04:00', DWELLTIME: 15.5, TRACT: 'D'},
			]
			expected_output = [None] + ['B']*2 + [None]*5 + ['D'] + [None]*16
			self.assertEqual(expected_output, dwells_to_trajectory_vector(dwells, 
				time_period_days=time_period_days, startday=startday))



class TestRelabelingVocab(unittest.TestCase):

	def test_empty(self):
		int_vectors, label_to_int_dict = to_int_vocab([])
		self.assertEqual(int_vectors, [])
		self.assertEqual(label_to_int_dict, {None: 0})

	def test_nonempty(self):
		vectors = [
				['a','b','c','a'],
				['d','a','a','f'],
				[None,'a','a','a']
			]
		int_vectors, label_to_int_dict = to_int_vocab(vectors)
		self.assertEqual(int_vectors,[
				[1,2,3,1],
				[4,1,1,5],
				[0,1,1,1]
			])
		# Test mapping it back
		relabeled_vectors = vectors_from_int_vocab(int_vectors, label_to_int_dict)
		self.assertEqual(vectors, relabeled_vectors)


if __name__ == '__main__':
	unittest.main()
