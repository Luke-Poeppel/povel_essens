# -*- coding: utf-8 -*-
####################################################################################################
# File:     cscore.py
# Purpose:  Implementation of Povel and Essens' algorithm for calculating the C-score of a rhythmic
#           fragment. From "Perception of Temporal Patterns" (1985). 
# 
# Author:   Luke Poeppel
#
# Location: Kent, CT 2020 / Frankfurt, DE 2020 / NYC, 2021
####################################################################################################
"""
PE Algorithm:
(1) Transform the temporal pattern into binary time-scale notation.
(2) Mark accents as 2 according to the accent observations (see below).
(3) Generate all clocks. Longest time-scale is floor(duration/2), so there are floor(duration/2) 
possiblities. 
(4) Generate the ev dicts for each clock. 
(5) Calculate the C-score
(6) Choose the best clock.

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
Povel and Essens Accent Heuristic:
(a) Relatively isolated events
(b) Second tone in a cluster of two
(c) Initial and final tones of a cluster consisting of >3 tones
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
"""
import copy
import math
import numpy as np
import pytest

from .accent import mark_accents

def transform_to_time_scale(temporal_pattern):
	"""
	Transforms a temporal pattern into time-scale notation.

	Args:
		temporal_pattern (array-like): quarter lengths for each value in the pattern.
	Returns:
		numpy.array: input pattern converted to binary time-scale notation.

	>>> udikshana = np.array([0.5, 1.5, 0.5, 0.5, 1.0])
	>>> transform_to_time_scale(temporal_pattern=udikshana)
	array([1, 1, 0, 0, 1, 1, 1, 0])
	"""
	total_durations = []
	shortest_value = min(temporal_pattern)
	if all(map(lambda x: x % shortest_value == 0, temporal_pattern)):
		total_durations = np.array([(x / shortest_value) for x in temporal_pattern])
	else:
		i = 2
		# This upper bound is arbitrary; something between 4 and 10 should suffice.
		while i < 6: 
			if all(map(lambda x: (x % (shortest_value / i)).is_integer(), temporal_pattern)):
				total_durations = np.array([(x / (shortest_value / i)) for x in temporal_pattern])
				break
			else:
				i += 1
	
	if len(total_durations) == 0:
		raise Exception("Something is wrong with the input quarter lengths!")

	result_out = []
	for this_elem in total_durations:
		if this_elem == 1.0:
			result_out.extend([1])
		else:
			result_out.extend([1])
			result_out.extend([0] * (int(this_elem) - 1))

	return np.array(result_out)

def _extend_clock_to_fit(time_scale_array, clock_unit):
	"""
	Sometimes, the clock does not fit the temporal pattern (i.e. there are too many ticks for the 
	total duration); in this case, we extend it.

	:param time_scale_array numpy.darray: temporal pattern in time-scale notation.
	:param clock_unit int: unit by which the clock is counted.
	:return: array holding clocks, extended to fit the input time_scale_array.
	:rtype: numpy.darray
	:raise ValueError: if the clock_unit is not an integer.

	>>> example3 = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
	>>> for this_clock_pos in _extend_clock_to_fit(time_scale_array=example3, clock_unit=5):
	...     print(this_clock_pos)
	[ 2  7 12]
	[ 3  8 13]
	[ 4  9 14]
	"""
	if type(clock_unit) != int:
		raise ValueError("Parameter :clock_unit: must be an integer!")
	clocks = []
	num_clicks = math.ceil(len(time_scale_array) / clock_unit)
	for i in range(0, clock_unit):
		this_clock = [j for j in range(i, len(time_scale_array), clock_unit)]
		if len(this_clock) != num_clicks:
			diff = num_clicks - len(this_clock)
			for _ in range(diff):
				this_clock.extend([this_clock[-1] + clock_unit])
			clocks.append(this_clock)
		else:
			pass

	return np.array(clocks)

def generate_all_clocks(time_scale_array):
	"""
	Generates all possible clocks for a time-scale array. 

	:param time_scale_array numpy.darray: temporal pattern in time-scale notation.
	:return: array holding all possible clocks for a given time_scale_array.
	:rtype: numpy.darray

	>>> example3 = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
	>>> for this_clock in generate_all_clocks(time_scale_array = example3):
	...     print(this_clock)
	[ 0  1  2  3  4  5  6  7  8  9 10 11]
	[ 0  2  4  6  8 10]
	[ 1  3  5  7  9 11]
	[0 3 6 9]
	[ 1  4  7 10]
	[ 2  5  8 11]
	[0 4 8]
	[1 5 9]
	[ 2  6 10]
	[ 3  7 11]
	[ 0  5 10]
	[ 1  6 11]
	[ 2  7 12]
	[ 3  8 13]
	[ 4  9 14]
	"""
	clocks = []
	accented_array = mark_accents(time_scale_array)
	duration = len(accented_array)
	max_unit = math.floor(duration / 2) - 1
	all_units = range(1, max_unit + 1)
	for this_unit in all_units:
		num_clicks = math.ceil(len(accented_array) / this_unit)
		for i in range(0, this_unit):
			clock = np.array(range(i, 12, this_unit))
			if len(clock) != num_clicks:
				for this_clock in _extend_clock_to_fit(time_scale_array = accented_array, clock_unit = this_unit):
					clocks.append(this_clock)
			else:
				clocks.append(clock)
	
	uniques = []
	for clock in clocks:
		if not any(np.array_equal(clock, unique_arr) for unique_arr in uniques):
			uniques.append(clock)

	return uniques

def get_ev_dict(time_scale_array, clock):
	"""
	Returns an "ev" dictionary given a time-scale array and clock.

	:param time_scale_array numpy.darray: temporal pattern in time-scale notation.
	:param clock numpy.darray: clock ticks for the temporal pattern.
	:return: dictionary holding ev information (keys: +ev, 0ev, 0ev).
	:rtype: dict

	>>> povel_ex3 = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
	>>> for this_clock in generate_all_clocks(time_scale_array=povel_ex3):
	...     print(get_ev_dict(time_scale_array = povel_ex3, clock = this_clock), this_clock)
	{'+ev': 5, '0ev': 2, '-ev': 5} [ 0  1  2  3  4  5  6  7  8  9 10 11]
	{'+ev': 0, '0ev': 2, '-ev': 4} [ 0  2  4  6  8 10]
	{'+ev': 5, '0ev': 0, '-ev': 1} [ 1  3  5  7  9 11]
	{'+ev': 2, '0ev': 2, '-ev': 0} [0 3 6 9]
	{'+ev': 2, '0ev': 0, '-ev': 2} [ 1  4  7 10]
	{'+ev': 1, '0ev': 0, '-ev': 3} [ 2  5  8 11]
	{'+ev': 0, '0ev': 1, '-ev': 2} [0 4 8]
	{'+ev': 3, '0ev': 0, '-ev': 0} [1 5 9]
	{'+ev': 0, '0ev': 1, '-ev': 2} [ 2  6 10]
	{'+ev': 2, '0ev': 0, '-ev': 1} [ 3  7 11]
	{'+ev': 1, '0ev': 1, '-ev': 1} [ 0  5 10]
	{'+ev': 1, '0ev': 1, '-ev': 1} [ 1  6 11]
	{'+ev': 1, '0ev': 1, '-ev': 1} [ 2  7 12]
	{'+ev': 2, '0ev': 0, '-ev': 1} [ 3  8 13]
	{'+ev': 1, '0ev': 0, '-ev': 2} [ 4  9 14]
	"""
	accented_array = mark_accents(time_scale_array)
	ev_dict = dict()
	pos_ev_count = 0
	zero_ev_count = 0
	neg_ev_count = 0
	for this_tick in clock:
		try:
			if accented_array[this_tick] == 2:
				pos_ev_count += 1
			if accented_array[this_tick] == 1:
				zero_ev_count += 1
			if accented_array[this_tick] == 0:
				neg_ev_count += 1
		except IndexError:
			diff = this_tick - len(accented_array) + 1
			new = np.append(accented_array, accented_array[:diff])

			if new[this_tick] == 2:
				pos_ev_count += 1
			if new[this_tick] == 1:
				zero_ev_count += 1
			if new[this_tick] == 0:
				neg_ev_count += 1

	ev_dict['+ev'] = pos_ev_count
	ev_dict['0ev'] = zero_ev_count
	ev_dict['-ev'] = neg_ev_count

	return ev_dict

# C-score calculation
def cscore(temporal_pattern, clock, W=4):
	"""
	Calculates the C-score of a temporal pattern. We use the convention of the original 
	paper by setting W = 4. 

	>>> fragment = np.array([0.75, 0.25, 0.25, 1.0, 0.5, 0.25])
	>>> clock_choice = np.array([2, 5, 8, 11])
	>>> cscore(fragment, clock_choice)
	8
	"""
	time_scale_array = transform_to_time_scale(temporal_pattern)
	ev_dict = get_ev_dict(time_scale_array, clock)
	neg_ev = ev_dict['-ev']
	zero_ev = ev_dict['0ev']

	return ((W * neg_ev) + (1 * zero_ev))

def get_best_clock(temporal_pattern):
	raise NotImplementedError