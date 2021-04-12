# -*- coding: utf-8 -*-
####################################################################################################
# File:     accent.py
# Purpose:  Accent helper functions. 
# 
# Author:   Luke Poeppel
#
# Location: NYC, 2021
####################################################################################################
import numpy as np

def get_indices_of_isolated_elements(time_scale_array):
	"""
	Returns the indices of isolated elements in an array. 

	:param time_scale_array numpy.darray: temporal pattern in time-scale notation.
	:return: array holding the indices of isolated elements in the time-scale input.
	:rtype: numpy.darray

	>>> paper_example = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
	>>> get_indices_of_isolated_elements(time_scale_array=paper_example)
	array([3, 9])
	"""
	if len(time_scale_array) == 1:
		return np.array([0])

	indices = []
	i = 0 
	while i <= len(time_scale_array) - 1:
		if time_scale_array[i] == 1:
			if i == 0:
				if time_scale_array[i + 1] == 0:
					indices.append(i)
			elif i == len(time_scale_array) - 1:
				if time_scale_array[-1] == 1 and time_scale_array[i - 1] == 0:
					indices.append(i)
			else:
				if time_scale_array[i - 1] == 0 and time_scale_array[i + 1] == 0:
					indices.append(i)
		
		i += 1

	return np.array(indices)

def get_indices_of_isolated_short_clusters(time_scale_array):
	"""
	Returns the second index of an isolated cluster of two values in an array. 

	:param time_scale_array numpy.darray: temporal pattern in time-scale notation.
	:return: array holding the second indices of the isolated 2-clusters in the time-scale input.
	:rtype: numpy.darray

	>>> povel_example = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
	>>> get_indices_of_isolated_short_clusters(time_scale_array=povel_example)
	array([1])
	>>> get_indices_of_isolated_short_clusters(np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]))
	array([1, 5])
	"""
	i = 0
	indices = []
	while i <= len(time_scale_array) - 2:
		this_window = time_scale_array[i:i+2]
		if this_window[0] == 1 and this_window[1] == 1:
			if i == 0 and time_scale_array[2] == 0:
				indices.append(i+1)
			else:
				try:
					if time_scale_array[i-1] == 0 and time_scale_array[i+2] == 0:
						indices.append(i+1)
				except IndexError:
					pass
		i += 1

	return np.array(indices)

def get_indices_of_isolated_long_clusters(time_scale_array):
	"""
	Returns an array of arrays, each of which holds the first & last indices of isolated clusters of length greater than >=3.

	:param time_scale_array numpy.darray: temporal pattern in time-scale notation.
	:return: array of arrays, each of which holds the first & last indices of isolated clusters of length greater than >=3.
	:rtype: numpy.darray

	>>> new_example = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
	>>> for this_cluster in get_indices_of_isolated_long_clusters(time_scale_array=new_example):
	...     print(this_cluster)
	[16 20]
	[0 3]
	[11 13]
	"""
	def _check_tuple_in_tuple_range(tup1, tup2):
		"""Checks if tuple 1 is in the range of tuple 2."""
		return (tup1[0] >= tup2[0] and tup1[0] <= tup2[1] and tup1[1] <= tup2[1])

	i = 3
	all_possibilities = []
	while i <= len(time_scale_array) - 1:
		j = 0
		while j <= len(time_scale_array) - i:
			this_window = time_scale_array[j:j+i]
			if len(set(this_window)) == 1 and this_window[0] == 1:
				all_possibilities.append([this_window, j, j + len(this_window) - 1])
			j += 1
		i += 1

	sorted_tuples = sorted(all_possibilities, key=lambda x: len(x[0]), reverse=True)
	out_sorted_tuples = copy.copy(sorted_tuples)
	k = 0
	while k <= len(out_sorted_tuples) - 1:
		curr = out_sorted_tuples[k]
		curr_indices = tuple(curr[1:])
		l = k+1
		while l <= len(out_sorted_tuples) - 1:
			if _check_tuple_in_tuple_range(tup1=tuple(out_sorted_tuples[l][1:]), tup2=curr_indices):
				del out_sorted_tuples[l]
			else:
				l += 1
		k += 1
	
	return np.array([x[1:] for x in out_sorted_tuples])

def mark_accents(time_scale_array):
	"""
	Marks accents in time-scale array based on the three rules. 

	:param time_scale_array numpy.darray: temporal pattern in time-scale notation.
	:return: a time-scale-like array with accents marked according to the rules put forth in the paper.
	:rtype: numpy.darray

	>>> povel_example2 = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0])
	>>> mark_accents(time_scale_array=povel_example2)
	array([2, 1, 1, 2, 0, 1, 2, 0, 0, 2, 0, 1, 2, 0, 0, 0, 2, 1, 1, 2, 0])

	>>> mark_accents(np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]))
	array([2, 1, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 1, 2, 0, 0])
	"""
	isolated_elements = get_indices_of_isolated_elements(time_scale_array)
	isolated_short_cluster = get_indices_of_isolated_short_clusters(time_scale_array)
	isolated_long_cluster = get_indices_of_isolated_long_clusters(time_scale_array)

	flatten = lambda l: [item for sublist in l for item in sublist]
	isolated_long_cluster = flatten(isolated_long_cluster)

	new = copy.copy(time_scale_array)
	for this_accent_index in isolated_elements:
		new[this_accent_index] = 2
	for this_accent_index in isolated_short_cluster:
		new[this_accent_index] = 2
	for this_accent_index in isolated_long_cluster:
		new[this_accent_index] = 2

	return new
