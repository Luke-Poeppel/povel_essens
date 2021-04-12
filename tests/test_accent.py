import pytest
import numpy as np

from povel_essens.accent import (
	get_indices_of_isolated_elements
)

def test_isolated_element():
	isolated_accents = get_indices_of_isolated_elements(np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]))
	assert list(isolated_accents) == [0, 2, 11]

# def test_udikshana_accent():
# 	udikshana = np.array([0.5, 1.5, 0.5, 0.5, 1.0])