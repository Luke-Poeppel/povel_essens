import pytest
import numpy as np

from povel_essens.cscore import transform_to_time_scale

def test_time_scale():
	transformed_1 = transform_to_time_scale(temporal_pattern=np.array([0.5, 0.5, 0.375, 0.375]))
	assert list(transformed_1) == [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]

	transformed_2 = transform_to_time_scale(np.array([1.0, 0.25, 0.25, 0.375]))
	assert list(transformed_2) == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]

	udikshana = np.array([0.5, 1.5, 0.5, 0.5, 1.0])
	transformed_3 = transform_to_time_scale(udikshana)
	assert list(transformed_3) == [1, 1, 0, 0, 1, 1, 1, 0]