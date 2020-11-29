Implementation of the algorithm for generating the C-score of a rhythm (i.e. temporal pattern) from "Perception of Temporal Patterns" (Povel & Essens, 1985). 

### Installation
At the moment, install with `python setup.py install`. 

### Usage 
```
>>> udikshana = np.array([0.5, 1.5, 0.5, 0.5, 1.0])
>>> transform_to_time_scale(udikshana)
array([1, 1, 0, 0, 1, 1, 1, 0])
```