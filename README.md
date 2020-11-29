Implementation of the algorithm for generating the C-score of a rhythm (i.e. temporal pattern) from ["Perception of Temporal Patterns"](https://online.ucpress.edu/mp/article-abstract/2/4/411/62235/Perception-of-Temporal-Patterns?redirectedFrom=fulltext) (Povel & Essens, 1985). 

### Installation
At the moment, install with `python setup.py install`. 

### Usage 
```
>>> udikshana = np.array([0.5, 1.5, 0.5, 0.5, 1.0])
>>> transform_to_time_scale(udikshana)
array([1, 1, 0, 0, 1, 1, 1, 0])
```
Povel & Essens provide a heuristic for marking accents in a temporal pattern according to three principles:
1. Relatively isolated events (use `get_indices_of_isolated_elements(temporal_pattern)`)
2. The second tone in a cluster of two (use `get_indices_of_isolated_short_clusters(temporal_pattern)`)
3. The initial and final tones of a cluster consisting of >3 tones (use `get_indices_of_isolated_long_clusters(temporal_pattern)`)

We can mark the accents in a temporal pattern with `mark_accents`. 

We can generate all the clocks of a fragment as follows:
```
>>> example = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
>>> for this_clock in generate_all_clocks(time_scale_array = example):
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
```

We calculate the C-score of a fragment (given a clock) as follows. Note that the default weight value `W` is set to 4, following the paper. 
```
>>> fragment = np.array([0.75, 0.25, 0.25, 1.0, 0.5, 0.25])
>>> clock_choice = np.array([2, 5, 8, 11])
>>> c_score(fragment, clock_choice, W=4)
8
```