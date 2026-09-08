# K-Means Clustering From Scratch

A simple implementation of the **K-Means clustering algorithm from scratch using Python and NumPy**.

## What is implemented?

This project implements:

- Random centroid initialization
- Squared Euclidean distance
- Cluster assignment
- Centroid updating
- Convergence checking
- Prediction for new data points
- Empty-cluster handling

## Example

```python
import numpy as np
from k_means import K_Means

np.random.seed(42)

data = np.array([
    [1, 1],
    [1, 2],
    [2, 1],

    [10, 10],
    [10, 11],
    [11, 10],

    [20, 20],
    [20, 21],
    [21, 20],
])

model = K_Means(k=3)

model.fit(data)

print("Centroids:")
print(model.centroids)

print("Predictions:")
print(model.predict(data))
```

Example output:

```text
Centroids:
[[20.33333333 20.33333333]
 [ 1.33333333  1.33333333]
 [10.33333333 10.33333333]]

Predictions:
[1, 1, 1, 2, 2, 2, 0, 0, 0]
```

## Algorithm

The algorithm works through four main steps:

```text
Initialize centroids
        ↓
Assign points
        ↓
Update centroids
        ↓
Check convergence
        ↓
      Repeat
```

A detailed explanation is available in:

[`algorithm.md`](algorithm.md)

## Requirements

```text
Python
NumPy
```

Install NumPy with:

```bash
pip install numpy
```

## Project Structure

```text
k-means-from-scratch/
│
├── README.md
├── algorithm.md
└── k_means.py
```

## Learning Goal

This project was created to understand how K-Means works internally by implementing the algorithm without using a machine-learning library such as scikit-learn.
