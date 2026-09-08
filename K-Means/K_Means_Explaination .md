# K-Means Clustering From Scratch

## What is K-Means?

K-Means is an **unsupervised machine learning algorithm** used for clustering.

## Ok What is Clustering ? :D

You can think of clustering as **Grouping** you have some data and you want to
put each one beside the relevante one.

## Example : Market Segmentation

Market segmentation means dividing a large group of customers into smaller groups (segments) based on similarities between them.

Imagine a supermarket has 10,000 customers. Each customer has different characteristics:

- Age
- Income
- Amount they spend
- Products they buy
- How often they shop

| Cluster | Customer type       | Characteristics                          |
| ------- | ------------------- | ---------------------------------------- |
| 1       | Budget shoppers     | Low spending, price-sensitive            |
| 2       | Premium shoppers    | High spending, prefer expensive products |
| 3       | Frequent shoppers   | Shop very often                          |
| 4       | Occasional shoppers | Shop only sometimes                      |

## Now how can we determine if two items are similar or not ?

suppose we represent these items as points with (x, y) in coordinates .. the less the distance between these points
the more the relevance.

For example, given:

```text
[1, 1]    [1, 2]    [2, 1]

[10, 10]  [10, 11]  [11, 10]

[20, 20]  [20, 21]  [21, 20]
```

K-Means can discover three clusters:

```text
Cluster 1          Cluster 2          Cluster 3

[1,1]              [10,10]            [20,20]
[1,2]              [10,11]            [20,21]
[2,1]              [11,10]            [21,20]
```

Notice that we did not provide labels such as `cluster_1`, `cluster_2`, etc.

This is why K-Means is an **unsupervised learning** algorithm.

## What does K mean?

`K` represents the number of clusters that we want to create.

For example:

```python
model = K_Means(k=3)
```

means:

> "Divide the data into 3 clusters."

K-Means does not automatically decide the value of `K` in the basic algorithm.

---

# How does K-Means Works ?

K-Means is one of the most easiest and intuative algorithms you might see.
the process of k-means is an iterative process contains 4 steps

```text
1. Initialize centroids
          ↓
2. Assign points to nearest centroid
          ↓
3. Calculate new centroids
          ↓
4. Check convergence
          ↓
     Not converged?
          │
          └──────→ Repeat
```

## let's go through these steps

## Step 1: Initialize the Centroids

First, we randomly select `K` points from the dataset to act as the initial centroids.

In our implementation:

```python
indices = np.random.choice(
    data.shape[0],
    self.k,
    replace=False
)

self.centroids = data[indices].copy().astype(float)
```

For example, if `K = 3`, we might initially select:

```text
Centroid 0 = [1, 1]
Centroid 1 = [10, 10]
Centroid 2 = [20, 20]
```

The initialization is random, so different runs can potentially start with different centroids.

For reproducible experiments, we can use:

```python
np.random.seed(42)
```

---

# Step 2: Assign Each Point to the Nearest Centroid

For every data point, we calculate its distance from every centroid.

We use **squared Euclidean distance**:

```python
np.sum((point - centroid) ** 2)
```

The normal Euclidean distance is:

```text
sqrt((x1-x2)² + (y1-y2)²)
```

We don't actually need the square root because we only care about which distance is smaller.

For example:

```text
Point = [2, 1]

Centroid 0 = [1, 1]
Centroid 1 = [10, 10]
Centroid 2 = [20, 20]
```

Distances:

```text
Distance to centroid 0 = 1
Distance to centroid 1 = 145
Distance to centroid 2 = 685
```

Therefore:

```text
[2,1] → Cluster 0
```

In the code:

```python
index = np.argmin(distance)
```

`np.argmin()` returns the index of the smallest distance.

---

# Step 3: Update the Centroids

After assigning all points to clusters, we calculate a new centroid for each cluster.

The centroid is simply the **mean of all points belonging to that cluster**.

For example:

```text
Cluster:

[1,1]
[1,2]
[2,1]
```

The new centroid is:

```text
x = (1 + 1 + 2) / 3 = 1.333

y = (1 + 2 + 1) / 3 = 1.333
```

Therefore:

```text
Centroid = [1.333, 1.333]
```

Our implementation uses:

```python
np.mean(points, axis=0)
```

The `axis=0` means:

> Calculate the mean separately for each feature/column.

---

# Step 4: Check for Convergence

After updating the centroids, we check whether they have stopped changing.

We save the previous centroids:

```python
prev = self.centroids.copy()
```

Then update them:

```python
self.update()
```

Finally:

```python
if np.allclose(self.centroids, prev, rtol=1e-4):
    break
```

If the old and new centroids are sufficiently close, we consider the algorithm converged and stop iterating.

There is also a maximum number of iterations:

```python
max_iter = 500
```

This prevents the algorithm from running forever.

---

# Complete Algorithm

The complete K-Means algorithm can be summarized as:

```text
Choose K

Randomly initialize K centroids

Repeat:

    1. Assign every point
       to its nearest centroid

    2. Calculate the mean of
       every cluster

    3. Move each centroid
       to its cluster mean

    4. Check whether centroids
       have stopped moving

Until:

    Centroids converge
    OR
    Maximum iterations are reached
```

---
