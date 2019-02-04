# CDbw
Compute the S_Dbw validity index  
S_Dbw validity index is defined by equation:
##### CDbw = compactness\*cohesion*separation
**Highest value -> better clustering.**
______________________________________________

#### Installation:

```shell
pip install --upgrade cdbw
```

### Usage:

```python
from cdbw import CDbw
score = CDbw(X, labels, metric="euclidean", alg_noise='comb', 
     intra_dens_inf=False, s=3, multipliers=False)

```

### Parameters:
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)
    metric : str,
        The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’.
    alg_noise : str,
        Algorithm for recording noise points.
        'comb' - combining all noise points into one cluster (default)
        'sep' - definition of each noise point as a separate cluster
        'bind' -  binding of each noise point to the cluster nearest from it
        'filter' - filtering noise points
    intra_dens_inf : bool,
        If False (default) CDbw index = 0 for cohesion or compactness - inf or nan.
    s : int,
        Number of art representative points. (>2)
    multipliers : bool,
        Format of output. False (default) - only CDbw index, True - tuple (compactness, cohesion, separation, CDbw)

### Returns:
    cdbw : float,
        The resulting CDbw validity index.

References:
-----------
1. M. Halkidi and M. Vazirgiannis, “A density-based cluster validity approach using multi-representatives”
        Pattern Recognition Letters 29 (2008) 773–786.
