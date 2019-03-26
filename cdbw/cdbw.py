# -*- coding: utf-8 -*-

import importlib
import math
from collections import defaultdict, Counter

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def gen_dist_func(metric):
    """
    Obtain the distances function from scipy.spatial.distance package

    Parameters
    ----------
    metric : str,
        The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’.

    Returns
    -------
    func : function (array_like, array_like)
        Function for calculation distance between two points in n-dimensional space.
    """
    mod = importlib.import_module("scipy.spatial.distance")
    func = getattr(mod, metric)
    return func


def filter_noise_lab(X, labels):
    """
    Filter noise points

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)

    Returns
    -------
    filterLabel : array-like, shape (n_samples,)
        Filtered predicted labels for each sample.
    filterXYZ : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point. Data points which label = -1 was removed.
    """
    filterLabel = labels[labels != -1]
    filterXYZ = X[labels != -1]
    return filterLabel, filterXYZ


def bind_noise_lab(X, labels, metric):
    """
    Bind noise points to nearest cluster

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)
    metric : str,
        The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’. Default is ‘euclidean’.

    Returns
    -------
    labels : array-like, shape (n_samples,)
        Modified predicted labels for each sample. to a single data point.
        Data points which label = -1 was bound to nearest clusters.
    """

    labels = labels.copy()
    if -1 not in set(labels):
        return labels
    if len(set(labels)) == 1 and -1 in set(labels):
        raise ValueError('Labels contains noise point only')
    label_id = []
    label_new = []
    for i in range(len(labels)):
        if labels[i] == -1:
            point = np.array([X[i]])
            dist = cdist(X[labels != -1], point, metric=metric)
            lid = np.where(np.all(X == X[labels != -1][np.argmin(dist), :], axis=1))[0][0]
            label_id.append(i)
            label_new.append(labels[lid])
    labels[np.array(label_id)] = np.array(label_new)
    return labels


def comb_noise_lab(labels):
    """
    Combining all noise points into one cluster

    Parameters
    ----------
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)

    Returns
    -------
    labels : array-like, shape (n_samples,)
        Modified predicted labels for each sample. to a single data point.
        All data points which label = -1 was combined into a one cluster.
    """
    labels = labels.copy()
    max_label = np.max(labels)
    j = max_label + 1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = j
    return labels


def prep(X, labels):
    """
    Calculation necessary parameters

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    n_clusters : int,
        Number of clusters.
    stdev : float,
        Average standard deviation of the considered clusters.
    dimension : int,
        Dimension of Data Set.
    n_points_in_cl : array_like shape (n_clusters,)
        Number of points in considered clusters     
    n_max: int,
        Maximal number of points in considered clusters    
    coord_in_cl : array-like,shape (n_clusters, n_max, dimension)
        Coordinates of points in each  cluster.
    labels_in_cl :array-like,shape (n_clusters, n_max)
        labels of points in each  cluster.    
    """
    dimension = X.shape[1]
    n_clusters = labels.max() + 1
    n_points_in_cl = np.zeros(n_clusters, dtype=int)
    stdv1_cl = np.zeros(shape=(n_clusters, dimension), dtype=float)
    std1_cl = np.zeros(shape=n_clusters, dtype=float)
    for i in range(n_clusters):
        n_points_in_cl[i] = Counter(labels).get(i)
        stdv1_cl[i] = np.std(X[labels == i], axis=0)
        std1_cl[i] = np.dot(stdv1_cl[i].T, stdv1_cl[i])
        std1_cl[i] = math.sqrt(std1_cl[i] / dimension)
    n_max = max(n_points_in_cl)
    coord_in_cl = np.full((n_clusters, n_max, dimension), np.nan)
    labels_in_cl = np.full((n_clusters, n_max), -1)
    for i in range(n_clusters):
        for j in range(n_clusters):
            stdev = np.power(np.mean([std1_cl[i] ** 2, std1_cl[j] ** 2]), 0.5)
    for i in range(n_clusters):
        coord_in_cl[i, 0:n_points_in_cl[i], 0:dimension] = X[labels == i]
        labels_in_cl[i, 0:n_points_in_cl[i]] = np.where(labels == i)[0]
    return n_clusters, stdev, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl


def rep(n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl):
    """
    Select of representative points for each clusters

    Parameters
    ----------
    n_clusters : int,
        Number of clusters.
    dimension : int,
        Dimension of Data Set.
    n_points_in_cl : array-like, shape ( n_clusters)   
        Number of points in considered clusters.
    coord_in_cl :  array-like,shape (n_clusters, n_max, dimension)
        Coordinates of points in each of clusters.
    labels_in_cl :  array-like,shape (n_clusters, n_max)
        labels of points in each of clusters.
        
    Returns
    -------    
    mean_arr : array_like shape (n_clusters, dimension)
        Coordinates of the centroid of each cluster.
    n_rep : array_like shape (n_clusters,)
        Number of representative points in each cluster.
    n_rep_max : int,
    Maximal number of points in clusters
    rep_in_cl : array_like shape (n_clusters,n_rep_max)
        Continuous numbers of representatives in each cluster         
    """
    mean_arr = np.zeros(shape=(n_clusters, dimension), dtype=float)
    n_rep = np.zeros(shape=(n_clusters), dtype=int)
    for i in range(n_clusters):
        if n_points_in_cl[i] >= 4:
            ch = ConvexHull(coord_in_cl[i, 0:n_points_in_cl[i], 0:dimension])
            n_rep[i] = ch.vertices.size
        else:
            n_rep[i] = n_points_in_cl[i]
    n_rep_max = np.max(n_rep)
    rep_in_cl = np.full((n_clusters, n_rep_max), -1)
    for i in range(n_clusters):
        if n_points_in_cl[i] >= 4:
            ch = ConvexHull(coord_in_cl[i, 0:n_points_in_cl[i], 0:dimension])
            rep_in_cl[i, 0:n_rep[i]] = labels_in_cl[i, 0:n_points_in_cl[i]][ch.vertices]
        else:
            rep_in_cl[i, 0:n_rep[i]] = labels_in_cl[i, 0:n_points_in_cl[i]]
        mean_arr[i] = np.mean(coord_in_cl[i, 0:n_points_in_cl[i], 0:dimension], axis=0)
    return mean_arr, n_rep, n_rep_max, rep_in_cl


def closest_rep(X, n_clusters, rep_in_cl, n_rep, metric, distvec):
    """    
    Select of the closest representative points for two clusters

    Parameters
    ----------

    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    n_clusters : int,
        Number of clusters.
    rep_in_cl : array_like shape (n_clusters,n_rep_max)
        Continuous numbers of representatives in each cluster 
    n_rep : array_like shape (n_clusters,)
        Number of representative points in each cluster.
    metric : str,
        The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’,
        ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’,
        ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, 
        ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’.
    distvec : function (array_like, array_like)
        Function for calculation distance between two points in n-dimensional
        space.

    Returns
    -------
    dist_min : defaultdict {tuple (i, j) : list [float]}
        List of distances between of of closest representative points for 
        each pair of clusters,
        i - No. of 1 cluster, j - No. of 2 cluster.
    middle_point : defaultdict {tuple (i, j) : list [array_like (dimension)]}
        List of coordinates of middle points for two closest representative
        points of each pair of clusters,
        i - No. of 1 cluster, j - No. of 2 cluster.
    n_cl_rep : dict {tuple (i, j): indexes}
        Indices of closest representative points for each two clusters, 
        i - No. of 1 cluster, j - No. of 2 cluster
        indexes - array of indexes.
    """
    b1 = {}
    b2 = {}
    dist_arr = {}
    min_value1 = {}
    min_value0 = {}
    min_index0 = {}
    min_index1 = {}
    min_index2 = {}
    min_index_r = {}
    min_index_c = {}
    cl_r = {}
    s1 = []
    s2 = []
    s2_t = []
    t1 = []
    t2 = []
    v1 = []
    v2 = []
    dist_min = defaultdict(list)
    middle_point = defaultdict(list)
    n_cl_rep = {}
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                dist_arr[(i, j)] = cdist(X[rep_in_cl[i, 0:n_rep[i]]], X[rep_in_cl[j, 0:n_rep[j]]], metric=metric)
                min_value1[(i, j)] = dist_arr[(i, j)].min(axis=1)
                min_value0[(i, j)] = dist_arr[(i, j)].min(axis=0)
                min_index1[(i, j)] = dist_arr[(i, j)].argmin(axis=1)
                min_index0[(i, j)] = dist_arr[(i, j)].argmin(axis=0)
                min_index_r[(i, j)] = np.add(np.arange(0, n_rep[i] * n_rep[j], n_rep[j]), min_index1[(i, j)])
                min_index_c[(i, j)] = np.add(np.arange(0, n_rep[i] * n_rep[j], n_rep[i]), min_index0[(i, j)])
                t1 += [n_rep[i]]
                t2 += [n_rep[j]]
                for k in range(n_rep[i]):
                    s1.append(np.unravel_index(min_index_r[(i, j)][k],
                                               (n_rep[i], n_rep[j])))
                for n in range(n_rep[j]):
                    s2.append(np.unravel_index(min_index_c[(i, j)][n],
                                               (n_rep[j], n_rep[i])))
                    s2_t = [(x[1], x[0]) for x in s2]
    p = 0
    for m in range(len(t1)):
        p += t1[m]
        v1.append(p)
    p = 0
    for m in range(len(t2)):
        p += t2[m]
        v2.append(p)
    min_index1[(1, 0)] = s1[0:v1[0]]
    min_index2[(1, 0)] = s2_t[0:v2[0]]
    m = 0
    for i in range(2, n_clusters):
        for j in range(n_clusters):
            if i > j:
                min_index1[(i, j)] = s1[v1[m]:v1[m + 1]]
                min_index2[(i, j)] = s2_t[v2[m]:v2[m + 1]]
                m += 1
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                b1[(i, j)] = set(min_index1[(i, j)])
                b2[(i, j)] = set(min_index2[(i, j)])
                cl_r[(i, j)] = list(b1[(i, j)] & b2[(i, j)])
                n_cl_rep[(i, j)] = len(cl_r[i, j])
                for u, v in cl_r[(i, j)]:
                    middle_point[(i, j)].append((X[rep_in_cl[i, 0:n_rep[i]][u]] + X[rep_in_cl[j, 0:n_rep[j]][v]]) / 2)
                    dist_min[(i, j)].append(distvec(X[rep_in_cl[i, 0:n_rep[i]][u]], X[rep_in_cl[j, 0:n_rep[j]][v]]))
    return middle_point, dist_min, n_cl_rep


def art_rep(X, n_clusters, rep_in_cl, n_rep, n_rep_max, mean_arr, s, dimension):
    """
    Calculate of the art representative points

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    n_clusters : int,
        Number of clusters.
    rep_in_cl : array_like shape (n_clusters,n_rep_max)
        Continuous numbers of representatives in each cluster 
    n_rep : array_like shape (n_clusters,)
        Number of representative points in each cluster.
    n_rep_max : int,
        Maximal number of points in clusters    
    mean_arr : array_like shape (n_clusters, dimension)
        Coordinates of the centroid of each cluster.
    s : int,
        Number of art representative points. (>2)
    dimension : int,
        Dimension of Data Set.

    Returns
    -------
    a_rep_shell : array_like (n_clusters,s,n_rep_max,dimension)
        array of n-dimensional coordinates of art representative points for
        pair i, k where i - No. of cluster,
        k - No. of shell.
    """
    a_rep_shell = np.full((n_clusters, s, n_rep_max, dimension), np.nan)
    for i in range(n_clusters):
        if n_rep[i] == 1:
            raise ValueError('Cluster No. {:d} obtain only 1 point'.format(i))
        for k in range(0, s):
            a_rep_shell[i, k, 0:n_rep[i], 0:dimension] = (1 - (k / s)) * X[rep_in_cl[i, 0:n_rep[i]]] + (k / s) * \
                                                         mean_arr[i]
    return a_rep_shell


def compactness(n_clusters, stdev, a_rep_shell, n_rep, n_points_in_cl, s, coord_in_cl, n_max, n_rep_max, metric):
    """
    Clusters compactness and cohesion evaluation

    Parameters
    ----------   
    n_clusters : integer
        Number of clusters.
    stdev : float
        Average standard deviation of the considered clusters.
    a_rep_shell : defaultdict {tuple (i, k) : list [array_like (dimension)]}
        List of n-dimensional coordinates of art representative points for
        pair i, k where i - No. of cluster,
        k - No. of shell.
    n_rep : array_like shape (n_clusters,)
        Number of representative points in each cluster.
    n_points_in_cl : array_like shape (n_clusters,)
        Number of points in each cluster
    s : integer
        Number of art representative points. (>2)
    coord_in_cl :  array-like,shape (n_clusters, n_max, dimension)
        Coordinates of points in each of clusters.
    n_max : int,
        Maximal number of points in clusters.     
    n_rep_max : int,
        Maximal number of points in clusters
    metric : string,
        Type of metric in distance calculations    
    Returns
    -------
    compact : float
        Compactness of clusters.
    cohesion : float
        Cohesion of clusters.
    """
    intra_dens_shell = np.zeros((n_clusters, s), dtype=float)
    dist = np.full((n_clusters, s, n_max, n_rep_max), np.nan)
    arr_ones = np.zeros((n_clusters, s, n_max, n_rep_max), dtype=int)
    for i in range(0, n_clusters):
        for k in range(0, s):
            dist[i, k, 0:n_points_in_cl[i], 0:n_rep[i]] = cdist(coord_in_cl[i, 0:n_points_in_cl[i], :],
                                                                a_rep_shell[i, k, 0:n_rep[i], :], metric=metric)
            arr_ones[i, k, 0:n_points_in_cl[i], 0:n_rep[i]] = np.array(
                dist[i, k, 0:n_points_in_cl[i], 0:n_rep[i]] < stdev, dtype=int)
    card = np.sum(arr_ones, axis=(2, 3))
    for i in range(n_clusters):
        for k in range(s):
            intra_dens_shell[i, k] = card[i, k] / (n_rep[i] * n_points_in_cl[i] * stdev)
    intra_dens = np.sum(intra_dens_shell, axis=0) / n_clusters
    compact = np.sum(intra_dens) / s
    intra_change = 0
    for l in range(s - 1):
        intra_change += abs(intra_dens[l + 1] - intra_dens[l]) / (s - 1)
    cohesion = compact / (1 + intra_change)
    return compact, cohesion


def separation(n_clusters, stdev, middle_point, dist_min, n_cl_rep, n_points_in_cl, coord_in_cl):
    """
    Clusters separation evaluation

    Parameters
    ----------    
    n_clusters : int,
        Number of clusters.
    stdev : float.
        Average standard deviation of the considered clusters.
    middle_point : defaultdict {tuple (i, j) : list [array_like (dimension)]},
        List of coordinates of middle points for two closest 
        representative points of each pair of clusters,
        i - No. of 1 cluster, j - No. of 2 cluster.
    dist_min : defaultdict {tuple (i, j) : list [float]}
        List of distances between of of closest representative points
        for each pair of clusters,
        i - No. of 1 cluster, j - No. of 2 cluster.
    n_cl_rep : dict {tuple (i, j): indexes}
        Indices of closest representative points for each two clusters,
        i - No. of 1 cluster, j - No. of 2 cluster
        indexes - array of indexes.
    n_points_in_cl : array_like shape (n_clusters,)
        Number of points in each cluster
    coord_in_cl :  array-like,shape (n_clusters, n_max, dimension)
        Coordinates of points in each of clusters.

    Returns
    -------
    sep : float,
        Separation of clusters.
    """
    dist_mm = np.zeros((int(n_clusters * (n_clusters - 1) / 2)))
    dist_mmm = np.zeros(n_clusters)
    n_cl_rep_arr = np.zeros((1), dtype=int)
    n_cl_arr = np.zeros((1), dtype=int)
    n_cl_rep_arr[0] = n_cl_rep[(1, 0)]
    n_cl_arr[0] = n_points_in_cl[1] + n_points_in_cl[0]
    coord_f = np.vstack((coord_in_cl[1, 0:n_points_in_cl[1], :], coord_in_cl[0, 0:n_points_in_cl[0], :]))
    for i in range(2, n_clusters):
        for j in range(n_clusters):
            if i > j:
                n_cl_rep_arr = np.hstack((n_cl_rep_arr, n_cl_rep[(i, j)]))
                n_cl_arr = np.hstack((n_cl_arr, n_points_in_cl[i] + n_points_in_cl[j]))
                coord_f = np.vstack((coord_f, np.vstack(
                    (coord_in_cl[i, 0:n_points_in_cl[i], :], coord_in_cl[j, 0:n_points_in_cl[j], :]))))
    num = np.cumsum(n_cl_rep_arr)
    num = np.hstack([np.array([0]), num])
    num_f = np.cumsum(n_cl_arr)
    num_f = np.hstack([np.array([0]), num_f])
    num_max = num[-1]
    middle_point_arr = np.array(middle_point[(1, 0)])
    dist_min_arr = np.reshape(dist_min[(1, 0)], (n_cl_rep[1, 0], 1))
    for i in range(2, n_clusters):
        for j in range(n_clusters):
            if i > j:
                dist_min[i, j] = np.reshape(dist_min[(i, j)], (n_cl_rep[i, j], 1))
                middle_point_arr = np.vstack((middle_point_arr, middle_point[(i, j)]))
                dist_min_arr = np.vstack((dist_min_arr, dist_min[(i, j)]))
    ns = np.arange(n_clusters)
    card1 = np.zeros(num_max, dtype=int)
    nums = np.cumsum(ns)
    dens1 = np.zeros(int(n_clusters * (n_clusters - 1) / 2))
    for i in range(int(n_clusters * (n_clusters - 1) / 2)):
        dist = cdist(middle_point_arr[num[i]:num[i + 1], :], coord_f[num_f[i]:num_f[i + 1], :])
        arr_ones = np.array(dist < stdev, dtype=int)
        card = np.sum(arr_ones, axis=1, dtype=int)
        card1[num[i]:num[i + 1]] = card
        dist_mm[i] = np.sum(dist_min_arr[num[i]:num[i + 1]]) / n_cl_rep_arr[i]
    for i in range(n_clusters - 1):
        dist_mmm[i] = np.min(dist_mm[nums[i]:nums[i + 1]])
    dist_min_arr = dist_min_arr.reshape(1, num_max)
    dens = card1 * dist_min_arr
    for i in range(int(n_clusters * (n_clusters - 1) / 2)):
        dens1[i] = np.sum(dens[:, num[i]:num[i + 1]])
    dens_mean = dens1 / (stdev * n_cl_arr)
    inter_dens = np.sum(np.max(dens_mean)) / (n_clusters)
    dist_m = np.sum(dist_mmm) / n_clusters
    sep = dist_m / (1 + inter_dens)
    return sep


def CDbw(X, labels, metric="euclidean", alg_noise='comb', intra_dens_inf=False, s=3, multipliers=False):
    """
    Calculate CDbw-index for cluster validation, as defined in [1]

    CDbw = compactness*cohesion*separation

    Parameters
    ----------
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
        'bind' -  binding of each noise point to the cluster nearest from it
        'filter' - filtering noise points
    intra_dens_inf : bool,
        If False (default) CDbw index = 0 for cohesion or compactness - inf or nan.
    s : int,
        Number of art representative points. (>2)
    multipliers : bool,
        Format of output. False (default) - only CDbw index, True - tuple (compactness, cohesion, separation, CDbw)

    Returns
    -------
    cdbw : float,
        The resulting CDbw validity index.

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, “A density-based cluster validity approach using multi-representatives”
        Pattern Recognition Letters 29 (2008) 773–786.

    """
    if len(set(labels)) < 2 or len(set(labels)) > len(X) - 1:
        raise ValueError("No. of unique labels must be > 1 and < n_samples")
    if s < 2:
        raise ValueError("Parameter s must be > 2")
    elif alg_noise == 'bind':
        labels = bind_noise_lab(X, labels, metric=metric)
    elif alg_noise == 'comb':
        labels = comb_noise_lab(labels)
    elif alg_noise == 'filter':
        labels, X = filter_noise_lab(X, labels)
    labels = np.asarray(labels)
    distvec = gen_dist_func(metric)
    n_clusters, stdev, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl = prep(X, labels)
    mean_arr, n_rep, n_rep_max, rep_in_cl = rep(n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl)
    middle_point, dist_min, n_cl_rep = closest_rep(X, n_clusters, rep_in_cl, n_rep, metric, distvec)
    try:
        a_rep_shell = art_rep(X, n_clusters, rep_in_cl, n_rep, n_rep_max, mean_arr, s, dimension)
    except ValueError:
        return 0
    compact, cohesion = compactness(n_clusters, stdev, a_rep_shell, n_rep, n_points_in_cl, s, coord_in_cl, n_max,
                                    n_rep_max, metric)
    if (np.isinf(compact) or np.isnan(compact)) and not intra_dens_inf:
        return 0
    sep = separation(n_clusters, stdev, middle_point, dist_min, n_cl_rep, n_points_in_cl, coord_in_cl)
    cdbw = compact * cohesion * sep
    if multipliers:
        return compact, cohesion, sep, cdbw
    else:
        return cdbw


