# -*- coding: utf-8 -*-
"""Input 2 np.array : coord_arr,labels"""
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull, distance
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y


def prep(labels, coord_arr):
    """ Подготовка данных"""
    n_data = labels.size
    dimension = coord_arr.shape[1]
    n_clusters = labels.max() + 1
    stdv1_cl = np.zeros(shape=(n_clusters, dimension), dtype=float)
    std2_cl = np.zeros(shape=(n_clusters, n_clusters), dtype=float)
    for i in range(n_clusters):
        stdv1_cl[i] = np.std(coord_arr[labels == i], axis=0)
    std1_cl = np.mean(stdv1_cl, axis=1)
    for i in range(n_clusters):
        for j in range(n_clusters):
            std2_cl[i, j] = np.mean([std1_cl[i], std1_cl[j]], axis=0)
    return n_data, n_clusters, std2_cl, dimension


def rep(coord_arr, labels, n_clusters, dimension):
    """ Выбор представителей с помощью conhull"""
    rep_dic = {}
    mean_arr = np.zeros(shape=(n_clusters, dimension), dtype=float)
    n_rep = {}
    labels_in_cluster = {}
    n_points_in_cl = {}
    for i in range(n_clusters):
        labels_in_cluster[i] = np.where(labels == i)
        if labels_in_cluster[i][0].size >= 4:
            ch = ConvexHull(coord_arr[labels == i])
            rep_dic[i] = labels_in_cluster[i][0][ch.vertices]
        else:
            rep_dic[i] = labels_in_cluster[i][0]
        mean_arr[i] = np.mean(coord_arr[labels == i], axis=0)
        n_rep[i] = rep_dic[i].size
        n_points_in_cl[i] = labels_in_cluster[i][0].size
    return rep_dic, mean_arr, n_rep, n_points_in_cl


def closest_rep(rep_dic, n_rep, coord_arr, n_clusters):
    """Выбор ближайших представителей"""
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
                dist_arr[(i, j)] = distance.cdist(coord_arr[rep_dic[i]], coord_arr[rep_dic[j]])
                min_value1[(i, j)] = dist_arr[(i, j)].min(axis=1)
                min_value0[(i, j)] = dist_arr[(i, j)].min(axis=0)
                min_index1[(i, j)] = dist_arr[(i, j)].argmin(axis=1)
                min_index0[(i, j)] = dist_arr[(i, j)].argmin(axis=0)
                min_index_r[(i, j)] = np.add(np.arange(0, n_rep[i] * n_rep[j], n_rep[j]), min_index1[(i, j)])
                min_index_c[(i, j)] = np.add(np.arange(0, n_rep[i] * n_rep[j], n_rep[i]), min_index0[(i, j)])
                t1 += [n_rep[i]]
                t2 += [n_rep[j]]
                for k in range(n_rep[i]):
                    s1.append(np.unravel_index(min_index_r[(i, j)][k], (n_rep[i], n_rep[j])))
                for n in range(n_rep[j]):
                    s2.append(np.unravel_index(min_index_c[(i, j)][n], (n_rep[j], n_rep[i])))
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
    l = 0
    for i in range(2, n_clusters):
        for j in range(n_clusters):
            if i > j:
                min_index1[(i, j)] = s1[v1[l]:v1[l + 1]]
                min_index2[(i, j)] = s2_t[v2[l]:v2[l + 1]]
                l += 1
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                b1[(i, j)] = set(min_index1[(i, j)])
                b2[(i, j)] = set(min_index2[(i, j)])
                cl_r[(i, j)] = list(b1[(i, j)] & b2[(i, j)])
                n_cl_rep[(i, j)] = len(cl_r[i, j])
                for u, v in cl_r[(i, j)]:
                    middle_point[(i, j)].append((coord_arr[rep_dic[i][u]] + coord_arr[rep_dic[j][v]]) / 2)
                    dist_min[(i, j)].append(distance.euclidean(coord_arr[rep_dic[i][u]], coord_arr[rep_dic[j][v]]))
    return middle_point, dist_min, n_cl_rep


def art_rep(n_clusters, rep_dic, mean_arr, coord_arr):
    """Искусственные представители"""
    a_rep_shell = defaultdict(list)
    for i in range(n_clusters):
        for x in rep_dic[i]:
            for k in range(5):
                a_rep_shell[i, k].append((1 - k * 0.2) * coord_arr[x] + k * 0.2 * mean_arr[i])
    return a_rep_shell


def compactness(labels, coord_arr, a_rep_shell, n_clusters, n_rep, std2_cl, n_points_in_cl):
    """Оценка компактности и однородности кластеров"""
    card = defaultdict(lambda: 0)
    a_rep_shell1 = {}
    intra_dens_shell = np.zeros(shape=(n_clusters, 5), dtype=float)
    for i in range(n_clusters):
        for x in coord_arr[labels == i]:
            for k in range(5):
                a_rep_shell1[i, k] = np.array(a_rep_shell[i, k])
                for p in a_rep_shell1[i, k]:
                    dist = distance.euclidean(x, p)
                    if dist < std2_cl[i, i]:
                        card[k, i] += 1
                    else:
                        continue
    for i in range(n_clusters):
        for k in range(5):
            intra_dens_shell[i, k] = card[k, i] / (std2_cl[i, i] * n_clusters * n_rep[i] * n_points_in_cl[i])
    intra_dens = np.sum(intra_dens_shell, axis=0)
    compact = np.sum(intra_dens) / 5
    intra_change = 0
    for l in range(4):
        intra_change += np.sum(abs(intra_dens[l + 1] - intra_dens[l])) / 4
    cohesion = compact / (1 + intra_change)
    return compact, cohesion


def separation(middle_point, dist_min, n_cl_rep, std2_cl, coord_arr, n_clusters, labels, n_points_in_cl):
    """Оценка отделимости кластеров"""
    dens_mean = np.zeros((n_clusters, n_clusters))
    dist_mm = np.zeros((n_clusters, n_clusters))
    dist_mm[np.diag_indices_from(dist_mm)] = np.inf
    card1 = {k: [0 for _ in range(n)] for k, n in n_cl_rep.items()}
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                for s in range(n_cl_rep[i, j]):
                    for x in np.array(middle_point[(i, j)][s]):
                        for p in np.array(np.vstack([coord_arr[labels == i], coord_arr[labels == j]])):
                            dist1 = distance.euclidean(x, p)
                            if dist1 < std2_cl[(i, j)]:
                                card1[i, j][s] += 1
                            else:
                                continue
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                dens_mean[i, j] = np.mean(np.array(dist_min[(i, j)]) * np.array(card1[(i, j)])) / (
                        n_points_in_cl[i] + n_points_in_cl[j]) / (2 * std2_cl[(i, j)])
                dist_mm[i, j] = np.mean(dist_min[(i, j)])
            elif i < j:
                dens_mean[i, j] = np.mean(np.array(dist_min[(j, i)]) * np.array(card1[(j, i)])) / (
                        n_points_in_cl[j] + n_points_in_cl[i]) / (2 * std2_cl[(j, i)])
                dist_mm[i, j] = np.mean(dist_min[(j, i)])
    inter_dens = np.sum(np.max(dens_mean, axis=0)) / n_clusters
    dist_m = np.sum(np.min(dist_mm, axis=0)) / n_clusters
    sep = dist_m / (1 + inter_dens)
    return sep


def CDbw(coord_arr, labels, intra_dens_inf=False):
    """

    :param coord_arr:
    :param labels:
    :return:
    """
    if len(set(labels)) < 2 or len(set(labels)) > len(coord_arr) - 1:
        raise ValueError("No. of unique labels must be > 1 and < n_samples")
    coord_arr, labels = check_X_y(coord_arr, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_data, n_clusters, std2_cl, dimension = prep(labels, coord_arr)
    rep_dic, mean_arr, n_rep, n_points_in_cl = rep(coord_arr, labels, n_clusters, dimension)
    middle_point, dist_min, n_cl_rep = closest_rep(rep_dic, n_rep, coord_arr, n_clusters)
    a_rep_shell = art_rep(n_clusters, rep_dic, mean_arr, coord_arr)
    compact, cohesion = compactness(labels, coord_arr, a_rep_shell, n_clusters, n_rep, std2_cl, n_points_in_cl)
    if (np.isinf(compact) or np.isnan(compact)) and not intra_dens_inf:
        return 0
    sep = separation(middle_point, dist_min, n_cl_rep, std2_cl, coord_arr, n_clusters, labels, n_points_in_cl)
    cdbw = compact * cohesion * sep
    print(compact, cohesion, sep, cdbw)
    return cdbw


if __name__ == "__main__":
    coord_arr = np.load("xyz.npy")
    labels = np.load("labels.npy")
    print(CDbw(coord_arr, labels))
