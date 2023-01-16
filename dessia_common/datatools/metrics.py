"""
Distances operators for metrics on Datasets.
"""

import numpy as npy
from scipy.spatial.distance import mahalanobis


def diff_list(list_a, list_b):
    """
    Difference between to lists.

    :param list_a: First list
    :type list_a: List[float]

    :param list_b: Second list
    :type list_b: List[float]

    :return: a generator of the difference between each element
    :rtype: generator
    """
    return (a - b for a, b in zip(list_a, list_b))

def l1_norm(vector):
    """
    L1-norm of vector.

    :param vector: vector to get norm
    :type vector: List[float]

    :return: the l1-norm
    :rtype: float
    """
    return sum(map(abs, vector))

def l2_norm(vector):
    """
    L2-norm of vector.

    :param vector: vector to get norm
    :type vector: List[float]

    :return: the l2-norm
    :rtype: float
    """
    # better than numpy for len = 20000, nearly the same for len = 2000
    return sum(x*x for x in vector)**0.5

def lp_norm(vector, mink_power = 2):
    """
    Minkowski norm of vector.

    :param vector: vector to get norm
    :type vector: List[float]

    :param mink_power: the value of exponent in Minkowski norm
    :type mink_power: float

    :return: the Minkowski norm
    :rtype: float
    """
    return float(npy.linalg.norm(vector, ord=mink_power))

def inf_norm(vector):
    """
    Inifinite norm of vector.

    :param vector: vector to get norm
    :type vector: List[float]

    :return: maximum value of absolute values in vector
    :rtype: float
    """
    return max(abs(coord) for coord in vector)

def manhattan_distance(list_a, list_b):
    """
    Compute the l1 distance between list_a and list_b, i.e. the l1-norm of difference between list_a and list_b.

    :param list_a: First list
    :type list_a: List[float]

    :param list_b: Second list
    :type list_b: List[float]

    :return: the l1 distance between the two list
    :rtype: float
    """
    # faster than numpy
    return l1_norm(diff_list(list_a, list_b))

def euclidian_distance(list_a, list_b):
    """
    Compute the euclidian distance between list_a and list_b, i.e. the l2-norm of difference between list_a and list_b.\
    It is the natural distance of 3D space.

    :param list_a: First list
    :type list_a: List[float]

    :param list_b: Second list
    :type list_b: List[float]

    :return: the l2 distance between the two list
    :rtype: float
    """
    # faster than numpy for len = 20000, nearly the same for len = 2000
    return l2_norm(diff_list(list_a, list_b))

def minkowski_distance(list_a, list_b, mink_power = 2):
    """
    Compute the Minkowski distance between list_a and list_b, i.e. the lp-norm of difference between list_a and list_b.

    :param list_a: First list
    :type list_a: List[float]

    :param list_b: Second list
    :type list_b: List[float]

    :param mink_power: the value of exponent in Minkowski norm
    :type mink_power: float

    :return: the Minkowski distance between the two list
    :rtype: float
    """
    # faster than sum((a - b)**p for a, b in zip(list_a, list_b))**(1/p)
    return lp_norm(npy.array(list_a)-npy.array(list_b), mink_power=mink_power)

def mean(vector):
    """
    Mean of vector.

    :param vector: vector to get mean
    :type vector: List[float]

    :return: the mean of vector
    :rtype: float
    """
    return sum(vector)/len(vector)

def variance(vector):
    """
    Variance of vector.

    :param vector: vector to get variance
    :type vector: List[float]

    :return: the variance of vector
    :rtype: float
    """
    # faster than euclidian_distance(vector, [mean(vector)] * len(vector))**2 / len(vector)
    return float(npy.var(vector))

def covariance(vector_x, vector_y):
    """
    Covariance between vector_x and vector_y.

    :param vector_x: first vector to get covariance
    :type vector_x: List[float]

    :param vector_y: second vector to get covariance
    :type vector_y: List[float]

    :return: the covariance between vector_x and vector_y
    :rtype: float
    """
    # nearly as fast as numpy
    if len(vector_x) != len(vector_y):
        raise ValueError("vector_x and vector_y must be the same length to compute covariance.")
    mean_x = mean(vector_x)
    mean_y = mean(vector_y)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(vector_x, vector_y)) / len(vector_x)

def covariance_matrix(matrix):
    """
    Compute the covariance matrix of `matrix` of dimension `N x M`.

    :return: the covariance matrix of `matrix`
    :rtype: List[List[float]], `N x N`

    :Examples:
    >>> from dessia_common.datatools.metrics import covariance_matrix
    >>> from dessia_common.models import all_cars_wi_feat
    >>> matrix = Dataset(all_cars_wi_feat).matrix
    >>> cov_matrix = covariance_matrix(list(zip(*matrix)))
    >>> for row in cov_matrix[:2]: print(row[:2])
    [70.58986267712706, -0.6737370735267286]
    [-0.6737370735267286, 0.011008722272395539]
    """
    return npy.cov(matrix, dtype=float).tolist()

def std(vector):
    """
    Standard deviation of vector.

    :param vector: vector to get standard deviation
    :type vector: List[float]

    :return: the standard deviation of vector
    :rtype: float
    """
    # faster than euclidian_distance(vector, [mean(vector)] * len(vector)) / math.sqrt(len(vector))
    return float(npy.std(vector))

def mahalanobis_distance(list_a, list_b, cov_matrix):
    """
    Compute the Mahalanobis distance between list_a and list_b.

    This method computes distances considering the scale
    and the data repartition on each dimension (covariance matrix). It is adviced to use this method to compute
    distances in spaces constituted of very different dimensions in terms of scale and data repartition.

    :param list_a: First list
    :type list_a: List[float]

    :param list_b: Second list
    :type list_b: List[float]

    :param cov_matrix: the covariance matrix of data
    :type cov_matrix: List[List[float]]

    :return: the Mahalanobis distance between the two list
    :rtype: float
    """
    inv_cov_matrix = npy.linalg.pinv(cov_matrix)
    return mahalanobis(list_a, list_b, inv_cov_matrix)
