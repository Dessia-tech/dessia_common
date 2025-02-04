""" Distances operators for metrics on Datasets. """

from typing import Union

import numpy as npy
from scipy.spatial.distance import mahalanobis

Vector = list[float]
Matrix = list[Vector]

def diff_list(list_a, list_b):
    """
    Difference between to lists.

    :param list_a: First list
    :type list_a: list[float]

    :param list_b: Second list
    :type list_b: list[float]

    :return: a generator of the difference between each element
    :rtype: generator
    """
    return (a - b for a, b in zip(list_a, list_b))

def maximums(matrix: Union[Vector, Matrix]) -> Vector:
    """ Compute maximum values and store it in a list of length `len(matrix[0])`. """
    if not isinstance(matrix[0], list):
        return [max(matrix)]
    return [max(column) for column in zip(*matrix)]

def minimums(matrix: Union[Vector, Matrix]) -> Vector:
    """ Compute minimum values and store it in a list of length `len(matrix[0])`. """
    if not isinstance(matrix[0], list):
        return [min(matrix)]
    return [min(column) for column in zip(*matrix)]

def l1_norm(vector):
    """
    L1-norm of vector.

    :param vector: vector to get norm
    :type vector: list[float]

    :return: the l1-norm
    :rtype: float
    """
    return sum(map(abs, vector))


def l2_norm(vector):
    """
    L2-norm of vector.

    :param vector: vector to get norm
    :type vector: list[float]

    :return: the l2-norm
    :rtype: float
    """
    # better than numpy for length = 20000, nearly the same for length = 2000
    return sum(x*x for x in vector)**0.5


def lp_norm(vector, mink_power=2):
    """
    Minkowski norm of vector.

    :param vector: vector to get norm
    :type vector: list[float]

    :param mink_power: the value of exponent in Minkowski norm
    :type mink_power: float

    :return: the Minkowski norm
    :rtype: float
    """
    return float(npy.linalg.norm(vector, ord=mink_power))


def inf_norm(vector):
    """
    Infinite norm of vector.

    :param vector: vector to get norm
    :type vector: list[float]

    :return: maximum value of absolute values in vector
    :rtype: float
    """
    return max(abs(coord) for coord in vector)


def manhattan_distance(list_a, list_b):
    """
    Compute the l1 distance between list_a and list_b, i.e. the l1-norm of difference between list_a and list_b.

    :param list_a: First list
    :type list_a: list[float]

    :param list_b: Second list
    :type list_b: list[float]

    :return: the l1 distance between the two list
    :rtype: float
    """
    # faster than numpy
    return l1_norm(diff_list(list_a, list_b))


def euclidean_distance(list_a, list_b):
    """
    Compute the euclidean distance between list_a and list_b, i.e. the l2-norm of difference between list_a and list_b.

    It is the natural distance of 3D space.

    :param list_a: First list
    :type list_a: list[float]

    :param list_b: Second list
    :type list_b: list[float]

    :return: the l2 distance between the two list
    :rtype: float
    """
    # faster than numpy for length = 20000, nearly the same for length = 2000
    return l2_norm(diff_list(list_a, list_b))


def minkowski_distance(list_a, list_b, mink_power= 2):
    """
    Compute the Minkowski distance between list_a and list_b, i.e. the lp-norm of difference between list_a and list_b.

    :param list_a: First list
    :type list_a: list[float]

    :param list_b: Second list
    :type list_b: list[float]

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
    :type vector: list[float]

    :return: the mean of vector
    :rtype: float
    """
    return sum(vector)/len(vector)


def variance(vector):
    """
    Variance of vector.

    :param vector: vector to get variance
    :type vector: list[float]

    :return: the variance of vector
    :rtype: float
    """
    # faster than euclidean_distance(vector, [mean(vector)] * len(vector))**2 / len(vector)
    return float(npy.var(vector))


def covariance(vector_x, vector_y):
    """
    Covariance between vector_x and vector_y.

    :param vector_x: first vector to get covariance
    :type vector_x: list[float]

    :param vector_y: second vector to get covariance
    :type vector_y: list[float]

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
    :rtype: list[list[float]], `N x N`

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
    :type vector: list[float]

    :return: the standard deviation of vector
    :rtype: float
    """
    # faster than euclidean_distance(vector, [mean(vector)] * len(vector)) / math.sqrt(len(vector))
    return float(npy.std(vector))


def mahalanobis_distance(list_a, list_b, cov_matrix):
    """
    Compute the Mahalanobis distance between list_a and list_b.

    This method computes distances considering the scale
    and the data distribution on each dimension (covariance matrix). It is advised to use this method to compute
    distances in spaces constituted of very different dimensions in terms of scale and data distribution.

    :param list_a: First list
    :type list_a: list[float]

    :param list_b: Second list
    :type list_b: list[float]

    :param cov_matrix: the covariance matrix of data
    :type cov_matrix: list[list[float]]

    :return: the Mahalanobis distance between the two list
    :rtype: float
    """
    inv_cov_matrix = npy.linalg.pinv(cov_matrix)
    return mahalanobis(list_a, list_b, inv_cov_matrix)
