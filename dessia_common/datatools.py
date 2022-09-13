"""
Library for building clusters on data.
"""
from typing import List, Dict, Any
from copy import copy
import itertools

import numpy as npy
from sklearn import cluster, preprocessing
import matplotlib.pyplot as plt

try:
    from plot_data.core import Scatter, Histogram, MultiplePlots, Tooltip, ParallelPlot, PointFamily, EdgeStyle, Axis, \
        PointStyle
    from plot_data.colors import BLUE, GREY, LIGHTGREY, Color
except ImportError:
    pass
from dessia_common.exports import XLSXWriter
from dessia_common.core import DessiaObject, DessiaFilter, FiltersList, templates


class HeterogeneousList(DessiaObject):
    """
    Base object for handling a list of DessiaObjects.

    :param dessia_objects:
        --------
        List of DessiaObjects to store in HeterogeneousList
    :type dessia_objects: `List[DessiaObject]`, `optional`, defaults to `None`

    :param name:
        --------
        Name of HeterogeneousList
    :type name: `str`, `optional`, defaults to `''`

    :Properties:
        * **common_attributes:** (`List[str]`)
            --------
            Common attributes of DessiaObjects contained in the current `HeterogeneousList`

        * **matrix:** (`List[List[float]]`, `n_samples x n_features`)
            --------
            Matrix of data computed by calling the `to_vector` method of all `dessia_objects`

    **Built-in methods**:
        * __init__
            >>> from dessia_common.core import HeterogeneousList
            >>> from dessia_common.models import all_cars_wi_feat
            >>> hlist = HeterogeneousList(all_cars_wi_feat, name="init")

        * __str__
            >>> print(HeterogeneousList(all_cars_wi_feat[:3], name='printed'))
            HeterogeneousList printed: 3 samples, 5 features
            |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
            -----------------------------------------------------------------------------------------------------------
            |               18.0  |             0.307  |             130.0  |            3504.0  |              12.0  |
            |               15.0  |              0.35  |             165.0  |            3693.0  |              11.5  |
            |               18.0  |             0.318  |             150.0  |            3436.0  |              11.0  |

        * __len__
            >>> len(HeterogeneousList(all_cars_wi_feat))
            returns len(all_cars_wi_feat)

        * __get_item__
            >>> HeterogeneousList(all_cars_wi_feat)[0]
            returns <dessia_common.tests.CarWithFeatures object at 'memory_address'>
            >>> HeterogeneousList(all_cars_wi_feat)[0:2]
            returns HeterogeneousList(all_cars_wi_feat[0:2])
            >>> HeterogeneousList(all_cars_wi_feat)[[0,5,6]]
            returns HeterogeneousList([all_cars_wi_feat[idx] for idx in [0,5,6]])
            >>> booleans_list = [True, False,..., True] of length len(all_cars_wi_feat)
            >>> HeterogeneousList(all_cars_wi_feat)[booleans_list]
            returns HeterogeneousList([car for car, boolean in zip(all_cars_wi_feat, booleans_list) if boolean])

        * __add__
            >>> HeterogeneousList(all_cars_wi_feat) + HeterogeneousList(all_cars_wi_feat)
            HeterogeneousList(all_cars_wi_feat + all_cars_wi_feat)
            >>> HeterogeneousList(all_cars_wi_feat) + HeterogeneousList()
            HeterogeneousList(all_cars_wi_feat)
            >>> HeterogeneousList(all_cars_wi_feat).extend(HeterogeneousList(all_cars_wi_feat))
            HeterogeneousList(all_cars_wi_feat + all_cars_wi_feat)
    """
    _standalone_in_db = True
    _vector_features = ["name", "common_attributes"]

    def __init__(self, dessia_objects: List[DessiaObject] = None, name: str = ''):
        if dessia_objects is None:
            dessia_objects = []
        self.dessia_objects = dessia_objects
        self._common_attributes = None
        self._matrix = None
        DessiaObject.__init__(self, name=name)

    def __getitem__(self, key: Any):
        """
        Is added in a further release (feat/clists_metrics)

        """
        if len(self) == 0:
            return []
        if isinstance(key, int):
            return self.pick_from_int(key)
        if isinstance(key, slice):
            return self.pick_from_slice(key)
        if isinstance(key, list):
            if len(key) == 0:
                return self.__class__()
            if isinstance(key[0], bool):
                if len(key) == self.__len__():
                    return self.pick_from_boolist(key)
                raise ValueError(f"Cannot index {self.__class__.__name__} object of len {self.__len__()} with a "
                                 f"list of boolean of len {len(key)}")
            if isinstance(key[0], int):
                return self.pick_from_boolist(self._indexlist_to_booleanlist(key))

        raise NotImplementedError(f"key of type {type(key)} with {type(key[0])} elements not implemented for "
                                  f"indexing HeterogeneousLists")

    def __add__(self, other: 'HeterogeneousList'):
        """
        Is added in a further release (feat/clists_metrics)

        """
        if self.__class__ != HeterogeneousList or other.__class__ != HeterogeneousList:
            raise TypeError("Addition only defined for HeterogeneousList. A specific __add__ method is required for "
                            f"{self.__class__}")

        sum_hlist = self.__class__(dessia_objects=self.dessia_objects + other.dessia_objects,
                                   name=self.name[:5] + '_+_' + other.name[:5])

        if all(item in self.common_attributes for item in other.common_attributes):
            sum_hlist._common_attributes = self.common_attributes
            if self._matrix is not None and other._matrix is not None:
                sum_hlist._matrix = self._matrix + other._matrix
        return sum_hlist

    def extend(self, other: 'HeterogeneousList'):
        """
        Update a HeterogeneousList by adding b values to it

        :param b: HeterogeneousList to add to the current HeterogeneousList
        :type b: HeterogeneousList

        :return: None

        Examples
        --------
        >>> from dessia_common.core import HeterogeneousList
        >>> from dessia_common.models import all_cars_wi_feat
        >>> HeterogeneousList(all_cars_wi_feat).extend(HeterogeneousList(all_cars_wi_feat))
        HeterogeneousList(all_cars_wi_feat + all_cars_wi_feat)
        """
        # Not "self.dessia_objects += other.dessia_objects" to take advantage of __add__ algorithm
        self.__dict__.update((self + other).__dict__)

    def pick_from_int(self, idx: int):
        return self.dessia_objects[idx]

    def pick_from_slice(self, key: slice):
        new_hlist = self.__class__(dessia_objects=self.dessia_objects[key], name=self.name)
        new_hlist._common_attributes = copy(self._common_attributes)
        new_hlist.dessia_objects = self.dessia_objects[key]
        if self._matrix is not None:
            new_hlist._matrix = self._matrix[key]
        # new_hlist.name += f"_{key.start if key.start is not None else 0}_{key.stop}")
        return new_hlist

    def _indexlist_to_booleanlist(self, index_list: List[int]):
        boolean_list = [False] * len(self)
        for idx in index_list:
            boolean_list[idx] = True
        return boolean_list

    def pick_from_boolist(self, key: List[bool]):
        new_hlist = self.__class__(dessia_objects=DessiaFilter.apply(self.dessia_objects, key), name=self.name)
        new_hlist._common_attributes = copy(self._common_attributes)
        if self._matrix is not None:
            new_hlist._matrix = DessiaFilter.apply(self._matrix, key)
        # new_hlist.name += "_list")
        return new_hlist

    def __str__(self):
        size_col_label = self._set_size_col_label()
        attr_name_len = []
        attr_space = []
        prefix = self._write_str_prefix()

        if self.__len__() == 0:
            return prefix

        string = ""
        string += self._print_titles(attr_space, attr_name_len, size_col_label)
        string += "\n" + "-" * len(string)

        string += self._print_objects_slice(slice(0, 5), attr_space, attr_name_len,
                                            self._set_label_space(size_col_label))

        undispl_len = len(self) - 10
        string += (f"\n+ {undispl_len} undisplayed object" + "s" * (min([undispl_len, 2]) - 1) + "..."
                   if len(self) > 10 else '')

        string += self._print_objects_slice(slice(-5, len(self)), attr_space, attr_name_len,
                                            self._set_label_space(size_col_label))
        return prefix + "\n" + string + "\n"

    def _print_objects_slice(self, key: slice, attr_space: int, attr_name_len: int, label_space: int):
        string = ""
        for dessia_object in self.dessia_objects[key]:
            string += "\n" + " " * label_space
            string += self._print_objects(dessia_object, attr_space, attr_name_len)
        return string

    def _set_size_col_label(self):
        return 0

    def _set_label_space(self, size_col_label: int):
        if size_col_label:
            return 2 * size_col_label - 1
        return 0

    def _write_str_prefix(self):
        prefix = f"{self.__class__.__name__} {self.name if self.name != '' else hex(id(self))}: "
        prefix += f"{len(self)} samples, {len(self.common_attributes)} features"
        return prefix

    def _print_titles(self, attr_space: int, attr_name_len: int, size_col_label: int):
        string = ""
        if size_col_label:
            string += "|" + " " * (size_col_label - 1) + "n°" + " " * (size_col_label - 1)
        for idx, attr in enumerate(self.common_attributes):
            end_bar = ""
            if idx == len(self.common_attributes) - 1:
                end_bar = "|"
            # attribute
            attr_space.append(len(attr) + 6)
            name_attr = " " * 3 + f"{attr.capitalize()}" + " " * 3
            attr_name_len.append(len(name_attr))
            string += "|" + name_attr + end_bar
        return string

    def _print_objects(self, dessia_object: DessiaObject, attr_space: int, attr_name_len: int):
        string = ""
        for idx, attr in enumerate(self.common_attributes):
            end_bar = ""
            if idx == len(self.common_attributes) - 1:
                end_bar = "|"

            # attribute
            string += "|" + " " * (attr_space[idx] - len(str(getattr(dessia_object, attr))) - 1)
            string += f"{getattr(dessia_object, attr)}"[:attr_name_len[idx] - 3]
            if len(str(getattr(dessia_object, attr))) > attr_name_len[idx] - 3:
                string += "..."
            else:
                string += " "
            string += end_bar
        return string

    def __len__(self):
        """
        Is added in a further release (feat/clists_metrics)

        """
        return len(self.dessia_objects)

    def get_attribute_values(self, attribute: str):
        """
        Get a list of all values of dessia_objects of an attribute given by name

        :param attribute: Attribute to get all values
        :type attribute: str

        :return: A list of all values of the specified attribute of dessia_objects
        :rtype: List[Any]

        Examples
        --------
        >>> from dessia_common.core import HeterogeneousList
        >>> from dessia_common.models import all_cars_wi_feat
        >>> HeterogeneousList(all_cars_wi_feat[:10]).get_attribute_values("weight")
        [3504.0, 3693.0, 3436.0, 3433.0, 3449.0, 4341.0, 4354.0, 4312.0, 4425.0, 3850.0]
        """
        return [getattr(dessia_object, attribute) for dessia_object in self.dessia_objects]

    def get_column_values(self, index: int):
        """
        Get a list of all values of dessia_objects for an attribute given by its index common_attributes

        :param index: Index in common_attributes to get all values of dessia_objects
        :type index: int

        :return: A list of all values of the specified attribute of dessia_objects
        :rtype: List[float]

        Examples
        --------
        >>> from dessia_common.core import HeterogeneousList
        >>> from dessia_common.models import all_cars_wi_feat
        >>> HeterogeneousList(all_cars_wi_feat[:10]).get_column_values(2)
        [130.0, 165.0, 150.0, 150.0, 140.0, 198.0, 220.0, 215.0, 225.0, 190.0]
        """
        return [row[index] for row in self.matrix]

    def sort(self, key: Any, ascend: bool = True):  # TODO : Replace numpy with faster algorithms
        """
        Sort the current HeterogeneousList along the given key.

        :param key:
            --------
            The parameter on which to sort the HeterogeneousList. Can be an attribute or its index in \
                `common_attributes`
        :type key: `int` or `str`

        :param ascend:
            --------
            Whether to sort the HeterogeneousList in ascending (`True`) or descending (`False`) order
        :type key: `bool`, defaults to `True`

        :return: None

        Examples
        --------
        >>> from dessia_common.core import HeterogeneousList
        >>> from dessia_common.models import all_cars_wi_feat
        >>> example_list = HeterogeneousList(all_cars_wi_feat[:3], "sort_example")
        >>> example_list.sort("mpg", False)
        >>> print(example_list)
        HeterogeneousList sort_example: 3 samples, 5 features
        |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
        -----------------------------------------------------------------------------------------------------------
        |               18.0  |             0.318  |             150.0  |            3436.0  |              11.0  |
        |               18.0  |             0.307  |             130.0  |            3504.0  |              12.0  |
        |               15.0  |              0.35  |             165.0  |            3693.0  |              11.5  |
        >>> example_list.sort(2, True)
        >>> print(example_list)
        HeterogeneousList sort_example: 3 samples, 5 features
        |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
        -----------------------------------------------------------------------------------------------------------
        |               18.0  |             0.307  |             130.0  |            3504.0  |              12.0  |
        |               18.0  |             0.318  |             150.0  |            3436.0  |              11.0  |
        |               15.0  |              0.35  |             165.0  |            3693.0  |              11.5  |
        """
        if len(self) != 0:
            if isinstance(key, int):
                sort_indexes = npy.argsort(self.get_column_values(key))
            elif isinstance(key, str):
                sort_indexes = npy.argsort(self.get_attribute_values(key))
            self.dessia_objects = [self.dessia_objects[idx] for idx in (sort_indexes if ascend else sort_indexes[::-1])]
            if self._matrix is not None:
                self._matrix = [self._matrix[idx] for idx in (sort_indexes if ascend else sort_indexes[::-1])]

    @property
    def common_attributes(self):
        if self._common_attributes is None:
            if len(self) == 0:
                return []

            all_class = []
            one_instance = []
            for dessia_object in self.dessia_objects:
                if dessia_object.__class__ not in all_class:
                    all_class.append(dessia_object.__class__)
                    one_instance.append(dessia_object)

            all_attributes = sum((instance.vector_features() for instance in one_instance), [])
            set_attributes = set.intersection(*(set(instance.vector_features()) for instance in one_instance))

            # Keep order
            self._common_attributes = []
            for attr in all_attributes:
                if attr in set_attributes:
                    self._common_attributes.append(attr)
                    set_attributes.remove(attr)

        return self._common_attributes

    @property
    def matrix(self):
        if self._matrix is None:
            matrix = []
            for dessia_object in self.dessia_objects:
                temp_row = dessia_object.to_vector()
                vector_features = dessia_object.vector_features()
                matrix.append(list(temp_row[vector_features.index(attr)] for attr in self.common_attributes))
            self._matrix = matrix
        return self._matrix

    def filtering(self, filters_list: FiltersList):
        """
        Filter a HeterogeneousList given a FiltersList.
        Method filtering apply a FiltersList to the current HeterogeneousList.

        :param filters_list:
            FiltersList to apply on current HeterogeneousList
        :type filters_list: FiltersList

        :return: The filtered HeterogeneousList
        :rtype: HeterogeneousList

        Examples
        --------
        >>> from dessia_common.core import HeterogeneousList, DessiaFilter
        >>> from dessia_common.models import all_cars_wi_feat
        >>> filters = [DessiaFilter('weight', '<=', 1650.), DessiaFilter('mpg', '>=', 45.)]
        >>> filters_list = FiltersList(filters, "xor")
        >>> example_list = HeterogeneousList(all_cars_wi_feat, name="example")
        >>> filtered_list = example_list.filtering(filters_list)
        >>> print(filtered_list)
        HeterogeneousList example: 3 samples, 5 features
        |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
        -----------------------------------------------------------------------------------------------------------
        |               35.0  |             0.072  |              69.0  |            1613.0  |              18.0  |
        |               31.0  |             0.076  |              52.0  |            1649.0  |              16.5  |
        |               46.6  |             0.086  |              65.0  |            2110.0  |              17.9  |
        """
        booleans_index = filters_list.get_booleans_index(self.dessia_objects)
        return self.pick_from_boolist(booleans_index)

    def singular_values(self):
        """
        Computes the Singular Values Decomposition (SVD) of self.matrix.
        SVD factorizes self.matrix into two unitary matrices `U` and `Vh`, and a 1-D array `s` of singular values \
            (real, non-negative) such that ``a = U @ S @ Vh``, where S is diagonal such as `s1 > s2 >...> sn`.

        SVD gives indications on the dimensionality of a given matrix thanks to the normalized singular values: they \
            are stored in descending order and their sum is equal to 1. Thus, one can set a threshold value, e.g. \
                `0.95`, and keep only the `r` first normalized singular values which sum is greater than the threshold.

        `r` is the rank of the matrix and gives a good indication on the real dimensionality of the data contained in \
            the current HeterogeneousList. `r` is often much smaller than the current dimension of the studied data.
        This indicates that the used features can be combined into less new features, which do not necessarily \
            make sense for engineers.

        More informations: https://en.wikipedia.org/wiki/Singular_value_decomposition

        :return:
            **normalized_singular_values**: list of normalized singular values
            **singular_points**: list of points to plot in dimensionality plot. Does not add any information.
        :rtype: Tuple[List[float], List[Dict[str, float]]]

        """
        scaled_data = HeterogeneousList.scale_data(npy.array(self.matrix) - npy.mean(self.matrix, axis=0))
        _, singular_values, _ = npy.linalg.svd(npy.array(scaled_data).T, full_matrices=False)
        normalized_singular_values = singular_values / npy.sum(singular_values)

        singular_points = []
        for idx, value in enumerate(normalized_singular_values):
            # TODO (plot_data log_scale 0)
            singular_points.append({'Index of reduced basis vector': idx + 1,
                                    'Singular value': (value if value != 0. else 1e-16)})
        return normalized_singular_values, singular_points

    @staticmethod
    def scale_data(data_matrix: List[List[float]]):
        scaled_matrix = preprocessing.StandardScaler().fit_transform(data_matrix)
        return [list(map(float, row.tolist())) for row in scaled_matrix]

    def plot_data(self):
        """
        Plot data method.
        Plot a standard scatter matrix of all attributes in common_attributes and a dimensionality plot.
        """
        # Plot a correlation matrix : To develop
        # correlation_matrix = []
        # List datadict
        data_list = self._plot_data_list()
        # Dimensionality plot
        dimensionality_plot = self._plot_dimensionality()
        # Scattermatrix
        scatter_matrix = self._build_multiplot(data_list, self._tooltip_attributes(), axis=dimensionality_plot.axis,
                                               point_style=dimensionality_plot.point_style)
        # Parallel plot
        parallel_plot = self.parallel_plot(data_list)

        return [parallel_plot, scatter_matrix, dimensionality_plot]

    def _build_multiplot(self, data_list: List[Dict[str, float]], tooltip: List[str], **kwargs: Dict[str, Any]):
        subplots = []
        for line in self.common_attributes:
            for col in self.common_attributes:
                if line == col:
                    unic_values = set((getattr(dobject, line) for dobject in self.dessia_objects))
                    if len(unic_values) == 1:  # TODO (plot_data linspace axis between two same values)
                        subplots.append(Scatter(x_variable=line, y_variable=col))
                    else:
                        subplots.append(Histogram(x_variable=line))
                else:
                    subplots.append(Scatter(x_variable=line, y_variable=col, tooltip=Tooltip(tooltip), **kwargs))

        scatter_matrix = MultiplePlots(plots=subplots, elements=data_list, point_families=self._point_families(),
                                       initial_view_on=True)
        return scatter_matrix

    def parallel_plot(self, data_list: List[Dict[str, float]]):
        return ParallelPlot(elements=data_list, axes=self._parallel_plot_attr(), disposition='vertical')

    def _tooltip_attributes(self):
        return self.common_attributes

    def _plot_data_list(self):
        plot_data_list = []
        for row, _ in enumerate(self.dessia_objects):
            plot_data_list.append({attr: self.matrix[row][col] for col, attr in enumerate(self.common_attributes)})
        return plot_data_list

    def _point_families(self):
        return [PointFamily(BLUE, list(range(len(self))))]

    def _parallel_plot_attr(self):
        # TODO: Put it in plot_data
        (sorted_r2, sorted_association), constant_attributes = self._get_correlations()
        attr_series = self._get_attribute_trios(sorted_r2, sorted_association)
        return constant_attributes + self._trios_list_to_parallel_axes(attr_series)

    def _get_correlations(self):
        r2_scores = []
        association_list = []
        constant_attributes = []
        for idx, attr1 in enumerate(self.common_attributes):
            if len(set(self.get_attribute_values(attr1))) == 1:
                constant_attributes.append(attr1)
                continue
            for _, attr2 in enumerate(self.common_attributes[idx:]):
                if len(set(self.get_attribute_values(attr2))) == 1:
                    constant_attributes.append(attr2)
                    continue
                if attr1 != attr2:
                    correlation_matrix = npy.corrcoef(self.get_attribute_values(attr1),
                                                      self.get_attribute_values(attr2))
                    correlation_xy = correlation_matrix[0, 1]
                    r2_scores.append(correlation_xy**2)
                    association_list.append([attr1, attr2])
        # Returns list of list of associated attributes sorted along their R2 score and constant attributes
        return map(list, zip(*sorted(zip(r2_scores, association_list))[::-1])), list(set(constant_attributes))

    def _get_attribute_trios(self, sorted_r2, sorted_association):
        attr_series = []
        picked_attr = set()
        set_association = set(sum(sorted_association, []))
        while len(picked_attr) != len(set_association):
            first_association = sorted_association[0]
            attr_series.append(first_association)
            picked_attr.update(set(first_association))
            del sorted_r2[0], sorted_association[0]

            for idx, _ in enumerate(sorted_r2):
                if any(item in first_association for item in sorted_association[idx]):
                    attr_series[-1] += sorted_association[idx]
                    picked_attr.update(set(sorted_association[idx]))
                    del sorted_r2[idx], sorted_association[idx]
                    break
        return attr_series

    def _trios_list_to_parallel_axes(self, attribute_series):
        ordered_attr = []
        for attribute_serie in attribute_series:
            if not any(item in attribute_serie for item in ordered_attr):
                ordered_attr += self._new_attributes_trio(attribute_serie)
            else:
                ordered_attr = self._new_sided_attribute(ordered_attr, attribute_serie)
        return ordered_attr

    def _new_attributes_trio(self, attribute_serie):
        mid_index = [attribute_serie.count(attr) for attr in attribute_serie].index(2)
        mid_attr = attribute_serie[mid_index]
        remaining_attr = iter(set(attribute_serie).difference({mid_attr}))
        return [next(remaining_attr), mid_attr, next(remaining_attr)]

    def _new_sided_attribute(self, ordered_attr, attribute_serie):
        for side in [0, -1]:
            if ordered_attr[side] in attribute_serie:
                nb_instances = attribute_serie.count(ordered_attr[side])
                for ieme_instance in range(nb_instances):
                    idx_in_serie = (ieme_instance) * 2 + attribute_serie[ieme_instance * 2:].index(ordered_attr[side])
                    # 1 if idx_in_serie = 0, 0 if idx_in_serie = 1, 3 if idx_in_serie = 2, 2 if idx_in_serie = 3
                    idx_attr_to_add = idx_in_serie + 1 - 2 * (idx_in_serie % 2)
                    added_attr = []
                    if attribute_serie[idx_attr_to_add] not in ordered_attr:
                        added_attr = [attribute_serie[idx_attr_to_add]]
                        ordered_attr = (side + 1) * added_attr + ordered_attr + (-1 * side) * added_attr
                        break
        return ordered_attr

    def _plot_dimensionality(self):
        _, singular_points = self.singular_values()

        axis_style = EdgeStyle(line_width=0.5, color_stroke=GREY)
        axis = Axis(nb_points_x=len(singular_points), nb_points_y=len(singular_points), axis_style=axis_style)
        point_style = PointStyle(color_fill=BLUE, color_stroke=BLUE, stroke_width=0.1, size=2, shape='circle')

        dimensionality_plot = Scatter(elements=singular_points, x_variable='Index of reduced basis vector',
                                      y_variable='Singular value', log_scale_y=True, axis=axis, point_style=point_style)
        return dimensionality_plot

    def to_markdown(self):  # TODO: Custom this markdown
        """
        Render a markdown of the object output type: string
        """
        return templates.heterogeneouslist_markdown_template.substitute(name=self.name, class_=self.__class__.__name__)

    @staticmethod
    def _check_costs(len_data: int, costs: List[List[float]]):
        if len(costs) != len_data:
            if len(costs[0]) == len_data:
                return list(map(list, zip(*costs)))
            raise ValueError(f"costs is length {len(costs)} and the matching HeterogeneousList is length {len_data}. " +
                             "They should be the same length.")
        return costs

    @staticmethod
    def pareto_indexes(costs: List[List[float]]):
        """
        Find the pareto-efficient points

        :return: A (n_points, ) boolean list, indicating whether each point is Pareto efficient
        """
        is_efficient = npy.ones(len(costs), dtype=bool)
        # costs_array = npy.array(costs)
        costs_array = (costs - npy.mean(costs, axis=0)) / npy.std(costs, axis=0)
        for index, cost in enumerate(costs_array):
            if is_efficient[index]:
                # Keep any point with a lower cost
                is_efficient[is_efficient] = npy.any(costs_array[is_efficient] < cost, axis=1)
                # And keep self
                is_efficient[index] = True
        return is_efficient.tolist()

    def pareto_points(self, costs: List[List[float]]):
        """
        Find the pareto-efficient points

        :param costs:
            -----------
            costs on which the pareto points are computed
        :type costs: `List[List[float]]`, `n_samples x n_features`

        :return: a HeterogeneousList containing the selected points
        :rtype: HeterogeneousList
        """
        checked_costs = HeterogeneousList._check_costs(len(self), costs)
        return self[self.__class__.pareto_indexes(checked_costs)]

    def pareto_sheets(self, costs: List[List[float]], nb_sheets: int = 1):
        """
        Get successive pareto sheets (i.e. optimal points in a DOE for pre-computed costs).

        :param costs:
            Pre-computed costs of `len(self)`. Can be multi-dimensional.
        :type costs: `List[List[float]]`, `n_samples x n_costs` or `n_costs x n_samples`

        :param nb_sheets:
            Number of pareto sheets to pick
        :type nb_sheets: `int`, `optional`, default to `1`

        :return: The successive pareto sheets and not selected elements
        :rtype: `List[HeterogeneousList]`, `HeterogeneousList`
        """
        checked_costs = HeterogeneousList._check_costs(len(self), costs)
        non_optimal_costs = checked_costs[:]
        non_optimal_points = self.dessia_objects[:]
        pareto_sheets = []
        for idx in range(nb_sheets):
            pareto_sheet = HeterogeneousList.pareto_indexes(non_optimal_costs)
            pareto_sheets.append(HeterogeneousList(list(itertools.compress(non_optimal_points, pareto_sheet)),
                                                   self.name + f'_pareto_{idx}'))
            non_optimal_points = list(itertools.compress(non_optimal_points, map(lambda x: not x, pareto_sheet)))
            non_optimal_costs = list(itertools.compress(non_optimal_costs, map(lambda x: not x, pareto_sheet)))
        return pareto_sheets, HeterogeneousList(non_optimal_points, self.name)

    @staticmethod
    def pareto_frontiers(len_data: int, costs: List[List[float]]):
        # Experimental
        checked_costs = HeterogeneousList._check_costs(len_data, costs)
        pareto_indexes = HeterogeneousList.pareto_indexes(checked_costs)
        pareto_costs = npy.array(list(itertools.compress(checked_costs, pareto_indexes)))

        array_costs = npy.array(checked_costs)
        super_mini = npy.min(array_costs, axis=0)
        pareto_frontiers = []
        for x_dim in range(pareto_costs.shape[1]):
            for y_dim in range(pareto_costs.shape[1]):
                if x_dim != y_dim:
                    frontier_2d = HeterogeneousList._pareto_frontier_2d(x_dim, y_dim, pareto_costs,
                                                                        npy.max(array_costs[:, x_dim]), super_mini)
                    pareto_frontiers.append(frontier_2d)

        return pareto_frontiers

    @staticmethod
    def _pareto_frontier_2d(x_dim: int, y_dim: int, pareto_costs: List[List[float]], max_x_dim: float,
                            super_mini: List[float]):
        # Experimental
        minidx = npy.argmin(pareto_costs[:, y_dim])
        x_coord = pareto_costs[minidx, x_dim]
        y_coord = pareto_costs[minidx, y_dim]

        new_pareto = pareto_costs[x_coord - pareto_costs[:, x_dim] != 0., :]

        dir_coeffs = (y_coord - new_pareto[:, y_dim]) / (x_coord - new_pareto[:, x_dim])
        dir_coeffs[x_coord == new_pareto[:, x_dim]] = npy.max(dir_coeffs[x_coord != new_pareto[:, x_dim]])

        offsets = y_coord - dir_coeffs * x_coord
        approx_super_mini = dir_coeffs * super_mini[x_dim] + offsets
        chosen_line = npy.argmin(npy.absolute(approx_super_mini - super_mini[y_dim]))

        frontier_2d = npy.array([[super_mini[x_dim], max_x_dim], [approx_super_mini[chosen_line], max_x_dim *
                                                                  dir_coeffs[chosen_line] + offsets[chosen_line]]]).T
        return frontier_2d


class CategorizedList(HeterogeneousList):
    """
    Base object for handling a categorized (clustered) list of DessiaObjects.

    **CategorizedList should be instantiated with** `from_...` **methods.**

    **Do not use** `__init__` **to instantiate a CategorizedList.**

    :param dessia_objects:
        --------
        List of DessiaObjects to store in CategorizedList
    :type dessia_objects: `List[DessiaObject]`, `optional`, defaults to `None`

    :param labels:
        --------
        Labels of DessiaObjects' cluster stored in CategorizedList
    :type labels: `List[int]`, `optional`, defaults to `None`

    :param name:
        --------
        Name of CategorizedList
    :type name: `str`, `optional`, defaults to `""`

    :Properties:
        * **common_attributes:** (`List[str]`)
            --------
            Common attributes of DessiaObjects contained in the current `CategorizedList`
        * **matrix:** (`List[List[float]]`, `n_samples x n_features`)
            --------
            Matrix of data computed by calling the to_vector method of all dessia_objects
        * **n_cluster:** (`int`)
            --------
            Number of clusters in dessia_objects

    **Built-in methods**: See :func:`~HeterogeneousList`
    """
    _allowed_methods = ['from_agglomerative_clustering', 'from_kmeans', 'from_dbscan', 'from_pareto_sheets']

    def __init__(self, dessia_objects: List[DessiaObject] = None, labels: List[int] = None, name: str = ''):
        HeterogeneousList.__init__(self, dessia_objects=dessia_objects, name=name)
        if labels is None:
            labels = [0] * len(self)
        self.labels = labels
        self._n_clusters = None

    @property
    def n_clusters(self):
        if self._n_clusters is None:
            unic_labels = set(self.labels)
            unic_labels.discard(-1)
            self._n_clusters = len(unic_labels)
        return self._n_clusters

    def to_xlsx_stream(self, stream):
        """
        Exports the object to an XLSX to a given stream
        """
        if not isinstance(self.dessia_objects[0], HeterogeneousList):
            writer = XLSXWriter(self.clustered_sublists())
        else:
            writer = XLSXWriter(self)
        writer.save_to_stream(stream)

    def pick_from_slice(self, key: slice):
        new_hlist = HeterogeneousList.pick_from_slice(self, key)
        new_hlist.labels = self.labels[key]
        # new_hlist.name += f"_{key.start if key.start is not None else 0}_{key.stop}")
        return new_hlist

    def pick_from_boolist(self, key: List[bool]):
        new_hlist = HeterogeneousList.pick_from_boolist(self, key)
        new_hlist.labels = DessiaFilter.apply(self.labels, key)
        # new_hlist.name += "_list")
        return new_hlist

    def _print_objects_slice(self, key: slice, attr_space: int, attr_name_len: int, label_space: str):
        string = ""
        for label, dessia_object in zip(self.labels[key], self.dessia_objects[key]):
            string += "\n"
            space = label_space - len(str(label))
            string += "|" + " " * space + f"{label}" + " "
            string += self._print_objects(dessia_object, attr_space, attr_name_len)
        return string

    def _write_str_prefix(self):
        prefix = f"{self.__class__.__name__} {self.name if self.name != '' else hex(id(self))}: "
        prefix += (f"{len(self)} samples, {len(self.common_attributes)} features, {self.n_clusters} clusters")
        return prefix

    def _set_size_col_label(self):
        return 4

    def clustered_sublists(self):
        """
        Split a CategorizedList of labelled DessiaObjects into a CategorizedList of labelled HeterogeneousLists.

        :return: A CategorizedList of length n_cluster that store each cluster in a HeterogeneousList. Labels are \
            the labels of each cluster, i.e. stored HeterogeneousList
        :rtype: CategorizedList[HeterogeneousList]

        Examples
        --------
        >>> from dessia_common.core import HeterogeneousList
        >>> from dessia_common.models import all_cars_wi_feat
        >>> hlist = HeterogeneousList(all_cars_wi_feat, name="cars")
        >>> clist = CategorizedList.from_agglomerative_clustering(hlist, n_clusters=10, name="ex")
        >>> split_clist = clist.clustered_sublists()
        >>> print(split_clist[:3])
        CategorizedList ex_split: 3 samples, 2 features, 3 clusters
        |   n°   |   Name   |   Common_attributes   |
        ---------------------------------------------
        |      0 |     ex_0 |['mpg', 'displacemen...|
        |      1 |     ex_1 |['mpg', 'displacemen...|
        |      2 |     ex_2 |['mpg', 'displacemen...|
        >>> print(split_clist[3][:3])
        HeterogeneousList ex_3: 3 samples, 5 features
        |   Mpg   |   Displacement   |   Horsepower   |   Weight   |   Acceleration   |
        -------------------------------------------------------------------------------
        |    21.0 |              0.2 |           85.0 |     2587.0 |             16.0 |
        |    25.0 |             0.11 |           87.0 |     2672.0 |             17.5 |
        |    21.0 |            0.199 |           90.0 |     2648.0 |             15.0 |
        """
        sublists = []
        label_tags = sorted(list(map(str, set(self.labels).difference({-1}))))
        unic_labels = list(set(self.labels))
        for _ in range(self.n_clusters):
            sublists.append([])
        if -1 in self.labels:
            sublists.append([])
            label_tags.append("outliers")

        for idx, label in enumerate(self.labels):
            sublists[unic_labels.index(label)].append(self.dessia_objects[idx])

        new_dessia_objects = [HeterogeneousList(dessia_objects=sublist, name=self.name + f"_{label_tag}")
                              for label_tag, sublist in zip(label_tags, sublists)]

        return CategorizedList(new_dessia_objects,
                               list(set(self.labels).difference({-1})) + ([-1] if -1 in self.labels else []),
                               name=self.name + "_split")

    def _merge_sublists(self):
        merged_hlists = self.dessia_objects[0][:]
        merged_labels = [self.labels[0]] * len(merged_hlists)
        for dobject, label in zip(self.dessia_objects[1:], self.labels[1:]):
            merged_hlists.extend(dobject)
            merged_labels.extend([label] * len(dobject))
        plotted_clist = self.__class__(dessia_objects=merged_hlists.dessia_objects, labels=merged_labels)
        return plotted_clist

    def _tooltip_attributes(self):
        return self.common_attributes + ["Cluster Label"]

    def plot_data(self):
        if isinstance(self.dessia_objects[0], HeterogeneousList):
            plotted_clist = self._merge_sublists()
            return plotted_clist.plot_data()
        return HeterogeneousList.plot_data(self)

    def _plot_data_list(self):
        _plot_data_list = []
        for row, label in enumerate(self.labels):
            _plot_data_list.append({attr: self.matrix[row][col] for col, attr in enumerate(self.common_attributes)})
            _plot_data_list[-1]["Cluster Label"] = label
            # (label if label != -1 else "Excluded") plot_data "Excluded" -> NaN
        return _plot_data_list

    def _point_families(self):
        colormap = plt.cm.get_cmap('hsv', self.n_clusters + 1)(range(self.n_clusters + 1))
        point_families = []
        for i_cluster in range(self.n_clusters):
            color = Color(colormap[i_cluster][0], colormap[i_cluster][1], colormap[i_cluster][2])
            points_index = list(map(int, npy.where(npy.array(self.labels) == i_cluster)[0].tolist()))
            point_families.append(PointFamily(color, points_index, name="Cluster " + str(i_cluster)))

        if -1 in self.labels:
            color = LIGHTGREY
            points_index = list(map(int, npy.where(npy.array(self.labels) == -1)[0].tolist()))
            point_families.append(PointFamily(color, points_index, name="Excluded"))
        return point_families

    @classmethod
    def from_agglomerative_clustering(cls, data: HeterogeneousList, n_clusters: int = 2,
                                      affinity: str = 'euclidean', linkage: str = 'ward',
                                      distance_threshold: float = None, scaling: bool = False, name: str = ""):
        """
        Hierarchical clustering is a general family of clustering algorithms that
        build nested clusters by merging or splitting them successively.
        This hierarchy of clusters is represented as a tree (or dendrogram).
        The root of the tree is the unique cluster that gathers all the samples,
        the leaves being the clusters with only one sample. See the Wikipedia page
        for more details.

        The AgglomerativeClustering object performs a hierarchical clustering using
        a bottom up approach: each observation starts in its own cluster, and clusters
        are successively merged together. The linkage criteria determines the metric
        used for the merge strategy: Ward minimizes the sum of squared differences within all clusters.

        It is a variance-minimizing approach and in this sense is similar to the
        k-means objective function but tackled with an agglomerative hierarchical approach.
        Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
        Average linkage minimizes the average of the distances between all observations of pairs of clusters.
        Single linkage minimizes the distance between the closest observations of pairs of clusters.
        AgglomerativeClustering can also scale to large number of samples when it is used
        jointly with a connectivity matrix, but is computationally expensive when no connectivity
        constraints are added between samples: it considers at each step all the possible merges.

        See more : https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

        :param data: The future clustered data.
        :type data: List[DessiaObject]

        :param n_clusters:
            -------
            Number of wished clusters.

            Must be `None` if `distance_threshold` is not `None`
        :type n_clusters: `int`, `optional`, defaults to `2`

        :param affinity:
            -------
            Metric used to compute the linkage.
            Can be one of `['euclidean', 'l1', 'l2', 'manhattan', 'cosine', or 'precomputed']`.

            If linkage is `'ward'`, only `'euclidean'` is accepted.

            If `'precomputed'`, a distance matrix (instead of a similarity matrix) is needed as input for the \
                fit method.
        :type affinity: `str`, `optional`, defaults to `'euclidean'`

        :param linkage:
            --------
            |  Which linkage criterion to use. Can be one of `[‘ward’, ‘complete’, ‘average’, ‘single’]`
            |  The linkage criterion determines which distance to use between sets of observation.
            |  The algorithm will merge the pairs of cluster that minimize this criterion.
            |     - `'ward'` minimizes the variance of the clusters being merged.
            |     - `'average`' uses the average of the distances of each observation of the two sets.
            |     - `'complete`' or `'maximum`' linkage uses the maximum distances between all observations of the two \
                sets.
            |     - `'single`' uses the minimum of the distances between all observations of the two sets.
        :type linkage: `str`, `optional`, defaults to `'ward'`

        :param distance_threshold:
            --------
            The linkage distance above which clusters will not be merged.
            If not `None`, `n_clusters` must be `None`.
        :type distance_threshold: `float`, `optional`, defaults to `None`

        :param scaling:
            --------
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a CategorizedList that knows the data and their labels
        :rtype: CategorizedList

        """
        skl_cluster = cluster.AgglomerativeClustering(
            n_clusters=n_clusters, affinity=affinity, distance_threshold=distance_threshold, linkage=linkage)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_kmeans(cls, data: HeterogeneousList, n_clusters: int = 2, n_init: int = 10, tol: float = 1e-4,
                    scaling: bool = False, name: str = ""):
        """
        The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance,
        minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below).
        This algorithm requires the number of clusters to be specified. It scales well to large number
        of samples and has been used across a large range of application areas in many different fields.
        The k-means algorithm divides a set of samples into disjoint clusters , each described by the mean
        of the samples in the cluster. The means are commonly called the cluster “centroids”; note that
        they are not, in general, points from, although they live in the same space.
        The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster
        sum-of-squares criterion.

        See more : https://scikit-learn.org/stable/modules/clustering.html#k-means

        :param data: The future clustered data.
        :type data: List[DessiaObject]

        :param n_clusters:
            --------
            Number of wished clusters
        :type n_clusters: `int`, `optional`, defaults to `2`

        :param n_init:
            --------
            Number of time the k-means algorithm will be run with different centroid seeds.
            The final results will be the best output of n_init consecutive runs in terms of inertia.
        :type n_init: `int`, `optional`, defaults to `10`

        :param tol:
            --------
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two \
                consecutive iterations to declare convergence.
        :type tol: `float`, `optional`, defaults to `1e-4`

        :param scaling:
            --------
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a CategorizedList that knows the data and their labels
        :rtype: CategorizedList

        """
        skl_cluster = cluster.KMeans(n_clusters=n_clusters, n_init=n_init, tol=tol)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_dbscan(cls, data: HeterogeneousList, eps: float = 0.5, min_samples: int = 5, mink_power: float = 2,
                    leaf_size: int = 30, metric: str = "euclidean", scaling: bool = False, name: str = ""):
        """
        The DBSCAN algorithm views clusters as areas of high density separated by areas of low density.
        Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means
        which assumes that clusters are convex shaped. The central component to the DBSCAN is the concept
        of core samples, which are samples that are in areas of high density. A cluster is therefore a set
        of core samples, each close to each other (measured by some distance measure) and a set of non-core
        samples that are close to a core sample (but are not themselves core samples).
        There are two parameters to the algorithm, min_samples and eps, which define formally what we mean
        when we say dense. Higher min_samples or lower eps indicate higher density necessary to form a cluster.

        See more : https://scikit-learn.org/stable/modules/clustering.html#dbscan

        :param data: The future clustered data.
        :type data: List[DessiaObject]

        :param eps:
            --------
            The maximum distance between two samples for one to be considered as in the neighborhood \
            of the other. This is not a maximum bound on the distances of points within a cluster. This is the most \
            important DBSCAN parameter to choose appropriately for your data set and distance function
        :type eps: `float`, `optional`, defaults to `0.5`

        :param min_samples:
            --------
            The number of samples (or total weight) in a neighborhood for a point to be considered as \
            a core point. This includes the point itself
        :type min_samples: `int`, `optional`, defaults to 5

        :param mink_power:
            --------
            The power of the Minkowski metric to be used to calculate distance between points. \
            If `None`, then `mink_power=2` (equivalent to the Euclidean distance)
        :type mink_power: `float`, `optional`, defaults to `2`

        :param leaf_size:
            --------
            Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and \
            query, as well as the memory required to store the tree. The optimal value depends on the nature of the \
            problem
        :type leaf_size: `int`, `optional`, defaults to `30`

        :param metric:
            --------
            The metric to use when calculating distance between instances in a feature array. If metric is \
            a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for its \
            metric parameter. If metric is `'precomputed'`, X is assumed to be a distance matrix and must be square. \
            X may be a sparse graph, in which case only `'nonzero'` elements may be considered neighbors for DBSCAN.
        :type metric: `str`, or `callable`, default to `’euclidean’`


        :param scaling:
            --------
            Whether to scale the data or not before clustering.

            Formula is `scaled_x = ( x - mean )/standard_deviation`
        :type scaling: `bool`, `optional`, default to `False`

        :return: a CategorizedList that knows the data and their labels
        :rtype: CategorizedList

        """
        skl_cluster = cluster.DBSCAN(eps=eps, min_samples=min_samples, p=mink_power, leaf_size=leaf_size, metric=metric)
        skl_cluster = cls.fit_cluster(skl_cluster, data.matrix, scaling)
        return cls(data.dessia_objects, skl_cluster.labels_.tolist(), name=name)

    @classmethod
    def from_pareto_sheets(cls, h_list: HeterogeneousList, costs: List[List[float]], nb_sheets: int = 1):
        """
        Get successive pareto sheets (i.e. optimal points in a DOE for pre-computed costs) and put them in a
        `CategorizedList` where each label is the index of a pareto sheet.

        :param h_list:
            --------
            The HeterogeneousList in which to pick optimal points.
        :type h_list: `HeterogeneousList`

        :param costs:
            --------
            Pre-computed costs of `len(self)`. Can be multi-dimensional.
        :type costs: `List[List[float]]`, `n_samples x n_costs` or `n_costs x n_samples`

        :param nb_sheets:
            --------
            Number of pareto sheets to pick
        :type nb_sheets: `int`, `optional`, default to `1`

        :return: a CategorizedList where each element is labelled with its pareto_sheet. Elements outside a \
        pareto_sheet are labelled `n_sheets`
        :rtype: CategorizedList

        """
        labels = []
        dessia_objects = []
        pareto_sheets, non_optimal_points = h_list.pareto_sheets(costs, nb_sheets)
        for label, pareto_sheet in enumerate(pareto_sheets):
            labels.extend([label] * len(pareto_sheet))
            dessia_objects.extend(pareto_sheet)
        dessia_objects.extend(non_optimal_points)
        labels.extend([len(pareto_sheets)] * len(non_optimal_points))
        return cls(dessia_objects, labels)

    @staticmethod
    def fit_cluster(skl_cluster: cluster, matrix: List[List[float]], scaling: bool):
        if scaling:
            scaled_matrix = HeterogeneousList.scale_data(matrix)
        else:
            scaled_matrix = matrix
        skl_cluster.fit(scaled_matrix)
        return skl_cluster

# Function to implement, to find a good eps parameter for dbscan
# def nearestneighbors(self):
#     vectors = []
#     for machine in self.machines:
#         vector = machine.to_vector()
#         vectors.append(vector)
#     neigh = NearestNeighbors(n_neighbors=14)
#     vectors = StandardScaler().fit_transform(vectors)
#     nbrs = neigh.fit(vectors)
#     distances, indices = nbrs.kneighbors(vectors)
#     distances = npy.sort(distances, axis=0)
#     distances = distances[:, 1]
#     plt.plot(distances)
#     plt.show()
