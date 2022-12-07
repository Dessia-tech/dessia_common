Datatools
****

Dataset
===============

A Dataset is a container object for DessiaObjetcs.

It implements many features that can help engineers to explore the data contained in a list.
Dataset includes, among others, a plot_data method, filtering capabilities, data exploration features, metrics, statistics
and can be clustered into a ClusteredDataset to help in clustering similar data.

.. autoclass:: dessia_common.datatools.dataset.Dataset
   :members:

Sampler
===============

A Sampler is an object that allows to generate a DOE from a class and bounds for its attributes.

.. autoclass:: dessia_common.datatools.sampling.ClassSampler
   :members:

Clustered Dataset
===============

A Clustered Dataset is a container object for DessiaObjetcs.

It implements many features that can help engineers to explore clusters in data.
ClusteredDataset includes, among others, a plot_data method, user friendly clusters manipulation and metrics.

.. autoclass:: dessia_common.datatools.cluster.ClusteredDataset
   :members:

Modeling
===============

Modeling objects from sklearn.

.. automodule:: dessia_common.datatools.modeling
   :members:
