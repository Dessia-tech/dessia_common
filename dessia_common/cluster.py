
import cma
from typing import List
import numpy as npy
from time import sleep
import dessia_common.core as dc
import dessia_common.typings as dct
from sklearn import cluster, mixture


class ClusterResult(dc.DessiaObject):
    """
    Cluster object to instantiate and compute clusters on data
    """
    _standalone_in_db = True

    def __init__(self, data: List[dc.DessiaObject], labels: List[int], name: str = ''):
        dc.DessiaObject.__init__(self, name=name)
        self.data = data
        self.labels = labels
        
        
    @classmethod
    def fromAgglomerativeClustering(cls, data: List[dc.DessiaObject], 
                                    n_clusters: int = 2):#, affinity: str = 'euclidean', memory: str = None, 
                                    # connectivity=None, compute_full_tree='auto', linkage='ward', 
                                    # distance_threshold=None, compute_distances=False):
        
        skl_cluster = cluster.AgglomerativeClustering(n_clusters=n_clusters)
        # , affinity=affinity, memory=memory, 
        #                                               connectivity=connectivity, compute_full_tree=compute_full_tree, 
        #                                               linkage=linkage, distance_threshold=distance_threshold, compute_distances=compute_distances)
        
        skl_cluster.fit(cls.to_matrix(data))
        return cls(data, skl_cluster.labels_.tolist())
   
    
    @staticmethod
    def to_matrix(data: List[dc.DessiaObject]):
        if 'to_vector' not in dir(data[0]):
            raise NotImplementedError("\nDessiaObject must have a 'to_vector method'.")
            
        data_matrix = []
        for element in data:
            data_matrix.append(element.to_vector())
        return data_matrix
    
    
    @staticmethod 
    # Is it really pertinent to have a staticmethod for that since we will only call it when having a ClusterResult
    def data_to_clusters(data: List[dc.DessiaObject], labels: npy.ndarray):
        clusters_list = []
        for i in range(npy.max(labels) + 1):
            clusters_list.append([])
            
        for i, label in enumerate(labels):
            clusters_list[label].append(data[i])
            
        return clusters_list
    
    
    
    # def plot_data():
    #     1 couleur par cluster
    #     multidimensional_scaling()
        


# class DessiaCluster(dc.DessiaObject):
#     """
#     Cluster object to instantiate and compute clusters on data
#     """
#     _standalone_in_db = True
#     _non_serializable_attributes = ['cluster']

#     def __init__(self, method: str = '', hyperparameters: dict = {}, ignore_controls: bool = False, name: str = ''):
#         dc.DessiaObject.__init__(self, name=name)
#         self.method = (method if ignore_controls else self.check_method_name(method))
#         self.hyperparameters = (hyperparameters if ignore_controls else self.check_hyperparameters(hyperparameters))
#         self.cluster = self.init_cluster_object()
    
    
#     def set_hyperparameters(self, hyperparameters: dict = {}, **kwargs):
#         """
#         Update sklearn cluster hyperparameters with a dict, keywords or the two of them.
#         kwargs are stronger than hyperparameters :
#             - DessiaCluster.set_hyperparemeters({'n_clusters': 4}, n_clusters = 10) will set n_clusters = 10
        
#         Parameters
#         ----------
#         hyperparameters : dict, optional
#             dict of hyperparameters. The default is {}.
#         **kwargs : dict
#             keywords argument (declared as fun(required_args, key = keyword_arg).

#         """
#         if len(hyperparameters.keys()) != 0 and len(kwargs.keys()) == 0:
#             new_hyperparameters = hyperparameters
#         elif len(hyperparameters.keys()) == 0 and len(kwargs.keys()) != 0:
#             new_hyperparameters = kwargs
#         else:
#             new_hyperparameters = {**hyperparameters, **kwargs}
        
#         for key, value in new_hyperparameters.items():
#             setattr(self.cluster, key, value)
            
#         return
        
    
#     def check_hyperparameters(self, hyperparameters: dict):
#         users_param_names = list(hyperparameters.keys())
#         real_param_names = clustering_params()
#         for users_param in users_param_names:
#             if users_param not in real_param_names[self.method]:
#                 raise NotImplementedError("\n" + users_param + " is not a hyperparameter for " + self.method + " of scikit-learn.\n" +
#                                           "Expected hyperparameters are : " + str(real_param_names[self.method]) + ".")
        
#         return hyperparameters
            
    
#     def check_method_name(self, method: str):
#         clusterings_dict = clustering_methods()
#         if method.lower() in clusterings_dict['AffinityPropagation']:
#             new_method_name = 'AffinityPropagation'
            
#         elif method.lower() in clusterings_dict['AgglomerativeClustering']:
#             new_method_name = 'AgglomerativeClustering'
            
#         elif method.lower() in clusterings_dict['Birch']:
#             new_method_name = 'Birch'
            
#         elif method.lower() in clusterings_dict['DBSCAN']:
#             new_method_name = 'DBSCAN'
            
#         elif method.lower() in clusterings_dict['KMeans']:
#             new_method_name = 'KMeans'
            
#         elif method.lower() in clusterings_dict['MeanShift']:
#             new_method_name = 'MeanShift'
            
#         elif method.lower() in clusterings_dict['OPTICS']:
#             new_method_name = 'OPTICS'
            
#         elif method.lower() in clusterings_dict['SpectralClustering']:
#             new_method_name = 'SpectralClustering'

#         elif method.lower() in clusterings_dict['GaussianMixture']:
#             new_method_name = 'GaussianMixture'
#             raise NotImplementedError("GaussianMixture not implemented yet.")
            
#         return new_method_name
    
    
#     def init_cluster_object(self):
#         if self.method == 'AffinityPropagation':
#             cluster_object = cluster.AffinityPropagation(**self.hyperparameters)
            
#         elif self.method == 'AgglomerativeClustering':
#             cluster_object = cluster.AgglomerativeClustering(**self.hyperparameters)
            
#         elif self.method == 'Birch':
#             cluster_object = cluster.Birch(**self.hyperparameters)
            
#         elif self.method == 'DBSCAN':
#             cluster_object = cluster.DBSCAN(**self.hyperparameters)
            
#         elif self.method == 'KMeans':
#             cluster_object = cluster.KMeans(**self.hyperparameters)
            
#         elif self.method == 'MeanShift':
#             cluster_object = cluster.MeanShift(**self.hyperparameters)
            
#         elif self.method == 'OPTICS':
#             cluster_object = cluster.OPTICS(**self.hyperparameters)
            
#         elif self.method == 'SpectralClustering':
#             cluster_object = cluster.SpectralClustering(**self.hyperparameters)

#         elif self.method == 'GaussianMixture':
#             cluster_object = mixture.GaussianMixture(**self.hyperparameters)
            
#         return cluster_object
    
    
#     def fit(self, data):
#         """
#         Call the fit method of the choosen clustering method

#         Parameters
#         ----------
#         data : DessiaObject or list or npy.ndarray:
#                 - list of n_samples DessiaObjects : [DessiaObject_1, DessiaObject_2,...,DessiaObject_n_samples].
#                 - DessiaObjects in list must have a to_vector() method which transform it in a list of numerical values.
#                 - The size of the list generated with to_vector() is n_features
#                 - list case : list of n_samples list of length n_features
#                 - npy.ndarray case : matrix n_samples x n_features
#         """
#         if isinstance(data, dc.DessiaObject):
#             if 'to_matrix' in dir(data):
#                 self.cluster.fit(data.to_matrix())
#             else:
#                 raise NotImplementedError("\nDessiaObject must have a 'to_matrix method'.")
#         elif isinstance(data, list) or isinstance(data, npy.ndarray):
#             self.cluster.fit(self.to_matrix(data))
#         else:
#             raise NotImplementedError("\nType " + type(data) + " not implemented. Must be a DessiaObject or a list of list or a npy.ndarray.")
#         return
    
    
#     def to_matrix(self, data):
#         data_matrix = []
#         for element in data:
#             data_matrix.append(element.to_vector())
#         return data_matrix
            
    
#     def export_clusters(self, data):
#         labels = npy.unique(self.cluster.labels_)
#         for i, label in labels:
#             a=1
            
#         return
    
                                                    
            

# def clustering_methods():
#     """
#     Documentation : 
#         - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
#         - https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
#     """
#     return {'AffinityPropagation': 
#                 ['affinitypropagation', 'affinity', 'affinity_propagation', 'ap'],
#             'AgglomerativeClustering': 
#                 ['agg', 'agglomerative', 'agglomerativeclustering', 'ag', 'aglomerative', 'aglomerativeclustering', 'agglomerative_clustering'], 
#             'Birch': 
#                 ['birch'],
#             'DBSCAN': 
#                 ['dbscan', 'db', 'db_scan', 'scandb'],
#             'KMeans': 
#                 ['kmeans', 'km', 'k_means'],
#             'MeanShift': 
#                 ['meanshift', 'ms', 'mean_shift'],
#             'OPTICS': 
#                 ['optics', 'optic'],
#             'SpectralClustering':
#                 ['spectralclustering', 'spectral_clustering', 'sc', 'spectral'],
#             'GaussianMixture': 
#                 ['gaussianmixture', 'mixture', 'gaussian', 'gm']
#             }
        

# def clustering_params():
#     """
#     Documentation : 
#         - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
#         - https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
#     """
#     return {'AffinityPropagation': 
#                 ['damping', 'max_iter', 'convergence_iter', 'copy', 'preference', 'affinity', 'verbose', 'random_state'],
#             'AgglomerativeClustering': 
#                 ['n_clusters', 'affinity', 'memory', 'connectivity', 'compute_full_tree', 'linkage', 'distance_threshold', 'compute_distances'], 
#             'Birch': 
#                 ['threshold', 'branching_factor', 'n_clusters', 'compute_labels', 'copy'],
#             'DBSCAN': 
#                 ['eps', 'min_samples', 'metric', 'metric_params', 'algorithm', 'leaf_size', 'p', 'n_jobs'],
#             'KMeans': 
#                 ['n_clusters', 'init', 'n_init', 'max_iter', 'tol', 'verbose', 'random_state', 'copy_x', 'algorithm'],
#             'MeanShift': 
#                 ['bandwidth', 'seeds', 'bin_seeding', 'min_bin_freq', 'cluster_all', 'n_jobs', 'max_iter'],
#             'OPTICS': 
#                 ['min_samples', 'max_eps', 'metric', 'p', 'metric_params', 'cluster_method', 'eps', 'xi', 'predecessor_correction', 
#                  'min_cluster_size', 'algorithm', 'leaf_size', 'memory', 'n_jobs'],
#             'SpectralClustering':
#                 ['n_clusters', 'eigen_solver', 'n_components', 'random_state', 'n_init', 'gamma', 'affinity', 'n_neighbors', 'eigen_tol', 
#                  'assign_labels', 'degree', 'coef0', 'kernel_params', 'n_jobs', 'verbose'],
#             'GaussianMixture': 
#                 ['n_components', 'covariance_type', 'tol', 'reg_covar', 'max_iter', 'n_init', 'init_params', 'weights_init', 'means_init', 
#                  'precisions_init', 'random_state', 'warm_start', 'verbose', 'verbose_interval']
#             }