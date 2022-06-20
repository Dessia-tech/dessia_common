"""
Library for building clusters on data.
"""
from typing import List

import numpy as npy
from sklearn import cluster, manifold

import matplotlib.pyplot as plt

import plot_data
import dessia_common.core as dc


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
    def from_agglomerative_clustering(cls, data: List[dc.DessiaObject], n_clusters: int = 2, 
                                    affinity: str = 'euclidean', distance_threshold: float = None):
        
        skl_cluster = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, 
                                                      distance_threshold=distance_threshold)
        skl_cluster.fit(cls.to_matrix(data))
        return cls(data, skl_cluster.labels_.tolist())
    
    
    @classmethod
    def from_kmeans(cls, data: List[dc.DessiaObject], n_clusters: int = 2, 
                   n_init: int = 10, tolerance: float = 1e-4):
        
        skl_cluster = cluster.KMeans(n_clusters=n_clusters, n_init=n_init, tol=tolerance)
        skl_cluster.fit(cls.to_matrix(data))
        return cls(data, skl_cluster.labels_.tolist())
    
    
    @classmethod
    def from_dbscan(cls, data: List[dc.DessiaObject], eps: float = 0.5, min_samples: int = 5, 
                   norm_number: float = 2, leaf_size: int = 30):
        
        skl_cluster = cluster.DBSCAN(eps=eps, min_samples=min_samples, p=norm_number, leaf_size=leaf_size)
        skl_cluster.fit(cls.to_matrix(data))
        
        if npy.max(skl_cluster.labels_) == -1:
            skl_cluster.labels_ += 1
            
        # if npy.max(skl_cluster.labels_) == -1:
        #     raise ValueError("\nAll labels are -1 valued which means DBSCAN 
        # did not add any element to any cluster.\n" +
        #                      "Try to change 'eps' hyperparamerer.")

        return cls(data, skl_cluster.labels_.tolist())
   
    
    @staticmethod
    def to_matrix(data: List[dc.DessiaObject]):
        if 'to_vector' not in dir(data[0]):
            raise NotImplementedError("\nDessiaObject must have a 'to_vector' method'.")
            
        data_matrix = []
        for element in data:
            data_matrix.append(element.to_vector())
        return npy.array(data_matrix)
    
    
    @staticmethod 
    # Is it really pertinent to have a staticmethod for that since we will only call it when having a ClusterResult
    def data_to_clusters(data: List[dc.DessiaObject], labels: npy.ndarray):
        clusters_list = []
        for i in range(npy.max(labels) + 1):
            clusters_list.append([])
            
        for i, label in enumerate(labels):
            clusters_list[label].append(data[i])
            
        return clusters_list
        
    
    def check_dimensionality(self, data: List[dc.DessiaObject]):
        _, singular_values, _ = npy.linalg.svd(self.to_matrix(data))
        normed_singular_values = singular_values/npy.sum(singular_values)
        plt.figure()
        plt.semilogy(normed_singular_values, linestyle = 'None', marker = 'o')
        plt.grid()
    
    
    def plot_data(self):
        n_clusters = npy.max(self.labels) + 1
        encoding_mds = manifold.MDS(metric = True, n_jobs = -1, n_components = 2 )
        matrix_mds = encoding_mds.fit_transform(self.to_matrix(self.data))
                
        elements = []
        for i in range(len(matrix_mds)):
            elements.append({"X_MDS": matrix_mds[i, 0].tolist(), 
                             "Y_MDS": matrix_mds[i, 1]})
            
        dataset_list = []
        for i in range(n_clusters):
            dataset_list.append([])
        for i, label in enumerate(self.labels):
            dataset_list[label].append({"X_MDS": matrix_mds[i, 0].tolist(), 
                                        "Y_MDS": matrix_mds[i, 1]})

        cmp_f = plt.cm.get_cmap('jet', n_clusters)(range(n_clusters))
        edge_style = plot_data.EdgeStyle(line_width = 0.0001)
        for i in range(n_clusters):
            color = plot_data.colors.Color(cmp_f[i][0], cmp_f[i][1], cmp_f[i][2])
            point_style = plot_data.PointStyle(color_fill=color, color_stroke=color)
            dataset_list[i] = plot_data.Dataset(elements=dataset_list[i], 
                                                edge_style=edge_style, 
                                                point_style=point_style)
        
        scatter_plot = plot_data.Graph2D(x_variable = "X_MDS", 
                                         y_variable = "Y_MDS", 
                                         graphs=dataset_list)
        
        return plot_data.plot_canvas(plot_data_object=scatter_plot, debug_mode=True)
    
