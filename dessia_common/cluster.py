
from typing import List
import numpy as npy
import dessia_common.core as dc
from sklearn import cluster


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
    def fromAgglomerativeClustering(cls, data: List[dc.DessiaObject], n_clusters: int = 2, 
                                    affinity: str = 'euclidean', distance_threshold: float = None):
        
        skl_cluster = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, 
                                                      distance_threshold=distance_threshold)
        skl_cluster.fit(cls.to_matrix(data))
        return cls(data, skl_cluster.labels_.tolist())
    
    
    @classmethod
    def fromKMeans(cls, data: List[dc.DessiaObject], n_clusters: int = 2, 
                   n_init: int = 10, tolerance: float = 1e-4):
        
        skl_cluster = cluster.KMeans(n_clusters=n_clusters, n_init=n_init, tol=tolerance)
        skl_cluster.fit(cls.to_matrix(data))
        return cls(data, skl_cluster.labels_.tolist())
    
    
    @classmethod
    def fromDBSCAN(cls, data: List[dc.DessiaObject], eps: float = 0.5, min_samples: int = 5, 
                   norm_number: float = 2, leaf_size: int = 30):
        
        skl_cluster = cluster.DBSCAN(eps=eps, min_samples=min_samples, p=norm_number, leaf_size=leaf_size)
        skl_cluster.fit(cls.to_matrix(data))
        if npy.max(skl_cluster.labels_) == -1:
            raise NotImplementedError("\nAll labels are -1 valued which means DBSCAN did not add any element to any cluster.\n" +
                                      "Try to change eps paramerer.")
            
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