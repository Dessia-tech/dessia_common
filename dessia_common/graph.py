#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:16:30 2021

@author: steven
"""

# import random
import networkx as nx


# class GraphExplorer:

#     def __init__(self, graph: nx.DiGraph):
#         self.graph = graph
#         self.exploration_order = []
#         self.number_nodes = self.graph.number_of_nodes()
#         self.successors = {}

#         self.remaining_nodes = list(self.graph.nodes)
#         self.explored = {n: False for n in self.remaining_nodes}

#     def are_neighbors_explored(self, node):
#         try:
#             node_successors = self.successors[node]
#         except KeyError:
#             node_successors = list(self.graph.successors(node))
#             self.successors[node] = node_successors

#         for out_node in node_successors:
#             if not self.explored[out_node]:
#                 return False
#         return True

#     def add_node_to_order(self, node):
#         self.explored[node] = True
#         self.number_nodes -= 1
#         self.exploration_order.append(node)
#         self.remaining_nodes.remove(node)

#     def add_node_to_order_with_index(self, node, index=None):
#         self.explored[node] = True
#         self.number_nodes -= 1
#         self.exploration_order.append(node)
#         del self.remaining_nodes[index]

#     def explore_tree_from_leaves(self):
#         while self.number_nodes:
#             found_node = False
#             # Regular find of a node
#             for node_index, node in enumerate(self.remaining_nodes):
#                 if self.are_neighbors_explored(node):
#                     # Mark explored
#                     self.add_node_to_order_with_index(node, node_index)
#                     found_node = True
#                     current_node = node
#                     break

#             if found_node:
#                 parent_found = True
#                 while parent_found:
#                     # Searching if parents are unlocked
#                     parent_found = False
#                     for parent in self.graph.predecessors(current_node):
#                         if self.are_neighbors_explored(parent):
#                             self.add_node_to_order(parent)
#                             current_node = parent
#                             parent_found = True

#             else:
#                 raise ValueError('Can not find a node')
#         return self.exploration_order


def explore_tree_from_leaves(graph):
    if not nx.is_directed_acyclic_graph(graph):
        raise NotImplementedError('Cycles in jsonpointers not handled')

    return list(nx.topological_sort(graph.reverse()))


def cut_tree_final_branches(graph: nx.DiGraph):

    number_node_removed = 1
    while number_node_removed:
        nodes_to_delete = []
        # number_node_removed = 0
        for node in graph.nodes():
            if graph.out_degree(node) == 0:
                # It is a leaf
                degree_in = graph.in_degree(node)
                current_node = node
                while degree_in == 1:
                    nodes_to_delete.append(current_node)
                    current_node = graph.predecessors(current_node)
                    degree_in = graph.in_degree(current_node)
        number_node_removed = len(nodes_to_delete)
        if number_node_removed:
            new_nodes = [n for n in graph.nodes if n not in nodes_to_delete]
            graph = nx.subgraph(graph, new_nodes)

    return graph


def extract_region(networkx_graph: nx.Graph, nodes, distance: int = 5):
    """
    Extract all nodes that are at distance from node
    """

    region_nodes = nodes[:]
    for node in nodes:
        if node not in networkx_graph:
            raise ValueError(f'node {node} is not in graph')
        succs = list(nx.dfs_successors(networkx_graph, source=node, depth_limit=distance))
        preds = list(nx.dfs_predecessors(networkx_graph, source=node, depth_limit=distance))
        ancs = list(nx.ancestors(networkx_graph, source=node))
        print(f'found {len(succs)} successors and {len(preds)} predecesors {len(ancs)} ancestors')
        region_nodes.extend(succs)
        region_nodes.extend(preds)
        region_nodes.extend(ancs)

    return networkx_graph.subgraph(region_nodes)
