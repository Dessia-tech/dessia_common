#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:16:30 2021

@author: steven
"""

import networkx as nx


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


def explore_tree_from_leaves(graph: nx.DiGraph):
    if not nx.is_directed_acyclic_graph(graph):
        raise NotImplementedError('Cycles in jsonpointers not handled')

    return reversed(list(nx.topological_sort(graph)))


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
