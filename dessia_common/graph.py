#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:16:30 2021

@author: steven
"""
import igraph
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


def explore_tree_from_leaves(graph: igraph.Graph):
    if not graph.is_dag():
        raise NotImplementedError('Cycles in jsonpointers not handled')

    t_sorting = graph.topological_sorting()
    z = [graph.vs[x]['name'] for x in t_sorting]

    return list(reversed(z))