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
    exploration_order = []
    explored = {n: False for n in graph.nodes}
    number_nodes = graph.number_of_nodes()
    successors = {}

    # ns = 0
    while number_nodes:
        found_node = False
        # Finding a starting node
        for node in graph.nodes:
            if not explored[node]:
                neighbors_explored = True
                if node in successors:
                    node_successors = successors[node]
                else:
                    node_successors = list(graph.successors(node))
                    successors[node] = node_successors
                    # ns += 1

                for out_node in node_successors:
                    if not explored[out_node]:
                        neighbors_explored = False
                        break

                if neighbors_explored:
                    # Mark explored
                    explored[node] = True
                    exploration_order.append(node)
                    number_nodes -= 1
                    found_node = True
                    break
        if not found_node:
            raise ValueError('Can not find a node')
    return exploration_order
