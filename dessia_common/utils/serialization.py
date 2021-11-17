#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:25:54 2021

@author: steven
"""

import warnings
import inspect

import dessia_common as dc
import dessia_common.utils.types as dcty
from dessia_common.breakdown import get_in_object_from_path
import networkx as nx


def deserialize(serialized_element, sequence_annotation: str = 'List',
                global_dict=None, pointers_memo=None):#, enforce_pointers=False):
    if pointers_memo is None:
        pointers_memo = None
    
    if isinstance(serialized_element, dict):
        try:
            return dict_to_object(serialized_element, global_dict=global_dict,
                                  pointers_memo=pointers_memo)
        except TypeError:
            warnings.warn('specific dict_to_object of class {} should implement global_dict and pointers_memo arguments'.format(serialized_element.__class__.__name__), Warning)
            return dict_to_object(serialized_element)
    elif dcty.is_sequence(serialized_element):
        return deserialize_sequence(sequence=serialized_element,
                                    annotation=sequence_annotation,
                                    global_dict=global_dict,
                                    pointers_memo=pointers_memo)
    return serialized_element

def deserialize_sequence(sequence, annotation=None, global_dict=None, pointers_memo=None):
    # TODO: rename to deserialize sequence? Or is this a duplicate ?
    origin, args = dcty.unfold_deep_annotation(typing_=annotation)
    deserialized_sequence = []
    for elt in sequence:
        deserialized_element = deserialize(elt, args, global_dict=global_dict, pointers_memo=pointers_memo)
        deserialized_sequence.append(deserialized_element)
    if origin is tuple:
        # Keeping as a tuple
        return tuple(deserialized_sequence)
    return deserialized_sequence

def dict_to_object(dict_, class_=None, force_generic: bool = False,
                   global_dict=None, pointers_memo=None):
        
    if '$ref' in dict_ and dict_['$ref'] in pointers_memo:
        return pointers_memo[dict_['$ref']]
    
    class_argspec = None

    if pointers_memo is None:
        pointers_memo = {}

    if global_dict is None:
        global_dict = dict_
        # dc_ser.pointer_graph(global_dict)
        pointers_memo.update(dereference_jsonpointers(dict_))

        
        
    working_dict = dict_

    if class_ is None and 'object_class' in working_dict:
        class_ = dcty.get_python_class_from_class_name(working_dict['object_class'])


    if class_ is not None and hasattr(class_, 'dict_to_object'):
        different_methods = (class_.dict_to_object.__func__
                             is not dc.DessiaObject.dict_to_object.__func__)

        if different_methods and not force_generic:
            try:
                obj = class_.dict_to_object(dict_, global_dict=global_dict)
            except TypeError:
                warn_msg = 'specific dict_to_object of class {} should implement global_dict arguments'.format(class_.__name__)
                warnings.warn(warn_msg, Warning)
                obj = class_.dict_to_object(dict_)
            return obj

        if class_._init_variables is None:
            class_argspec = inspect.getfullargspec(class_)
            init_dict = {k: v for k, v in working_dict.items()
                         if k in class_argspec.args}
        else:
            init_dict = {k: v for k, v in working_dict.items()
                         if k in class_._init_variables}
        # TOCHECK Class method to generate init_dict ??
    else:
        init_dict = working_dict
        
    
    subobjects = {}
    memo = {}
    for key, value in init_dict.items():
        if class_argspec is not None and key in class_argspec.annotations:
            annotation = class_argspec.annotations[key]
        else:
            annotation = None
        
        subobjects[key] = deserialize(value, annotation, global_dict=global_dict, pointers_memo=pointers_memo)#, enforce_pointers=False)

    # if enforce_pointers:
    # subobjects = enforce_pointers(subobjects, dict_, subobjects)

    if class_ is not None:
        obj = class_(**subobjects)
    else:
        obj = subobjects
    
    return obj


def pointer_graph(value):
    nodes, edges = pointer_graph_elements(value)
    # print(nodes)
    # for e in edges:
    #     print('edge: ', e)
    # print(len(nodes), len(edges))
    # plt.S()
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # pos = nx.kamada_kawai_layout(graph)
    # nx.draw_networkx_nodes(graph, pos)
    # nx.draw_networkx_edges(graph, pos)
    # nx.draw_networkx_labels(graph, pos)
    # if 'object_class' in value:
    #     ax.set_title(value['object_class'])

    return graph
    

def dereference_jsonpointers(value):#, global_dict):
    graph = pointer_graph(value)
    
    pointers_memo = {}
    if '#' in graph.nodes:
        for _, ref in list(nx.bfs_edges(graph, '#'))[::-1]:
            # print(node1, node2)
            serialized_element = get_in_object_from_path(value, ref)
            pointers_memo[ref] = deserialize(serialized_element=serialized_element,
                                             global_dict=value, pointers_memo=pointers_memo)
            # print('\nref', ref, pointers_memo[ref])
    # print(pointers_memo.keys())
    return pointers_memo
            
   
#     if isinstance(value, (list, tuple)):
#         return dereference_jsonpointers_sequence(value, global_dict)
#     elif isinstance(value, dict):
#         return dereference_jsonpointers_dict(value, global_dict)
#     else:
#         return value
    
# def dereference_jsonpointers_dict(dict_, global_dict):
#     """
#     Dereference a dict_ by inserting dicts references (not objects!)
#     To be used before deserialization
#     """
#     if '$ref' in dict_:
#         path = dict_['$ref']
#         return get_in_object_from_path(global_dict, path)
#     else:
#         deref_dict = {}
#         for key, value in dict_.items():
#             deref_dict[key] = dereference_jsonpointers(value, global_dict)
#         return deref_dict
    

# def dereference_jsonpointers_sequence(sequence, global_dict):
#     deref_sequence = []
#     for element in sequence:
#         deref_sequence.append(dereference_jsonpointers(element, global_dict))
#     return deref_sequence




def pointer_graph_elements(value, path='#'):
    # edges = []
    # nodes = []

    if isinstance(value, dict):
        return pointer_graph_elements_dict(value, path)
    if dcty.isinstance_base_types(value):
        return [], []
    elif dcty.is_sequence(value):
        return pointer_graph_elements_sequence(value, path)
    else:
        print(value)
        raise ValueError(value)



def pointer_graph_elements_sequence(seq, path='#'):
    if isinstance(seq, str):
        raise ValueError


    edges = []
    nodes = []
    for ie, element in enumerate(seq):
        path_value = '{}/{}'.format(path, ie)
        value_nodes, value_edges = pointer_graph_elements(element, path=path_value)
        if value_edges:
            nodes.append(path_value)
            nodes.extend(value_nodes)
            
            edges.append((path, path_value))
            edges.extend(value_edges)


    return nodes, edges

def pointer_graph_elements_dict(dict_, path='#'):
    
    
    if '$ref' in dict_:
        return [path], [(path, dict_['$ref'])]
    
    edges = []
    nodes = []
    for key, value in dict_.items():
        path_value = '{}/{}'.format(path, key)
        value_nodes, value_edges = pointer_graph_elements(value, path=path_value)
        if value_edges:
            nodes.append(path_value)
            nodes.extend(value_nodes)
    
            edges.append((path, path_value))
            edges.extend(value_edges)

    return nodes, edges


# def enforce_pointers(object_, serialized_dict, global_object=None, memo=None):
#     """
#     Enforce python pointers with respect to jsonpointers in serialized_dict
#     Enforcing pointers aims at keeping same links accross the object
#     To use after naive deserialization with broken links
#     """
#     # print(object_, is_sequence(object_))
    
#     print('\n%%Enforce pointers', global_object)
#     if global_object is None:
#         raise ValueError('uuUUU')
#         global_object = object_
        
#     if memo is None:
#         memo = {}
        
#     if isinstance_base_types(object_):
#         return object_
#     elif is_sequence(object_):
#         return enforce_pointers_in_sequence(object_, serialized_dict, global_object=global_object, memo=memo)
#     elif isinstance(object_, dict):
#         return enforce_pointers_in_dict(object_, serialized_dict, global_object=global_object, memo=memo)
#     else:
#         return enforce_pointers_in_object(object_, serialized_dict, global_object=global_object, memo=memo)

# def enforce_pointers_in_sequence(seq, serialized_seq, global_object=None, memo=None):
#     # print('enforcing in ', object_, global_object)

#     # if global_object is None:
#     #     global_object = seq
        
#     if memo is None:
#         memo = {}


#     enforced_seq = []
#     for i, (seq_value, seq_serialized_value) in enumerate(zip(seq, serialized_seq)):
#         # seq[i] = choose_enforce_pointers(seq_value, seq_serialized_value, global_object))
#         enforced_seq.append(enforce_pointers(seq_value, seq_serialized_value,
#                                              global_object=global_object,
#                                              memo=memo))
#     return enforced_seq



# def enforce_pointers_in_dict(dict_, serialized_dict, global_object=None,
#                              memo=None):
#     if memo is None:
#         memo = {}

#     if '$ref' in serialized_dict:
#         return get_in_object_from_path(global_object, serialized_dict['$ref'])

#     dict_2 = {}
#     for key, serialized_value in serialized_dict.items():
#         if key in dict_:
#             object_value = dict_[key]
#             enforced_value = enforce_pointers(object_value, serialized_value, global_object=global_object)            
#             # TODO: Enforce key as well? how to do it 
#             dict_2[key] = enforced_value
#     return dict_2
    
    
# def enforce_pointers_in_object(object_, serialized_dict, global_object=None,
#                                memo=None):
#     """
#     handles objects
#     """
#     # print('\n\n## enforcing in ', object_, global_object)

#     if global_object is None:
#         global_object = object_
#         print('\n#enforcing top level ', object_)
        
#     if memo is None:
#         memo = {}

#     # object2 = object_.copy()
#     if '$ref' in serialized_dict:
#         return get_in_object_from_path(global_object, serialized_dict['$ref'])
        
#     for key, serialized_value in serialized_dict.items():
#         if hasattr(object_, key):
#             object_value = getattr(object_, key)
            
            
#             # if is_sequence(serialized_value):
#             #     enforced_value = []
#             #     for seq_value, seq_serialized_value in zip(object_value, serialized_value):
#             #         enforced_value.append(enforce_pointers(seq_value, seq_serialized_value, global_object=global_object))
                
#             # elif isinstance(serialized_value, dict):
#             #     if '$ref' in serialized_value:
#             #         path = serialized_value['$ref']
#             #         enforced_value = get_in_object_from_path(global_object, path)
#             #     else:
#             #         enforced_value = enforce_pointers_in_object(object_value, serialized_value, global_object=global_object)
#             # elif isinstance_base_types(object_value):
#             #     enforced_value = object_value
#             # # elif isinstance(serialized_value, type):
#             # #     pass
#             # else:
#             #     pass
#             #     # raise NotImplementedError(object_value, 'of type', type(object_value))
#             enforced_value = enforce_pointers(object_value, serialized_value, global_object=global_object)
#             # # Overriding value
#             setattr(object_, key, enforced_value)
#     return object_
