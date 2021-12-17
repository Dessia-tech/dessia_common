#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:25:54 2021

@author: steven
"""

import warnings
import inspect

import dessia_common as dc
import dessia_common.errors as dc_err
import dessia_common.utils.types as dcty
from dessia_common.graph import explore_tree_from_leaves#, cut_tree_final_branches
from dessia_common.breakdown import get_in_object_from_path
import networkx as nx

def serialize_with_pointers(deserialized_element, memo=None, path='#'):
    if memo is None:
        memo = {}
    if isinstance(deserialized_element, dc.DessiaObject):
        try:
            try:
                serialized = deserialized_element.to_dict(use_pointers=True, memo=memo, path=path)
            except TypeError:
                serialized = deserialized_element.to_dict()
                
        except TypeError:
            # warnings.warn('specific to_dict should implement memo and path arguments', Warning)
            serialized, memo = serialize_dict_with_pointers(deserialized_element.to_dict(), memo, path)
            
    elif isinstance(deserialized_element, dict):
        serialized, memo = serialize_dict_with_pointers(deserialized_element, memo, path)
    elif dcty.is_sequence(deserialized_element):
        serialized, memo = serialize_sequence_with_pointers(deserialized_element, memo, path)
    else:
        serialized = deserialized_element
    return serialized, memo


def serialize_dict_with_pointers(dict_, memo, path):
        
    serialized_dict = {}
    dict_attrs_keys = []
    seq_attrs_keys = []
    for key, value in dict_.items():
        value_path = '{}/{}'.format(path, key)
        if hasattr(value, 'to_dict'):
            # object
            if value in memo:
                serialized_dict[key] = {"$ref": memo[value]}
            else:
                try:
                    serialized_dict[key] = value.to_dict(use_pointers=True, path=value_path, memo=memo)
                except TypeError:
                    # warnings.warn('specific to_dict should implement memo and path arguments', Warning)
                    serialized_dict[key] = value.to_dict()
                memo[value] = value_path
        elif isinstance(value, dict):
            dict_attrs_keys.append(key)
        elif isinstance(value, (list, tuple)):
            seq_attrs_keys.append(key)
        else:
            if not dcty.is_jsonable(value):
                msg = 'Attribute {} of value {} is not json serializable'
                raise dc_err.SerializationError(msg.format(key, value))
            serialized_dict[key] = value
        
    # Handle seq & dicts afterwards
    for key in seq_attrs_keys:
        value_path = '{}/{}'.format(path, key)
        serialized_dict[key], memo = serialize_sequence_with_pointers(dict_[key], memo=memo, path=value_path)

    for key in dict_attrs_keys:
        value_path = '{}/{}'.format(path, key)
        serialized_dict[key], memo = serialize_dict_with_pointers(dict_[key], memo=memo, path=value_path)
    return serialized_dict, memo


def serialize_sequence_with_pointers(seq, memo, path):
    serialized_sequence = []
    for ival, value in enumerate(seq):
        value_path = '{}/{}'.format(path, ival)
        if hasattr(value, 'to_dict'):
            if value in memo:
                serialized_value = {"$ref": memo[value]}
            else:
                try:
                    serialized_value = value.to_dict(use_pointers=True, path=value_path, memo=memo)
                except TypeError:
                    # warnings.warn('specific to_dict should implement memo and path arguments', Warning)
                    serialized_value = value.to_dict()
                memo[value] = value_path
            serialized_sequence.append(serialized_value)
        elif isinstance(value, dict):
            serialized_value, memo = serialize_dict_with_pointers(value, memo=memo, path=value_path)
            serialized_sequence.append(serialized_value)
        elif isinstance(value, (list, tuple)):
            serialized_value, memo = serialize_sequence_with_pointers(value, memo=memo, path=value_path)
            serialized_sequence.append(serialized_value)
        else:
            serialized_sequence.append(value)
    return serialized_sequence, memo



def deserialize(serialized_element, sequence_annotation: str = 'List',
                global_dict=None, pointers_memo=None, path='#'):#, enforce_pointers=False):
    # if pointers_memo is None:
    #     pointers_memo = {}
    if pointers_memo is not None:
        if path in pointers_memo:
            return pointers_memo[path]
    
    if isinstance(serialized_element, dict):
        try:
            return dict_to_object(serialized_element, global_dict=global_dict,
                                  pointers_memo=pointers_memo,
                                  path=path)
        except TypeError:
            # warnings.warn('specific dict_to_object of class {}'
            #               ' should implement global_dict and'
            #               ' pointers_memo arguments'.format(serialized_element.__class__.__name__),
            #               Warning)
            return dict_to_object(serialized_element)
    elif dcty.is_sequence(serialized_element):
        return deserialize_sequence(sequence=serialized_element,
                                    annotation=sequence_annotation,
                                    global_dict=global_dict,
                                    pointers_memo=pointers_memo,
                                    path=path)
    return serialized_element

def deserialize_sequence(sequence, annotation=None,
                         global_dict=None, pointers_memo=None,
                         path='#'):
    # TODO: rename to deserialize sequence? Or is this a duplicate ?
    origin, args = dcty.unfold_deep_annotation(typing_=annotation)
    deserialized_sequence = []
    for ie, elt in enumerate(sequence):
        path_elt = '{}/{}'.format(path, ie)
        deserialized_element = deserialize(elt, args,
                                           global_dict=global_dict,
                                           pointers_memo=pointers_memo,
                                           path=path_elt)
        deserialized_sequence.append(deserialized_element)
    if origin is tuple:
        # Keeping as a tuple
        return tuple(deserialized_sequence)
    return deserialized_sequence

def dict_to_object(dict_, class_=None, force_generic: bool = False,
                   global_dict=None, pointers_memo=None, path='#'):
        
    
    if '$ref' in dict_:
        # and dict_['$ref'] in pointers_memo:
        # print(dict_['$ref'])
        # print('This is a ref', path, pointers_memo[dict_['$ref']])
        return pointers_memo[dict_['$ref']]
    
    class_argspec = None

    if pointers_memo is None:
        pointers_memo = {}

    if global_dict is None:
        global_dict = dict_
        pointers_memo.update(dereference_jsonpointers(dict_))
        
        
    working_dict = dict_

    if class_ is None and 'object_class' in working_dict:
        class_ = dcty.get_python_class_from_class_name(working_dict['object_class'])


    if class_ is not None and hasattr(class_, 'dict_to_object'):
        different_methods = (class_.dict_to_object.__func__
                             is not dc.DessiaObject.dict_to_object.__func__)

        if different_methods and not force_generic:
            try:
                obj = class_.dict_to_object(dict_,
                                            global_dict=global_dict,
                                            pointers_memo=pointers_memo)
            except TypeError:
                # warn_msg = 'specific dict_to_object of class {} should implement global_dict arguments'.format(class_.__name__)
                # warnings.warn(warn_msg, Warning)
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
    for key, value in init_dict.items():
        if class_argspec is not None and key in class_argspec.annotations:
            annotation = class_argspec.annotations[key]
        else:
            annotation = None
        
        key_path = '{}/{}'.format(path, key)
        if key_path in pointers_memo:
            subobjects[key] = pointers_memo[key_path]
        else:
            subobjects[key] = deserialize(value, annotation,
                                          global_dict=global_dict,
                                          pointers_memo=pointers_memo,
                                          path=key_path)#, enforce_pointers=False)

    if class_ is not None:
        obj = class_(**subobjects)
    else:
        obj = subobjects
    
    return obj

def find_references(value, path='#'):
    if isinstance(value, dict):
        return find_references_dict(value, path)
    if dcty.isinstance_base_types(value):
        return []
    elif dcty.is_sequence(value):
        return find_references_sequence(value, path)
    else:
        raise ValueError(value)


def find_references_sequence(seq, path='#'):
    if isinstance(seq, str):
        raise ValueError


    references = []
    for ie, element in enumerate(seq):
        path_value = '{}/{}'.format(path, ie)
        references.extend(find_references(element, path=path_value))
        # if value_nodes or value_edges:
        
    return references

def find_references_dict(dict_, path='#'):
    
    if '$ref' in dict_:
        return [(path, dict_['$ref'])]
    
    references = []
    for key, value in dict_.items():
        if not dcty.isinstance_base_types(value):
            path_value = '{}/{}'.format(path, key)
            references.extend(find_references(value, path=path_value))
    return references


def pointer_graph(value):

    nodes = set()    
    edges = set()
    # print(find_references(value))
    for path, reference in find_references(value):
        segments = path.split('/')
        nodes.add(segments[0])
        previous_node = segments[0]
        for s in segments[1:]:
            node = '{}/{}'.format(previous_node, s)
            nodes.add(node)            
            edges.add((previous_node, node))
            previous_node = node
        edges.add((path, reference))
        # print(path,'->', reference)
            
    
            
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
        

    return graph
    

def dereference_jsonpointers(value):#, global_dict):
    graph = pointer_graph(value)
    
    pointers_memo = {}
    if '#' in graph.nodes:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            for cycle in cycles:
                print(cycle)
            raise NotImplementedError('Cycles in ref not handled')
            
        order = list(explore_tree_from_leaves(graph))
        if '#' in order:
            order.remove('#')
            
        # for ref in order:
        #     print('ref', ref)

        for ref in order:
            serialized_element = get_in_object_from_path(value, ref)
            # print(serialized_element)
            # try:
            pointers_memo[ref] = deserialize(serialized_element=serialized_element,
                                     global_dict=value,
                                     pointers_memo=pointers_memo,
                                     path=ref)
            # except:
            #     print(ref)
            #     # print('\n', serialized_element)
            #     # print('\n\n', pointers_memo)
            #     return serialized_element, pointers_memo
            #     raise RuntimeError('jjj')
            # print('\nref', ref, pointers_memo[ref])
    # print(pointers_memo.keys())
    return pointers_memo
            




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
        raise ValueError(value)



def pointer_graph_elements_sequence(seq, path='#'):
    if isinstance(seq, str):
        raise ValueError


    edges = []
    nodes = []
    for ie, element in enumerate(seq):
        path_value = '{}/{}'.format(path, ie)
        value_nodes, value_edges = pointer_graph_elements(element, path=path_value)
        # if value_nodes or value_edges:
        
        nodes.append(path_value)
        nodes.extend(value_nodes)
        
        edges.append((path, path_value))
        edges.extend(value_edges)


    return nodes, edges

def pointer_graph_elements_dict(dict_, path='#'):
    
    
    if '$ref' in dict_:
        return [path, dict_['$ref']], [(path, dict_['$ref'])]
    
    edges = []
    nodes = []
    for key, value in dict_.items():
        if not dcty.isinstance_base_types(value):
            path_value = '{}/{}'.format(path, key)
            value_nodes, value_edges = pointer_graph_elements(value, path=path_value)
            # if value_nodes or value_edges:        
            nodes.append(path_value)
            nodes.extend(value_nodes)
        
            edges.append((path, path_value))
            edges.extend(value_edges)

    return nodes, edges


def pointers_analysis(obj):
    if isinstance(obj, dict):
        dict_ = obj
    else:
        dict_ = obj.to_dict()
    
    class_number = {}
    composed_by = {}
    class_from_path = {}
    graph = pointer_graph(dict_)
    for path1, path2 in graph.edges():
        if path1 != '#':
            # print(path1, path2)
            if path2 in class_from_path:
                val2_class = class_from_path[path2]
            else:
                val2 = get_in_object_from_path(obj, path2)
                val2_class = dcty.full_classname(val2)
                # val2_class = val2['object_class']
                class_from_path[path2] = val2_class
    
            # if 'object_class' in val2:
            # val2_class = val2['object_class']
            class_number[val2_class] = class_number.get(val2_class, 0) + 1
    
            
            if path1 in class_from_path:
                val1_class = class_from_path[path1]
            else:
                val1 = get_in_object_from_path(obj, path1)
                val1_class = dcty.full_classname(val1)
                # val1_class = val1['object_class']
                class_from_path[path1] = val1_class
    
            
            if val1_class != val2_class:
                if not val2_class in composed_by:
                    composed_by[val2_class] = {}
                
                if not val1_class in composed_by[val2_class]:
                    composed_by[val2_class][val1_class] = 1
                else:
                    composed_by[val2_class][val1_class] += 1
                
    return class_number, composed_by
                
    