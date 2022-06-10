#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serialization Tools

"""

from ast import literal_eval
import warnings
import inspect
import collections
from typing import get_origin, get_args, Union, Any, TextIO, BinaryIO
import dessia_common as dc
import dessia_common.errors as dc_err
from dessia_common.files import StringFile, BinaryFile
import dessia_common.utils.types as dcty
from dessia_common.typings import InstanceOf
from dessia_common.graph import explore_tree_from_leaves  # , cut_tree_final_branches
from dessia_common.breakdown import get_in_object_from_path
import networkx as nx


def serialize_dict(dict_):
    """
    Serialize a dict into a dict (values are serialized)
    """
    serialized_dict = {}
    for key, value in dict_.items():
        serialized_dict[key] = serialize(value)
    return serialized_dict


def serialize_sequence(seq):
    """
    Serialize a sequence (list or sequence) into a list of dicts
    """
    serialized_sequence = []
    for value in seq:
        serialized_sequence.append(serialize(value))
    return serialized_sequence


def serialize(value):
    """
    Main function for serialization without pointers
    Calls recursively itself serialize_sequence and serialize_dict
    """
    if isinstance(value, dc.DessiaObject):
        try:
            serialized_value = value.to_dict(use_pointers=False)
        except TypeError:
            warnings.warn(f'specific to_dict of class {value.__class__} '
                          'should implement use_pointers, memo and path arguments', Warning)
            serialized_value = value.to_dict()
    elif isinstance(value, dict):
        serialized_value = serialize_dict(value)
    elif dcty.is_sequence(value):
        serialized_value = serialize_sequence(value)
    elif isinstance(value, (BinaryFile, StringFile)):
        serialized_value = value
    elif isinstance(value, type) or dcty.is_typing(value):
        return dcty.serialize_typing(value)
    else:
        if not dcty.is_jsonable(value):
            msg = f'Element of value {value} is not json serializable'
            raise dc_err.SerializationError(msg)
        serialized_value = value
    return serialized_value


def serialize_with_pointers(value, memo=None, path='#'):
    """
    Main function for serialization with pointers
    """
    if memo is None:
        memo = {}
    if isinstance(value, dc.DessiaObject):
        if value in memo:
            return {'$ref': memo[value]}, memo
        try:
            serialized = value.to_dict(use_pointers=True, memo=memo, path=path)

        except TypeError:
            warnings.warn('specific to_dict should implement use_pointers, memo and path arguments', Warning)
            serialized = value.to_dict()
        memo[value] = path
    elif hasattr(value, 'to_dict'):
        serialized = value.to_dict()
    elif isinstance(value, dict):
        serialized, memo = serialize_dict_with_pointers(value, memo, path)
    elif dcty.is_sequence(value):
        serialized, memo = serialize_sequence_with_pointers(value, memo, path)
    elif isinstance(value, (BinaryFile, StringFile)):
        serialized = value
    else:
        if not dcty.is_jsonable(value):
            msg = f'Element of value {value} is not json serializable'
            raise dc_err.SerializationError(msg)
        serialized = value
    return serialized, memo


def serialize_dict_with_pointers(dict_, memo, path):
    '''
    Serialize a dict recursively with jsonpointers using a memo dict at a given path of the top level object
    '''
    serialized_dict = {}
    dict_attrs_keys = []
    seq_attrs_keys = []
    other_keys = []
    # Detecting type of keys
    for key, value in dict_.items():
        value_path = f'{path}/{key}'
        if isinstance(value, dict):
            dict_attrs_keys.append(key)
        elif dcty.is_sequence(value):
            seq_attrs_keys.append(key)
        else:
            other_keys.append(key)

    for key in other_keys:
        value_path = f'{path}/{key}'
        serialized_dict[key], memo = serialize_with_pointers(dict_[key], memo=memo, path=value_path)
    # Handle seq & dicts afterwards
    for key in seq_attrs_keys:
        value_path = f'{path}/{key}'
        serialized_dict[key], memo = serialize_sequence_with_pointers(dict_[key], memo=memo, path=value_path)

    for key in dict_attrs_keys:
        value_path = f'{path}/{key}'
        serialized_dict[key], memo = serialize_dict_with_pointers(dict_[key], memo=memo, path=value_path)
    return serialized_dict, memo


def serialize_sequence_with_pointers(seq, memo, path):
    '''
    Serialize a sequence (list or tuple) using jsonpointers
    '''
    serialized_sequence = []
    for ival, value in enumerate(seq):
        value_path = '{}/{}'.format(path, ival)
        serialized_value, memo = serialize_with_pointers(value, memo, path=value_path)
        serialized_sequence.append(serialized_value)

    return serialized_sequence, memo


def deserialize(serialized_element, sequence_annotation: str = 'List',
                global_dict=None, pointers_memo=None, path: str = '#'):
    """
    Main function for deserialization, handle pointers
    """

    if pointers_memo is not None:
        if path in pointers_memo:
            return pointers_memo[path]

    if isinstance(serialized_element, dict):
        # try:
        return dict_to_object(serialized_element, global_dict=global_dict,
                              pointers_memo=pointers_memo,
                              path=path)
        # except TypeError:
        #     warnings.warn(f'specific dict_to_object of class {serialized_element.__class__.__name__}'
        #                   ' should implement global_dict and'
        #                   ' pointers_memo arguments',
        #                   Warning)
        #     return dict_to_object(serialized_element)
    if dcty.is_sequence(serialized_element):
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
        path_elt = f'{path}/{ie}'
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
    """
    Transform a dict to an object
    """

    class_argspec = None

    if global_dict is None or pointers_memo is None:
        global_dict = dict_

        if pointers_memo is None:
            pointers_memo = {}

        pointers_memo.update(dereference_jsonpointers(dict_))

    if '$ref' in dict_:
        return pointers_memo[dict_['$ref']]

    if class_ is None and 'object_class' in dict_:
        class_ = dcty.get_python_class_from_class_name(dict_['object_class'])

    # Create init_dict
    if class_ is not None and hasattr(class_, 'dict_to_object'):
        different_methods = (class_.dict_to_object.__func__ is not dc.DessiaObject.dict_to_object.__func__)

        if different_methods and not force_generic:
            try:
                obj = class_.dict_to_object(dict_,
                                            global_dict=global_dict,
                                            pointers_memo=pointers_memo,
                                            path=path)
            except TypeError:
                warnings.warn(f'specific to_dict of class {class_.__name__} '
                              'should implement use_pointers, memo and path arguments', Warning)
                obj = class_.dict_to_object(dict_)

            return obj

        class_argspec = inspect.getfullargspec(class_)
        init_dict = {k: v for k, v in dict_.items() if k in class_argspec.args + class_argspec.kwonlyargs}
        # TOCHECK Class method to generate init_dict ??
    else:
        init_dict = dict_

    subobjects = {}
    for key, value in init_dict.items():
        if class_argspec is not None and key in class_argspec.annotations:
            annotation = class_argspec.annotations[key]
        else:
            annotation = None

        key_path = f'{path}/{key}'

        if key_path in pointers_memo:
            subobjects[key] = pointers_memo[key_path]
        else:
            subobjects[key] = deserialize(value, annotation, global_dict=global_dict,
                                          pointers_memo=pointers_memo, path=key_path)  # , enforce_pointers=False)
    if class_ is not None:
        obj = class_(**subobjects)
    else:
        obj = subobjects

    return obj


def deserialize_with_type(type_, value):
    if type_ in dcty.TYPES_STRINGS.values():
        return literal_eval(type_)(value)
    if isinstance(type_, str):
        class_ = dcty.get_python_class_from_class_name(type_)
        if inspect.isclass(class_):
            return class_.dict_to_object(value)
        raise NotImplementedError(f'Cannot get class from name {type_}')

    if isinstance(type_, (list, tuple)):
        return [deserialize_with_type(t, v) for t, v in zip(type_, value)]
    if type_ is None:
        return value

    raise NotImplementedError(type_)


def deserialize_with_typing(type_, argument):
    """
    Deserialize an object with a typing info
    """
    origin = get_origin(type_)
    args = get_args(type_)
    if origin is Union:
        # Check for Union false Positive (Default value = None)
        if len(args) == 2 and type(None) in args:
            return deserialize_argument(type_=args[0], argument=argument)

        # Type union
        classes = list(args)
        instantiated = False
        while instantiated is False:
            # Find the last class in the hierarchy
            hierarchy_lengths = [len(cls.mro()) for cls in classes]
            max_length = max(hierarchy_lengths)
            children_class_index = hierarchy_lengths.index(max_length)
            children_class = classes[children_class_index]
            try:
                # Try to deserialize
                # Throws KeyError if we try to put wrong dict into
                # dict_to_object. This means we try to instantiate
                # a children class with a parent dict_to_object
                deserialized_arg = children_class.dict_to_object(argument)

                # If it succeeds we have the right
                # class and instantiated object
                instantiated = True
            except KeyError:
                # This is not the right class, we should go see the parent
                classes.remove(children_class)
    elif origin in [list, collections.Iterator]:
        # Homogenous sequences (lists)
        sequence_subtype = args[0]
        deserialized_arg = [deserialize_argument(sequence_subtype, arg) for arg in argument]
        if origin is collections.Iterator:
            deserialized_arg = iter(deserialized_arg)

    elif origin is tuple:
        # Heterogenous sequences (tuples)
        deserialized_arg = tuple(deserialize_argument(t, arg) for (t, arg) in zip(args, argument))
    elif origin is dict:
        # Dynamic dict
        deserialized_arg = argument
    elif origin is InstanceOf:
        classname = args[0]
        object_class = dc.full_classname(object_=classname, compute_for='class')
        class_ = dcty.get_python_class_from_class_name(object_class)
        deserialized_arg = class_.dict_to_object(argument)
    else:
        msg = "Deserialization of typing {} is not implemented"
        raise NotImplementedError(msg.format(type_))
    return deserialized_arg


def deserialize_argument(type_, argument):
    """
    Deserialize an argument of a function with the type
    """
    if argument is None:
        return None
    if dcty.is_typing(type_):
        return deserialize_with_typing(type_, argument)
    if type_ in [TextIO, BinaryIO] or issubclass(type_, (StringFile, BinaryFile)):
        deserialized_arg = argument
    else:
        if type_ in dcty.TYPING_EQUIVALENCES.keys():
            if isinstance(argument, type_):
                deserialized_arg = argument
            else:
                if isinstance(argument, int) and type_ == float:
                    # Explicit conversion in this case
                    deserialized_arg = float(argument)
                else:
                    msg = f"Given built-in type and argument are incompatible: " \
                          f"{type(argument)} and {type_} in {argument}"
                    raise TypeError(msg)
        elif type_ is Any:
            # Any type
            deserialized_arg = argument
        elif inspect.isclass(type_) and issubclass(type_, dc.DessiaObject):
            # Custom classes
            deserialized_arg = type_.dict_to_object(argument)
        else:
            raise TypeError(f"Deserialization of type {type_} is Not Implemented")
    return deserialized_arg


def find_references(value, path='#'):
    """
    Traverse recursively the value to find reference (pointers) in it
    Calls recursively find_references_sequence and find_references_dict
    """
    if isinstance(value, dict):
        return find_references_dict(value, path)
    if dcty.isinstance_base_types(value):
        return []
    if dcty.is_sequence(value):
        return find_references_sequence(value, path)
    if isinstance(value, (BinaryFile, StringFile)):
        return []
    raise ValueError(value)


def find_references_sequence(seq, path='#'):
    if isinstance(seq, str):
        raise ValueError

    references = []
    for ie, element in enumerate(seq):
        path_value = f'{path}/{ie}'
        references.extend(find_references(element, path=path_value))
        # if value_nodes or value_edges:

    return references


def find_references_dict(dict_, path='#'):
    if '$ref' in dict_:
        return [(path, dict_['$ref'])]

    references = []
    for key, value in dict_.items():
        if not dcty.isinstance_base_types(value):
            path_value = f'{path}/{key}'
            references.extend(find_references(value, path=path_value))
    return references


def pointer_graph(value):
    """
    Create a graph of subattributes of an object with edge representing either:
     * the hierarchy of an subattribute to an attribute
     * the pointer link between the 2 elements
    """

    nodes = set()
    edges = set()
    for path, reference in find_references(value):
        # Slitting path & reference to add missing nodes
        # For path
        path_segments = path.split('/')
        nodes.add(path_segments[0])
        previous_node = path_segments[0]
        for segment in path_segments[1:]:
            node = f'{previous_node}/{segment}'
            nodes.add(node)
            edges.add((previous_node, node))
            previous_node = node

        # For reference
        reference_segments = reference.split('/')
        nodes.add(reference_segments[0])
        previous_node = reference_segments[0]
        for segment in reference_segments[1:]:
            node = f'{previous_node}/{segment}'
            nodes.add(node)
            edges.add((previous_node, node))
            previous_node = node

        edges.add((path, reference))

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph


def deserialization_order(dict_):
    """
    Analyse a dict representing an object and give a deserialization order
    """
    graph = pointer_graph(dict_)
    if '#' in graph.nodes:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            # import dessia_common.displays
            # dessia_common.displays.draw_networkx_graph(graph)
            for cycle in cycles:
                print(cycle)
            raise NotImplementedError('Cycles in jsonpointers not handled')

        order = list(explore_tree_from_leaves(graph))
        if '#' in order:
            order.remove('#')
        return order
    return []


def dereference_jsonpointers(dict_):  # , global_dict):
    """
    Analyse the given dict to:
    - find jsonpointers
    - deserialize them in the right order to respect pointers graph
    :returns: a dict with key the path of the item and the value is the python object
    """

    order = deserialization_order(dict_)
    # print('\norder of')
    # if 'object_class' in dict_:
    #     print(dict_['object_class'])

    pointers_memo = {}
    for ref in order:
        serialized_element = get_in_object_from_path(dict_, ref)
        pointers_memo[ref] = deserialize(serialized_element=serialized_element,
                                         global_dict=dict_,
                                         pointers_memo=pointers_memo,
                                         path=ref)

    return pointers_memo


def pointer_graph_elements(value, path='#'):

    if isinstance(value, dict):
        return pointer_graph_elements_dict(value, path)
    if dcty.isinstance_base_types(value):
        return [], []
    if dcty.is_sequence(value):
        return pointer_graph_elements_sequence(value, path)

    raise ValueError(value)


def pointer_graph_elements_sequence(seq, path='#'):
    """
    Compute
    """
    if isinstance(seq, str):
        raise ValueError

    edges = []
    nodes = []
    for ie, element in enumerate(seq):
        path_value = f'{path}/{ie}'
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
            path_value = f'{path}/{key}'
            value_nodes, value_edges = pointer_graph_elements(value, path=path_value)
            # if value_nodes or value_edges:
            nodes.append(path_value)
            nodes.extend(value_nodes)

            edges.append((path, path_value))
            edges.extend(value_edges)

    return nodes, edges


def pointers_analysis(obj):
    """
    Analyse on object to output stats on pointer use in the object
    :returns: a tuple of 2 dicts: one giving the number of pointer use by class
    """
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
