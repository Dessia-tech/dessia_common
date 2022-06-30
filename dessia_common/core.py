#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import time
import sys
import warnings
import math
import random
from functools import reduce
import collections
from copy import deepcopy, copy
import inspect
import json

from typing import List, Dict, Any, Tuple, get_type_hints
import traceback as tb

from importlib import import_module
from ast import literal_eval

import dessia_common.errors
from dessia_common.utils.diff import data_eq, diff, dict_hash, list_hash
from dessia_common.utils.serialization import dict_to_object, serialize_dict_with_pointers,\
    serialize_dict, deserialize_argument, serialize
from dessia_common.utils.types import full_classname, is_sequence, is_bson_valid, TYPES_FROM_STRING
from dessia_common.utils.copy import deepcopy_value
from dessia_common.utils.jsonschema import default_dict, jsonschema_from_annotation,\
    JSONSCHEMA_HEADER, set_default_value
from dessia_common.utils.docstrings import parse_docstring, FAILED_DOCSTRING_PARSING
from dessia_common.exports import XLSXWriter
from dessia_common.typings import JsonSerializable
from dessia_common import templates
from dessia_common.displays import DisplayObject, DisplaySetting
from dessia_common.breakdown import attrmethod_getter


_FORBIDDEN_ARGNAMES = ['self', 'cls', 'progress_callback', 'return']


def deprecated(use_instead=None):
    def decorated(function):
        def wrapper(*args, **kwargs):
            deprecation_warning(function.__name__, 'Function', use_instead)
            print('Traceback : ')
            tb.print_stack(limit=2)
            return function(*args, **kwargs)

        return wrapper

    return decorated


def deprecation_warning(name, object_type, use_instead=None):
    warnings.simplefilter('once', DeprecationWarning)
    msg = f"\n\n{object_type} {name} is deprecated.\n"
    msg += "It will be removed in a future version.\n"
    if use_instead is not None:
        msg += f"Use {use_instead} instead.\n"
    warnings.warn(msg, DeprecationWarning)
    return msg


class DessiaObject:
    """
    Base class for Dessia's platform compatible objects.
    Gathers generic methods and attributes

    :cvar bool _standalone_in_db:
        Indicates wether class objects should be independant
        in database or not.
        If False, object will only exist inside its parent.
    :cvar bool _eq_is_data_eq:
        Indicates which type of equality check is used:
        strict equality or equality based on data. If False, Python's
        object __eq__ method is used (ie. strict),
        else, user custom data_eq is used (ie. data)
    :cvar List[str] _non_serializable_attributes:
        [Advanced] List of instance attributes that should not
        be part of serialization with to_dict method. These will not
        be displayed in platform object tree, for instance.
    :cvar List[str] _non_data_eq_attributes:
        [Advanced] List of instance attributes that should not
        be part of equality check with data__eq__ method
        (if _eq_is_data_eq is True).
    :cvar List[str] _non_data_hash_attributes:
        [Advanced] List of instance attributes that should not
        be part of hash computation with data__hash__ method
        (if _eq_is_data_eq is True).
    :cvar List[str] _ordered_attributes:
        Documentation not available yet.
    :cvar List[str] _titled_attributes:
        Documentation not available yet.
    :cvar List[str] _init_variables:
        Documentation not available yet.
    :cvar List[str] _export_formats:
        List of all available export formats. Class must define a
        export_[format] for each format in _export_formats
    :cvar List[str] _allowed_methods:
        List of all methods that are runnable from platform.
    :cvar List[str] _whitelist_attributes:
        Documentation not available yet.
    :cvar List[str] _whitelist_attributes: List[str]


    :ivar str name: Name of object.
    :ivar Any kwargs: Additionnal user metadata
    """
    _standalone_in_db = False
    _non_serializable_attributes = []
    _non_editable_attributes = []
    _non_data_eq_attributes = ['name']
    _non_data_hash_attributes = ['name']
    _ordered_attributes = []
    _titled_attributes = []
    _eq_is_data_eq = True

    _init_variables = None
    _allowed_methods = []
    _whitelist_attributes = []

    def __init__(self, name: str = '', **kwargs):
        """
        Generic init of DessiA Object. Only store name in self. To be overload
        and call in specific class init
        """
        self.name = name
        for property_name, property_value in kwargs.items():
            setattr(self, property_name, property_value)

    def __hash__(self):
        """
        Compute a int from object
        """
        if self._eq_is_data_eq:
            return self._data_hash()
        return object.__hash__(self)

    def __eq__(self, other_object):
        """
        Generic equality of two objects. behavior can be controled by class
        attribute _eq_is_data_eq to tell if we must use python equality (based on memory addresses)
        (_eq_is_data_eq = False) or a data equality (True)
        """
        if self._eq_is_data_eq:
            if self.__class__.__name__ != other_object.__class__.__name__:
                return False
            if self._data_hash() != other_object._data_hash():
                return False
            return self._data_eq(other_object)
        return object.__eq__(self, other_object)

    def _data_eq(self, other_object):
        return data_eq(self, other_object)

    def _data_hash(self):
        hash_ = 0
        forbidden_keys = (self._non_data_eq_attributes
                          + self._non_data_hash_attributes
                          + ['package_version', 'name'])
        for key, value in self._serializable_dict().items():
            if key not in forbidden_keys:
                if is_sequence(value):
                    hash_ += list_hash(value)
                elif isinstance(value, dict):
                    hash_ += dict_hash(value)
                elif isinstance(value, str):
                    hash_ += sum(ord(v) for v in value)
                else:
                    hash_ += hash(value)
        return int(hash_ % 1e5)

    def _data_diff(self, other_object):
        """
        Make a diff between two objects
        returns: different values, missing keys in other object
        """
        # return diff(self, other_object)
        return diff(self, other_object)

    @property
    def full_classname(self):
        """
        Full classname of class like: package.module.submodule.classname
        """
        return full_classname(self)

    def base_dict(self):
        """
        A base dict for to_dict: put name, object class and version in a dict
        """
        package_name = self.__module__.split('.')[0]
        if package_name in sys.modules:
            package = sys.modules[package_name]
            if hasattr(package, '__version__'):
                package_version = package.__version__
            else:
                package_version = None
        else:
            package_version = None

        object_class = self.__module__ + '.' + self.__class__.__name__
        dict_ = {'name': self.name, 'object_class': object_class}
        if package_version:
            dict_['package_version'] = package_version
        return dict_

    def _serializable_dict(self):
        """
        Returns a dict of attribute_name, values (still python, not serialized)
        Keys are filtered with non serializable attributes controls
        """

        dict_ = {k: v for k, v in self.__dict__.items()
                 if k not in self._non_serializable_attributes
                 and not k.startswith('_')}
        return dict_

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#') -> JsonSerializable:
        """
        Generic to_dict method
        """
        if memo is None:
            memo = {}

        # Default to dict
        serialized_dict = self.base_dict()
        dict_ = self._serializable_dict()
        if use_pointers:
            serialized_dict.update(serialize_dict_with_pointers(dict_, memo, path)[0])
        else:
            serialized_dict.update(serialize_dict(dict_))

        return serialized_dict

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'DessiaObject':
        """
        Generic dict_to_object method
        """

        if cls is not DessiaObject:
            obj = dict_to_object(dict_=dict_, class_=cls,
                                 force_generic=force_generic,
                                 global_dict=global_dict,
                                 pointers_memo=pointers_memo,
                                 path=path)
            return obj

        if 'object_class' in dict_:
            obj = dict_to_object(dict_=dict_, force_generic=force_generic,
                                 global_dict=global_dict,
                                 pointers_memo=pointers_memo,
                                 path=path)
            return obj

        raise NotImplementedError('No object_class in dict')

    @classmethod
    def base_jsonschema(cls):
        jsonschema = deepcopy(JSONSCHEMA_HEADER)
        jsonschema['properties']['name'] = {
            'type': 'string',
            "title": "Object Name",
            "description": "Object name",
            "editable": True,
            "default_value": "Object Name"
        }
        return jsonschema

    @classmethod
    def jsonschema(cls):
        """
        Jsonschema of class: transfer python data structure to web standard
        """
        if hasattr(cls, '_jsonschema'):
            _jsonschema = cls._jsonschema
            return _jsonschema

        # Get __init__ method and its annotations
        init = cls.__init__
        if cls._init_variables is None:
            annotations = get_type_hints(init)
        else:
            annotations = cls._init_variables

        # Get ordered variables
        if cls._ordered_attributes:
            ordered_attributes = cls._ordered_attributes
        else:
            ordered_attributes = list(annotations.keys())

        unordered_count = 0

        # Parse docstring
        try:
            docstring = cls.__doc__
            parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        except Exception:
            parsed_docstring = FAILED_DOCSTRING_PARSING
        parsed_attributes = parsed_docstring['attributes']

        # Initialize jsonschema
        _jsonschema = deepcopy(JSONSCHEMA_HEADER)

        required_arguments, default_arguments = inspect_arguments(method=init,
                                                                  merge=False)
        _jsonschema['required'] = required_arguments
        _jsonschema['standalone_in_db'] = cls._standalone_in_db
        _jsonschema['description'] = parsed_docstring['description']
        _jsonschema['python_typing'] = str(cls)

        # Set jsonschema
        for annotation in annotations.items():
            name = annotation[0]
            if name in ordered_attributes:
                order = ordered_attributes.index(name)
            else:
                order = len(ordered_attributes) + unordered_count
                unordered_count += 1
            if name in cls._titled_attributes:
                title = cls._titled_attributes[name]
            else:
                title = None

            if name != 'return':
                editable = name not in cls._non_editable_attributes
                annotation_type = type_from_annotation(annotation[1], cls)
                annotation = (name, annotation_type)
                jss_elt = jsonschema_from_annotation(
                    annotation=annotation, jsonschema_element={}, order=order,
                    editable=editable, title=title, parsed_attributes=parsed_attributes
                )
                _jsonschema['properties'].update(jss_elt)
                if name in default_arguments.keys():
                    default = set_default_value(_jsonschema['properties'],
                                                name,
                                                default_arguments[name])
                    _jsonschema['properties'].update(default)

        _jsonschema['classes'] = [cls.__module__ + '.' + cls.__name__]
        _jsonschema['whitelist_attributes'] = cls._whitelist_attributes
        return _jsonschema

    @property
    def _method_jsonschemas(self):
        """
        Generates dynamic jsonschemas for methods of class
        """
        jsonschemas = {}
        class_ = self.__class__

        # TOCHECK Backward compatibility. Will need to be changed
        if hasattr(class_, '_dessia_methods'):
            allowed_methods = class_._dessia_methods
        else:
            allowed_methods = class_._allowed_methods

        valid_method_names = [m for m in dir(class_)
                              if not m.startswith('_')
                              and m in allowed_methods]

        for method_name in valid_method_names:
            method = getattr(class_, method_name)

            if not isinstance(method, property):
                required_args, default_args = inspect_arguments(method=method, merge=False)
                annotations = get_type_hints(method)
                if annotations:
                    jsonschemas[method_name] = deepcopy(JSONSCHEMA_HEADER)
                    jsonschemas[method_name]['required'] = []
                    jsonschemas[method_name]['method'] = True
                    for i, annotation in enumerate(annotations.items()):
                        # TOCHECK Not actually ordered
                        argname = annotation[0]
                        if argname not in _FORBIDDEN_ARGNAMES:
                            if argname in required_args:
                                jsonschemas[method_name]['required'].append(str(i))
                            jsonschema_element = \
                                jsonschema_from_annotation(annotation, {}, i)[argname]

                            jsonschemas[method_name]['properties'][str(i)] = jsonschema_element
                            if argname in default_args.keys():
                                default = set_default_value(
                                    jsonschemas[method_name]['properties'],
                                    str(i),
                                    default_args[argname]
                                )
                                jsonschemas[method_name]['properties'].update(default)
        return jsonschemas

    def method_dict(self, method_name=None, method_jsonschema=None):
        """
        Return a jsonschema of a method arguments
        """
        if method_name is None and method_jsonschema is None:
            msg = 'No method name nor jsonschema provided'
            raise NotImplementedError(msg)

        if method_name is not None and method_jsonschema is None:
            method_jsonschema = self._method_jsonschemas[method_name]

        dict_ = default_dict(method_jsonschema)
        return dict_

    def dict_to_arguments(self, dict_, method):
        """
        Transform serialized argument of a method to python objects ready to use in method evaluation
        """
        method_object = getattr(self, method)
        args_specs = inspect.getfullargspec(method_object)
        allowed_args = args_specs.args[1:]

        arguments = {}
        for i, arg in enumerate(allowed_args):
            if str(i) in dict_:
                arg_specs = args_specs.annotations[arg]
                value = dict_[str(i)]
                try:
                    deserialized_value = deserialize_argument(arg_specs, value)
                except TypeError as err:
                    msg = 'Error in deserialisation of value: '
                    msg += f'{value} of expected type {arg_specs}'
                    raise TypeError(msg) from err
                arguments[arg] = deserialized_value
        return arguments

    def save_to_file(self, filepath: str, indent: int = 2):
        """
        Save object to a JSON file
        :param filepath: either a string reprensenting the filepath or a stream
        """
        if not filepath.endswith('.json'):
            filepath += '.json'
            print(f'Changing name to {filepath}')
        with open(filepath, 'w', encoding='utf-8') as file:
            self.save_to_stream(file, indent=indent)

    def save_to_stream(self, stream, indent: int = 2):
        try:
            dict_ = self.to_dict(use_pointers=True)
        except TypeError:
            dict_ = self.to_dict()

        json.dump(dict_, stream, indent=indent)

    @classmethod
    def load_from_stream(cls, stream):
        dict_ = json.loads(stream.read().decode('utf-8'))
        return cls.dict_to_object(dict_)

    @classmethod
    def load_from_file(cls, filepath: str):
        """
        Load object from a json file
        :param filepath: either a string reprensenting the filepath or a stream
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            dict_ = json.load(file)

        return cls.dict_to_object(dict_)

    def is_valid(self):
        return True

    def copy(self, deep=True, memo=None):
        if deep:
            return deepcopy(self, memo=memo)
        return copy(self)

    def __copy__(self):
        """
        Generic copy use inits of objects
        """
        class_argspec = inspect.getfullargspec(self.__class__)
        dict_ = {}
        for arg in class_argspec.args:
            if arg != 'self':
                value = self.__dict__[arg]
                if hasattr(value, '__copy__'):
                    dict_[arg] = value.__copy__()
                else:
                    dict_[arg] = value
        return self.__class__(**dict_)

    def __deepcopy__(self, memo=None):
        """
        Generic deep copy use inits of objects
        """
        class_argspec = inspect.getfullargspec(self.__class__)
        if memo is None:
            memo = {}
        dict_ = {}
        for arg in class_argspec.args:
            if arg != 'self':
                dict_[arg] = deepcopy_value(getattr(self, arg), memo=memo)
        return self.__class__(**dict_)

    def plot_data(self):
        return []

    def plot(self, **kwargs):
        """
        Generic plot getting plot_data function to plot
        """
        if hasattr(self, 'plot_data'):
            import plot_data
            for data in self.plot_data(**kwargs):
                plot_data.plot_canvas(plot_data_object=data,
                                      canvas_id='canvas',
                                      width=1400, height=900,
                                      debug_mode=False)
        else:
            msg = 'Class {} does not implement a plot_data method' \
                  ' to define what to plot'
            raise NotImplementedError(msg.format(self.__class__.__name__))

    def mpl_plot(self, **kwargs):
        """
        Plot with matplotlib using plot_data function
        """
        axs = []
        if hasattr(self, 'plot_data'):
            try:
                plot_datas = self.plot_data(**kwargs)
            except TypeError as error:
                raise TypeError(f'{self.__class__.__name__}.{error}') from error
            for data in plot_datas:
                if hasattr(data, 'mpl_plot'):
                    ax = data.mpl_plot()
                    axs.append(ax)
        else:
            msg = 'Class {} does not implement a plot_data method' \
                  'to define what to plot'
            raise NotImplementedError(msg.format(self.__class__.__name__))

        return axs

    @staticmethod
    def display_settings() -> List[DisplaySetting]:
        """
        Returns a list of json describing how to call subdisplays
        """
        return [DisplaySetting('markdown', 'markdown', 'to_markdown', None),
                DisplaySetting('plot_data', 'plot_data', 'plot_data', None, serialize_data=True)]

    def _display_from_selector(self, selector: str, **kwargs) -> DisplayObject:
        """
        Generate the display from the selector
        """
        reference_path = kwargs.get('reference_path', '')

        display_setting = self._display_settings_from_selector(selector)
        track = ''
        try:
            data = attrmethod_getter(self, display_setting.method)(**display_setting.arguments)
        except:
            data = None
            track = tb.format_exc()

        if display_setting.serialize_data:
            data = serialize(data)
        return DisplayObject(type_=display_setting.type, data=data, reference_path=reference_path, traceback=track)

    def _display_settings_from_selector(self, selector: str):
        for display_setting in self.display_settings():
            if display_setting.selector == selector:
                return display_setting
        raise ValueError(f"No such selector '{selector}' in display of class '{self.__class__.__name__}'")

    def _displays(self, **kwargs) -> List[JsonSerializable]:
        """
        Generate displays of the object to be plot in the DessiA Platform
        """
        reference_path = kwargs.get('reference_path', '')

        displays = []
        for display_setting in self.display_settings():
            display_ = self._display_from_selector(display_setting.selector, reference_path=reference_path)
            displays.append(display_.to_dict())
        return displays

    def to_markdown(self) -> str:
        """
        Render a markdown of the object output type: string
        """
        return templates.dessia_object_markdown_template.substitute(name=self.name, class_=self.__class__.__name__)

    def _performance_analysis(self):
        """
        Prints time of rendering some commons operations (serialization, hash, displays)
        """
        data_hash_time = time.time()
        self._data_hash()
        data_hash_time = time.time() - data_hash_time
        print(f'Data hash time: {round(data_hash_time, 3)} seconds')

        todict_time = time.time()
        dict_ = self.to_dict()
        todict_time = time.time() - todict_time
        print(f'to_dict time: {round(todict_time, 3)} seconds')

        dto_time = time.time()
        self.dict_to_object(dict_)
        dto_time = time.time() - dto_time
        print(f'dict_to_object time: {round(dto_time, 3)} seconds')

        for display_setting in self.display_settings():
            display_time = time.time()
            self._display_from_selector(display_setting.selector)
            display_time = time.time() - display_time
            print(f'Generation of display {display_setting.selector} in: {round(display_time, 6)} seconds')

    def _check_platform(self):
        """
        Reproduce lifecycle on platform (serialization, display)
        raise an error if something is wrong
        """
        try:
            dict_ = self.to_dict(use_pointers=True)
        except TypeError:
            dict_ = self.to_dict()
        json_dict = json.dumps(dict_)
        decoded_json = json.loads(json_dict)
        deserialized_object = self.dict_to_object(decoded_json)
        if not deserialized_object._data_eq(self):
            print('data diff: ', self._data_diff(deserialized_object))
            raise dessia_common.errors.DeserializationError('Object is not equal to itself'
                                                            ' after serialization/deserialization')
        copied_object = self.copy()
        if not copied_object._data_eq(self):
            print('data diff: ', self._data_diff(copied_object))
            raise dessia_common.errors.CopyError('Object is not equal to itself'
                                                 ' after copy')

        valid, hint = is_bson_valid(stringify_dict_keys(dict_))
        if not valid:
            raise ValueError(hint)
        json.dumps(self._displays())
        json.dumps(self._method_jsonschemas)

    def to_xlsx(self, filepath: str):
        """
        Exports the object to an XLSX file given by the filepath
        """
        with open(filepath, 'wb') as file:
            self.to_xlsx_stream(file)

    def to_xlsx_stream(self, stream):
        """
        Exports the object to an XLSX to a given stream
        """
        writer = XLSXWriter(self)
        writer.save_to_stream(stream)

    def _export_formats(self):
        formats = [{"extension": "json", "method_name": "save_to_stream", "text": True, "args": {}},
                   {"extension": "xlsx", "method_name": "to_xlsx_stream", "text": False, "args": {}}]
        return formats


class PhysicalObject(DessiaObject):
    """
    Represent an object with CAD capabilities
    """

    @staticmethod
    def display_settings():
        """
        Returns a list of json describing how to call subdisplays
        """
        display_settings = DessiaObject.display_settings()
        display_settings.append(DisplaySetting(selector='cad', type_='babylon_data',
                                               method='volmdlr_volume_model().babylon_data',
                                               serialize_data=True))
        return display_settings

    def volmdlr_primitives(self):
        """
        Return a list of volmdlr primitives to build up volume model
        """
        return []

    def volmdlr_volume_model(self, **kwargs):
        """
        Gives the volmdlr VolumeModel
        """
        import volmdlr as vm  # !!! Avoid circular imports, is this OK ?
        return vm.core.VolumeModel(self.volmdlr_primitives(**kwargs))

    def to_step(self, filepath: str):
        """
        Exports the CAD of the object to step. Works if the class define a custom volmdlr model
        :param filepath: a str representing a filepath
        """
        self.volmdlr_volume_model().to_step(filepath=filepath)
        return self.volmdlr_volume_model().to_step(filepath=filepath)

    def to_step_stream(self, stream):
        """
        Exports the CAD of the object to a stream in the STEP format. Works if the class define a custom volmdlr model
        """
        return self.volmdlr_volume_model().to_step_stream(stream=stream)

    def to_stl_stream(self, stream):
        """
        Exports the CAD of the object to STL to a given stream
        """
        return self.volmdlr_volume_model().to_stl_stream(stream=stream)

    def to_stl(self, filepath):
        """
        Exports the CAD of the object to STL. Works if the class define a custom volmdlr model
        :param filepath: a str representing a filepath
        """
        return self.volmdlr_volume_model().to_stl(filepath=filepath)

    # def _displays(self, **kwargs):
    #     """
    #     Compute the list of displays
    #     """
    #     return DessiaObject._displays(self, **kwargs)

    def babylonjs(self, use_cdn=True, debug=False, **kwargs):
        """
        Show the 3D volmdlr of an object by calling volmdlr_volume_model method
        and plot in in browser
        """
        self.volmdlr_volume_model(**kwargs).babylonjs(use_cdn=use_cdn,
                                                      debug=debug)

    def save_babylonjs_to_file(self, filename: str = None, use_cdn: bool = True,
                               debug: bool = False, **kwargs):
        self.volmdlr_volume_model(**kwargs).save_babylonjs_to_file(filename=filename,
                                                                   use_cdn=use_cdn,
                                                                   debug=debug)

    def _export_formats(self):
        formats = DessiaObject._export_formats(self)
        formats3d = [{"extension": "step", "method_name": "to_step_stream", "text": True, "args": {}},
                     {"extension": "stl", "method_name": "to_stl_stream", "text": False, "args": {}}]
        formats.extend(formats3d)
        return formats


class MovingObject(PhysicalObject):

    def volmdlr_primitives_step_frames(self):
        """
        Return a list of volmdlr primitives to build up volume model
        """
        raise NotImplementedError('Object inheriting MovingObject should implement volmdlr_primitives_step_frames')

    def volmdlr_volume_model(self, **kwargs):
        import volmdlr as vm  # !!! Avoid circular imports, is this OK ?
        return vm.core.MovingVolumeModel(
            self.volmdlr_primitives(**kwargs),
            self.volmdlr_primitives_step_frames(**kwargs)
        )


# class Catalog(DessiaObject):
#     def __init__(self, objects: List[DessiaObject], name: str = ''):
#         self.objects = objects
#         DessiaObject.__init__(self, name=name)


class Parameter(DessiaObject):
    def __init__(self, lower_bound, upper_bound, periodicity=None, name=''):
        DessiaObject.__init__(self, name=name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.periodicity = periodicity

    def random_value(self):
        """
        Sample a value within the bounds
        """
        return random.uniform(self.lower_bound, self.upper_bound)

    def are_values_equal(self, value1, value2, tol=1e-2):
        if self.periodicity is not None:
            value1 = value1 % self.periodicity
            value2 = value2 % self.periodicity

        return math.isclose(value1, value2, abs_tol=tol)

    def normalize(self, value):
        normalized_value = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
        return normalized_value

    def original_value(self, normalized_value):
        value = normalized_value * (self.upper_bound - self.lower_bound) + self.lower_bound
        return value

    def optimizer_bounds(self):
        if self.periodicity is not None:
            return (self.lower_bound - 0.5 * self.periodicity,
                    self.upper_bound + 0.5 * self.periodicity)
        return None


class ParameterSet(DessiaObject):
    def __init__(self, values, name=''):
        self.values = values

        DessiaObject.__init__(self, name=name)

    @property
    def parameters(self):
        parameters = [Parameter(min(v), max(v), name=k)
                      for k, v in self.values.items()]
        return parameters

    @property
    def means(self):
        means = {k: sum(v) / len(v) for k, v in self.values.items()}
        return means


class DessiaFilter(DessiaObject):
    def __init__(self, attribute: str, operator: str,
                 bound: float, name: str = ''):
        self.attribute = attribute
        self.operator = operator
        self.bound = bound

        DessiaObject.__init__(self, name=name)

    def __hash__(self):
        hash_ = len(self.attribute)
        hash_ += hash(self.operator)
        hash_ += hash(self.bound)
        return int(hash_)

    def __eq__(self, other: 'DessiaFilter'):
        same_attr = self.attribute == other.attribute
        same_op = self.operator == other.operator
        same_bound = self.bound == other.bound
        return same_attr and same_op and same_bound


# class Evolution(DessiaObject):
#     """
#     Defines a generic evolution

#     :param evolution: float list
#     :type evolution: list
#     """
#     _non_data_eq_attributes = ['name']
#     _non_data_hash_attributes = ['name']
#     _generic_eq = True

#     def __init__(self, evolution: List[float] = None, name: str = ''):
#         if evolution is None:
#             evolution = []
#         self.evolution = evolution

#         DessiaObject.__init__(self, name=name)

#     def _displays(self):
#         displays = [{'angular_component': 'app-evolution1d',
#                      'table_show': False,
#                      'evolution': [self.evolution],
#                      'label_y': ['evolution']}]
#         return displays

#     def update(self, evolution):
#         """
#         Update the evolution list
#         """
#         self.evolution = evolution


# class CombinationEvolution(DessiaObject):
#     _non_data_eq_attributes = ['name']
#     _non_data_hash_attributes = ['name']
#     _generic_eq = True

#     def __init__(self, evolution1: List[Evolution],
#                  evolution2: List[Evolution], title1: str = 'x',
#                  title2: str = 'y', name: str = ''):

#         self.evolution1 = evolution1
#         self.evolution2 = evolution2

#         self.x_, self.y_ = self.genere_xy()

#         self.title1 = title1
#         self.title2 = title2

#         DessiaObject.__init__(self, name=name)

#     def _displays(self):
#         displays = [{
#             'angular_component': 'app-evolution2d-combination-evolution',
#             'table_show': False,
#             'evolution_x': [self.x_], 'label_x': ['title1'],
#             'evolution_y': [self.y_], 'label_y': ['title2']
#         }]
#         return displays

#     def update(self, evol1, evol2):
#         """
#         Update the CombinationEvolution object

#         :param evol1: list
#         :param evol2: list
#         """
#         for evolution, ev1 in zip(self.evolution1, evol1):
#             evolution.update(ev1)
#         for evolution, ev2 in zip(self.evolution2, evol2):
#             evolution.update(ev2)
#         self.x_, self.y_ = self.genere_xy()

#     def genere_xy(self):
#         x, y = [], []
#         for evol in self.evolution1:
#             x = x + evol.evolution
#         for evol in self.evolution2:
#             y = y + evol.evolution
#         return x, y


def dict_merge(old_dct, merge_dct, add_keys=True, extend_lists=True):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    This version will return a copy of the dictionary and leave the original
    arguments untouched.

    The optional argument ``add_keys``, determines whether keys which are
    present in ``merge_dct`` but not ``dct`` should be included in the
    new dict.

    Args:
        old_dct (dict) onto which the merge is executed
        merge_dct (dict): dct merged into dct
        add_keys (bool): whether to add new keys
        extend_lists (bool) : wether to extend lists if keys are updated
                              and value is a list

    Returns:
        dict: updated dict
    """
    dct = deepcopy(old_dct)
    if not add_keys:
        merge_dct = {k: merge_dct[k]
                     for k in set(dct).intersection(set(merge_dct))}

    for key, value in merge_dct.items():
        if isinstance(dct.get(key), dict)\
                and isinstance(value, collections.Mapping):
            dct[key] = dict_merge(dct[key], merge_dct[key],
                                  add_keys=add_keys, extend_lists=extend_lists)
        elif isinstance(dct.get(key), list) and extend_lists:
            dct[key].extend(value)
        else:
            dct[key] = value

    return dct


def stringify_dict_keys(obj):
    if isinstance(obj, (list, tuple)):
        new_obj = []
        for elt in obj:
            new_obj.append(stringify_dict_keys(elt))

    elif isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            new_obj[str(key)] = stringify_dict_keys(value)
    else:
        return obj
    return new_obj


def getdeepattr(obj, attr):
    return reduce(getattr, [obj] + attr.split('.'))


def enhanced_deep_attr(obj, sequence):
    """
    Get deep attribute where Objects, Dicts and Lists
    can be found in recursion.

    :param obj: Parent object in which recursively find attribute
                represented by sequence
    :param sequence: List of strings and integers that represents
                     path to deep attribute.
    :return: Value of deep attribute
    """
    if isinstance(sequence, str):
        # Sequence is a string and not a sequence of deep attributes
        if '/' in sequence:
            # Is deep attribute reference
            sequence = deepattr_to_sequence(sequence)
            return enhanced_deep_attr(obj=obj, sequence=sequence)
        # Is direct attribute
        return enhanced_get_attr(obj=obj, attr=sequence)

    # Get direct attribute
    subobj = enhanced_get_attr(obj=obj, attr=sequence[0])
    if len(sequence) > 1:
        # Recursively get deep attributes
        subobj = enhanced_deep_attr(obj=subobj, sequence=sequence[1:])
    return subobj


def enhanced_get_attr(obj, attr):
    """
    Safely get attribute in obj. Obj can be of Object, Dict, or List type

    :param obj: Parent object in which find given attribute
    :param attr: String or integer that represents name or index of attribute
    :return: Value of attribute
    """
    try:
        return getattr(obj, attr)
    except (TypeError, AttributeError):
        classname = obj.__class__.__name__
        track = tb.format_exc()
        try:
            return obj[attr]
        except KeyError:
            try:
                attr = literal_eval(attr)
                return obj[attr]
            except KeyError:
                track += tb.format_exc()
                msg = f"'{classname}' object has no attribute '{attr}'."
        except TypeError:
            track += tb.format_exc()
            msg = f"Object of type '{classname}' is not subscriptable. Failed to deeply get '{attr}' from it"
    raise dessia_common.errors.DeepAttributeError(message=msg, traceback_=track)


def concatenate_attributes(prefix, suffix, type_: str = 'str'):
    wrong_prefix_format = 'Attribute prefix is wrongly formatted.'
    wrong_prefix_format += 'Is of type {}. Should be str or list'
    if type_ == 'str':
        if isinstance(prefix, str):
            return prefix + '/' + str(suffix)
        if is_sequence(prefix):
            return sequence_to_deepattr(prefix) + '/' + str(suffix)
        raise TypeError(wrong_prefix_format.format(type(prefix)))

    if type_ == 'sequence':
        if isinstance(prefix, str):
            return [prefix, suffix]
        if is_sequence(prefix):
            return prefix + [suffix]
        raise TypeError(wrong_prefix_format.format(type(prefix)))

    wrong_concat_type = 'Type {} for concatenation is not supported.'
    wrong_concat_type += 'Should be "str" or "sequence"'
    raise ValueError(wrong_concat_type.format(type_))


def deepattr_to_sequence(deepattr: str):
    sequence = deepattr.split('/')
    healed_sequence = []
    for attribute in sequence:
        try:
            healed_sequence.append(int(attribute))
        except ValueError:
            healed_sequence.append(attribute)
    return healed_sequence


def sequence_to_deepattr(sequence):
    healed_sequence = [str(attr) if isinstance(attr, int) else attr
                       for attr in sequence]
    return '/'.join(healed_sequence)


def is_bounded(filter_: DessiaFilter, value: float):
    bounded = True
    operator = filter_.operator
    bound = filter_.bound

    if operator == 'lte' and value > bound:
        bounded = False
    if operator == 'gte' and value < bound:
        bounded = False

    if operator == 'lt' and value >= bound:
        bounded = False
    if operator == 'gt' and value <= bound:
        bounded = False

    if operator == 'eq' and value != bound:
        bounded = False
    return bounded


def type_from_annotation(type_, module):
    """
    Clean up a proposed type if there are stringified
    """
    if isinstance(type_, str):
        # Evaluating types
        type_ = TYPES_FROM_STRING.get(type_, default=getattr(import_module(module), type_))
    return type_


def prettyname(namestr):
    """
    Creates a pretty name from as str
    """
    pretty_name = ''
    if namestr:
        strings = namestr.split('_')
        for i, string in enumerate(strings):
            if len(string) > 1:
                pretty_name += string[0].upper() + string[1:]
            else:
                pretty_name += string
            if i < len(strings) - 1:
                pretty_name += ' '
    return pretty_name


def inspect_arguments(method, merge=False):
    # Find default value and required arguments of class construction
    argspecs = inspect.getfullargspec(method)
    nargs, ndefault_args = split_argspecs(argspecs)

    default_arguments = {}
    arguments = []
    for iargument, argument in enumerate(argspecs.args[1:]):
        if argument not in _FORBIDDEN_ARGNAMES:
            if iargument >= nargs - ndefault_args:
                default_value = argspecs.defaults[ndefault_args - nargs + iargument]
                if merge:
                    arguments.append((argument, default_value))
                else:
                    default_arguments[argument] = default_value
            else:
                arguments.append(argument)
    return arguments, default_arguments


def split_argspecs(argspecs) -> Tuple[int, int]:
    nargs = len(argspecs.args) - 1

    if argspecs.defaults is not None:
        ndefault_args = len(argspecs.defaults)
    else:
        ndefault_args = 0
    return nargs, ndefault_args
