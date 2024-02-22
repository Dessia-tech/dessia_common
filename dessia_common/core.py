#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module to handle serialization for engineering objects. """

import time
import warnings
import operator
import math
import random
import itertools
import zipfile

from copy import deepcopy, copy
import inspect
import json

from typing import List, Tuple
import traceback as tb

from importlib import import_module

from dessia_common.utils.diff import data_eq, diff, choose_hash
from dessia_common.utils.types import is_bson_valid
from dessia_common.utils.copy import deepcopy_value
import dessia_common.schemas.core as dcs
from dessia_common.serialization import SerializableObject, deserialize_argument, serialize
from dessia_common.exports import XLSXWriter, MarkdownWriter, ExportFormat
from dessia_common.typings import JsonSerializable
from dessia_common import templates
import dessia_common.checks as dcc
from dessia_common.displays import DisplayObject, DisplaySetting
from dessia_common.breakdown import attrmethod_getter, get_in_object_from_path
import dessia_common.utils.helpers as dch
import dessia_common.files as dcf
from dessia_common.document_generator import DocxWriter
from dessia_common.decorators import get_decorated_methods, DISPLAY_DECORATORS, EXPORT_DECORATORS
from dessia_common.excel_reader import ExcelReader


def __getattr__(name):
    if name == "_FORBIDDEN_ARGNAMES":
        warnings.warn("Attribute '_FORBIDDEN_ARGNAMES' is deprecated. Use schemas.RESERVED_ARGNAMES instead",
                      DeprecationWarning)
        return dcs.RESERVED_ARGNAMES
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


_fullargsspec_cache = {}


class DessiaObject(SerializableObject):
    """
    Base class for Dessia's platform compatible objects.

    Gathers generic methods and attributes

    :cvar bool _standalone_in_db: Indicates whether class objects should be independent in database or not.
        If False, object will only exist inside its parent.

    :cvar bool _eq_is_data_eq:
        Indicates which type of equality check is used: strict equality or equality based on data.
        If False, Python's object __eq__ method is used (strict), else, user custom data_eq is used (data)

    :cvar List[str] _non_serializable_attributes:
        [Advanced] List of instance attributes that should not be part of serialization with to_dict method.
        These will not be displayed in platform object tree, for instance.

    :cvar List[str] _non_data_eq_attributes:
        [Advanced] List of instance attributes that should not be part of equality check with data__eq__ method
        (if _eq_is_data_eq is True).

    :cvar List[str] _non_data_hash_attributes:
        [Advanced] List of instance attributes that should not be part of hash computation with data__hash__ method
        (if _eq_is_data_eq is True).

    :cvar List[str] _allowed_methods: List of all methods that can be ran from platform.

    :param name: Name of object.
    """

    _non_editable_attributes = []
    _non_data_eq_attributes = ['name']
    _non_data_hash_attributes = ['name']
    _eq_is_data_eq = True
    _vector_features = None

    _init_variables = None
    _allowed_methods = []

    def __init__(self, name: str = '', **kwargs):
        self.name = name
        if kwargs:
            warnings.warn(('Providing attributes to DessiaObject __init__ to be stored in self is deprecated\n'
                           + 'Please store your attributes by yourself in your init'),
                          DeprecationWarning)

        # The code below has shown to be inefficient and should be remove in future version (0.16?)
        for property_name, property_value in kwargs.items():
            setattr(self, property_name, property_value)

    def base_dict(self):
        """ Base dict of the object, with just its name. """
        dict_ = SerializableObject.base_dict(self)
        dict_['name'] = self.name
        return dict_

    def __hash__(self):
        """ Compute a int from object. """
        if self._eq_is_data_eq:
            return self._data_hash()
        return object.__hash__(self)

    def __eq__(self, other_object):
        """
        Generic equality of two objects.

        Behavior can be controlled by class attribute _eq_is_data_eq to tell if we must use python equality (based on
        memory addresses) (_eq_is_data_eq = False) or a data equality (True).
        """
        if hash(self) != hash(other_object):
            return False
        if self._eq_is_data_eq:
            if self.__class__.__name__ != other_object.__class__.__name__:
                return False
            return self._data_eq(other_object)
        return object.__eq__(self, other_object)

    def _data_eq_dict(self):
        """ Returns a dict of what to look at for data eq. Keys in non data eq attributes are removed. """
        return {k: v for k, v in self._serializable_dict().items()
                if k not in self._non_data_eq_attributes + ['package_version', 'name']}

    def _data_eq(self, other_object) -> bool:
        """ Returns if the object is equal to the other object in the sense of data contained in the objects. """
        return data_eq(self, other_object)

    def _data_hash(self):
        """ Generic computation of hash based on data. """
        forbidden_keys = (self._non_data_eq_attributes + self._non_data_hash_attributes + ['package_version', 'name'])
        hash_ = sum(choose_hash(v) for k, v in self._serializable_dict().items() if k not in forbidden_keys)
        return int(hash_ % 1e5)

    def _data_diff(self, other_object):
        """
        Make a diff between two objects.

        returns: different values, missing keys in other object
        """
        return diff(self, other_object)

    def _get_from_path(self, path: str):
        """ Get object's deep attribute from given path. """
        return get_in_object_from_path(self, path)

    @classmethod
    def raw_schema(cls):
        """ Schema of class: transfer python data structure to web standard. """
        if hasattr(cls, '_jsonschema'):
            warnings.warn("Jsonschema is fully deprecated and you may want to use the new generic schema feature."
                          "Please consider so", DeprecationWarning)
            return cls._jsonschema
        return dcs.ClassSchema(cls)

    @classmethod
    def schema(cls):
        """ Serialized Schema as a dict. """
        return cls.raw_schema().to_dict()

    @classmethod
    def jsonschema(cls):
        """ Jsonschema of class: transfer python data structure to web standard. """
        warnings.warn("'jsonschema' method is deprecated. Use schema instead", DeprecationWarning)
        return cls.schema()

    @property
    def raw_method_schemas(self):
        """ Generate dynamic schemas for methods of class. """
        # TODO This should be a classmethod, but is a property to avoid collision with workflow.
        cls = self.__class__
        valid_method_names = [m for m in dir(cls) if not m.startswith('_') and m in cls._allowed_methods]
        schemas = {}
        for method_name in valid_method_names:
            method = getattr(cls, method_name)
            schemas[method_name] = dcs.MethodSchema(method)
        return schemas

    @property
    def method_schemas(self):
        """ Generate dynamic schemas for methods of class. """
        return {method_name: schema.to_dict() for method_name, schema in self.raw_method_schemas.items()}

    @property
    def _method_jsonschemas(self):
        """ Generates dynamic schemas for methods of class. """
        warnings.warn("method_jsonschema method is deprecated. Use method_schema instead", DeprecationWarning)
        return self.method_schemas

    def method_dict(self, method_name: str, method_jsonschema=None):
        """ Return a default dict for the given method name. """
        if method_jsonschema is not None:
            warnings.warn("method_jsonschema argument is deprecated and its use will be removed in a future version."
                          " Please remove it from your function call. Method name is sufficient to get schema",
                          DeprecationWarning)
        schema = self.raw_method_schemas[method_name]
        return schema.default_dict()

    def dict_to_arguments(self, dict_, method):
        """ Transform serialized argument of a method to python objects ready to use in method evaluation. """
        method_full_name = f"{self.full_classname}.{method}"
        if method_full_name in _fullargsspec_cache:
            args_specs = _fullargsspec_cache[method_full_name]
        else:
            method_object = getattr(self, method)
            args_specs = inspect.getfullargspec(method_object)
            _fullargsspec_cache[method_full_name] = args_specs

        allowed_args = args_specs.args[1:]

        arguments = {}
        for i, arg in enumerate(allowed_args):
            if str(i) in dict_:
                arg_specs = args_specs.annotations[arg]
                value = dict_[str(i)]
                try:
                    deserialized_value = deserialize_argument(arg_specs, value)
                except TypeError as err:
                    msg = "Error in deserialisation of value: "
                    msg += f"{value} of expected type {arg_specs}"
                    raise TypeError(msg) from err
                arguments[arg] = deserialized_value
        return arguments

    def save_to_file(self, filepath: str, indent: int = 2):
        """
        Save object to a JSON file.

        :param filepath: either a string representing the filepath or a stream
        """
        if not filepath.endswith('.json'):
            filepath += '.json'
            print(f'Changing name to {filepath}')
        with open(filepath, 'w', encoding='utf-8') as file:
            self.save_to_stream(file, indent=indent)

    def save_to_stream(self, stream, indent: int = 2):
        """ Write object to a stream. Default is to_dict, as a text like stream. """
        try:
            dict_ = self.to_dict(use_pointers=True)
        except TypeError:
            dict_ = self.to_dict()

        json.dump(dict_, stream, indent=indent)
        stream.write('\n')

    @classmethod
    def load_from_stream(cls, stream: dcf.JsonFile):
        """
        Generate object from stream using utf-8 encoding.

        Should be consistent with save_to_stream method.
        """
        warnings.warn("This method is deprecated and will be removed in a future version."
                      " Please use the `from_json_stream` method instead.", DeprecationWarning)
        return cls.from_json_stream(stream=stream)

    @classmethod
    def load_from_file(cls, filepath: str):
        """
        Load object from a JSON file.

        :param filepath: either a string representing the filepath or a stream
        """
        warnings.warn("This method is deprecated and will be removed in a future version."
                      " Please use the `from_json` method instead.", DeprecationWarning)
        return cls.from_json(filepath=filepath)

    @classmethod
    def from_json_stream(cls, stream: dcf.JsonFile):
        """
        Generate object from stream using utf-8 encoding.

        Should be consistent with save_to_stream method.
        """
        dict_ = json.loads(stream.read().decode('utf-8'))
        return cls.dict_to_object(dict_)

    @classmethod
    def from_json(cls, filepath: str):
        """
        Load object from a JSON file.

        :param filepath: either a string representing the filepath or a stream
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            dict_ = json.load(file)

        return cls.dict_to_object(dict_)

    @classmethod
    def from_xlsx_stream(cls, stream: dcf.BinaryFile):
        """
        Load object from a xlsx stream.

        :param stream: either a string representing the stream
        """
        reader = ExcelReader(stream)
        return reader.read_workbook()

    @classmethod
    def from_xlsx(cls, filepath: str):
        """
        Load object from a xlsx file.

        :param filepath: either a string representing the filepath
        """
        stream = dcf.BinaryFile.from_file(filepath=filepath)
        return cls.from_xlsx_stream(stream=stream)

    def check_list(self, level: str = 'error', check_platform: bool = True) -> dcc.CheckList:
        """ Return a list of potential info, warning and issues on the instance, that might be user custom. """
        check_list = dcc.CheckList([])

        if check_platform:
            check_list += self.check_platform()
        return check_list

    def is_valid(self, level: str = 'error') -> bool:
        """
        Return whether the object of valid 'above' given level.

        Default is error, but warnings can be forbidden.
        """
        return not self.check_list().checks_above_level(level=level)

    def copy(self, deep: bool = True, memo=None):
        """
        Copy object.

        :param deep: If False, perform a shallow copy. If True, perform a deep copy.
        :param memo: A dict that keep track of references.
        """
        if deep:
            return deepcopy(self, memo=memo)
        return copy(self)

    def __copy__(self):
        """ Generic copy use init of objects. """
        class_name = self.full_classname
        if class_name in _fullargsspec_cache:
            class_argspec = _fullargsspec_cache[class_name]
        else:
            class_argspec = inspect.getfullargspec(self.__class__)
            _fullargsspec_cache[class_name] = class_argspec

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
        """ Generic deep copy use init functions of objects. """
        class_name = self.full_classname
        if class_name in _fullargsspec_cache:
            class_argspec = _fullargsspec_cache[class_name]
        else:
            class_argspec = inspect.getfullargspec(self.__class__)
            _fullargsspec_cache[class_name] = class_argspec

        if memo is None:
            memo = {}
        dict_ = {}
        for arg in class_argspec.args:
            if arg != 'self':
                dict_[arg] = deepcopy_value(getattr(self, arg), memo=memo)
        return self.__class__(**dict_)

    def plot_data(self, reference_path: str = "#", **kwargs):
        """
        Base plot_data method. Overwrite this to display 2D or graphs on platform.

        Should return a list of plot_data's objects.
        """
        warnings.warn("Function 'plot_data' is deprecated and should not be used to compute graphs and 2D displays,"
                      "anymore. Please use Display Decorators, instead", DeprecationWarning)
        return []

    def plot(self, reference_path: str = "#", **kwargs):
        """ Generic plot getting plot_data function to plot. """
        if hasattr(self, 'plot_data'):
            import plot_data
            for data in self.plot_data(reference_path, **kwargs):
                plot_data.plot_canvas(plot_data_object=data,
                                      canvas_id='canvas',
                                      width=1400, height=900,
                                      debug_mode=False)
        else:
            msg = f"Class '{self.__class__.__name__}' does not implement a plot_data method to define what to plot"
            raise NotImplementedError(msg)

    def mpl_plot(self, **kwargs):
        """ Plot with matplotlib using plot_data function. """
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
            msg = f"Class '{self.__class__.__name__}' does not implement a plot_data method to define what to plot"
            raise NotImplementedError(msg)
        return axs

    @classmethod
    def display_settings(cls, **kwargs) -> List[DisplaySetting]:
        """ Return a list of objects describing how to call object displays. """
        settings = [DisplaySetting(selector="markdown", type_="markdown", method="to_markdown", load_by_default=True)]
        settings.extend(cls._display_settings_from_decorators())
        return settings

    @classmethod
    def _display_settings_from_decorator_name(cls, decorator_name: str):
        methods = get_decorated_methods(class_=cls, decorator_name=decorator_name)
        settings = []
        for method in methods:
            name = method.__name__
            type_ = getattr(method, "type_", False)
            serialize_data = getattr(method, "serialize_data", False)
            load_by_default = getattr(method, "load_by_default", False)
            selector = getattr(method, "selector", None)
            if selector is None:
                selector = name
            settings.append(DisplaySetting(selector=selector, type_=type_, method=name,
                                           serialize_data=serialize_data, load_by_default=load_by_default))
        return settings

    @classmethod
    def _display_settings_from_decorators(cls) -> List[DisplaySetting]:
        """ Return a list, computed from decorated functions, of objects describing how to call displays. """
        return [s for d in DISPLAY_DECORATORS for s in cls._display_settings_from_decorator_name(d)]

    def _display_from_selector(self, selector: str) -> DisplayObject:
        """ Generate the display from the selector. """
        display_setting = self._display_settings_from_selector(selector)
        track = ""
        try:
            data = attrmethod_getter(self, display_setting.method)(**display_setting.arguments)
        except Exception:
            data = None
            track = tb.format_exc()

        if display_setting.serialize_data:
            data = serialize(data)
        reference_path = display_setting.reference_path  # Trying this
        return DisplayObject(type_=display_setting.type, data=data, reference_path=reference_path, traceback=track)

    def _display_settings_from_selector(self, selector: str):
        """ Get display settings from given selector. """
        for display_setting in self.display_settings():
            if display_setting.selector == selector:
                return display_setting
        raise ValueError(f"No such selector '{selector}' in display of class '{self.__class__.__name__}'")

    def _displays(self) -> List[JsonSerializable]:
        """ Generate displays of the object to be plot in the DessiA Platform. """
        displays = []
        for display_setting in self.display_settings():
            display_ = self._display_from_selector(display_setting.selector)
            displays.append(display_.to_dict())
        return displays

    def to_markdown(self, **kwargs) -> str:
        """ Render a markdown of the object output type: string. """
        writer = MarkdownWriter(print_limit=25, table_limit=None)
        template = templates.dessia_object_markdown_template
        return template.substitute(name=self.name, class_=self.__class__.__name__, table=writer.object_table(self))

    def performance_analysis(self):
        """ Print time of rendering some commons operations (serialization, hash, displays). """
        print(f'### Performance analysis of object {self} ###')
        data_hash_time = time.time()
        self._data_hash()
        data_hash_time = time.time() - data_hash_time
        print(f'\t- data hash time: {round(data_hash_time, 3)} seconds')

        todict_time = time.time()
        dict_ = self.to_dict()
        todict_time = time.time() - todict_time
        print(f'\t- to_dict time: {round(todict_time, 3)} seconds')

        dto_time = time.time()
        self.dict_to_object(dict_)
        dto_time = time.time() - dto_time
        print(f'\t- dict_to_object time: {round(dto_time, 3)} seconds')

        for display_setting in self.display_settings():
            display_time = time.time()
            self._display_from_selector(display_setting.selector)
            display_time = time.time() - display_time
            print(f'\t- generation of display {display_setting.selector} in: {round(display_time, 6)} seconds')
        print('\n')

    def _check_platform(self, level='error'):
        return self.check_platform().raise_if_above_level(level=level)

    def check_platform(self):
        """ Reproduce life-cycle on platform (serialization, display). Raise an error if something is wrong. """
        serializable_results = dcc.check_serialization_process(object_=self, use_pointers=True)
        dict_ = serializable_results.pop("dict_")

        copy_results = dcc.check_copy(self)

        # Not refactoring this due to dependencies and cyclic imports
        print("Checking BSON validity...")
        start = time.time()
        valid, hint = is_bson_valid(stringify_dict_keys(dict_))
        if not valid:
            print("Failed.\n")
            check = dcc.FailedCheck(f"Object is not BSON valid {hint}")
        else:
            print("Object is BSON valid.")
            check = dcc.PassedCheck("Object is BSON valid")
        duration = time.time() - start
        print(f"Checked BSON validity in {duration}s.\n")
        bson_results = {"check": check, "duration": duration}

        display_results = dcc.check_displays(self)
        schemas_results = dcc.check_schemas(self)

        results = [serializable_results, copy_results, bson_results, display_results, schemas_results]
        duration = sum(r["duration"] for r in results)
        print(f"\nCompleted Platform Check in {duration}s.\n")
        return dcc.CheckList([r["check"] for r in results])

    def to_xlsx(self, filepath: str):
        """ Export the object to an XLSX file given by the filepath. """
        if not filepath.endswith('.xlsx'):
            filepath += '.xlsx'
            print(f'Changing name to {filepath}')

        with open(filepath, 'wb') as file:
            self.to_xlsx_stream(file)

    def to_xlsx_stream(self, stream):
        """ Export the object to an XLSX to a given stream. """
        writer = XLSXWriter(self)
        writer.save_to_stream(stream)

    def to_markdown_docx(self):
        """ Generates a word document from the object attributes, using a markdown-like syntax. """
        docx_writer = DocxWriter.from_markdown(markdown_text=self.to_markdown())
        return docx_writer

    def to_docx_stream(self, stream: dcf.BinaryFile):
        """ Saves the document to a binary stream. """
        document = self.to_markdown_docx()
        document.save_to_stream(stream=stream)

    def to_docx(self, filepath: str):
        """ Saves the document to a file. """
        if not filepath.endswith('.docx'):
            filepath += '.docx'
            print(f'Changing name to {filepath}')
        self.to_docx_stream(filepath)

    def zip_settings(self):
        """
        Returns a list of streams containing different representations of the object.

        Excel file stream generated by calling 'to_xlsx_stream' method.
        JSON file stream generated by calling 'save_to_stream' method.
        """
        streams = []
        for export_format in self._export_formats():
            if export_format.extension != "zip":
                method_name = export_format.method_name
                stream_class = dcf.StringFile if export_format.text else dcf.BinaryFile
                stream = stream_class(filename=f"export.{export_format.extension}")
                getattr(self, method_name)(stream)
                streams.append(stream)
        return streams

    def to_zip_stream(self, archive: dcf.BinaryFile) -> List[dcf.BinaryFile]:
        """ Generates a zipped archive of the streams returned by 'zip_settings' method. """
        streams = self.zip_settings()
        archive_name = 'export_zip'
        archive.filename = archive_name
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zip_archive:
            for value in streams:
                archive = dcf.generate_archive(zip_archive=zip_archive, value=value)
        return [archive]

    def to_zip(self, filepath: str):
        """ Saves the zipped archive generated by 'to_zip_stream' method to a file. """
        if not filepath.endswith('.zip'):
            filepath += '.zip'
            print(f'Changing name to {filepath}')
        archive = dcf.BinaryFile()
        self.to_zip_stream(archive)
        with open(filepath, 'wb') as file:
            file.write(archive.getbuffer())

    @classmethod
    def _export_formats_from_decorator_name(cls, decorator_name: str):
        methods = get_decorated_methods(class_=cls, decorator_name=decorator_name)
        settings = []
        for method in methods:
            extension = getattr(method, "extension")
            method_name = getattr(method, "method_name")
            text = getattr(method, "text")
            selector = getattr(method, "selector", None)
            if not selector:
                selector = extension
            settings.append(ExportFormat(selector=selector, extension=extension, method_name=method_name, text=text))
        return settings

    @classmethod
    def _export_formats_from_decorators(cls):
        """ Return a list, computed from decorated functions, of objects describing how to call exports. """
        return [s for d in EXPORT_DECORATORS for s in cls._export_formats_from_decorator_name(d)]

    def _export_formats(self) -> List[ExportFormat]:
        """ Return a list of objects describing how to call generic exports (.json, .xlsx). """
        formats = [ExportFormat(selector="json", extension="json", method_name="save_to_stream", text=True),
                   ExportFormat(selector="xlsx", extension="xlsx", method_name="to_xlsx_stream", text=False),
                   ExportFormat(selector="docx", extension="docx", method_name="to_docx_stream", text=False)]

        formats.extend(self._export_formats_from_decorators())
        formats.append(ExportFormat(selector="zip", extension="zip", method_name="to_zip_stream", text=False))
        return formats

    def save_export_to_file(self, selector: str, filepath: str):
        """ Generic export from selector to given filepath. Return real location filepath. """
        for export_format in self._export_formats():
            if export_format.selector == selector:
                if not filepath.endswith(f".{export_format.extension}"):
                    filepath += f".{export_format.extension}"
                    print(f'Renaming filepath to {filepath}')
                if export_format.text:
                    with open(filepath, mode="w", encoding="utf-8") as stream:
                        getattr(self, export_format.method_name)(stream, **export_format.args)
                else:
                    with open(filepath, mode="wb") as stream:
                        getattr(self, export_format.method_name)(stream, **export_format.args)
                return filepath
        raise ValueError(f'Export selector not found: {selector}')

    def to_vector(self):
        """ Compute vector from object. """
        vectored_objects = []
        for feature in self.vector_features():
            vectored_objects.append(get_in_object_from_path(self, feature.lower()))
        return vectored_objects

    @classmethod
    def vector_features(cls):
        """ Get a list of vector features, or generate a default one. """
        if cls._vector_features is None:
            return list(set(get_attribute_names(cls)).difference(get_attribute_names(DessiaObject)))
        return cls._vector_features


class PhysicalObject(DessiaObject):
    """ Represent an object with CAD capabilities. """

    @classmethod
    def display_settings(cls, **kwargs):
        """ Returns a list of DisplaySettings objects describing how to call sub-displays. """
        display_settings = super().display_settings()
        display_settings.append(DisplaySetting(selector='cad', type_='babylon_data',
                                               method='volmdlr_volume_model().babylon_data', serialize_data=True))
        return display_settings

    def volmdlr_primitives(self, **kwargs):
        """ Return a list of volmdlr primitives to build up volume model. """
        return []

    def volmdlr_volume_model(self, **kwargs):
        """ Gives the volmdlr VolumeModel. """
        import volmdlr as vm  # !!! Avoid circular imports, is this OK ?
        return vm.core.VolumeModel(self.volmdlr_primitives(**kwargs))

    def to_step(self, filepath: str):
        """
        Exports the CAD of the object to step. Works if the class define a custom volmdlr model.

        :param filepath: a str representing a filepath
        """
        if not filepath.endswith('.step'):
            filepath += '.step'
        with open(filepath, 'w', encoding='utf-8') as file:
            self.to_step_stream(stream=file)

    def to_step_stream(self, stream):
        """
        Export object CAD to given stream in STEP format.

        Works if the class define a custom volmdlr model.
        """
        return self.volmdlr_volume_model().to_step_stream(stream=stream)

    def to_html_stream(self, stream: dcf.StringFile):
        """ Exports Object CAD to given stream as HTML. """
        model = self.volmdlr_volume_model()
        babylon_data = model.babylon_data()
        script = model.babylonjs_script(babylon_data)
        stream.write(script)

        return stream

    def to_stl_stream(self, stream):
        """ Export Object CAD to given stream as STL. """
        return self.volmdlr_volume_model().to_stl_stream(stream=stream)

    def to_stl(self, filepath: str):
        """
        Exports the CAD of the object to STL. Works if the class define a custom volmdlr model.

        :param filepath: a str representing a filepath
        """
        if not filepath.endswith('.stl'):
            filepath += '.stl'

        with open(filepath, 'wb', encoding='utf-8') as file:
            self.to_stl_stream(stream=file)

    def babylonjs(self, use_cdn=True, debug=False, **kwargs):
        """ Show the 3D volmdlr of an object by calling volmdlr_volume_model method and plot in in browser. """
        self.volmdlr_volume_model(**kwargs).babylonjs(use_cdn=use_cdn, debug=debug)

    def save_babylonjs_to_file(self, filename: str = None, use_cdn: bool = True, debug: bool = False, **kwargs):
        """
        Saves the 3D volmdlr of an object in a file.

        :param filename: The file's name. Default value is None
        :type filename: str, optional
        :param use_cdn: Activates the use of a content delivery network. Default value is True
        :type use_cdn: bool, optional
        :param debug: Activates the debug mode. Default value is False
        :type debug: bool, optional
        """
        self.volmdlr_volume_model(**kwargs).save_babylonjs_to_file(filename=filename, use_cdn=use_cdn, debug=debug)

    def _export_formats(self) -> List[ExportFormat]:
        """ Return a list of objects describing how to call 3D exports. """
        formats = DessiaObject._export_formats(self)
        formats3d = [ExportFormat(selector="step", extension="step", method_name="to_step_stream", text=True),
                     ExportFormat(selector="stl", extension="stl", method_name="to_stl_stream", text=False),
                     ExportFormat(selector="html", extension="html", method_name="to_html_stream", text=True)]
        formats.extend(formats3d)
        return formats


class MovingObject(PhysicalObject):
    """ A 3D object which display can move down a path from according to defined steps. """

    def volmdlr_primitives_step_frames(self):
        """ Return a list of volmdlr primitives to build up volume model. """
        raise NotImplementedError('Object inheriting MovingObject should implement volmdlr_primitives_step_frames')

    def volmdlr_volume_model(self, **kwargs):
        """ Volume model of Moving Object. """
        import volmdlr as vm  # !!! Avoid circular imports, is this OK ?
        return vm.core.MovingVolumeModel(self.volmdlr_primitives(**kwargs), self.volmdlr_primitives_step_frames())


class Parameter(DessiaObject):
    """ A value from a Parameter Set. """

    def __init__(self, lower_bound, upper_bound, periodicity=None, name=''):
        DessiaObject.__init__(self, name=name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.periodicity = periodicity

    def random_value(self):
        """ Sample a value within the bounds. """
        return random.uniform(self.lower_bound, self.upper_bound)

    def are_values_equal(self, value1, value2, tol=1e-2):
        """ Check equality according to given tolerance. """
        if self.periodicity is not None:
            value1 = value1 % self.periodicity
            value2 = value2 % self.periodicity

        return math.isclose(value1, value2, abs_tol=tol)

    def normalize(self, value):
        """ Translate the value to a [0, 1] interval where 0 are the lower and upper bounds. """
        normalized_value = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
        return normalized_value

    def original_value(self, normalized_value):
        """ Revert normalized value to its original value. """
        value = normalized_value * (self.upper_bound - self.lower_bound) + self.lower_bound
        return value

    def optimizer_bounds(self):
        """ Compute optimize bounds. """
        if self.periodicity is not None:
            return self.lower_bound - 0.5 * self.periodicity, self.upper_bound + 0.5 * self.periodicity
        return None


class ParameterSet(DessiaObject):
    """ Object that can provide miscellaneous features around values Dataset. """

    def __init__(self, values, name=''):
        self.values = values

        DessiaObject.__init__(self, name=name)

    @property
    def parameters(self):
        """ Compute parameters from ParameterSet values. """
        parameters = [Parameter(min(v), max(v), name=k) for k, v in self.values.items()]
        return parameters

    @property
    def means(self):
        """ Compute means on all parameters defined in Set. """
        means = {k: sum(v) / len(v) for k, v in self.values.items()}
        return means


class DessiaFilter(DessiaObject):
    """
    Base class for filters working on lists of DessiaObjects (List[DessiaObject]).

    :param attribute: Name of attribute on which to filter
    :type attribute: str

    :param comparison_operator: Comparison operator
    :type comparison_operator: str

    :param bound: The bound value to compare `'attribute'` of DessiaObjects of a list with `'comparison_operator'`
    :type bound: float

    :param name: Name of filter
    :type name: `str`, `optional`, defaults to `''`

    :Comparison operators:
        * greater than: >=, gte, ge
        * greater: >, gt
        * lower than: <=, lte, le
        * lower: <, lt
        * lower: <, lt
        * equal: ==, eq
        * different: !=, ne
    """

    _REAL_OPERATORS = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le, '==': operator.eq,
                       '!=': operator.ne, 'gt': operator.gt, 'lt': operator.lt, 'ge': operator.ge, 'le': operator.le,
                       'eq': operator.eq, 'ne': operator.ne, 'gte': operator.ge, 'lte': operator.le}

    def __init__(self, attribute: str, comparison_operator: str, bound: float, name: str = ''):
        self.attribute = attribute
        self.comparison_operator = comparison_operator
        self.bound = bound
        DessiaObject.__init__(self, name=name)

    def __str__(self, offset_attr: int = 10, offset_boun: int = 0):
        offset_oper = 0
        if offset_boun == 0:
            offset_boun = len(str(self.bound)) + 2
        string_operator = {'>': '>', '<': '<', '>=': '>=', '<=': '<=', '==': '==', '!=': '!=', 'gt': '>', 'lt': '<',
                           'ge': '>=', 'le': '<=', 'eq': '==', 'ne': '!=', 'gte': '>=', 'lte': '<='}
        printed_operator = string_operator[self.comparison_operator]
        return (self.attribute + " " * (offset_attr - len(self.attribute)) +
                printed_operator + " " * (offset_oper - len(printed_operator)) +
                " " * (offset_boun - len(str(self.bound))) + str(self.bound))

    def __hash__(self):
        hash_ = len(self.attribute)
        hash_ += hash(self.comparison_operator)
        hash_ += hash(self.bound)
        return int(hash_)

    def __eq__(self, other: 'DessiaFilter'):
        same_attr = self.attribute == other.attribute
        same_op = self.comparison_operator == other.comparison_operator
        same_bound = self.bound == other.bound
        return same_attr and same_op and same_bound

    def _comparison_operator(self):
        return self._REAL_OPERATORS[self.comparison_operator]

    def _to_lambda(self):
        return lambda x: (self._comparison_operator()(get_in_object_from_path(value, f'#/{self.attribute}'),
                                                      self.bound) for value in x)

    def get_booleans_index(self, values: List[DessiaObject]):
        """
        Get the Boolean indexing of a filtered list.

        :param values: List of DessiaObjects to filter
        :type values: List[DessiaObject]

        :return: `list of length `len(values)` where the elements are `True` if kept by the filter, otherwise `False`.
        :rtype: List[bool]

        :Examples:
        >>> from dessia_common.core import DessiaFilter
        >>> from dessia_common.models import all_cars_no_feat
        >>> values = all_cars_no_feat[:5]
        >>> filter_ = DessiaFilter('weight', '<=', 3500.)
        >>> filter_.get_booleans_index(values)
        [False, False, True, True, True]
        """
        return list(self._to_lambda()(values))

    @staticmethod
    def booleanlist_to_indexlist(booleans_list: List[int]):  # TODO: Should it exist ?
        """
        Transform a Boolean list to an index list.

        :param booleans_list: list of length `len(values)` where elements are `True` if kept, otherwise `False`.
        :type booleans_list: List[int]

        :return: list of kept indexes
        :rtype: List[int]

        :Examples:
        >>> from dessia_common.core import DessiaFilter
        >>> from dessia_common.models import all_cars_no_feat
        >>> values = all_cars_no_feat[:5]
        >>> filter_ = DessiaFilter('weight', '<=', 3500.)
        >>> booleans_list = filter_.get_booleans_index(values)
        [False, False, True, True, True]
        >>> DessiaFilter.booleanlist_to_indexlist(booleans_list)
        [2, 3, 4]
        """
        return list(itertools.compress(range(len(booleans_list)), booleans_list))

    @staticmethod
    def apply(values: List[DessiaObject], booleans_list: List[List[bool]]):
        """
        Apply a Dessia Filter on a list of DessiaObjects.

        :param values: List of DessiaObjects to filter
        :type values: List[DessiaObject]

        :param booleans_list: list of length `len(values)` where elements are `True` if kept, otherwise `False`.
        :type booleans_list: List[List[bool]]

        :return: List of filtered values
        :rtype: List[DessiaObject]

        :Examples:
        >>> from dessia_common.core import DessiaFilter
        >>> from dessia_common.models import all_cars_no_feat
        >>> values = all_cars_no_feat[:5]
        >>> filter_ = DessiaFilter('weight', '<=', 3500.)
        >>> booleans_list = filter_.get_booleans_index(values)
        [False, False, True, True, True]
        >>> for car in DessiaFilter.apply(values, booleans_list): print(car.weight)
        3436.0, 3433.0, 3449.0
        """
        return list(itertools.compress(values, booleans_list))


class FiltersList(DessiaObject):
    """
    Combine several filters stored as a list of DessiaFilters with a logical operator.

    :param filters: List of DessiaFilters to combine
    :type filters: List[DessiaFilter]

    :param logical_operator: Logical operator to combine filters
    :type logical_operator: str

    :param name: Name of FiltersList
    :type name: `str`, `optional`, defaults to `''`

    :Logical operators: `'and'`, `'or'`, `'xor'`
    """

    _standalone_in_db = True

    def __init__(self, filters: List[DessiaFilter], logical_operator: str = 'and', name: str = ''):
        self.filters = filters
        self.logical_operator = logical_operator
        DessiaObject.__init__(self, name=name)

    def __len__(self):
        return len(self.filters)

    def __str__(self):
        print_lim = 15
        len_attr = max(map(len, [filter_.attribute for filter_ in self.filters]))
        len_numb = max(map(len, [str(filter_.bound) for filter_ in self.filters]))
        prefix = f"{self.__class__.__name__} {self.name if self.name != '' else hex(id(self))}: "
        prefix += f"{len(self)} filters combined with '" + self.logical_operator + "' operator :\n"
        string = ""
        for filter_ in self.filters[:print_lim]:
            string += " " * 3 + "- "
            string += filter_.__str__(len_attr + 2, len_numb + 2)
            string += "\n"
        return prefix + string

    @classmethod
    def from_filters_list(cls, filters: List[DessiaFilter], logical_operator: str = 'and', name: str = ''):
        """
        Compute a FilersList from an already built list of DessiaFilter.

        :param filters: List of DessiaFilters to combine
        :type filters: List[DessiaFilter]

        :param logical_operator: Logical operator to combine filters (`'and'`, `'or'` or `'xor'`)
        :type logical_operator: `str`, `optional`, defaults to `'and'`

        :param name: Name of FiltersList
        :type name: `str`, `optional`, defaults to `''`

        :return: A new instantiated list of DessiaFilter
        :rtype: FiltersList

        :Examples:
        >>> from dessia_common.core import DessiaFilter, FiltersList
        >>> filters = [DessiaFilter('weight', '<=', 3500.), DessiaFilter('mpg', '<=', 40.)]
        >>> filters_list = FiltersList(filters, logical_operator="or", name="example")
        >>> print(filters_list)
        FiltersList example: 2 filters combined with 'or' operator :
           - weight  <=  3500.0
           - mpg     <=    40.0
        """
        return cls(filters=filters, logical_operator=logical_operator, name=name)

    @staticmethod
    def combine_booleans_lists(booleans_lists: List[List[bool]], logical_operator: str = "and") -> List[bool]:
        """
        Combine a list of Boolean indices with the logical operator into a simple Boolean index.

        :param booleans_lists: List of boolean indices

        :param logical_operator: Logical operator to combine filters (`'or'`, `'and'` or `'xor'`)

        :raises NotImplementedError: If logical_operator is not one of `'and'`, `'or'`, `'xor'`, raises an error

        :return: Boolean indices of the filtered data

        :Examples:
        >>> from dessia_common.core import FiltersList
        >>> booleans_lists = [[True, True, False, False], [False, True, True, False]]
        >>> FiltersList.combine_booleans_lists(booleans_lists, logical_operator="xor")
        [True, False, True, False]
        """
        if logical_operator == 'and':
            return [all(booleans_tuple) for booleans_tuple in zip(*booleans_lists)]
        if logical_operator == 'or':
            return [any(booleans_tuple) for booleans_tuple in zip(*booleans_lists)]
        if logical_operator == 'xor':
            return [bool(sum(booleans_tuple)) for booleans_tuple in zip(*booleans_lists)]
        raise NotImplementedError(f"'{logical_operator}' str for 'logical_operator' attribute is not a use case")

    def get_booleans_index(self, dobjects_list: List[DessiaObject]):
        """
        Compute all the filters of `self.filters` on `dobjects_list` and return boolean indices of `dobjects_list`.

        :param dobject_list: List of data to filter
        :type dobject_list: List[DessiaObject]

        :return: A `booleans index` of `dobjects_list` of the list of data to filter (`dobjects_list`)
        :rtype: List[bool]

        :Examples:
        >>> from dessia_common.core import FiltersList
        >>> from dessia_common.models import all_cars_no_feat
        >>> dobjects_list = all_cars_no_feat[:5]
        >>> filters = [DessiaFilter('weight', '<=', 4000.), DessiaFilter('mpg', '>=', 30.)]
        >>> filters_list = FiltersList(filters, logical_operator="xor", name="example")
        >>> filters_list.get_booleans_index(dobjects_list)
        [True, True, True, True, True]
        """
        booleans_index = []
        for filter_ in self.filters:
            booleans_index.append(filter_.get_booleans_index(dobjects_list))
        return self.__class__.combine_booleans_lists(booleans_index, self.logical_operator)

    def apply(self, dobjects_list: List[DessiaObject]):
        """
        Apply a FiltersList on a list of DessiaObjects.

        :param dobjects_list: List of DessiaObjects to filter
        :type dobjects_list: List[DessiaObject]

        :return: List of filtered values
        :rtype: List[DessiaObject]

        :Examples:
        >>> from dessia_common.core import FiltersList
        >>> from dessia_common.datatools.dataset import Dataset
        >>> from dessia_common.models import all_cars_wi_feat
        >>> filters = [DessiaFilter('weight', '<=', 1650.), DessiaFilter('mpg', '>=', 45.)]
        >>> filters_list = FiltersList(filters, logical_operator="xor", name="example")
        >>> filtered_cars = filters_list.apply(all_cars_wi_feat)
        >>> print(Dataset(filtered_cars, name="example"))
        Dataset example: 3 samples, 5 features
        |         Mpg         |    Displacement    |     Horsepower     |       Weight       |    Acceleration    |
        -----------------------------------------------------------------------------------------------------------
        |               35.0  |             0.072  |              69.0  |            1613.0  |              18.0  |
        |               31.0  |             0.076  |              52.0  |            1649.0  |              16.5  |
        |               46.6  |             0.086  |              65.0  |            2110.0  |              17.9  |
        """
        booleans_index = self.get_booleans_index(dobjects_list)
        return DessiaFilter.apply(dobjects_list, booleans_index)


def stringify_dict_keys(obj):
    """ Stringify dict keys. """
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


def type_from_annotation(type_, module):
    """ Clean up a proposed type if there are stringified. """
    if isinstance(type_, str):
        # Evaluating types
        type_ = dcs.TYPES_FROM_STRING.get(type_, default=getattr(import_module(module), type_))
    return type_


def prettyname(namestr):
    """ Create a pretty name from as str. """
    warnings.warn("prettyname function has been moved to 'helpers' module. Use it instead", DeprecationWarning)
    return dch.prettyname(namestr)


def inspect_arguments(method, merge=False):
    """
    Find default value and required arguments of class construction.

    Get method arguments and default arguments as sequences while removing forbidden ones (self, cls...).
    """
    warnings.warn("Method 'inspect_arguments' have been moved to dessia_common/schemas."
                  "Use it instead instead", DeprecationWarning)
    return dcs.inspect_arguments(method=method, merge=merge)


def split_default_args(argspecs, merge: bool = False):
    """
    Find default value and required arguments of class construction.

    Get method arguments and default arguments as sequences while removing forbidden ones (self, cls...).
    """
    warnings.warn("Method 'split_default_args' have been moved to dessia_common/schemas."
                  "Use it instead instead", DeprecationWarning)
    return dcs.split_default_args(argspecs=argspecs, merge=merge)


def split_argspecs(argspecs) -> Tuple[int, int]:
    """ Get number of regular arguments as well as arguments with default values. """
    warnings.warn("Method 'split_argspecs' have been moved to dessia_common/schemas."
                  "Use it instead instead", DeprecationWarning)
    return dcs.split_argspecs(argspecs=argspecs)


def get_attribute_names(object_class):
    """ Get all attributes of a class which are present in __init__ or numeric attributes and not in parent class. """
    attributes = [attribute[0] for attribute in inspect.getmembers(object_class, lambda x: not inspect.isroutine(x))
                  if not attribute[0].startswith('__')
                  and not attribute[0].endswith('__')
                  and isinstance(attribute[1], (float, int, complex, bool))]
    subclass_attributes = {name: param for name, param in inspect.signature(object_class.__init__).parameters.items()
                           if type in inspect.getmro(param.annotation.__class__)}
    subclass_numeric_attributes = [name for name, param in subclass_attributes.items()
                                   if any(item in inspect.getmro(param.annotation)
                                          for item in [float, int, bool, complex])]
    attributes += [a for a in subclass_numeric_attributes if a not in dcs.RESERVED_ARGNAMES]
    return attributes
