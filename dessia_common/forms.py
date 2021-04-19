"""
This is a module that aims to list all
possibilities of data formats offered by Dessia.
It should be used as a repertory of rules and available typings.

Some general rules :

- Lists are homogeneous, ie, they should not
  contain several types of elements
    ex : List[int], List[str], List[CustomClass], ...
- Tuples can, therefore, be used as heterogenous sequences,
  thanks to the fact that they are immutables.
    ex : Tuple[str, int, CustomClass] is a tuple like this :
        t = ('tuple', 1, custom_class_object)
- Dict is used whenever a dynamic structure should be defined.
  It takes only one possible type for keys
  and one possible type for values.
    ex : Dict[str, bool] is a dict like :
        d = {'key0': True, 'key1': False, 'another_key': False,...}
- As opposed to this, TypedDict defines a static structure,
  with a defined number of given, expected keys & types of their values.
    ex :
    class StaticDict(TypedDict):
        name: str
        value: float

In addition to types & genericity (brought by DessiaObject),
this module can also be seen as a template for Dessia's
coding/naming style & convention.
"""

from math import floor, ceil
from typing import Dict, List, Tuple, Union

try:
    import volmdlr as vm
    from volmdlr.wires import Contour2D
    from volmdlr import primitives2d as p2d
    from volmdlr import primitives3d as p3d
    import plot_data
    from plot_data.colors import *
except:
    pass

from dessia_common import DessiaObject
from dessia_common.typings import InstanceOf, Distance
from dessia_common.vectored_objects import Catalog

try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7

from numpy import linspace
from math import cos


class StandaloneSubobject(DessiaObject):
    _standalone_in_db = True
    _generic_eq = True

    def __init__(self, floatarg: Distance, name: str = 'Standalone Subobject'):
        self.floatarg = floatarg

        DessiaObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'StandaloneSubobject':
        floatarg = Distance(1.7 * seed)
        name = 'StandaloneSubobject' + str(seed)
        return cls(floatarg=floatarg, name=name)

    @classmethod
    def generate_many(cls, seed: int) -> List['StandaloneSubobject']:
        subobjects = [cls.generate((i+1)*1000) for i in range(seed)]
        return subobjects

    def contour(self):
        points = [vm.Point2D(0, 0), vm.Point2D(0, 1),
                  vm.Point2D(1, 1), vm.Point2D(1, 0)]

        crls = p2d.ClosedRoundedLineSegments2D(points=points, radius={})
        return vm.wires.Contour2D(crls.primitives)

    def voldmlr_primitives(self):
        contour = self.contour()
        volumes = [p3d.ExtrudedProfile(vm.O3D, vm.X3D, vm.Z3D,
                                       contour, [], vm.Y3D)]
        return volumes


DEF_SS = StandaloneSubobject.generate(1)


class EnhancedStandaloneSubobject(StandaloneSubobject):
    def __init__(self, floatarg: Distance, boolarg: bool,
                 name: str = 'Standalone Subobject'):
        self.boolarg = boolarg

        StandaloneSubobject.__init__(self, floatarg=floatarg, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'EnhancedStandaloneSubobject':
        floatarg = Distance(1.2 * seed)
        boolarg = floatarg.is_integer()
        name = 'EnhancedStandaloneSubobject' + str(seed)
        return cls(floatarg=floatarg, boolarg=boolarg, name=name)


DEF_ESS = EnhancedStandaloneSubobject.generate(1)


class InheritingStandaloneSubobject(StandaloneSubobject):
    def __init__(self, floatarg: Distance, strarg: str,
                 name: str = 'Inheriting Standalone Subobject'):
        self.strarg = strarg

        StandaloneSubobject.__init__(self, floatarg=floatarg, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'InheritingStandaloneSubobject':
        floatarg = Distance(0.7 * seed)
        strarg = str(-seed)
        name = 'Inheriting Standalone Subobject' + str(seed)
        return cls(floatarg=floatarg, strarg=strarg, name=name)


DEF_ISS = InheritingStandaloneSubobject.generate(1)


class EmbeddedSubobject(DessiaObject):
    def __init__(self, embedded_list: List[int] = None,
                 name: str = 'Embedded Subobject'):
        if embedded_list is None:
            self.embedded_list = [1, 2, 3]
        else:
            self.embedded_list = embedded_list

        DessiaObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'EmbeddedSubobject':
        if not bool(seed % 2):
            embedded_list = list(range(int(seed/2)))
        else:
            embedded_list = None
        name = 'Embedded Subobject' + str(seed)
        return cls(embedded_list=embedded_list, name=name)

    @classmethod
    def generate_many(cls, seed: int) -> List['EmbeddedSubobject']:
        return [cls.generate(i) for i in range(ceil(seed/3))]


DEF_ES = EmbeddedSubobject.generate(10)


class StaticDict(DessiaObject):
    def __init__(self, float_value: float, int_value: int,
                 is_valid: bool, name: str = ''):
        self.float_value = float_value
        self.int_value = int_value
        self.is_valid = is_valid
        DessiaObject.__init__(self, name=name)


DEF_SD = StaticDict(name="Default SD Name", float_value=1.3,
                    int_value=0, is_valid=True)


UnionArg = Union[StandaloneSubobject, EnhancedStandaloneSubobject]


class StandaloneObject(DessiaObject):
    """
    Dev Object for testing purpose

    :param standalone_subobject: A dev subobject that is standalone_in_db
    :type standalone_subobject: StandaloneSubobject
    :param embedded_subobject: A dev subobject that isn't standalone_in_db
    :type embedded_subobject: EmbeddedSubobject
    :param dynamic_dict: A variable length dict
    :type dynamic_dict: Dict[str, bool]
    :param static_dict: A 1-level structurewith only builtin values & str keys
    :type static_dict: StaticDict
    :param tuple_arg: A heterogeneous sequence
    :type tuple_arg: tuple
    """
    _standalone_in_db = True
    _generic_eq = True
    _allowed_methods = ['add_standalone_object',
                        'add_embedded_object', 'add_float']

    def __init__(self, standalone_subobject: StandaloneSubobject,
                 embedded_subobject: EmbeddedSubobject,
                 dynamic_dict: Dict[str, bool], static_dict: StaticDict,
                 tuple_arg: Tuple[str, int], intarg: int, strarg: str,
                 object_list: List[StandaloneSubobject],
                 subobject_list: List[EmbeddedSubobject],
                 builtin_list: List[int],
                 union_arg: UnionArg,
                 subclass_arg: InstanceOf[StandaloneSubobject],
                 name: str = 'Standalone Object Demo'):
        self.union_arg = union_arg
        self.builtin_list = builtin_list
        self.subobject_list = subobject_list
        self.object_list = object_list
        self.tuple_arg = tuple_arg
        self.strarg = strarg
        self.intarg = intarg
        self.static_dict = static_dict
        self.dynamic_dict = dynamic_dict
        self.standalone_subobject = standalone_subobject
        self.embedded_subobject = embedded_subobject
        self.subclass_arg = subclass_arg

        DessiaObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int):
        is_even = not bool(seed % 2)
        standalone_subobject = StandaloneSubobject.generate(seed)
        embedded_subobject = EmbeddedSubobject.generate(seed)
        dynamic_dict = {'n'+str(i): bool(seed % 2) for i in range(seed)}
        static_dict = StaticDict(name='Object'+str(seed),
                                 float_value=seed * 1.3,
                                 int_value=seed, is_valid=is_even)
        tuple_arg = ('value', seed * 3)
        intarg = seed
        strarg = str(seed) * floor(seed/3)
        object_list = StandaloneSubobject.generate_many(seed)
        subobject_list = EmbeddedSubobject.generate_many(seed)
        builtin_list = [seed]*seed
        union_arg = EnhancedStandaloneSubobject.generate(seed)
        if is_even:
            subclass_arg = StandaloneSubobject.generate(-seed)
        else:
            subclass_arg = InheritingStandaloneSubobject.generate(seed)
        return cls(standalone_subobject=standalone_subobject,
                   embedded_subobject=embedded_subobject,
                   dynamic_dict=dynamic_dict, static_dict=static_dict,
                   tuple_arg=tuple_arg, intarg=intarg, strarg=strarg,
                   object_list=object_list, subobject_list=subobject_list,
                   builtin_list=builtin_list, union_arg=union_arg,
                   subclass_arg=subclass_arg)

    def add_standalone_object(self, object_: StandaloneSubobject):
        """
        This methods adds a standalone object to object_list.
        It doesn't return anything, hence, API will update object when
        computing from frontend
        """
        self.object_list.append(object_)

    def add_embedded_object(self, object_: EmbeddedSubobject):
        """
        This methods adds an embedded object to subobject_list.
        It doesn't return anything, hence, API will update object
        when computing from frontend
        """
        self.subobject_list.append(object_)

    def add_float(self, value: float) -> StandaloneSubobject:
        """
        This methods adds value to its standalone subobject
        floatarg property and returns it.
        API should replace standalone_subobject as it is returned
        """
        self.standalone_subobject.floatarg += value
        return self.standalone_subobject

    def volmdlr_primitives(self):
        return self.standalone_subobject.voldmlr_primitives()

    def plot_data(self):
        attributes = ['cx', 'cy']

        # Contour
        contour = self.standalone_subobject.contour().plot_data()
        primitives_group = plot_data.PrimitiveGroup(primitives=[contour],
                                                    name='Contour')

        # Scatter Plot
        bounds = {'x': [0, 6], 'y': [100, 2000]}
        catalog = Catalog.random_2d(bounds=bounds, threshold=8000)
        points = [plot_data.Point2D(cx=v[0], cy=v[1], name='Point'+str(i))
                  for i, v in enumerate(catalog.array)]
        axis = plot_data.Axis()
        tooltip = plot_data.Tooltip(to_disp_attribute_names=attributes,
                                    name='Tooltips')
        scatter_plot = plot_data.Scatter(axis=axis, tooltip=tooltip,
                                         elements=points,
                                         to_disp_attribute_names=attributes,
                                         name='Scatter Plot')

        # Parallel Plot
        attributes = ['cx', 'cy', 'color_fill', 'color_stroke']
        parallel_plot = plot_data.ParallelPlot(elements=points,
                                               to_disp_attribute_names=attributes,
                                               name='Parallel Plot')

        # Multi Plot
        objects = [scatter_plot, parallel_plot]
        sizes = [plot_data.Window(width=560, height=300),
                 plot_data.Window(width=560, height=300)]
        coords = [(0, 0), (300, 0)]
        multi_plot = plot_data.MultiplePlots(elements=points, plots=objects,
                                             sizes=sizes, coords=coords,
                                             name='Multiple Plot')

        attribute_names = ['time', 'electric current']
        tooltip = plot_data.Tooltip(to_disp_attribute_names=attribute_names)
        time1 = linspace(0, 20, 20)
        current1 = [t ** 2 for t in time1]
        elements1 = []
        for time, current in zip(time1, current1):
            elements1.append({'time': time, 'electric current': current})

        # The previous line instantiates a dataset with limited arguments but
        # several customizations are available
        point_style = plot_data.PointStyle(color_fill=RED, color_stroke=BLACK)
        edge_style = plot_data.EdgeStyle(color_stroke=BLUE, dashline=[10, 5])

        custom_dataset = plot_data.Dataset(elements=elements1, name='I = f(t)',
                                           tooltip=tooltip,
                                           point_style=point_style,
                                           edge_style=edge_style)

        # Now let's create another dataset for the purpose of this exercice
        time2 = linspace(0, 20, 100)
        current2 = [100 * (1 + cos(t)) for t in time2]
        elements2 = []
        for time, current in zip(time2, current2):
            elements2.append({'time': time, 'electric current': current})

        dataset2 = plot_data.Dataset(elements=elements2, name='I2 = f(t)')

        graph2d = plot_data.Graph2D(graphs=[custom_dataset, dataset2],
                                    to_disp_attribute_names=attribute_names)
        return [primitives_group, scatter_plot,
                parallel_plot, multi_plot, graph2d]

    def maldefined_method(self, arg0, arg1=1, arg2: int = 10, arg3 = 3):
        """
        Defining a docstring for testing parsing purpose
        """
        nok_string = "This is a bad coding behavior"
        ok_string = "This could be OK as temporary attr"
        self.maldefined_attr = nok_string
        self._ok_attribute = ok_string

        computation = nok_string + 'or' + ok_string

        return computation


DEF_SO = StandaloneObject.generate(1)


class StandaloneObjectWithDefaultValues(StandaloneObject):
    def __init__(self, standalone_subobject: StandaloneSubobject = DEF_SS,
                 embedded_subobject: EmbeddedSubobject = DEF_ES,
                 dynamic_dict: Dict[str, bool] = None,
                 static_dict: StaticDict = DEF_SD,
                 tuple_arg: Tuple[str, int] = ("Default Tuple", 0),
                 intarg: int = 1, strarg: str = "Default Strarg",
                 object_list: List[StandaloneSubobject] = None,
                 subobject_list: List[EmbeddedSubobject] = None,
                 builtin_list: List[int] = None,
                 union_arg: UnionArg = DEF_ESS,
                 subclass_arg: InstanceOf[StandaloneSubobject] = DEF_ISS,
                 name: str = 'Standalone Object Demo'):
        if dynamic_dict is None:
            dynamic_dict = {}
        if object_list is None:
            object_list = [DEF_SS]
        if subobject_list is None:
            subobject_list = [DEF_ES]
        if builtin_list is None:
            builtin_list = [1, 2, 3, 4, 5]
        StandaloneObject.__init__(
            self, standalone_subobject=standalone_subobject,
            embedded_subobject=embedded_subobject, dynamic_dict=dynamic_dict,
            static_dict=static_dict, tuple_arg=tuple_arg, intarg=intarg,
            strarg=strarg, object_list=object_list,
            subobject_list=subobject_list, builtin_list=builtin_list,
            union_arg=union_arg, subclass_arg=subclass_arg, name=name
        )


DEF_SOWDV = StandaloneObjectWithDefaultValues()


class Generator(DessiaObject):
    def __init__(self, parameter: int, nb_solutions: int = 25, name: str = ''):
        self.parameter = parameter
        self.nb_solutions = nb_solutions
        self.models = None

        DessiaObject.__init__(self, name=name)

    def generate(self) -> List[StandaloneObject]:
        # submodels = [Submodel(self.parameter * i)
        #              for i in range(self.nb_solutions)]
        self.models = [StandaloneObject.generate(self.parameter + i)
                       for i in range(self.nb_solutions)]
        return self.models


class Optimizer(DessiaObject):
    def __init__(self, model_to_optimize: StandaloneObject, name: str = ''):
        self.model_to_optimize = model_to_optimize

        DessiaObject.__init__(self, name=name)

    def optimize(self, optimization_value: int = 3) -> int:
        self.model_to_optimize.intarg += optimization_value
        return self.model_to_optimize.intarg
