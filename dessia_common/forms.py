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
from dessia_common.typings import Subclass
from dessia_common.vectored_objects import Catalog

try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7


class StandaloneSubobject(DessiaObject):
    _standalone_in_db = True
    _generic_eq = True

    def __init__(self, floatarg: float, name: str = 'Standalone Subobject'):
        self.floatarg = floatarg

        DessiaObject.__init__(self, name=name)

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


class EnhancedStandaloneSubobject(StandaloneSubobject):
    def __init__(self, floatarg: float, boolarg: bool,
                 name: str = 'Standalone Subobject'):
        self.boolarg = boolarg

        StandaloneSubobject.__init__(self, floatarg=floatarg, name=name)


class InheritingStandaloneSubobject(StandaloneSubobject):
    def __init__(self, floatarg: float, strarg: str,
                 name: str = 'Inheriting Standalone Subobject'):
        self.strarg = strarg

        StandaloneSubobject.__init__(self, floatarg=floatarg, name=name)


class EmbeddedSubobject(DessiaObject):
    def __init__(self, embedded_list: List[int] = None,
                 name: str = 'Embedded Subobject'):
        if embedded_list is None:
            self.embedded_list = [1, 2, 3]
        else:
            self.embedded_list = embedded_list

        DessiaObject.__init__(self, name=name)


class StaticDict(TypedDict):
    name: str
    value: float
    is_valid: bool
    subobject: EmbeddedSubobject


class StandaloneObject(DessiaObject):
    _standalone_in_db = True
    _generic_eq = True

    def __init__(self, standalone_subobject: StandaloneSubobject,
                 embedded_subobject: EmbeddedSubobject,
                 dynamic_dict: Dict[str, bool], static_dict: StaticDict,
                 tuple_arg: Tuple[str, int], intarg: int, strarg: str,
                 object_list: List[StandaloneSubobject],
                 subobject_list: List[EmbeddedSubobject],
                 builtin_list: List[int],
                 union_arg: Union[StandaloneSubobject,
                                  EnhancedStandaloneSubobject],
                 inheritance_list: List[Subclass[StandaloneSubobject]],
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
        self.inheritance_list = inheritance_list

        DessiaObject.__init__(self, name=name)

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

    def add_float(self, value) -> StandaloneSubobject:
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
        multi_plot = plot_data.MultiplePlots(elements=points, objects=objects,
                                             sizes=sizes, coords=coords,
                                             name='Multiple Plot')
        # return [scatter_plot]
        # return [parallel_plot]
        # return [multi_plot]
        # return [scatter_plot, parallel_plot]
        return [primitives_group, scatter_plot, parallel_plot, multi_plot]

