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
- As opposed to this, a non-standalone_in_db class defines a static structure,
  with a defined number of given, expected keys & types of their values.

In addition to types & genericity (brought by DessiaObject),
this module can also be seen as a template for Dessia's
coding/naming style & convention.
"""

from math import floor, ceil, cos
from typing import Dict, List, Tuple, Union
from numpy import linspace

try:
    import volmdlr as vm
    from volmdlr import primitives2d as p2d
    from volmdlr import primitives3d as p3d
    import plot_data
    import plot_data.colors
except ImportError:
    pass

from dessia_common import DessiaObject, PhysicalObject
from dessia_common.typings import InstanceOf, Distance
from dessia_common.vectored_objects import Catalog


from dessia_common.files import BinaryFile, StringFile


class StandaloneSubobject(PhysicalObject):
    _standalone_in_db = True
    _generic_eq = True

    def __init__(self, floatarg: Distance, name: str = 'Standalone Subobject'):
        self.floatarg = floatarg

        PhysicalObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'StandaloneSubobject':
        floatarg = Distance(1.7 * seed)
        name = 'StandaloneSubobject' + str(seed)
        return cls(floatarg=floatarg, name=name)

    @classmethod
    def generate_many(cls, seed: int) -> List['StandaloneSubobject']:
        subobjects = [cls.generate((i + 1) * 1000) for i in range(seed)]
        return subobjects

    def contour(self):
        origin = self.floatarg
        points = [vm.Point2D(origin, origin), vm.Point2D(origin, origin + 1),
                  vm.Point2D(origin + 1, origin + 1), vm.Point2D(origin + 1, origin)]

        crls = p2d.ClosedRoundedLineSegments2D(points=points, radius={})
        return crls

    def voldmlr_primitives(self):
        contour = self.contour()
        volumes = [p3d.ExtrudedProfile(vm.O3D, vm.X3D, vm.Z3D, contour, [], vm.Y3D)]
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
            embedded_list = list(range(int(seed / 2)))
        else:
            embedded_list = None
        name = 'Embedded Subobject' + str(seed)
        return cls(embedded_list=embedded_list, name=name)

    @classmethod
    def generate_many(cls, seed: int) -> List['EmbeddedSubobject']:
        return [cls.generate(i) for i in range(ceil(seed / 3))]


class EnhancedEmbeddedSubobject(EmbeddedSubobject):
    def __init__(self, embedded_list: List[int] = None,
                 embedded_array: List[List[float]] = None,
                 name: str = 'Enhanced Embedded Subobject'):
        self.embedded_array = embedded_array

        EmbeddedSubobject.__init__(self, embedded_list=embedded_list, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'EnhancedEmbeddedSubobject':
        embedded_list = [seed]
        embedded_array = [[seed, seed * 10, seed * 10]] * seed
        name = 'Embedded Subobject' + str(seed)
        return cls(embedded_list=embedded_list, embedded_array=embedded_array, name=name)


DEF_ES = EmbeddedSubobject.generate(10)
DEF_EES = EnhancedEmbeddedSubobject.generate(3)


UnionArg = Union[EmbeddedSubobject, EnhancedEmbeddedSubobject]


class StandaloneObject(PhysicalObject):
    """
    Dev Object for testing purpose

    :param standalone_subobject: A dev subobject that is standalone_in_db
    :type standalone_subobject: StandaloneSubobject
    :param embedded_subobject: A dev subobject that isn't standalone_in_db
    :type embedded_subobject: EmbeddedSubobject
    :param dynamic_dict: A variable length dict
    :type dynamic_dict: Dict[str, bool]
    :param tuple_arg: A heterogeneous sequence
    :type tuple_arg: tuple
    """
    _standalone_in_db = True
    _generic_eq = True
    _allowed_methods = ['add_standalone_object', 'add_embedded_object',
                        'add_float', 'generate_from_text', 'generate_from_bin']

    def __init__(self, standalone_subobject: StandaloneSubobject, embedded_subobject: EmbeddedSubobject,
                 dynamic_dict: Dict[str, bool], float_dict: Dict[str, float], string_dict: Dict[str, str],
                 tuple_arg: Tuple[str, int], intarg: int, strarg: str,
                 object_list: List[StandaloneSubobject], subobject_list: List[EmbeddedSubobject],
                 builtin_list: List[int], union_arg: List[UnionArg],
                 subclass_arg: InstanceOf[StandaloneSubobject], array_arg: List[List[float]],
                 name: str = 'Standalone Object Demo'):
        self.union_arg = union_arg
        self.builtin_list = builtin_list
        self.subobject_list = subobject_list
        self.object_list = object_list
        self.tuple_arg = tuple_arg
        self.strarg = strarg
        self.intarg = intarg
        self.dynamic_dict = dynamic_dict
        self.float_dict = float_dict
        self.string_dict = string_dict
        self.standalone_subobject = standalone_subobject
        self.embedded_subobject = embedded_subobject
        self.subclass_arg = subclass_arg
        self.array_arg = array_arg

        PhysicalObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int, name: str = 'Standalone Object Demo') -> 'StandaloneObject':
        is_even = not bool(seed % 2)
        standalone_subobject = StandaloneSubobject.generate(seed)
        embedded_subobject = EmbeddedSubobject.generate(seed)
        dynamic_dict = {'n' + str(i): bool(seed % 2) for i in range(seed)}
        float_dict = {'k' + str(i): seed * 1.09 for i in range(seed)}
        string_dict = {'key' + str(i): 'value' + str(i) for i in range(seed)}
        tuple_arg = ('value', seed * 3)
        intarg = seed
        strarg = str(seed) * floor(seed / 3)
        object_list = StandaloneSubobject.generate_many(seed)
        subobject_list = EmbeddedSubobject.generate_many(seed)
        builtin_list = [seed] * seed
        array_arg = [builtin_list] * 3
        union_arg = [EnhancedEmbeddedSubobject.generate(seed),
                     EmbeddedSubobject.generate(seed)]
        if is_even:
            subclass_arg = StandaloneSubobject.generate(-seed)
        else:
            subclass_arg = InheritingStandaloneSubobject.generate(seed)
        return cls(standalone_subobject=standalone_subobject, embedded_subobject=embedded_subobject,
                   dynamic_dict=dynamic_dict, float_dict=float_dict, string_dict=string_dict, tuple_arg=tuple_arg,
                   intarg=intarg, strarg=strarg, object_list=object_list, subobject_list=subobject_list,
                   builtin_list=builtin_list, union_arg=union_arg, subclass_arg=subclass_arg,
                   array_arg=array_arg, name=name)

    @classmethod
    def generate_from_bin(cls, stream: BinaryFile):
        # User need to decode the binary as he see fit
        my_string = stream.read().decode('utf8')
        my_file_name = stream.filename
        _, raw_seed = my_string.split(",")
        seed = int(raw_seed.strip())
        return cls.generate(seed=seed, name=my_file_name)

    @classmethod
    def generate_from_text(cls, stream: StringFile):
        my_text = stream.getvalue()
        my_file_name = stream.filename
        _, raw_seed = my_text.split(",")
        seed = int(raw_seed.strip())
        return cls.generate(seed=seed, name=my_file_name)

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
        primitives_group = plot_data.PrimitiveGroup(primitives=[contour], name='Contour')

        # Scatter Plot
        bounds = {'x': [0, 6], 'y': [100, 2000]}
        catalog = Catalog.random_2d(bounds=bounds, threshold=8000)
        points = [plot_data.Point2D(cx=v[0], cy=v[1], name='Point' + str(i)) for i, v in enumerate(catalog.array)]
        axis = plot_data.Axis()
        tooltip = plot_data.Tooltip(attributes=attributes, name='Tooltips')
        scatter_plot = plot_data.Scatter(axis=axis, tooltip=tooltip, x_variable=attributes[0],
                                         y_variable=attributes[1], name='Scatter Plot')

        # Parallel Plot
        attributes = ['cx', 'cy', 'color_fill', 'color_stroke']
        parallel_plot = plot_data.ParallelPlot(elements=points, axes=attributes, name='Parallel Plot')

        # Multi Plot
        objects = [scatter_plot, parallel_plot]
        sizes = [plot_data.Window(width=560, height=300), plot_data.Window(width=560, height=300)]
        coords = [(0, 0), (300, 0)]
        multi_plot = plot_data.MultiplePlots(elements=points, plots=objects, sizes=sizes,
                                             coords=coords, name='Multiple Plot')

        attribute_names = ['time', 'electric current']
        tooltip = plot_data.Tooltip(attributes=attribute_names)
        time1 = linspace(0, 20, 20)
        current1 = [t ** 2 for t in time1]
        elements1 = []
        for time, current in zip(time1, current1):
            elements1.append({'time': time, 'electric current': current})

        # The previous line instantiates a dataset with limited arguments but several customizations are available
        point_style = plot_data.PointStyle(color_fill=plot_data.colors.RED, color_stroke=plot_data.colors.BLACK)
        edge_style = plot_data.EdgeStyle(color_stroke=plot_data.colors.BLUE, dashline=[10, 5])

        custom_dataset = plot_data.Dataset(elements=elements1, name='I = f(t)', tooltip=tooltip,
                                           point_style=point_style, edge_style=edge_style)

        # Now let's create another dataset for the purpose of this exercice
        time2 = linspace(0, 20, 100)
        current2 = [100 * (1 + cos(t)) for t in time2]
        elements2 = []
        for time, current in zip(time2, current2):
            elements2.append({'time': time, 'electric current': current})

        dataset2 = plot_data.Dataset(elements=elements2, name='I2 = f(t)')

        graph2d = plot_data.Graph2D(graphs=[custom_dataset, dataset2],
                                    x_variable=attribute_names[0], y_variable=attribute_names[1])
        return [primitives_group, scatter_plot,
                parallel_plot, multi_plot, graph2d]

    def maldefined_method(self, arg0, arg1=1, arg2: int = 10, arg3=3):
        """
        Defining a docstring for testing parsing purpose
        """
        nok_string = "This is a bad coding behavior"
        ok_string = "This could be OK as temporary attr"
        self.maldefined_attr = nok_string
        self._ok_attribute = ok_string

        computation = nok_string + 'or' + ok_string

        return computation

    def to_markdown(self):
        contents = """
        # Quem Stygios dumque

        ## Recursus erat aere decus Lemnicolae

        Lorem markdownum laetum senior quod Libys utroque
        *mirantibus teneat aevo*, aquis.
        Procumbit eandem ensis, erigor intercepta, quae habitabat nostro
        *et hoc que* enim: inpulit.
        Mecum ferat **fecissem** vale per myricae suis
        quas turba potentior mentita.
        Annis nunc, picae erat quis minatur dare Diana redimitus
        [Clymene venisses sinat](http://est.net/umbram.html)
        protinus pulchra, sucos! Tanta haec varios tuaque,
        nisi Erigonen si aquae Hippomene inguine murmur.

        1. Poma enim dextra icta capillis extinctum foedera
        2. Mediis requirit exercita ascendere fecisse sola
        3. Sua externis tigride saevarum
        4. Aves est pendebant sume latentis

        ## Suum videre quondam generis dolentem simul femineos

        Ille lacus progenitore Cycnum pressa, excidit silva
        [crudus](http://www.domino.com/nequevox), boum ducem vocari,
        ne monte tanto harenae.
        Opus Aesone excipit adempto.
        Inpius illa latratu atque sed praedam,
        ille construit intravit concipit,
        concha dedit, qua audit calathosque.
        Dedit putrefacta cortex.
        Tenet aut carmina quod proditione media; pro ense medicina
        vita repetit adrectisque inops e sentiat.

        > Imagine caesaries superbos muneraque *ne terras* cunctis.
        Diversae Hesioneque
        > numinis regia at anima nascuntur Iovis.
        Sua fama quoque capillos lugubris
        > **egimus**, a ingenti [Ericthonio](http://raptos.org/lucem)
        iubebat!

        ## Ponderis venit veteris mihi tofis

        Propensum discedunt, iacere dedisti; lene potest caelo,
        felix flamma caecus decet excipit.
        *Aurum occiderat*, retro cum, quorum *Diana timuere At*.
        Ait Labros hasta mundi, **ut est** ruit nosse o gravet!

        ## Qui aether undis nulla

        Homines oppidaque nominibus devexo genitoris quoque,
        praesensque rota Saturnia.
        Auras cecinit fera quae mirantum imbris,
        Gratia verba incesto, sed visa contigit
        saepe adicit trepidant.
        [Siqua radiis quod](http://www.naris-pectebant.org/comeset)
        ad duabus alienisque, sponte; dum.

        Occidit Babylonia dubitare. Vultus cui: erat dea!
        Iam ense forma est se, tibi pedem adfectat nec nostra.
        Armenta socium nutrix [precatur](http://in-fraxinus.io/)
        aderam, quam mentem Elin labor auctor
        potentia prodidit inmitibus duo di?
        Verum a, tuo quoque nec Mysum per posses;
        vigor danda meruit: tecum audire responsa
        [conplexae](http://quis.io/disrestat.html) et alios.

        Agros grata illo animo mei nova, in magis furens et
        [modo](http://pondere.com/aquis) dimittere ubi neque es!
        Sua qua ac ire una facit Alcmene coepere
        arduus quae vestigia aliquis; meritorum Dorylas, scindunt.
        """
        return contents


DEF_SO = StandaloneObject.generate(1)


class StandaloneObjectWithDefaultValues(StandaloneObject):
    _non_editable_attributes = ['intarg', 'strarg']

    def __init__(self, standalone_subobject: StandaloneSubobject = DEF_SS,
                 embedded_subobject: EmbeddedSubobject = DEF_ES,
                 dynamic_dict: Dict[str, bool] = None,
                 float_dict: Dict[str, float] = None,
                 string_dict: Dict[str, str] = None,
                 tuple_arg: Tuple[str, int] = ("Default Tuple", 0),
                 intarg: int = 1, strarg: str = "Default Strarg",
                 object_list: List[StandaloneSubobject] = None,
                 subobject_list: List[EmbeddedSubobject] = None,
                 builtin_list: List[int] = None,
                 union_arg: List[UnionArg] = None,
                 subclass_arg: InstanceOf[StandaloneSubobject] = DEF_ISS,
                 array_arg: List[List[float]] = None,
                 name: str = 'Standalone Object Demo'):
        if dynamic_dict is None:
            dynamic_dict = {}
        if float_dict is None:
            float_dict = {}
        if string_dict is None:
            string_dict = {}
        if object_list is None:
            object_list = [DEF_SS]
        if subobject_list is None:
            subobject_list = [DEF_ES]
        if builtin_list is None:
            builtin_list = [1, 2, 3, 4, 5]
        if union_arg is None:
            union_arg = [DEF_EES, DEF_ES]
        if array_arg is None:
            array_arg = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        StandaloneObject.__init__(self, standalone_subobject=standalone_subobject,
                                  embedded_subobject=embedded_subobject, dynamic_dict=dynamic_dict,
                                  float_dict=float_dict, string_dict=string_dict, tuple_arg=tuple_arg, intarg=intarg,
                                  strarg=strarg, object_list=object_list, subobject_list=subobject_list,
                                  builtin_list=builtin_list, union_arg=union_arg, subclass_arg=subclass_arg,
                                  array_arg=array_arg, name=name)


DEF_SOWDV = StandaloneObjectWithDefaultValues()


class Generator(DessiaObject):
    """
    A class that allow to generate several StandaloneObjects from different parameters

    :param parameter: An "offset" for the seed that will be used in generation
    :type parameter: int
    :param nb_solutions: The max number of solutions that will be generated
    :type nb_solutions: int
    :param name: The name of the Generator. It is not used in object generation
    :type name: str
    """
    _standalone_in_db = True

    def __init__(self, parameter: int, nb_solutions: int = 25, models: List[StandaloneObject] = None, name: str = ''):
        self.parameter = parameter
        self.nb_solutions = nb_solutions
        self.models = models

        DessiaObject.__init__(self, name=name)

    def generate(self) -> List[StandaloneObject]:
        """
        Generates a list of Standalone objects
        """
        self.models = [StandaloneObject.generate(self.parameter + i) for i in range(self.nb_solutions)]
        return self.models


class Optimizer(DessiaObject):
    """
    Mock an optimization process

    :param model_to_optimize: An object which will be modified (one of its attributes)
    :type model_to_optimize: StandaloneObject
    :param name: Name of the optimizer. Will not be used in the optimization process
    :type name: str
    """
    _standalone_in_db = True

    def __init__(self, model_to_optimize: StandaloneObject, name: str = ''):
        self.model_to_optimize = model_to_optimize

        DessiaObject.__init__(self, name=name)

    def optimize(self, optimization_value: int = 3) -> int:
        """
        Sums model value with given one

        :param optimization_value: value that will be added to model's intarg attribute
        :type optimization_value: int
        """
        self.model_to_optimize.intarg += optimization_value
        return self.model_to_optimize.intarg


class Container(DessiaObject):
    _standalone_in_db = True
    _allowed_methods = ["generate_from_text_files"]

    def __init__(self, models: List[StandaloneObject] = None, name: str = ""):
        if models is None:
            self.models = []
        else:
            self.models = models

        DessiaObject.__init__(self, name=name)

    @classmethod
    def generate_from_text_files(cls, files: List[StringFile], name: str = "Generated from text files"):
        models = [StandaloneObject.generate_from_text(file) for file in files]
        return cls(models=models, name=name)
