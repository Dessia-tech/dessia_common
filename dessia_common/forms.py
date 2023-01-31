"""
A module that aims to list all possibilities of data formats offered by Dessia.

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
from typing import Dict, List, Tuple, Union, Any
import time
import random
from numpy import linspace

try:
    import volmdlr as vm
    from volmdlr import primitives2d as p2d
    from volmdlr import primitives3d as p3d
    import plot_data
    import plot_data.colors
except ImportError:
    pass

from dessia_common.core import DessiaObject, PhysicalObject, MovingObject
from dessia_common.typings import InstanceOf
from dessia_common.measures import Distance
from dessia_common.exports import MarkdownWriter

from dessia_common.files import BinaryFile, StringFile


class EmbeddedBuiltinsSubobject(PhysicalObject):
    """
    An object that is not standalone and gather builtins types (float, int, bool, str & Distance).

    :param distarg: A Distance with units
    :type distarg: Distance

    :param floatarg: A float
    :type floatarg: float

    :param intarg: An integer
    :type intarg: int

    :param boolarg: A boolean
    :type boolarg: bool

    :param name: Object's name
    :type name: str
    """
    _standalone_in_db = False

    def __init__(self, distarg: Distance, floatarg: float, intarg: int,
                 boolarg: bool, name: str = 'Standalone Subobject'):
        self.distarg = distarg
        self.floatarg = floatarg
        self.intarg = intarg
        self.boolarg = boolarg

        PhysicalObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'EmbeddedBuiltinsSubobject':
        """ Generate an embedded subobject with default values computed from a seed. """
        floatarg = 0.3
        distarg = Distance(1.7 * floatarg * seed)
        intarg = seed
        boolarg = bool(seed % 2)
        name = 'EmbeddedSubobject' + str(seed)
        return cls(distarg=distarg, floatarg=floatarg, intarg=intarg, boolarg=boolarg, name=name)

    @classmethod
    def generate_many(cls, seed: int) -> List['EmbeddedBuiltinsSubobject']:
        """ Generate many embedded subobjects with default values computed from a seed. """
        return [cls.generate((i + 1) * 1000) for i in range(seed)]

    def contour(self):
        """ Square contour of an embedded subobject, for testing purpose. """
        origin = self.floatarg
        points = [vm.Point2D(origin, origin), vm.Point2D(origin, origin + 1),
                  vm.Point2D(origin + 1, origin + 1), vm.Point2D(origin + 1, origin)]
        return p2d.ClosedRoundedLineSegments2D(points=points, radius={})

    def plot_data(self, **kwargs):
        """ Bare text, for testing purpose. """
        primitives = [plot_data.Text(comment="Test with text", position_x=0, position_y=0),
                      plot_data.Text(comment="Test with text", position_x=0, position_y=0)]
        primitives_group = plot_data.PrimitiveGroup(primitives=primitives)
        return [primitives_group]

    def voldmlr_primitives(self):
        """ Volmdlr primitives of the squared contour. """
        contour = self.contour()
        volumes = [p3d.ExtrudedProfile(vm.O3D, vm.X3D, vm.Z3D, contour, [], vm.Y3D)]
        return volumes


class StandaloneBuiltinsSubobject(EmbeddedBuiltinsSubobject):
    """
    Overwrite EmbeddedBuiltinsObject to make it standalone.

    :param distarg: A Distance with units
    :type distarg: Distance

    :param floatarg: A float
    :type floatarg: float

    :param intarg: An integer
    :type intarg: int

    :param boolarg: A boolean
    :type boolarg: bool

    :param name: Object's name
    :type name: str
    """
    _standalone_in_db = True

    def __init__(self, distarg: Distance, floatarg: float, intarg: int,
                 boolarg: bool, name: str = 'Standalone Subobject'):

        EmbeddedBuiltinsSubobject.__init__(self, distarg=distarg, floatarg=floatarg,
                                           intarg=intarg, boolarg=boolarg, name=name)


DEF_EBS = EmbeddedBuiltinsSubobject.generate(1)
DEF_SBS = StandaloneBuiltinsSubobject.generate(1)


class EnhancedStandaloneSubobject(StandaloneBuiltinsSubobject):
    """ Overwrite StandaloneSubobject, principally for InstanceOf and Union typings testing purpose. """

    def __init__(self, floatarg: Distance, name: str = 'Standalone Subobject'):
        StandaloneBuiltinsSubobject.__init__(self, distarg=floatarg, floatarg=floatarg, intarg=floor(floatarg),
                                             boolarg=floatarg.is_integer(), name=name)

    @classmethod
    def generate(cls, seed: int) -> 'EnhancedStandaloneSubobject':
        """ Generate an enhanced subobject with default values computed from a seed. """
        floatarg = Distance(1.2 * seed)
        name = f"EnhancedStandaloneSubobject{seed}"
        return cls(floatarg=floatarg, name=name)


DEF_ESS = EnhancedStandaloneSubobject.generate(1)


class InheritingStandaloneSubobject(StandaloneBuiltinsSubobject):
    """ Overwrite StandaloneSubobject, principally for InstanceOf and Union typings testing purpose. """

    def __init__(self, distarg: Distance, floatarg: float, intarg: int, boolarg: bool, strarg: str,
                 name: str = 'Inheriting Standalone Subobject'):
        self.strarg = strarg

        StandaloneBuiltinsSubobject.__init__(self, distarg=distarg, floatarg=floatarg, intarg=intarg,
                                             boolarg=boolarg, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'InheritingStandaloneSubobject':
        """ Generate an inheriting subobject with default values computed from a seed. """
        distarg = Distance(0.7 * seed)
        floatarg = 0.1 * seed
        strarg = str(-seed)
        intarg = seed * 3
        boolarg = bool(intarg % 2)
        name = 'Inheriting Standalone Subobject' + str(seed)
        return cls(floatarg=floatarg, distarg=distarg, intarg=seed * 3, boolarg=boolarg, strarg=strarg, name=name)


DEF_ISS = InheritingStandaloneSubobject.generate(1)


class EmbeddedSubobject(DessiaObject):
    """ EmbeddedSubobject, in order to test non standalone features. """

    def __init__(self, embedded_list: List[int] = None, name: str = 'Embedded Subobject'):
        if embedded_list is None:
            self.embedded_list = [1, 2, 3]
        else:
            self.embedded_list = embedded_list

        DessiaObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'EmbeddedSubobject':
        """ Generate an embedded subobject with default values computed from a seed. """
        if not bool(seed % 2):
            embedded_list = list(range(int(seed / 2)))
        else:
            embedded_list = None
        name = 'Embedded Subobject' + str(seed)
        return cls(embedded_list=embedded_list, name=name)

    @classmethod
    def generate_many(cls, seed: int) -> List['EmbeddedSubobject']:
        """ Generate many embedded subobjects with default values computed from a seed. """
        return [cls.generate(i) for i in range(ceil(seed / 3))]


class EnhancedEmbeddedSubobject(EmbeddedSubobject):
    """ Overwrite EmbeddedSubobject, principally for InstanceOf and Union typings testing purpose. """

    def __init__(self, embedded_list: List[int] = None, embedded_array: List[List[float]] = None,
                 name: str = 'Enhanced Embedded Subobject'):
        self.embedded_array = embedded_array

        EmbeddedSubobject.__init__(self, embedded_list=embedded_list, name=name)

    @classmethod
    def generate(cls, seed: int) -> 'EnhancedEmbeddedSubobject':
        """ Generate an embedded subobject with default values computed from a seed. """
        embedded_list = [seed]
        embedded_array = [[seed, seed * 10, seed * 10]] * seed
        name = f"Embedded Subobject{seed}"
        return cls(embedded_list=embedded_list, embedded_array=embedded_array, name=name)


DEF_ES = EmbeddedSubobject.generate(10)
DEF_EES = EnhancedEmbeddedSubobject.generate(3)


UnionArg = Union[EmbeddedSubobject, EnhancedEmbeddedSubobject]


class StandaloneObject(MovingObject):
    """
    Standalone Object for testing purpose.

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
    _allowed_methods = ['add_standalone_object', 'add_embedded_object', "count_until",
                        'add_float', 'generate_from_text', 'generate_from_bin']

    def __init__(self, standalone_subobject: StandaloneBuiltinsSubobject, embedded_subobject: EmbeddedSubobject,
                 dynamic_dict: Dict[str, bool], float_dict: Dict[str, float], string_dict: Dict[str, str],
                 tuple_arg: Tuple[str, int], object_list: List[StandaloneBuiltinsSubobject],
                 subobject_list: List[EmbeddedSubobject], builtin_list: List[int], union_arg: List[UnionArg],
                 subclass_arg: InstanceOf[StandaloneBuiltinsSubobject], array_arg: List[List[float]],
                 name: str = 'Standalone Object Demo'):
        self.union_arg = union_arg
        self.builtin_list = builtin_list
        self.subobject_list = subobject_list
        self.object_list = object_list
        self.tuple_arg = tuple_arg
        self.dynamic_dict = dynamic_dict
        self.float_dict = float_dict
        self.string_dict = string_dict
        self.standalone_subobject = standalone_subobject
        self.embedded_subobject = embedded_subobject
        self.subclass_arg = subclass_arg
        self.array_arg = array_arg

        MovingObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int, name: str = 'Standalone Object Demo') -> 'StandaloneObject':
        """ Generate an object with default values computed from a seed. """
        dynamic_dict = {'n' + str(i): bool(seed % 2) for i in range(seed)}
        float_dict = {'k' + str(i): seed * 1.09 for i in range(seed)}
        string_dict = {'key' + str(i): 'value' + str(i) for i in range(seed)}
        builtin_list = [seed] * seed
        array_arg = [builtin_list] * 3
        union_arg = [EnhancedEmbeddedSubobject.generate(seed), EmbeddedSubobject.generate(seed)]
        if not bool(seed % 2):
            subclass_arg = StandaloneBuiltinsSubobject.generate(-seed)
        else:
            subclass_arg = InheritingStandaloneSubobject.generate(seed)
        return cls(standalone_subobject=StandaloneBuiltinsSubobject.generate(seed),
                   embedded_subobject=EmbeddedSubobject.generate(seed),
                   dynamic_dict=dynamic_dict, float_dict=float_dict, string_dict=string_dict,
                   tuple_arg=('value', seed * 3), object_list=StandaloneBuiltinsSubobject.generate_many(seed),
                   subobject_list=EmbeddedSubobject.generate_many(seed), builtin_list=builtin_list, union_arg=union_arg,
                   subclass_arg=subclass_arg, array_arg=array_arg, name=name)

    @classmethod
    def generate_with_many_subobjects(cls, seed: int, number: int, name: str = "Populated SO"):
        """ Generate an object with a lot of subobjects. """
        standalone_object = cls.generate(seed=seed, name=name)
        subobjects = StandaloneBuiltinsSubobject.generate_many(seed=seed * number)
        standalone_object.subclass_arg = subobjects
        return standalone_object

    @classmethod
    def generate_from_bin(cls, stream: BinaryFile):
        """ Generate an object from bin file in order to test frontend forms and backend streams. """
        # User need to decode the binary as he see fit
        my_string = stream.read().decode('utf8')
        my_file_name = stream.filename
        _, raw_seed = my_string.split(",")
        seed = int(raw_seed.strip())
        return cls.generate(seed=seed, name=my_file_name)

    @classmethod
    def generate_from_text(cls, stream: StringFile):
        """ Generate an object from text file in order to test frontend forms and backend streams. """
        my_text = stream.getvalue()
        my_file_name = stream.filename
        _, raw_seed = my_text.split(",")
        seed = int(raw_seed.strip())
        return cls.generate(seed=seed, name=my_file_name)

    def add_standalone_object(self, object_: StandaloneBuiltinsSubobject):
        """
        Add a standalone object to object_list.

        It doesn't return anything, hence, API will update object when computing from frontend.
        """
        self.object_list.append(object_)

    def add_embedded_object(self, object_: EmbeddedSubobject):
        """
        Add an embedded object to subobject_list.

        It doesn't return anything, hence, API will update object when computing from frontend.
        """
        self.subobject_list.append(object_)

    def add_float(self, value: float) -> StandaloneBuiltinsSubobject:
        """
        Add value to its standalone subobject floatarg property and return it.

        API should replace standalone_subobject as it is returned
        """
        self.standalone_subobject.floatarg += value
        return self.standalone_subobject

    def append_union_arg(self, object_: UnionArg):
        """ Append an object with corresponding type to union_arg, for testing purpose. """
        self.union_arg.append(object_)

    def contour(self):
        """ Squared contour. """
        intarg = self.standalone_subobject.intarg
        points = [vm.Point2D(intarg, intarg), vm.Point2D(intarg, intarg + 1),
                  vm.Point2D(intarg + 1, intarg + 1), vm.Point2D(intarg + 1, 0)]

        crls = p2d.ClosedRoundedLineSegments2D(points=points, radius={})
        return crls

    def volmdlr_primitives(self):
        """ Volmdlr primitives of a cube. """
        subcube = self.standalone_subobject.voldmlr_primitives()[0]
        contour = self.contour()
        cube = p3d.ExtrudedProfile(plane_origin=vm.Point3D(0, 1, -1), x=vm.X3D, y=vm.Z3D,
                                   outer_contour2d=contour, inner_contours2d=[], extrusion_vector=vm.Y3D)
        return [subcube, cube]

    def volmdlr_primitives_step_frames(self):
        """ Steps to show the cube moving on frontend. """
        frame0 = vm.Frame3D(vm.O3D.copy(), vm.X3D.copy(), vm.Y3D.copy(), vm.Z3D.copy())
        frame11 = frame0.rotation(center=vm.O3D, axis=vm.Y3D, angle=0.7)
        frame21 = frame11.translation(offset=vm.Y3D)
        frame31 = frame21.rotation(center=vm.O3D, axis=vm.Y3D, angle=0.7)

        frame12 = frame0.translation(offset=vm.Z3D)
        frame22 = frame12.translation(offset=vm.X3D)
        frame32 = frame22.translation(offset=vm.X3D)
        return [[frame0, frame0], [frame11, frame12], [frame21, frame22], [frame31, frame32]]

    @staticmethod
    def scatter_plot():
        """ Test scatter plots. """
        attributes = ['cx', 'cy']
        tooltip = plot_data.Tooltip(attributes=attributes, name='Tooltips')
        return plot_data.Scatter(axis=plot_data.Axis(), tooltip=tooltip, x_variable=attributes[0],
                                 y_variable=attributes[1], name='Scatter Plot')

    def plot_data(self, **kwargs):
        """ Full plot data definition with lots of graphs and 2Ds. For frontend testing purpose. """
        # Contour
        contour = self.standalone_subobject.contour().plot_data()
        primitives_group = plot_data.PrimitiveGroup(primitives=[contour], name='Contour')

        rand = random.randint
        samples = [plot_data.Sample(values={"cx": rand(0, 600) / 100, "cy": rand(100, 2000) / 100}, name=f"Point{i}")
                   for i in range(500)]

        # Scatter Plot
        scatterplot = self.scatter_plot()

        # Parallel Plot
        parallelplot = plot_data.ParallelPlot(elements=samples, axes=['cx', 'cy', 'color_fill', 'color_stroke'],
                                              name='Parallel Plot')

        # Multi Plot
        objects = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300), plot_data.Window(width=560, height=300)]
        multiplot = plot_data.MultiplePlots(elements=samples, plots=objects, sizes=sizes,
                                            coords=[(0, 0), (300, 0)], name='Multiple Plot')

        attribute_names = ['timestep', 'electric current']
        tooltip = plot_data.Tooltip(attributes=attribute_names)
        timesteps = linspace(0, 20, 20)
        current1 = [t ** 2 for t in timesteps]
        elements1 = []
        for timestep, current in zip(timesteps, current1):
            elements1.append({'timestep': timestep, 'electric current': current})

        # The previous line instantiates a dataset with limited arguments but several customizations are available
        point_style = plot_data.PointStyle(color_fill=plot_data.colors.RED, color_stroke=plot_data.colors.BLACK)
        edge_style = plot_data.EdgeStyle(color_stroke=plot_data.colors.BLUE, dashline=[10, 5])

        custom_dataset = plot_data.Dataset(elements=elements1, name='I = f(t)', tooltip=tooltip,
                                           point_style=point_style, edge_style=edge_style)

        # Now let's create another dataset for the purpose of this exercice
        timesteps = linspace(0, 20, 100)
        current2 = [100 * (1 + cos(t)) for t in timesteps]
        elements2 = []
        for timestep, current in zip(timesteps, current2):
            elements2.append({'timestep': timestep, 'electric current': current})

        dataset2 = plot_data.Dataset(elements=elements2, name='I2 = f(t)')

        graph2d = plot_data.Graph2D(graphs=[custom_dataset, dataset2],
                                    x_variable=attribute_names[0], y_variable=attribute_names[1])
        return [primitives_group, scatterplot, parallelplot, multiplot, graph2d]

    def maldefined_method(self, arg0, arg1=1, arg2: int = 10, arg3=3):
        """ Define a docstring for testing parsing purpose. """
        nok_string = "This is a bad coding behavior"
        ok_string = "This could be OK as temporary attr"
        self.maldefined_attr = nok_string
        self._ok_attribute = ok_string

        computation = nok_string + 'or' + ok_string

        return computation

    # @staticmethod
    # def method_with_faulty_typing(arg0: Iterator[int]):
    #     return arg0

    def to_markdown(self):
        """ Write a standard markdown of StandaloneObject. """
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
        contents += "\n## Attribute Table\n\n"
        contents += MarkdownWriter(print_limit=25, table_limit=None).object_table(self)
        return contents

    def count_until(self, duration: float, raise_error: bool = False):
        """
        Test long execution with a customizable duration.

        :param duration: Duration of the method in s
        :type duration: float

        :param raise_error: Wether the computation should raise an error or not at the end
        :type raise_error: bool
        """
        starting_time = time.time()
        current_time = time.time()
        last_duration = round(current_time - starting_time)
        while current_time - starting_time <= duration:
            current_time = time.time()
            current_duration = current_time - starting_time
            if current_duration > last_duration + 1:
                last_duration = round(current_time - starting_time)
                print(round(current_duration))

        if raise_error:
            raise RuntimeError(f"Evaluation stopped after {duration}s")


DEF_SO = StandaloneObject.generate(1)


class StandaloneObjectWithDefaultValues(StandaloneObject):
    """ Overwrite StandaloneObject to set default values to it. For frontend's forms testing purpose. """
    _non_editable_attributes = ['intarg', 'strarg']

    def __init__(self, standalone_subobject: StandaloneBuiltinsSubobject = DEF_SBS,
                 embedded_subobject: EmbeddedSubobject = DEF_ES,
                 dynamic_dict: Dict[str, bool] = None,
                 float_dict: Dict[str, float] = None,
                 string_dict: Dict[str, str] = None,
                 tuple_arg: Tuple[str, int] = ("Default Tuple", 0),
                 object_list: List[StandaloneBuiltinsSubobject] = None,
                 subobject_list: List[EmbeddedSubobject] = None,
                 builtin_list: List[int] = None,
                 union_arg: List[UnionArg] = None,
                 subclass_arg: InstanceOf[StandaloneBuiltinsSubobject] = DEF_ISS,
                 array_arg: List[List[float]] = None,
                 name: str = 'Standalone Object Demo'):
        if dynamic_dict is None:
            dynamic_dict = {}
        if float_dict is None:
            float_dict = {}
        if string_dict is None:
            string_dict = {}
        if object_list is None:
            object_list = [DEF_SBS]
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
                                  float_dict=float_dict, string_dict=string_dict, tuple_arg=tuple_arg,
                                  object_list=object_list, subobject_list=subobject_list, builtin_list=builtin_list,
                                  union_arg=union_arg, subclass_arg=subclass_arg, array_arg=array_arg, name=name)


DEF_SOWDV = StandaloneObjectWithDefaultValues()

# class ObjectWithFaultyTyping(DessiaObject):
#     """
#     Dummy class to test faulty typing jsonschema
#     """
#     def __init__(self, faulty_attribute: Iterator[int], name: str = ""):
#         self.faulty_attribute = faulty_attribute
#
#         DessiaObject.__init__(self, name=name)


class ObjectWithOtherTypings(DessiaObject):
    """ Dummy class to test some typing jsonschemas. """

    def __init__(self, undefined_type_attribute: Any, name: str = ""):
        self.undefined_type_attribute = undefined_type_attribute

        DessiaObject.__init__(self, name=name)


class MovingStandaloneObject(MovingObject):
    """ Overwrite StandaloneObject to make its 3D move. """
    _standalone_in_db = True

    def __init__(self, origin: float, name: str = ""):
        self.origin = origin
        MovingObject.__init__(self, name=name)

    @classmethod
    def generate(cls, seed: int):
        """ Generate an object with default values from seed. """
        return cls(origin=1.3 * seed, name=f"moving_{seed}")

    def contour(self):
        """ Squared contour. """
        points = [vm.Point2D(self.origin, self.origin), vm.Point2D(self.origin, self.origin + 1),
                  vm.Point2D(self.origin + 1, self.origin + 1), vm.Point2D(self.origin + 1, 0)]

        crls = p2d.ClosedRoundedLineSegments2D(points=points, radius={})
        return crls

    def volmdlr_primitives(self):
        """ A cube. """
        contour = self.contour()
        volume = p3d.ExtrudedProfile(plane_origin=vm.O3D, x=vm.X3D, y=vm.Z3D, outer_contour2d=contour,
                                     inner_contours2d=[], extrusion_vector=vm.Y3D)
        return [volume]

    def volmdlr_primitives_step_frames(self):
        """ A moving cube. """
        frame1 = vm.Frame3D(vm.O3D.copy(), vm.X3D.copy(), vm.Y3D.copy(), vm.Z3D.copy())
        frame2 = frame1.rotation(center=vm.O3D, axis=vm.Y3D, angle=0.7)
        frame3 = frame2.translation(offset=vm.Y3D)
        frame4 = frame3.rotation(center=vm.O3D, axis=vm.Y3D, angle=0.7)
        return [[frame1], [frame2], [frame3], [frame4]]


class Generator(DessiaObject):
    """
    A class that allow to generate several StandaloneObjects from different parameters.

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
        """ Generate a list of Standalone objects in order to emulate bots' generators. """
        self.models = [StandaloneObject.generate(self.parameter + i) for i in range(self.nb_solutions)]
        return self.models


class Optimizer(DessiaObject):
    """
    Mock an optimization process. Emulates bots Optimizers.

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
        Sum model value with given one.

        :param optimization_value: value that will be added to model's intarg attribute
        :type optimization_value: int
        """
        self.model_to_optimize.standalone_subobject.intarg += optimization_value
        return self.model_to_optimize.standalone_subobject.intarg


class Container(DessiaObject):
    """ Gather a list of Standalone objects as a container. For 'Catalog' behavior testing purpose. """
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
        """ Generate catalog from several text files. """
        models = [StandaloneObject.generate_from_text(file) for file in files]
        return cls(models=models, name=name)


class NotStandalone(DessiaObject):
    """A simple non-standalone class."""

    def __init__(self, attribute: int, name: str = ""):
        self.attribute = attribute
        DessiaObject.__init__(self, name=name)


class BottomLevel(DessiaObject):
    """A simple class at the bottom of the data structure."""
    _standalone_in_db = True

    def __init__(self, attributes: List[NotStandalone] = None, name: str = ""):
        if attributes is None:
            self.attributes = []
        else:
            self.attributes = attributes

        DessiaObject.__init__(self, name=name)


class MidLevel(DessiaObject):
    """A simple class at the mid level of the data structure."""
    _standalone_in_db = True
    _allowed_methods = ["generate_with_references"]

    def __init__(self, bottom_level: BottomLevel = None, name: str = ""):
        self.bottom_level = bottom_level

        DessiaObject.__init__(self, name=name)

    @classmethod
    def generate_with_references(cls, name: str = "Result Name"):
        """A fake class generator."""
        object1 = NotStandalone(attribute=1, name="1")
        object2 = NotStandalone(attribute=1, name="2")
        bottom_level = BottomLevel([object1, object2])
        return cls(bottom_level=bottom_level, name=name)
