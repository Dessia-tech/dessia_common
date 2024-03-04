# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## 0.16.1

### Added

- Export : Define export method with decorator 

### Performance

- Schema : Add new unittest to test get_import_names methods


## 0.16.0

###

- Excel Reader

### Build

- Global : remove usage of pkg_resources to use importlib.resources instead

### Fix

- Serialization : Fix wrong path in serialization that result in pointers cycles in some cases
- Workflow : Fix export workflow in .py script

### Refactor

- Export: Add all export formats in zip file by iterating over _export_formats()
- Serialization : Mutualize DessiaObject and Regular objects pointer serialization


## 0.15.4

### Fix

- Workflow : Add display block class discovery


## 0.15.3

### Fix

- Workflow : Add backward compatibility for variables


## 0.15.2


### Build

- Drone : Use pip install instead of setup.py install
- Drone : Remove non-working try on failure ignore


## 0.15.1

### Removed

- Workflow : remove useless code

### Build

- Drone : Try to ignore pypi upload failure


## 0.15.0


### Added

- Documentation : Rewrite from scratch
- Forms : Update with last volmdlr
- Workflow : Variables now have a pretty type to display on frontend
- Workflow : add documentation to workflow when export in .py
- Workflow : Add Label to variables
- WorkflowState : add memory usage monitoring & markdown
- Schemas : Property file relevance. Replace 0.13.7 hot with this

### Changed

- PhysicalObject: Pool to_stem and to_step_stream
- Workflow : Block input and output names have been enhanced
- Workflow : Inputs now have entries for Imposed Variable Values

### Fixed

- Document generator : Fix add_picture to .docx

### Removed

- Workflow : jointjs plot


## 0.14.2


###  Fix

- Set right path in pointers for non DessiaObject equal elements


## 0.14.1


### Changed

- Multiplot block takes selector as a string

### Fixed

- 'reference_path' is now passed on by Display block while evaluating


## 0.14.0


### Added

- Blocks : GetModelAttribute block which will replace ModelAttribute in a few releases
- Blocks : Display blocks are now more configurable (Custom selector)
- DessiaObjegitct : Add type to load_from_file method arguments
- Displays : Displays can now be defined with decorators
- Document generator : New module to write in docx file
- Document generator : New class Table
- Files : Functions to init StringFile and BinaryFile from local path
- Files : .doc & .docx files typings
- MarkdownWriter : New functions (table_of_contents, header)
- Schemas refactor : Support of Ellipsed tuple (Tuple[T, ...])
- Schemas refactor : More Error Checks
- Schemas refactor : JSON export to method schemas for low-code implementations
- Schemas refactor : Default value to method types
- Schemas refactor : Standalone in db property
- Schemas refactor : Support for Display View Types
- Schemas refactor : Mutualize Display View Types and Attribute Types
- Typings : AttributeType, ClassAttributeType and View Types
- Workflow : Tasks display


### Changed
 
- Blocks : Add the possibility to have TypedValue in SetModelAttribute
- Blocks : ! Display blocks definition have changed to adapt to new View Decorator paradigm
- Dataset : Default color of points from blue to grey to make the difference between selected points and the other ones
- DessiaObject : Rename load_from_file and load_from_stream to from_json and from_json_stream
- Export : Export all 3d formats in zip file
- Workflow : WorkflowRun now has smart display by default computation
  - Documentation is disabled when at least one block is displayed by default, enabled otherwise
  - Workflow always shows Documentation and its Display by default
- Workflow : Memorize display pipes from init


### Fixed

- Dataset : Allow to specify attributes of subojects for creating dataset matrix ('subobject/attr')
- Exports : Add trailing line at the end of JSON export
- Schemas refactor : Allow incomplete schemas
- Schemas refactor : Old jsonschema with magic Method Type serialized value
- Schemas refactor : Sequence schema uses args_schemas instead of unique items_schemas
- Workflow : Propagate progress_callback to blocks
- Workflow Blocks : New display selectors are now correctly deserialized


### Refactored

- Tests : Change tests. Unittests are in 'tests' folder. Other tests in scripts
- DessiaObject : Check Platform is now verbose and split into several functions
- Document generator : Refactor document_generator module
- Schemas : Whole jsonschema generation process. It now uses Oriented Object algorithms. Most of jsonschema module is now deprecated
- Schemas : Remove Any support in schemas
- Schemas : Use of Schemas to serialize typings
- Schemas : Change serialize_typing function name to serialize_annotation
- Workflow : to_dict method do not use pointers anymore
- Workflow : Remove some attributes from serialization


### Removed

- Serialization : Remove warning for dict_to_object if class is not inheriting from SerializableObject

### Chore

- Fix Spelling (x2)
- Pylint : Fix iterator error
- Object : Add Backward Compatibality over method_dict to cover old frontend calls
- Workflow : Add Backward Compatibility over ModelMethod, ClassMethod, GetModelAttribute & SetModelAttribute blocks


### Build

- CI : Upload coverage is now optional


## 0.13.7


### Fix

- Workflow: Ignore Optional File-Like Sequence inputs in WorkflowRun serialization


## 0.13.6


### Fix

- Workflow: Ignore Optional File-Like Sequence inputs in WorkflowRun serialization


## 0.13.5 [09/25/2023]


### Fix

- add python 3.9 minimum requirement to avoid install issues


## 0.13.4 [07/31/2023]


### Added
- Add rotational speed to measures


## 0.13.3 [05/04/2023]


### Changed

- Add rotation speed in measures
- License changed from GPL to Lesser GPL
- package_version is removed from serialization

### Fix

- Fixes a bug when generating a script from a workflow : names containing special quote characters are now properly escaped
- Workflow name correction: correct the name if it contains an apostrophe.


## 0.13.2 [03/01/2023]


### Fix

- Workflow state/run to_dict fix on adding references
- Handle serialization pointers of non-standalone objects
- hash fix: calling hash instead of data hash in eq.

### Added

- Display settings now have load_by_default config option


## 0.13.1


### Fix

- Handle serialization pointers of non-standalone objects
- WorkflowRun settings now sets the right method to call for workflow display


## 0.13.0 [02/14/2023]


### Chore

- Tag for release candidate
- Toggle some D2xx errors

### Fix

- Do not take into account non-eq attributes

### CI

- tutorials/ci_tutorials.py added to check runnability of .ipynb files inside this new folder
- automatic upload of coverage
- spellcheck with pyenchant integrated to pylint
- fixing versions of pylint and pydocstyle to avoid uncontrolled new errors

### Performance

- Change sequence hash to check only first and last element recursively
- Change dict hash to check only first and last element recursively
- Refactor search_memo function to improve copy performance
- Add pre-checks for non list-like element in is_sequence function
- For is_serializable, not using pointers while trying to_dict
- Refactor workflow run values for serializability checks


### Chore

- Toggle some D2xx errors
- Tag for release candidate

### Fixed

- fix str of Dataset

## 0.12.0 [01/20/2023]


### Changed

- Reference path is now given all the way down to plot_data
- DessiaObject kwargs in init are deprecated

### Added
- Save WorkflowRun to a python script

### Fixed

- serialization (to_dict) use some uuids instead of paths for references.
- Workflow : Serialize Imposed Variable Values
- Workflow : to_script Export now write only class name instead of full_classname
- Hot fix for is_valid method of workflow's Block
- Core : add a boolean for platform checking in check_list. Fixes a problem with testing classes not existing
- Reference path for datatools Dataset and ClusteredDataset

### Refactor

- Move ./utils/serialization to ./serialization to avoid cyclic-imports

### CI

- add a check to enforce update of changelog in PR
- code_pydocstyle.py checks daily instead of weekly
- Add a time decrease effect for pylint

### Performance

- Conform doc for many parts of dessia_common
- 100% coverage for clusters module
- cache inspect.fullargspecs calls
- Add trivial checks for simple types
- Avoid redundant serialization in display blocks

### Build

- Update python minimum version to 3.8 (3.7 was not supported in fact)
- Update scikit learn to minimum 1.2.0 for the parameter metric in Clustering

### Tests

- Add backend backward import tests in order to warn when imports are changed

### Chores
- Merge back master to dev
- Docs weekly decrease
- Fixed all remaining pydocstyle errors
- Docs new rules
- More docs


## 0.11.0 [12/19/2022]

### Fixed

- Workflow pipe order when copy
- Diff Dict is now robust to uneven arguments commutation
- Fix path deepth when dict misses keys

### Changed
- Refactor copy_pipes and nbv checking

### Performance

- Cache for fullargspec and deserialize sequence as comprehension list

## v0.10.2

- non serializable attributes were not working
- wrong import to sklearn -> scikit-learn

## v0.10.1

### Fixed
- jsonschema bug
- time rendering on workflow

## v0.10.0

### Added
- Generic save to file
- Sampler class
- (De)Serialization handles 'typing.Type'
- Workflow: handle position in to_dict / dict_to_object process
- sub_matrix method to Dataset


### Breaking Changes
- Measures moved to dessia_common.measures
- HeterogeneousList becomes Dataset
- CategorizedList becomes ClusteredDataset
- Change file organization for datatools:
    * File datatools.py becomes directory datatools
    * class Dataset is now coded in file dataset.py
    * class ClusteredDataset is now coded in file cluster.py
    * class Sampler is now coded in file sampling.py
    * Metrics function are now coded in file metrics.py
- Retrocompatibility is supported for the present time, with a big warning
- pareto methods of datatools are now called with a list of attributes and not anymore a costs matrix


### Changes
- Workflow: improve layout method

### Performance
- switch is_jsonable to orjson

### Fixed
- Workflow: improve to_script


## v0.10.0 [9/26/2022]

### Added
- FiltersList class
- Easy access to Dataset with getitem, len, add, extend, sort
- filtering method in Dataset that calls a FiltersList
- Documentation for Dataset (previously HeterogeneousList), ClusteredDataset (previously CategorizedList), DessiaFilter, FiltersList
- pareto front and parallel plot for Dataset
- Sort parallel plot axis with correlation coefficient (experimental algorithm that seems to work)
- Metrics for Dataset (previously HeterogeneousList) and ClusteredDataset (previously CategorizedList)
- Centroids for ClusteredDataset (previously CategorizedList)
- Nearly all required tests for all these developments

### Breaking Changes
- Change attribute "operator" of DessiaFilter to "comparison_operator"
- Change name of file "cluster.py" to "datatools.py"
- Move Dataset in "datatools.py"

### Changes
- Add 'logical_operator="and"' attribute to workflow.block.Filter
- Improve workflow._data_eq

### Performance
- DessiaFilters and FiltersList: A priori optimized access to elements so it is really faster than before

### Fixed
- Excel Export now used proper length of the cell value
- Fix workflow.copy() issue where a nbv with several pipes became several nbv with one pipe  

## v0.9.0 [7/20/2022]

### Added
- Clustering classes

### Fixed
- Implement to_script method for workflow class
- Prevent foreach name from being it's iter_input name
- Temporarly remove workflow-state from workflow run display settings

## v0.8.0

### Added
- performance analysis function

### Fixed
- babylon display fix
- Any typing does not trigger error with subclass anymore
- workflow: imposed variable values fixes

### Performance
- types: caching type from calling import_module

## v0.7.0

### Fixed
- FileTypes looks for subclass in jsonschema computation instead of wrong isinstance

### Added
- Physical Object: splitting CAD capabilities from DessiaObject
- Workflow to script (for a few blocks to begin with)
- Separate evaluation of displays with display settings feature
- workflow: port matching

### Changed
- Enhanced get attr use literal eval to try to get dict keys
- moved blocks in dessia_common.workflow.blocks, retrocompatbility with previous behavior

### Deleted
- Import Block in workflow

## v0.6.0
### Added
- Exports in workflows
- Workflow inputs documentation from docstrings
- Workflow description and documentation
- Custom Dessia FileTypes

### Changed
- split export functions in two: one to write to a file, one to save in a stream
- to_dict use jsonpointers, des

## v0.5.1
### Fixed
- copy problems
- more tests

## v0.5.0

### Added
- Workflow stop/start first features
- Files Typings
- Inputs can now search for upstream nonblock_variables
- Add input_values addition bulk methods (from block_index & indice sequence)
- Can compute jsonschema from Any annotation
- Add more structure to dev objects
- ClassMethod Block now supports MethodType
- WorkflowState add_input_values activates them
- Several variables index computation methods

### Changed
- data_eq checks for a dual non insinstance test before returning false
- Moved errors to submodule dessia_common.errors
- Workflow.variable_from_indices is not a classmethod anymore
- Full support of MethodType for ModelMethod Block

### Fixed
- Re-instantiate nonblock_variable when copying workflow
- WorkflowState now serialize its values keys in to_dict
- deepcopy of workflow

### Refactor
- Separate in utils module

## v0.4.7
## Added
- File typings
- Support for Iterator typing
- Object's default_dicts now implement objct_class

## v0.4.6

### Added
### Fixed
- Display block faulty definition (removed inputs as init argument & fixed to_dict)
- Workflow & WorkflowRun uses implemented data_eq
- WorkflowRun data_eq when output_value is a sequence
- ForEach checks for workflow_block equivalence instead of equality



#### New Features
* Block Display: propagation of kwargs to model _displays
* MethodType
* Fields in workflows jsonschema that were missing
* Names can now be set at Workflows run and run_again
* Dev Objects implement markdown display
* Support for None argument deserialization
* Support for InstanceOf argument deserialization
* add stl export
* (workflow): add kwargs to block Display subcall
* (markdown): clean indents inside DisplayObject Init
* (workflow): use method typing for ModelMethod
* (typing): add method typing
* (dev): add markdown to dev-objects
* (workflow): add name input to run arguments
* (deserialize_arg): add InstanceOf support
* (dict_to_object): add force_generic argument
* adding some units
* save to file handles stringio
* export formats
* (optimization): fixed attributes
* (forms): add Distance to forms objects
* (core): python_typing mandatory in jsonschema
* restric extrapolation in interpolation_from_istep
* (core): serialize Dict typing
* (core): add standalone_in_db entry to generic union jsonschema
* add InstanceOf typing
* add builtins default
* (typings): add deprecation warning on Subclass
* (core): add exception to prevent docstring parsing from failing platform
* separating __getstate__ and serailizable dict features in DessiaObject
* (forms): add graph2d to dev objects
* rename type datatype as class datatype
* callback in run_again
* (forms): add doc to dev objects
* add docstring parsing & description to jsonschema
* (core): add datatype for Type typing
* check platform
* (workflow): a couple features for workflows
* (forms): allow methods for dev purpose
* add method flag ton method_jsonschema
* (core): default_dict based on datatype
* (core): add standalone_in_db to jsonschema_from_annotation
* add standalone_in_db in DessiaObject's jsonschema method
* kwargs in volmdlr primitives
* Kwargs in plot_data
* (workflow): add input/output connections to ForEach
* (forms): add optionnal sequence
* propose support for default_value type serialization
* add subclass typed arg for dev purpose
* (core): raise TypeError if structure is used as static dict value
* (core): raise ValueError if plot_data is not a sequence
* (workflow): add input/output_connections to WorkflowBlock
* (core): add jsonschema_from_annonation Optional support
* (core): add is_typing function
* (workflow): positions in to_dict()
* (workflow): workflow.displays_() calls workflow_blocks.workflow's display_()
* (typings): add units to jsonschema if Measure
* (workflow): add types to block init

#### Fixes
* workflow ClassMethod in no returns in annotation
* (display_block): fix kwargs propagation
* (filter): remove TypedDict to use conventionnel non standalone class
* (workflow): add name argument to run_again
* (forms): remove extra " form multiline string
* finalize generation
* verbose typing for workflow
* (workflow): MultiPlot inputs
* (workflow): TypedVariable dict_to_object typo
* (deseriliaze_arguments): add supoport of None & Optional
* (workflow): display block inputs and to_dict
* working decision tree generator
* (workflow): fix WorkflowRun data_eq when output is sequence
* (workflow): workflow uses data_eq & ForEach use block.equivalent
* (makefile): revert command
* extrapolate in utils, biggest canvas for plot_data
* (core): remove raise serialise typing error
* sorted in istep
* getstate to dict
* default value for variable-length objects
* moving volume model
* (workflow): recursion error fix
* (workflow): refresh position before setting it in dict
* (workflow): workflowblock position
* (typings): add include_extra to get_type_hints
* (forms): add name and elements to scatterplot & forms workflow
* (core): add missing data to tuple from annotation
* (workflow): dict_to_arguments issues

#### Refactorings
* (workflow): remove pareto from workflow inputs
* (core): propagate subclass name change to jsonschema
* use InstanceOf instead of Subclass
* (workflow): import from dessia_common instead of whole module
* (core): format parsed docstring
* (workflow): change type-arguments typings to Type
* remove default_values on dev objects
* (workflow): set input/output connections to None
* (core): clean code
* (core): proposition on new default_dict
* (core): replace some jsonschema attributes out of loops
* (workflow): comment Function block out
* remove include_extras argument
* (core): add some more introspection helpers
* (workflow): class InstantiateModel instead of InstanciateModel
* (typings): use get_type_hints instead of __annotations__

#### Others
* (plot_data): update to 0.5.1


### [0.4.5]
## Added
- Dev Objects : add maldefined method
- Typings : add Mass typing
- Add force_generic argument to dict_to_object to avoid recursion when generic computation is needed
- Dict typing serilization/deserialization
- All typings serialization
- python_typing is set in all jsonschema

## [0.4.4]
### Added
- InstanceOf typing. Subclass is Deprecated
- Docstring parsing & failure prevention
- Description of class and attributes in jsonschema

### Changed
- Union cannot implement two classes with non coherent standalone_in_db attributes anymore

### Removed
- TypedDict not supported anymore

## [0.4.3]
### Added
- Datatype from jsonschema method
- Method flag in jsonschema
- Add is_builtin function
- Raise ValueError if plot_data is not a sequence
- compute_for argument to full_classname to allow classname computing for class object

### Fixed
- dict_to_arguments of workflows have now right signature

### Changed
- Default values are based on datatypes
- Complex structure as static dict value is not supported anymore
- Remove type from Sequence block
- Use Subclass instead of Type for typings
- Use of instrospection helpers for jsonschema computation from typings (get_type_hints, get_args, get_origin)
- WorkflowRun method_jsonschemas implements Workflow method

## 0.4.2
### Changed
- BREAKING CHANGE : _display_angular is renamed _displays
- BREAKING CHANGE : Block ParallelPlot is renamed MultiPlot
- Kwargs are added to _displays in order to pass args such as reference_attribute
- Display block is now a base class for all display blocks
- ForEach block implementation : workflow_block_input -> iter_input_index

### Added
- Serialization tuples
- DisplayObject
- _displayable_input as Display block class attribute

### Fix
- Fix wrong type check with __origin__ in deserialize argument

## [v0.4.1]


## [v0.4]
### Fix
- _eq_is_eq_data as False in workflow class definitions

### Changed
- introducing _eq_is_data_eq instead of _generic eq
- __origin__ used in serialize_typing instead of _name. For typing type checking

### Added
- _data_diff method to DessiaObject

### Removed
- Compiled parts of package, switching to full open-source

## [0.3.10]
### Fixed
- Run again takes input_values arguments as it should
- Changed workflow to workflow in to_dict/dict_to_object
- Support for new data types
- Added mypy_extensions to setup.py

### Added
- Relevant Error raised in getting deep_attribute when object has no attribute
- is_sequence function
- Documentation on blocks
_ Rerun method for WorkflowRuns & prerequisite (input_values in __init__ et method_jsonschema)
- Method dict attribute
- Generic serialize function
- ParallelPlot block
- Memorize attribute in Variables and intermediate variables values in workflow_run
- Non Standalone object default value in default dict support
- Demo classes for forms data format
- Add unit_tests.py script + exec in drone
- Recursive/complex deepattr function

### Changed
- Add a more explicit message to list as default value error
- Filter block doesn't write _display_angular anymore. See ParallelPlot
- Catalog _display_angular to conform to ParallelPlot Block

## [0.3.8]
### Added
- Change rerun method name to run_again
- Unpacker for sequence workflow block
- Cars dataset as imported models
- Models & Templates added to MANIFEST.in

### Changed
- ModelAttribute use dessia_common's getdeepattr to get attributes in subobjects
- Test directions and signs of coeff for maximization in find_best_objectives
- BestCoefficients names changed to remove space in it
- Directions argument is now mandatory

## [0.3.7]
### Fixed
- coefficients_from_angles list was used as dictionnary in from_angles method

## [0.3.6]
### Added
- Support of dict (keys and values) copy in generic deepcopy
- Find best objective

### Changed
- Export csv to fit vectored_objects removal
- All set variables in _display_angular config instead of just choice_variables
- Strategy for scale and custom value scaling

## [0.3.5]
### Changed
- Objective settings is now inside Objective and not global inside Catalog
- Datasets are reordered to properly display in front
- Ordered attributes for catalog

### Fixed
- Classes in every type object jsonschema
- n_near_values now gives indices in display and is workings

## [0.3.4]
### Changed
- Changes in WorklowRun _display_angular to handle new datasets structure
- Removed VectoredObject. Catalog now know List of List as data array
- Datasets values are now indices of corresponding points in all values of data array

## [0.3.3]
## Added
- WorkflowBlock
- Type checking in workflow
- imposed variable values

## [0.3.2]
### Added
- Plot data in display of DessiaObject
- Deprecation decorator

### Changed
- dessia_methods to allowed_methods
- return & progress_callback removed from _method_jsonschema
- copy as one DessiaObject method with deep attribute
- _editable_variables to _non_editable_attributes
- handling of new frontend display values

### Fixed
- (Quickfix) Check if output_value is None
- revert to working version of vectored_object scale

## [0.3.1]
### Added
- First version of sphinx documentation
- generic dict to arguments
- copy/deeepcopy

## [0.3.0]
### Added
- VectoredObjects

## [0.2.3]
### Added
- nonblock variables in workflow
- jointjs enhancements: layout colors for ports...

## [0.2.2]
### Added
- SerializationError
- is_jsonable
- get_python_class_from_class_name
- base_dict returns class name
### Changed
- Checks for volume_model implementation

## [0.2.1]

### Added
- Evolution class
- ConsistencyError in core

## [0.2.0]

### Added
- _export_formats to DessiaObject

### Fixed
- Check equality of methods working on classes

## [0.1.2] - 2019-12-20

### Added
- Generic eq,
- Copy volmdlr support
