# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fix
- Fix wrong type check with __origin__ in deserialize argument

### Changed
- ForEach block implementation : workflow_block_input -> iter_input_index

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

