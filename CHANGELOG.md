# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


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


### Fixed
- (Quickfix) Check if output_value is None


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

