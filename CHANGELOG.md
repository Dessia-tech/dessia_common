# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Export csv to fit vectored_objects removal
- All set variables in _display_angular config instead of just choice_variables

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

