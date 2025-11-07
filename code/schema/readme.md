# VR Foraging NWB Event's table documentation

## Definition

This document defines the naming conventions used for the columns in the Events table of the VR Foraging NWB files. The naming conventions are designed to provide clarity and consistency across different datasets and to facilitate data analysis.

Formal definition can be found in `./specification.yml` and a corresponding json-schema for validation and autocompletion `./schema.json`.

## Nomenclature

* suffixes and prefixes are separated by underscores `_`
* modifiers are separated by a single underscore `_`
* snake casing for all names (e.g. `start_time`, `is_rewarded`)

### suffixes

- `time` - Absolute time of an event, in seconds. All timestamps are considered high-precision (<1ms resolution) unless otherwise specified.
- `duration` - Duration of an event, in seconds
- `position` - Absolute position in the VR environment, in cm
- `length` - Length of an epoch in cm
- `label` - Categorical, human-readable, label for an event. Must be hashable and assumed to not support comparison operators
- `index` - Ordinal number for an event, non-negative integer. Always 0-indexed.
- `count` - Count of occurrences of an event, non-negative integer


### modifiers

- `enable`, `onset` - The enabling or starting of an event
- `disable`, `offset` - The disabling or ending of an event
- `by` - Indicates that the event is grouped or categorized by the modifier (e.g. `reward_by_type`)


### prefixes

- `is`/`has` - Used for boolean values
- `last` - The last occurrence of an event
- `first` - The first occurrence of an event

## References to json-schema/pydantic model

From here https://allenneuraldynamics.github.io/Aind.Behavior.VrForaging/api.task_logic.html
