# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pydantic",
#     "pyyaml",
# ]
# ///

import datetime
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field
import json
import os
import yaml

KNOWN_JSON_TYPES = Literal[
    "string", "number", "integer", "boolean", "array", "object", "null"
]


class NwbEvent(BaseModel):
    """Individual event definition within the NWB events specification."""

    data_type: KNOWN_JSON_TYPES = Field(
        description="JSON Schema type - can be a single type or nullable type [type, null]"
    )
    is_nullable: bool = Field(
        default=False, description="Indicates whether the event can be null"
    )
    description: str = Field(description="Human-readable description of the item")
    implementation_details: Optional[str] = Field(
        default=None, description="Implementation detail or reference"
    )
    unit: Optional[str] = Field(
        default=None, description="Unit of measurement or specification"
    )


class NwbEventsSpec(BaseModel):
    """NWB Event Description Schema - top-level specification for NWB events."""

    date: datetime.date = Field(description="Date and time when the data was recorded")
    events: Dict[str, NwbEvent] = Field(
        description="Dictionary of event definitions keyed by event name"
    )


if __name__ == "__main__":
    # Build schema
    schema = NwbEventsSpec.model_json_schema()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(script_dir, "schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"Schema saved to: {schema_path}")

    # Read and deserialize YAML specification
    with open(
        os.path.join(script_dir, "specification.yml"), "r", encoding="utf-8"
    ) as f:
        yaml_data = yaml.safe_load(f)

    spec = NwbEventsSpec(**yaml_data)

    print(f"Successfully parsed specification with {len(spec.events)} events")
    print(f"Date: {spec.date}")

    # Generate markdown table
    markdown_lines = []
    markdown_lines.append("# NWB Events Specification")
    markdown_lines.append("")
    markdown_lines.append(
        "<!-- DO NOT MANUALLY EDIT THIS FILE. THIS IS AUTO-GENERATED USING `specification.yml`!!!! -->"
    )
    markdown_lines.append("")

    markdown_lines.append(f"**Date:** {spec.date}")
    markdown_lines.append("")
    markdown_lines.append(
        "| Event Name | Data Type | Nullable | Unit | Description | Implementation Details |"
    )
    markdown_lines.append(
        "|------------|-----------|----------|------|-------------|------------------------|"
    )

    for event_name, event_data in spec.events.items():
        # Escape pipe characters in descriptions and implementation details
        description = (
            event_data.description.replace("|", "\\|") if event_data.description else ""
        )
        impl_details = (
            event_data.implementation_details.replace("|", "\\|")
            if event_data.implementation_details
            else ""
        )
        unit = event_data.unit or ""
        nullable = "Yes" if event_data.is_nullable else "No"

        markdown_lines.append(
            f"| {event_name} | {event_data.data_type} | {nullable} | {unit} | {description} | {impl_details} |"
        )

    # Save markdown table
    table_path = os.path.join(script_dir, "table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    print(f"Markdown table saved to: {table_path}")
