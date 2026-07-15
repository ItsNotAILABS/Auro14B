"""JSON Schema definitions for hub tools, labs, and API endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolSchema:
    """Schema definition for a research tool.

    Attributes:
        tool_id: Unique tool identifier.
        name: Human-readable name.
        description: What the tool does.
        input_schema: JSON Schema for input parameters.
        output_schema: JSON Schema for output.
        category: Tool category.
    """

    tool_id: str
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "category": self.category,
        }


@dataclass
class LabSchema:
    """Schema definition for a lab environment.

    Attributes:
        domain: Lab domain identifier.
        name: Human-readable lab name.
        capabilities: List of supported operations.
        operations: Detailed schema per operation.
    """

    domain: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    operations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "name": self.name,
            "capabilities": self.capabilities,
            "operations": self.operations,
        }


class HubSchema:
    """Central schema registry for the research hub.

    Manages discovery of all available tools, labs, and their schemas.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSchema] = {}
        self._labs: Dict[str, LabSchema] = {}

    def register_tool(self, schema: ToolSchema) -> None:
        self._tools[schema.tool_id] = schema

    def register_lab(self, schema: LabSchema) -> None:
        self._labs[schema.domain] = schema

    def get_tool(self, tool_id: str) -> Optional[ToolSchema]:
        return self._tools.get(tool_id)

    def get_lab(self, domain: str) -> Optional[LabSchema]:
        return self._labs.get(domain)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._tools.values()]

    def list_labs(self) -> List[Dict[str, Any]]:
        return [l.to_dict() for l in self._labs.values()]

    def openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI-compatible specification."""
        paths: Dict[str, Any] = {}

        for tool_id, schema in self._tools.items():
            paths[f"/tools/{tool_id}"] = {
                "post": {
                    "summary": schema.name,
                    "description": schema.description,
                    "requestBody": {
                        "content": {
                            "application/json": {"schema": schema.input_schema}
                        }
                    },
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {"schema": schema.output_schema}
                            }
                        }
                    },
                }
            }

        for domain, schema in self._labs.items():
            paths[f"/labs/{domain}"] = {
                "get": {
                    "summary": schema.name,
                    "description": f"Lab capabilities: {', '.join(schema.capabilities)}",
                }
            }
            for op in schema.capabilities:
                paths[f"/labs/{domain}/{op}"] = {
                    "post": {
                        "summary": f"{schema.name} — {op}",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": schema.operations.get(op, {})
                                }
                            }
                        },
                    }
                }

        return {
            "openapi": "3.1.0",
            "info": {
                "title": "MESIE Research Hub API",
                "version": "1.0.0",
                "description": "Universal Research Lab Platform",
            },
            "paths": paths,
        }
