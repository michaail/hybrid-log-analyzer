from pydantic import BaseModel, Field
from typing import List


class TemplateField(BaseModel):
    name: str = Field(description="Field name or placeholder position (e.g. 'block_id', 'src')")
    description: str = Field(description="What this field represents in the system")
    anomaly_relevance: str = Field(description="How unexpected values in this field signal an anomaly")


class FailureMode(BaseModel):
    name: str = Field(description="Short name for the failure mode (e.g. 'Network Partition')")
    description: str = Field(description="How this failure manifests in the logs")
    observable_signal: str = Field(description="Specific log pattern or absence that indicates this failure")


class EnrichedTemplate(BaseModel):
    component: str = Field(description="System component that generated the log (e.g. 'DataNode$DataXceiver')")
    component_role: str = Field(description="Brief description of what this component does in the system")
    log_level: str = Field(description="Log level extracted from the template (e.g. INFO, WARN, ERROR)")
    purpose: str = Field(description="What this log event means in the context of normal system operation")
    fields: List[TemplateField] = Field(description="Semantic description of each variable field in the template")
    expected_sequence: List[str] = Field(
        description="Ordered list of log template patterns that should precede and follow this event in normal operation"
    )
    failure_modes: List[FailureMode] = Field(
        description="Failure modes detectable by monitoring this template and its surrounding sequence"
    )
    anomaly_indicators: List[str] = Field(
        description="Concrete observable signals that this event or its absence is part of an anomaly"
    )
    related_templates: List[str] = Field(
        description="Other log template patterns (using <*> notation) that should be correlated with this one for RCA"
    )
