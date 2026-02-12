"""
Patient Profile Data Structure for EXPRAG Workflow (Phase 1)

This module defines the standardized data structure for patient profiles
used in the Experience Retrieval-Augmentation system.
"""

from typing import List, Dict, Union, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PatientProfile(BaseModel):
    """
    Standardized patient profile for colonoscopy clinical data.
    
    This structure enables structured comparison between patients
    for cohort-based retrieval.
    
    Attributes:
        case_id: Unique identifier for this patient case/analysis
        indication: Primary clinical indication (e.g., "IBD", "screening")
        polyp_history: List of polyp types or dictionary with polyp details
        procedures: List of procedure codes (e.g., ["colonoscopy", "biopsy"])
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "case_id": "analysis_id_001",
                "indication": "IBD",
                "polyp_history": ["tubular", "sessile"],
                "procedures": ["colonoscopy", "biopsy", "polypectomy"]
            }
        }
    )
    
    case_id: str = Field(..., description="Unique case/analysis identifier")
    indication: str = Field(..., description="Primary clinical indication")
    age: Optional[float] = Field(None, description="Patient age at colonoscopy")
    sum_pmayo: Optional[float] = Field(None, description="Sum of Partial Mayo Score")
    dz_location: Optional[str] = Field(None, description="Disease location (active only)")
    hb: Optional[float] = Field(None, description="Hemoglobin levels")
    rectal_bleed: Optional[int] = Field(None, description="Rectal bleeding score (from Mayo)")
    
    polyp_history: Union[List[str], Dict[str, Any]] = Field(
        default_factory=list,
        description="Polyp history as list of types or structured dict"
    )
    procedures: List[str] = Field(
        default_factory=list,
        description="List of procedure codes or names"
    )
    
    @field_validator('indication')
    @classmethod
    def indication_must_not_be_empty(cls, v):
        """Ensure indication is not empty."""
        if not v or not v.strip():
            raise ValueError('indication cannot be empty')
        return v.strip().lower()
    
    @field_validator('dz_location')
    @classmethod
    def normalize_dz_location(cls, v):
        """Normalize disease location to lowercase."""
        return v.strip().lower() if v else None
    
    @field_validator('polyp_history')
    @classmethod
    def normalize_polyp_history(cls, v):
        """Normalize polyp history to consistent format."""
        if isinstance(v, dict):
            # Keep as dict if already structured
            return v
        elif isinstance(v, list):
            # Normalize strings to lowercase
            return [str(item).lower().strip() for item in v if item]
        else:
            # Convert single value to list
            return [str(v).lower().strip()] if v else []
    
    @field_validator('procedures')
    @classmethod
    def normalize_procedures(cls, v):
        """Normalize procedure codes to lowercase."""
        return [str(proc).lower().strip() for proc in v if proc]
    
    def to_comparison_dict(self) -> Dict[str, Any]:
        """
        Convert profile to standardized format for comparison.
        
        Returns:
            Dictionary with comparable attributes (sets or numerical values)
        """
        # Convert polyp history to set
        if isinstance(self.polyp_history, dict):
            polyp_set = set(str(v).lower() for v in self.polyp_history.values() if v)
        else:
            polyp_set = set(self.polyp_history)
        
        return {
            'indication': {self.indication},
            'polyp_history': polyp_set,
            'procedures': set(self.procedures),
            'age': self.age,
            'sum_pmayo': self.sum_pmayo,
            'dz_location': {self.dz_location} if self.dz_location else set(),
            'hb': self.hb,
            'rectal_bleed': self.rectal_bleed
        }
