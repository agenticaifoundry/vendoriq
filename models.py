"""
VendorIQ Data Models
====================
All Pydantic models for the entire system live here.
Every LLM output is parsed through one of these models before
touching any other part of the system.

Rule: Never use raw LLM text in business logic. Always parse through Pydantic first.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime, timezone
from enum import Enum
from datetime import datetime, timezone


# ── Enums ──────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    PORT_CLOSURE    = "port_closure"
    FACTORY_FIRE    = "factory_fire"
    GEOPOLITICAL    = "geopolitical"
    LOGISTICS       = "logistics"
    WEATHER         = "weather"
    FINANCIAL       = "financial"
    LABOUR_DISPUTE  = "labour_dispute"
    OTHER           = "other"


class SeverityLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class TransportMode(str, Enum):
    SEA  = "sea"
    AIR  = "air"
    ROAD = "road"
    RAIL = "rail"


# ── Core Event Models ──────────────────────────────────────────────────────────

class DisruptionEvent(BaseModel):
    """
    Represents a supply chain disruption event.
    Extracted by Claude from unstructured news text.
    This is the entry point for every VendorIQ analysis.
    """
    event_type: EventType
    location: str = Field(min_length=2, max_length=200)
    severity: SeverityLevel
    estimated_duration_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=8760,  # max 1 year
        description="Estimated duration in hours. None if unknown."
    )
    affected_transport_modes: List[TransportMode] = Field(
        min_length=1,
        description="At least one transport mode must be affected"
    )
    summary: str = Field(
        min_length=10,
        max_length=500,
        description="One sentence description of the event"
    )
    immediate_action_required: bool
    raw_input: Optional[str] = Field(
        default=None,
        description="The original news text this was extracted from"
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="When this event was extracted"
    )

    @field_validator('location')
    @classmethod
    def location_must_be_meaningful(cls, v: str) -> str:
        if v.lower() in ['unknown', 'n/a', 'none', '']:
            raise ValueError('Location must be a real place, not unknown or n/a')
        return v

    @field_validator('summary')
    @classmethod
    def summary_must_be_sentence(cls, v: str) -> str:
        if not v[0].isupper():
            v = v.capitalize()
        return v

    @property
    def is_sea_affected(self) -> bool:
        return TransportMode.SEA in self.affected_transport_modes

    @property
    def severity_score(self) -> int:
        """Numeric score for severity. Useful for sorting and filtering."""
        scores = {
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4
        }
        return scores[self.severity]


# ── Vendor Models (used from Section 11 onwards) ──────────────────────────────

class VendorTier(str, Enum):
    TIER_1 = "tier1"  # Strategic — sole source or critical
    TIER_2 = "tier2"  # Important — limited alternatives
    TIER_3 = "tier3"  # Commodity — many alternatives


class Vendor(BaseModel):
    """Represents a vendor in the VendorIQ knowledge base."""
    id: str = Field(pattern=r"^VND-\d{3}$", description="Format: VND-001")
    name: str
    tier: VendorTier
    country: str
    primary_port: Optional[str] = None
    risk_score: float = Field(ge=0.0, le=1.0)
    is_sole_source: bool = False
    annual_spend_usd: Optional[float] = Field(default=None, ge=0)


class Component(BaseModel):
    """A component or raw material supplied by a vendor."""
    id: str = Field(pattern=r"^CMP-\d{3}$", description="Format: CMP-001")
    name: str
    vendor_id: str
    is_sole_source: bool = False
    lead_time_days: int = Field(ge=1, le=365)
    unit_cost_usd: float = Field(ge=0)


class SKU(BaseModel):
    """A finished sellable product that uses components."""
    id: str = Field(pattern=r"^SKU-\d{4}$", description="Format: SKU-0001")
    name: str
    component_ids: List[str]
    current_stock: int = Field(ge=0)
    daily_demand: float = Field(ge=0)

    @property
    def days_of_cover(self) -> Optional[float]:
        if self.daily_demand == 0:
            return None
        return self.current_stock / self.daily_demand


# ── Risk Assessment Model (used from Section 13 onwards) ──────────────────────

class RiskAssessment(BaseModel):
    """
    The final output of a VendorIQ analysis.
    Produced by the SynthesisAgent from all specialist agent findings.
    """
    disruption_event: DisruptionEvent
    affected_vendor_ids: List[str]
    affected_sku_ids: List[str]
    affected_customer_count: int = Field(ge=0)
    overall_risk_level: SeverityLevel
    estimated_financial_impact_usd: Optional[float] = None
    force_majeure_eligible_vendor_ids: List[str] = []
    recommended_actions: List[str] = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    total_analysis_cost_usd: Optional[float] = None
    total_analysis_latency_ms: Optional[int] = None
