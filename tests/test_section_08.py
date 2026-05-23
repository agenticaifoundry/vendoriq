"""
Section 8 Tests — LLMs & Prompt Engineering
Run with: pytest tests/test_section_08.py -v
"""

import pytest
from agents.base_llm import extract_disruption_event, calculate_cost, MODEL_FAST, MODEL_SMART
from models import DisruptionEvent, SeverityLevel, EventType, TransportMode


# ── Test Data ──────────────────────────────────────────────────────────────────

NEWS_SAMPLES = [
    {
        "input": "Port of Shanghai closed for 72 hours due to Typhoon Bebinca.",
        "expected_event_type": EventType.WEATHER,
        "expected_severity": SeverityLevel.HIGH,
        "expected_sea_affected": True,
    },
    {
        "input": "Factory fire at VND-042 manufacturing plant in Shenzhen. Facility shut down indefinitely.",
        "expected_event_type": EventType.FACTORY_FIRE,
        "expected_severity": SeverityLevel.CRITICAL,
        "expected_action_required": True,
    },
    {
        "input": "Minor customs delay at Rotterdam port. Expected to clear within 6 hours.",
        "expected_event_type": EventType.LOGISTICS,
        "expected_severity": SeverityLevel.LOW,
        "expected_action_required": False,
    }
]


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_returns_disruption_event():
    """The function should return a valid DisruptionEvent."""
    event = await extract_disruption_event(NEWS_SAMPLES[0]["input"])
    assert isinstance(event, DisruptionEvent)


@pytest.mark.asyncio
async def test_shanghai_typhoon_extraction():
    """Shanghai typhoon should be classified as weather, high severity."""
    sample = NEWS_SAMPLES[0]
    event = await extract_disruption_event(sample["input"])

    assert event.event_type == sample["expected_event_type"]
    assert event.severity == sample["expected_severity"]
    assert event.is_sea_affected == sample["expected_sea_affected"]
    assert any(x in event.location for x in ["Shanghai", "China"])
    assert event.raw_input == sample["input"]


@pytest.mark.asyncio
async def test_factory_fire_extraction():
    """Factory fire should be critical with immediate action required."""
    sample = NEWS_SAMPLES[1]
    event = await extract_disruption_event(sample["input"])

    assert event.event_type == sample["expected_event_type"]
    assert event.severity == sample["expected_severity"]
    assert event.immediate_action_required == sample["expected_action_required"]


@pytest.mark.asyncio
async def test_minor_delay_extraction():
    """Minor delay should be low severity, no immediate action."""
    sample = NEWS_SAMPLES[2]
    event = await extract_disruption_event(sample["input"])

    assert event.severity == sample["expected_severity"]
    assert event.immediate_action_required == sample["expected_action_required"]


@pytest.mark.asyncio
async def test_raw_input_preserved():
    """The original news text should be preserved in the model."""
    # Use a real news headline — LLM functions need realistic inputs
    news = "Port of Shanghai closed for 72 hours due to Typhoon Bebinca."
    event = await extract_disruption_event(news)
    assert event.raw_input == news


@pytest.mark.asyncio
async def test_severity_score_ordering():
    """Higher severity should have higher numeric score."""
    low_event = await extract_disruption_event(NEWS_SAMPLES[2]["input"])
    high_event = await extract_disruption_event(NEWS_SAMPLES[0]["input"])
    assert high_event.severity_score > low_event.severity_score


def test_cost_calculation():
    """Cost calculation should return a positive float."""
    cost = calculate_cost(1000, 500, MODEL_FAST)
    assert cost > 0
    assert isinstance(cost, float)


def test_cost_haiku_cheaper_than_sonnet():
    """Haiku should cost less than Sonnet for the same token count."""
    haiku_cost = calculate_cost(1000, 500, MODEL_FAST)
    sonnet_cost = calculate_cost(1000, 500, MODEL_SMART)
    assert haiku_cost < sonnet_cost
