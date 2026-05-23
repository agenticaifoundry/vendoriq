"""
VendorIQ Base LLM Module
========================
The foundation of every agent in VendorIQ.

All LLM calls go through this module. Never call the Anthropic SDK
directly from agents, orchestrators, or API routes.

This module handles:
- Structured output extraction from news/documents
- Markdown fence stripping (Claude sometimes wraps JSON in fences)
- Retry logic for rate limit errors
- Cost tracking per call
- Async support for concurrent agent execution
"""

import json
import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
import anthropic
from models import DisruptionEvent

load_dotenv()

# ── Client Setup ───────────────────────────────────────────────────────────────
client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL_FAST      = "claude-haiku-4-5-20251001"
MODEL_SMART     = "claude-sonnet-4-20250514"
MODEL_POWERFUL  = "claude-opus-4-6"


# ── System Prompts ─────────────────────────────────────────────────────────────

DISRUPTION_EXTRACTION_PROMPT = """You are a senior procurement analyst at a global manufacturing company.

When given a supply chain disruption event, return ONLY a JSON object.
No explanation. No markdown. No preamble. No code fences. Just the raw JSON.

Return exactly this structure:
{
  "event_type": "port_closure|factory_fire|geopolitical|logistics|weather|financial|labour_dispute|other",
  "location": "specific city and country",
  "severity": "low|medium|high|critical",
  "estimated_duration_hours": number or null if unknown,
  "affected_transport_modes": ["sea", "air", "road", "rail"] — include all that apply,
  "summary": "one sentence description of the disruption",
  "immediate_action_required": true or false
}

Severity guidelines:
- low: minor delays, workarounds available, <24hr duration
- medium: significant delays, some alternatives available, 24-72hr duration
- high: major disruption, limited alternatives, 72hr-2 week duration
- critical: catastrophic disruption, no alternatives, >2 weeks or unknown duration

If you cannot determine a value with confidence, use null for optional fields."""


# ── Utility Functions ──────────────────────────────────────────────────────────

def clean_llm_json(text: str) -> str:
    """
    Strip markdown code fences that Claude sometimes wraps around JSON.

    Claude occasionally returns:
```json
        { "key": "value" }
```

    Even when told not to. This function handles all variations:
    - ```json ... ```
    - ``` ... ```
    - Plain JSON with no fences (passes through unchanged)
    """
    text = text.strip()

    # Remove ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        # Remove the opening fence line (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline:].strip()

        # Remove the closing fence
        if text.endswith("```"):
            text = text[:-3].strip()

    return text


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str
) -> float:
    """
    Calculate the cost of an LLM call in USD.

    Prices per million tokens (2025):
    - Haiku:  $0.80 input / $4.00 output
    - Sonnet: $3.00 input / $15.00 output
    - Opus:   $15.00 input / $75.00 output
    """
    pricing = {
        MODEL_FAST:     (0.80,  4.00),
        MODEL_SMART:    (3.00,  15.00),
        MODEL_POWERFUL: (15.00, 75.00),
    }

    input_price, output_price = pricing.get(model, (3.00, 15.00))

    return (
        (input_tokens / 1_000_000) * input_price +
        (output_tokens / 1_000_000) * output_price
    )


# ── Core Functions ─────────────────────────────────────────────────────────────

async def extract_disruption_event(
    news_text: str,
    max_retries: int = 3
) -> DisruptionEvent:
    """
    Takes raw news text and returns a validated DisruptionEvent.

    This is the entry point for every VendorIQ analysis.
    The analyst pastes news, this function returns structured data
    that every downstream agent can work with.

    Args:
        news_text: Raw news headline, article, or description
        max_retries: How many times to retry if Claude returns invalid JSON

    Returns:
        DisruptionEvent: Validated Pydantic model ready for downstream processing

    Raises:
        ValueError: If Claude returns invalid data after all retries
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=MODEL_FAST,
                max_tokens=500,
                temperature=0,
                system=DISRUPTION_EXTRACTION_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Extract disruption data from this news. "
                        f"Return raw JSON only, no markdown fences:\n\n{news_text}"
                    )
                }]
            )

            raw_text = response.content[0].text

            # Strip markdown fences if Claude added them
            cleaned_text = clean_llm_json(raw_text)

            # Parse JSON
            data = json.loads(cleaned_text)

            # Validate through Pydantic
            event = DisruptionEvent(
                **data,
                raw_input=news_text
            )

            return event

        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed on attempt {attempt + 1}: {e}"
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

        except Exception as e:
            error_str = str(e)

            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                last_error = f"Rate limited. Retrying in {wait_time}s..."

            elif "ValidationError" in type(e).__name__:
                last_error = f"Validation failed on attempt {attempt + 1}: {e}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

            else:
                raise

    raise ValueError(
        f"Failed to extract disruption event after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


async def ask_claude(
    question: str,
    system: str,
    model: str = MODEL_SMART,
    max_tokens: int = 2048,
    temperature: float = 0.3
) -> str:
    """
    General-purpose Claude call for free-form text output.
    Used by agents that need reasoning or explanation, not structured data.
    """
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text
