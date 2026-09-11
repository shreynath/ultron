"""Application composition for capability-owned evidence providers.

The runtime receives the resulting registry through a generic interface.  This
module is the explicit application wiring point for currently installed
capabilities, not verification policy.
"""
from __future__ import annotations

from app.brain.reference_capability import ReferenceEvidenceProvider
from app.brain.verification import EvidenceProviderRegistry, WeatherEvidenceProvider


def create_application_evidence_providers() -> EvidenceProviderRegistry:
    """Return providers for the capabilities installed in this application."""

    return EvidenceProviderRegistry((
        WeatherEvidenceProvider(),
        ReferenceEvidenceProvider("pi_prefix"),
        ReferenceEvidenceProvider("positional_lookup"),
        ReferenceEvidenceProvider("science_reference"),
    ))
