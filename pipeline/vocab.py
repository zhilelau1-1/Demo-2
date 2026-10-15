"""Controlled vocabularies: the single source of truth for every enum the pipeline accepts."""

DEFECT_CATEGORIES = (
    "solder_voiding",
    "pin_misalignment",
    "surface_scratch",
    "no_material_issue",
    "uncertain",
)
SEVERITIES = ("critical", "high", "medium", "low")
DISPOSITIONS = ("hold", "reinspect", "monitor", "pass")
OWNER_TEAMS = ("Quality", "Process Engineering", "Manufacturing", "Supplier Quality")
PATTERN_STRENGTHS = ("strong", "moderate", "weak")
ALERT_RATINGS = ("green", "yellow", "orange", "red")

DEFECT_CATEGORY_DEFINITIONS = {
    "solder_voiding": "Gas voids or dark regions inside solder joints (typically X-ray), e.g. under BGAs or connector joints.",
    "pin_misalignment": "Pins, leads or connector contacts shifted, skewed or offset from their nominal position.",
    "surface_scratch": "Scratches, line marks or abrasion on a part surface or board edge.",
    "no_material_issue": "Evidence shows no defect of concern; the part appears acceptable.",
    "uncertain": "Evidence is contradictory or insufficient to assign any other category.",
}
SEVERITY_DEFINITIONS = {
    "critical": "Likely functional or reliability failure; evidence clearly beyond acceptance and/or escalating.",
    "high": "At or above the reference reject threshold; probable nonconformance.",
    "medium": "Borderline: near the reference threshold, needs confirmation.",
    "low": "Well below concern level; cosmetic or traceability only.",
}
DISPOSITION_DEFINITIONS = {
    "hold": "Quarantine the unit (and suspect lot) pending human review.",
    "reinspect": "Re-examine the unit with a second method or inspector before deciding.",
    "monitor": "Release path is open, but track for recurrence.",
    "pass": "No action required.",
}
PATTERN_STRENGTH_DEFINITIONS = {
    "strong": "Several reports share the same signature (category, location, part and/or station) with measurements at or above threshold or worsening.",
    "moderate": "A consistent signature, but measurements are borderline or stations/locations are mixed.",
    "weak": "A plausible link with limited or inconsistent evidence.",
}
OWNER_TEAM_RESPONSIBILITIES = {
    "Quality": "Disposition decisions, containment/quarantine, review sign-off, customer-impact assessment.",
    "Process Engineering": "Root cause in process parameters: reflow profile, paste/stencil, placement programs, fixtures.",
    "Manufacturing": "Line operations, handling, rework, tooling and operator practice; executes reinspection.",
    "Supplier Quality": "Incoming material, component or lot issues and supplier corrective actions.",
}

# Reference thresholds (shared by the scoring formula and the severity guidance given to the LLM).
MEASUREMENT_THRESHOLDS = {
    "void_ratio": 0.20,
    "misalignment_mm": 0.20,
    "scratch_length_mm": 6.0,
}
CATEGORY_MEASUREMENT = {
    "solder_voiding": "void_ratio",
    "pin_misalignment": "misalignment_mm",
    "surface_scratch": "scratch_length_mm",
}
FUNCTIONAL_CATEGORIES = ("solder_voiding", "pin_misalignment")

VOCABULARIES = {
    "defect_category": DEFECT_CATEGORIES,
    "severity": SEVERITIES,
    "disposition": DISPOSITIONS,
    "team": OWNER_TEAMS,
    "pattern_strength": PATTERN_STRENGTHS,
    "alert_rating": ALERT_RATINGS,
}


def vocab_error(field, value):
    """Return an error message if value is not in the vocabulary for field, else None."""
    allowed = VOCABULARIES[field]
    if value not in allowed:
        return f"{field}={value!r} is not in the controlled vocabulary {list(allowed)}"
    return None
