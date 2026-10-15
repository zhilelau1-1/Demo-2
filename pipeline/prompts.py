"""Prompt templates and JSON output schemas for every LLM stage.

Each builder returns (instructions, schema). Vocabulary enums are embedded in both the prompt text
and the schema; outputs are still re-validated in code after every call.
"""

from . import vocab

PROMPT_VERSION = "2026-10-15.1"

SYSTEM_PROMPT = """You are a manufacturing quality analyst working inside an auditable inspection-triage pipeline for manufactured parts.
Rules:
- Answer only with JSON that matches the provided JSON schema.
- Use only the controlled vocabulary values listed in the instructions; never invent new values.
- Refer only to report IDs and pattern IDs that appear in the input data.
- Never produce numeric priority scores; scoring is done by deterministic code.
- Be concise and factual, and ground every statement in the input data."""


def _defs(definitions):
    return "\n".join(f"  - {k}: {v}" for k, v in definitions.items())


def _thresholds():
    return ", ".join(f"{k} >= {v}" for k, v in vocab.MEASUREMENT_THRESHOLDS.items())


def _obj(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or properties.keys()),
        "additionalProperties": False,
    }


def _str():
    return {"type": "string"}


def _enum(values):
    return {"type": "string", "enum": list(values)}


def _list_of(item, key):
    return _obj({key: {"type": "array", "items": item}})


def translation(report_ids):
    instructions = """TASK: language detection and translation of operator notes (runs before interpretation).
For every note in the input:
1. original_language: the ISO 639-1 code of the note's SUBSTANTIVE content (the defect description), e.g. "en", "ms", "es", "zh". A short English label or prefix (for example "Visual check in Malay:") does not make a note English.
2. is_english: true only when the substantive content is English.
3. english_text: if is_english is true, repeat the note unchanged. Otherwise translate the whole note into clear English. Keep any English prefix, technical terms, part/location references (e.g. J4, BGA), numbers and hedging words (e.g. "slightly", "maybe"). Do not add interpretation.
Return exactly one entry per input note."""
    item = _obj({
        "report_id": _enum(report_ids),
        "original_language": _str(),
        "is_english": {"type": "boolean"},
        "english_text": _str(),
    })
    return instructions, _list_of(item, "notes")


def interpretation(report_ids):
    instructions = f"""TASK: Stage 1 report interpretation.
Interpret every inspection report in the input using BOTH the operator note (note_for_interpretation, already in English) AND the structured data (model_finding, confidence, measurements, image_count, station, part).

Output one object per report (every report exactly once) with these fields:
- report_id: copied from the input.
- defect_category: exactly one of
{_defs(vocab.DEFECT_CATEGORY_DEFINITIONS)}
- severity: exactly one of
{_defs(vocab.SEVERITY_DEFINITIONS)}
- disposition: exactly one of
{_defs(vocab.DISPOSITION_DEFINITIONS)}
- reason_short: at most 25 words citing the specific evidence (measurement versus reference threshold, note wording).

Measurement reference thresholds (at or above = reject-level evidence for that defect type): {_thresholds()}.
Values just under a threshold are borderline. Purely cosmetic surface marks rarely justify "critical".
The fields translated and original_language are filled in by code from the normalisation step; do not output them."""
    item = _obj({
        "report_id": _enum(report_ids),
        "defect_category": _enum(vocab.DEFECT_CATEGORIES),
        "severity": _enum(vocab.SEVERITIES),
        "disposition": _enum(vocab.DISPOSITIONS),
        "reason_short": _str(),
    })
    return instructions, _list_of(item, "interpretations")


def pattern_grouping(report_ids):
    instructions = f"""TASK: Stage 2 recurring pattern grouping.
The input contains every report with its Stage 1 interpretation (defect_category, severity, disposition, reason_short), part metadata, station, timestamp, confidence, measurements and the English note used for interpretation.
Identify recurring defect patterns that may indicate a PROCESS issue rather than isolated defects.
- A pattern needs at least 2 reports that share a defect signature: the same defect category plus a shared part, location (e.g. joint J4, BGA region, bracket arm), station or time cluster.
- Do not create patterns from a single report. Isolated reports may belong to no pattern.
- Only put a report in more than one pattern when the patterns are genuinely distinct (for example a part-specific pattern and a cross-part pattern).
- Reports interpreted as no_material_issue or with low severity should only be included when they clearly extend a signature.

For each pattern output:
- title: a short descriptive title.
- supporting_report_ids: the report IDs in the pattern.
- pattern_strength: exactly one of
{_defs(vocab.PATTERN_STRENGTH_DEFINITIONS)}
- likely_driver: one sentence on the most plausible process cause, hedged appropriately.
- recommended_action: one sentence with the next concrete action.
pattern_id values are assigned by code; do not output them."""
    item = _obj({
        "title": _str(),
        "supporting_report_ids": {"type": "array", "items": _enum(report_ids)},
        "pattern_strength": _enum(vocab.PATTERN_STRENGTHS),
        "likely_driver": _str(),
        "recommended_action": _str(),
    })
    return instructions, _list_of(item, "patterns")


def routing(flagged_ids):
    instructions = f"""TASK: Stage 3 case routing.
The input contains the top flagged cases (with their deterministic priority score breakdowns), all detected recurring patterns, and the owner-team vocabulary.
For EVERY flagged case output:
- report_id: copied from the input.
- teams: one or more owner teams, each exactly one of
{_defs(vocab.OWNER_TEAM_RESPONSIBILITIES)}
  Route to every team with a real action to take; a case may go to several teams.
- briefing_note: a concise internal handoff (at most 50 words): what was found, why it matters (pattern membership, measurements), and what the receiving teams should do first."""
    item = _obj({
        "report_id": _enum(flagged_ids),
        "teams": {"type": "array", "items": _enum(vocab.OWNER_TEAMS)},
        "briefing_note": _str(),
    })
    return instructions, _list_of(item, "routes")


def explanation(flagged_ids):
    instructions = """TASK: reviewer explanations for flagged cases.
The input contains each flagged case with its interpretation, its deterministic priority score and the score breakdown, its pattern membership and its routing.
For every flagged case output:
- report_id: copied from the input.
- why_prioritised: 2-3 plain-English sentences explaining why the case ranks where it does. Use only the numbers in the score breakdown; never recompute or change them.
- verification_focus: one sentence naming the specific evidence a human reviewer must verify before any production decision (e.g. re-measure a value, re-image a joint, confirm a translation)."""
    item = _obj({
        "report_id": _enum(flagged_ids),
        "why_prioritised": _str(),
        "verification_focus": _str(),
    })
    return instructions, _list_of(item, "explanations")


def shift_trend_summary():
    instructions = """TASK: shift-level trend summary.
The input contains deterministic shift statistics: hourly defect distribution, per-pattern measurement trends (with a code-computed trend label), and station/part activity.
Write:
- summary: 2-4 sentences describing how defect activity developed over the shift and which station/part family is most affected.
- worsening_assessment: 1-2 sentences on whether any recurring issue appears to be worsening, citing pattern IDs and measurement values. Do not contradict the code-computed trend labels."""
    return instructions, _obj({"summary": _str(), "worsening_assessment": _str()})


def disagreement_review(candidate_ids):
    instructions = """TASK: low-confidence / disagreement review.
Each input candidate was flagged by deterministic rules because the model finding, the measurements, the confidence and/or the interpreted result appear weakly aligned or uncertain. The rule signals are listed per candidate.
For every candidate output:
- report_id: copied from the input.
- disagreement_reason: one or two sentences explaining the misalignment in plain English, citing the specific values.
- recommended_follow_up: one sentence with the concrete follow-up check."""
    item = _obj({
        "report_id": _enum(candidate_ids),
        "disagreement_reason": _str(),
        "recommended_follow_up": _str(),
    })
    return instructions, _list_of(item, "items")


def alert_justification():
    instructions = """TASK: lot/shift alert justification.
The alert rating was assigned by a deterministic rubric (the triggered rules are in the input). Do NOT change the rating.
Write:
- justification: 3-5 sentences justifying the rating with specific report IDs, recurring pattern data, top priority scores, and whether multiple stations or parts are affected.
- operational_posture: 1-2 sentences with the recommended operational posture for the line (e.g. containment, increased sampling, normal running)."""
    return instructions, _obj({"justification": _str(), "operational_posture": _str()})


def reinspection_plan(pattern_ids):
    instructions = """TASK: next-shift reinspection plan.
The input contains the detected patterns with code-derived sampling priorities, stations, measurement thresholds and stop/continue criteria.
Write:
- plan_summary: 2-3 sentences summarising the reinspection focus for the next shift.
- watch_notes: for every pattern, one sentence describing the defect signature inspectors should watch for (location, appearance, measurement)."""
    note = _obj({"pattern_id": _enum(pattern_ids), "signature_to_watch": _str()})
    return instructions, _obj({"plan_summary": _str(), "watch_notes": {"type": "array", "items": note}})
