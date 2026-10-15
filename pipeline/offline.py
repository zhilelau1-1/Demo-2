"""Deterministic rule-based stand-in for the LLM, used when no Anthropic credentials are available.

It answers the same stage payloads with the same output schemas as the live model, so the pipeline,
validation and logging behave identically. Its translation is glossary-based and covers only a few
languages; notes in other languages are reported as untranslated rather than guessed.
"""

import re

from . import vocab

RULES_VERSION = "rules-v1"

# --------------------------------------------------------------------------------------------------
# Language detection and glossary translation
# --------------------------------------------------------------------------------------------------

_STOPWORDS = {
    "en": set("the and is are but near looks look of to on in with a an this that not below above still another "
              "same from has have be could probably possible visible seems slightly off edge left right under over "
              "present logging for at by unit units board similar area shape case region third today morning".split()),
    "ms": set("di dan yang pada sebelah kiri kanan nampak sedikit mungkin perlu semakan kedua penjajaran senget ini "
              "itu tidak ada dengan untuk atas bawah lompang pateri calar permukaan berhampiran dekat sama lagi "
              "tiada kecil besar teruk rosak kawasan tepi garisan tanda papan sambungan agak terdapat".split()),
    "es": set("el la los las de del en y con una un parece posible pero muy superficie cerca soldadura pines "
              "izquierda derecha ligeramente desalineado rayon rayones vacios borde placa segunda revision".split()),
    "fr": set("le la les de des du et avec une un semble possible mais tres surface pres soudure broches gauche "
              "droite legerement rayure vides bord carte".split()),
    "de": set("der die das und mit ein eine scheint moglich aber sehr oberflache nahe lotstelle stifte links "
              "rechts leicht kratzer lunker kante platine".split()),
}

_PHRASES = {
    "ms": [
        ("penjajaran pin", "pin alignment"),
        ("di sebelah kiri", "on the left side"),
        ("di sebelah kanan", "on the right side"),
        ("nampak senget sedikit", "looks slightly skewed"),
        ("senget sedikit", "slightly skewed"),
        ("mungkin perlu", "may need"),
        ("semakan kedua", "a second check"),
        ("lompang pateri", "solder voids"),
        ("calar permukaan", "surface scratch"),
        ("pemeriksaan visual", "visual inspection"),
    ],
    "es": [
        ("vacios de soldadura", "solder voids"),
        ("lado izquierdo", "left side"),
        ("lado derecho", "right side"),
        ("segunda revision", "second check"),
    ],
}
_WORDS = {
    "ms": {
        "penjajaran": "alignment", "pin": "pin", "di": "at", "sebelah": "side", "kiri": "left", "kanan": "right",
        "nampak": "looks", "senget": "skewed", "sedikit": "slightly", "mungkin": "maybe", "perlu": "needs",
        "semakan": "check", "kedua": "second", "lompang": "voids", "pateri": "solder", "calar": "scratch",
        "permukaan": "surface", "panjang": "long", "retak": "crack", "unit": "unit", "papan": "board",
        "sambungan": "joint", "berhampiran": "near", "dekat": "near", "sama": "same", "lagi": "another",
        "ada": "present", "tiada": "none", "kecil": "small", "besar": "large", "teruk": "severe",
        "rosak": "damaged", "dan": "and", "yang": "that", "pada": "on", "ini": "this", "itu": "that",
        "tidak": "not", "dengan": "with", "untuk": "for", "kawasan": "area", "tepi": "edge", "garisan": "line",
        "tanda": "mark", "agak": "rather", "terdapat": "there is",
    },
    "es": {
        "el": "the", "la": "the", "los": "the", "las": "the", "de": "of", "del": "of the", "en": "on", "y": "and",
        "con": "with", "una": "a", "un": "a", "parece": "looks", "posible": "possible", "pero": "but",
        "muy": "very", "superficie": "surface", "cerca": "near", "soldadura": "solder", "pines": "pins",
        "izquierda": "left", "derecha": "right", "ligeramente": "slightly", "desalineado": "misaligned",
        "desalineados": "misaligned", "rayon": "scratch", "rayones": "scratches", "vacios": "voids",
        "borde": "edge", "placa": "board", "segunda": "second", "revision": "check",
    },
}
_LANGUAGE_NAMES = {"en": "English", "ms": "Malay", "es": "Spanish", "fr": "French", "de": "German"}
_SCRIPT_LANGUAGES = [
    (r"[぀-ヿ]", "ja"), (r"[一-鿿]", "zh"), (r"[가-힯]", "ko"),
    (r"[Ѐ-ӿ]", "ru"), (r"[฀-๿]", "th"), (r"[؀-ۿ]", "ar"),
]


def _tokens(text):
    return re.findall(r"[a-zà-ÿ']+", text.lower())


def detect_language(text):
    for pattern, lang in _SCRIPT_LANGUAGES:
        if re.search(pattern, text):
            return lang
    tokens = _tokens(text)
    counts = {lang: sum(t in words for t in tokens) for lang, words in _STOPWORDS.items()}
    best = max((l for l in counts if l != "en"), key=lambda l: counts[l])
    # Judge by substantive content: a non-English language wins when it dominates the tokens,
    # even if the note carries a short English prefix.
    if counts[best] >= 3 and counts[best] >= counts["en"]:
        return best
    return "en"


def translate(text, lang):
    """Glossary translation (phrases first, then words). Returns None for unsupported languages."""
    if lang not in _WORDS:
        return None
    out = text
    for src, dst in _PHRASES.get(lang, []):
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    words = _WORDS[lang]
    return re.sub(r"[A-Za-zÀ-ÿ']+", lambda m: words.get(m.group(0).lower(), m.group(0)), out)


# --------------------------------------------------------------------------------------------------
# Interpretation helpers
# --------------------------------------------------------------------------------------------------

_FINDING_KEYWORDS = [
    ("solder_voiding", ("void", "voids", "voiding", "dark region", "dark regions")),
    ("pin_misalignment", ("misalign", "shift", "skew", "alignment", "offset", "bent pin")),
    ("surface_scratch", ("scratch", "line mark", "surface mark", "abrasion", "scuff")),
    ("no_material_issue", ("no defect", "no issue", "acceptable", "clean")),
]
_REPEAT_WORDS = ("another", "again", "repeated", "third", "second", "similar", "same region", "still")
_LOCATION_RE = re.compile(r"\b(BGA|[A-Z]{1,2}\d{1,3})\b|(left edge|right edge|board edge|bracket arm|left side|right side|connector pins)", re.IGNORECASE)


def keyword_category(text):
    text = (text or "").lower()
    for category, words in _FINDING_KEYWORDS:
        if any(w in text for w in words):
            return category
    return None


def measurement_ratio(category, measurements):
    name = vocab.CATEGORY_MEASUREMENT.get(category)
    if not name:
        return None
    return float(measurements.get(name, 0) or 0) / vocab.MEASUREMENT_THRESHOLDS[name]


def _dominant_measurement(measurements):
    best, ratio = None, 0.0
    for category in vocab.CATEGORY_MEASUREMENT:
        r = measurement_ratio(category, measurements)
        if r > ratio:
            best, ratio = category, r
    return best, ratio


def _interpret(report):
    measurements = report.get("measurements", {})
    note = report.get("note_for_interpretation", "")
    category = keyword_category(report.get("model_finding")) or keyword_category(note)
    if category is None:
        dominant, ratio = _dominant_measurement(measurements)
        category = dominant if ratio >= 0.5 else ("no_material_issue" if ratio < 0.25 else "uncertain")
    ratio = measurement_ratio(category, measurements)
    if category in vocab.CATEGORY_MEASUREMENT and ratio < 0.25:
        dominant, dom_ratio = _dominant_measurement(measurements)
        if dom_ratio >= 1.0:
            category, ratio = "uncertain", None  # finding and measurements point to different defects
    confidence = float(report.get("confidence", 0) or 0)

    if category == "no_material_issue":
        severity, disposition = "low", "pass"
    elif category == "uncertain":
        severity, disposition = "medium", "reinspect"
    else:
        if ratio >= 1.15 and confidence >= 0.85 and category in vocab.FUNCTIONAL_CATEGORIES:
            severity = "critical"
        elif ratio >= 1.0:
            severity = "high"
        elif ratio >= 0.75:
            severity = "medium"
        else:
            severity = "low"
        disposition = {
            "critical": "hold",
            "high": "hold" if category in vocab.FUNCTIONAL_CATEGORIES else "reinspect",
            "medium": "reinspect",
            "low": "monitor",
        }[severity]

    name = vocab.CATEGORY_MEASUREMENT.get(category)
    if name and ratio is not None:
        value = measurements.get(name, 0)
        threshold = vocab.MEASUREMENT_THRESHOLDS[name]
        relation = "at/above" if ratio >= 1.0 else "below"
        reason = f"{report.get('model_finding')}; {name} {value} {relation} reference {threshold}; confidence {confidence:.2f}."
    else:
        reason = f"Finding '{report.get('model_finding')}' not supported consistently by measurements; confidence {confidence:.2f}."
    return {
        "report_id": report["report_id"],
        "defect_category": category,
        "severity": severity,
        "disposition": disposition,
        "reason_short": reason,
    }


def _location(notes):
    found = []
    for note in notes:
        for m in _LOCATION_RE.finditer(note or ""):
            loc = m.group(0)
            loc = loc.upper() if m.group(1) else loc.lower()
            if loc not in found:
                found.append(loc)
    return found


_DRIVERS = {
    "solder_voiding": "Reflow profile, solder paste condition or stencil aperture issue producing voids at {loc}",
    "pin_misalignment": "Placement or fixture offset during connector assembly shifting pins at {loc}",
    "surface_scratch": "Handling or fixture contact damage during transport or rework at {loc}",
}
_ACTIONS = {
    "solder_voiding": "Verify the reflow profile and paste/stencil condition on the affected line and X-ray the next units at {loc}.",
    "pin_misalignment": "Check placement/fixture calibration and measure pin position on the next units with a calibrated method.",
    "surface_scratch": "Audit handling and fixture contact points along the route and add protective handling at {loc}.",
}
_CATEGORY_LABEL = {
    "solder_voiding": "solder voiding",
    "pin_misalignment": "pin misalignment",
    "surface_scratch": "surface scratching",
}


def _patterns(reports):
    groups = {}
    for r in reports:
        if r["defect_category"] in _CATEGORY_LABEL and r["severity"] != "low":
            groups.setdefault((r["part_id"], r["defect_category"]), []).append(r)
    patterns = []
    for (part_id, category), members in groups.items():
        if len(members) < 2:
            continue
        stations = sorted({m["station"] for m in members})
        ratios = [measurement_ratio(category, m["measurements"]) for m in members]
        mean_ratio = sum(ratios) / len(ratios)
        single_station = len(stations) == 1
        if len(members) >= 3 or (single_station and mean_ratio >= 0.95):
            strength = "strong"
        elif single_station or mean_ratio >= 0.95:
            strength = "moderate"
        else:
            strength = "weak"
        loc = ", ".join(_location([m["note_for_interpretation"] for m in members])) or "the same area"
        part_name = members[0]["part_name"]
        patterns.append({
            "title": f"Recurring {_CATEGORY_LABEL[category]} on {part_name} ({loc})",
            "supporting_report_ids": [m["report_id"] for m in members],
            "pattern_strength": strength,
            "likely_driver": _DRIVERS[category].format(loc=loc) + f" (stations: {', '.join(stations)}).",
            "recommended_action": _ACTIONS[category].format(loc=loc),
        })
    return patterns


_ROUTING = {
    "solder_voiding": ["Process Engineering", "Quality"],
    "pin_misalignment": ["Manufacturing", "Process Engineering"],
    "surface_scratch": ["Manufacturing"],
    "no_material_issue": ["Quality"],
    "uncertain": ["Quality"],
}
_SUPPLIER_WORDS = ("lot", "supplier", "vendor", "incoming")


def _route(case, patterns_by_id):
    teams = list(_ROUTING[case["defect_category"]])
    if case["disposition"] in ("hold", "reinspect") and "Quality" not in teams:
        teams.append("Quality")
    note = (case.get("note_for_interpretation") or "").lower()
    if any(re.search(rf"\b{w}\b", note) for w in _SUPPLIER_WORDS):
        teams.append("Supplier Quality")
    pattern_text = ""
    if case["pattern_ids"]:
        p = patterns_by_id[case["pattern_ids"][0]]
        pattern_text = f" Part of {p['pattern_id']} ({p['pattern_strength']}). Next: {p['recommended_action']}"
    else:
        pattern_text = " Isolated case; confirm the measurement before disposition."
    brief = (
        f"{case['report_id']} {case['part_name']} at {case['station']}: {case['defect_category']}, "
        f"{case['severity']}/{case['disposition']}, score {case['priority_score']}.{pattern_text}"
    )
    return {"report_id": case["report_id"], "teams": teams, "briefing_note": brief}


_VERIFY = {
    "solder_voiding": "Re-image the joint on X-ray and confirm the void ratio of {value} against the 0.20 acceptance limit.",
    "pin_misalignment": "Measure pin position with a calibrated gauge and confirm the {value} mm offset against drawing tolerance.",
    "surface_scratch": "Inspect the {value} mm mark under controlled lighting to confirm whether it breaks the coating or is cosmetic.",
}


def _explain(case):
    c = case["score_components"]
    parts = [f"severity {case['severity']} (+{c['base_priority']})"]
    for hit in case["measurement_hits"]:
        parts.append(f"{hit['measurement']} {hit['value']} at/above {hit['threshold']} (+{hit['bonus']})")
    if c["pattern_membership_bonus"]:
        parts.append(f"membership in {', '.join(case['pattern_ids'])} (+{c['pattern_membership_bonus']})")
    parts.append(f"{case['disposition']} disposition (+{c['disposition_bonus']})")
    why = (
        f"Ranked #{case['rank']} with a deterministic score of {case['priority_score']}. "
        f"The main contributors are {', '.join(parts)}. "
        f"Model confidence {case['confidence']} adds +{c['confidence_bonus']} and {case['image_count']} image(s) add +{c['evidence_bonus']}."
    )
    name = vocab.CATEGORY_MEASUREMENT.get(case["defect_category"])
    if name:
        focus = _VERIFY[case["defect_category"]].format(value=case["measurements"].get(name, 0))
    else:
        focus = "Have a second inspector re-examine the unit and original images, as the interpretation is not conclusive."
    if case.get("translated"):
        focus += f" Also confirm the translation of the original {case.get('original_language')} operator note."
    return {"report_id": case["report_id"], "why_prioritised": why, "verification_focus": focus}


_FOLLOW_UP = {
    "finding_category_mismatch": "Have a second inspector classify the defect from the original images.",
    "measurement_contradicts_severity": "Re-measure the unit before acting on the interpreted severity.",
    "borderline_measurement": "Re-measure with a calibrated method to settle which side of the threshold it falls.",
    "low_confidence": "Re-run inspection or add images to raise confidence before disposition.",
    "finding_not_supported": "Confirm on re-inspection whether a material defect is present at all.",
    "uncertain_category": "Escalate to Quality for manual classification.",
}


def _disagreement(candidate):
    reasons = [s["detail"] for s in candidate["signals"]]
    codes = [s["code"] for s in candidate["signals"] if s["code"] in _FOLLOW_UP]
    follow = " ".join(dict.fromkeys(_FOLLOW_UP[c] for c in codes)) or "Review the unit manually."
    return {
        "report_id": candidate["report_id"],
        "disagreement_reason": " ".join(reasons),
        "recommended_follow_up": follow,
    }


def _trend_summary(facts):
    most = facts["most_affected"]
    worsening = [p for p in facts["pattern_trends"] if p["trend"] == "worsening"]
    summary = (
        f"{facts['report_count']} reports between {facts['shift_window']['start']} and {facts['shift_window']['end']}; "
        f"category totals {facts['category_distribution']}. "
        f"Most affected station is {most['station']['name']} and most affected part family is "
        f"{most['part']['name']} ({most['part']['product_line']})."
    )
    if worsening:
        detail = "; ".join(f"{p['pattern_id']} {p['measurement']} {p['values']}" for p in worsening)
        assessment = f"Worsening: {detail}."
    else:
        assessment = "No recurring pattern shows a worsening measurement trend this shift."
    return {"summary": summary, "worsening_assessment": assessment}


def _alert(payload):
    ev = payload["evidence"]
    top = ", ".join(f"{s['report_id']} ({s['priority_score']})" for s in ev["top_priority_scores"])
    pats = "; ".join(f"{p['pattern_id']} {p['pattern_strength']}/{p['trend']} ({', '.join(p['supporting_report_ids'])})" for p in ev["patterns"]) or "none"
    justification = (
        f"Rating {payload['rating']} triggered by: {'; '.join(payload['triggered_rules'])}. "
        f"Top scores: {top}. Patterns: {pats}. "
        f"Parts affected: {', '.join(ev['parts_affected'])}; stations: {', '.join(ev['stations_affected'])}."
    )
    posture = {
        "red": "Contain: hold affected lots, stop shipping suspect parts and escalate to Process Engineering and Quality before continuing.",
        "orange": "Run with containment: hold flagged units, increase sampling on affected stations and review at next shift handover.",
        "yellow": "Continue running with heightened monitoring and reinspection of flagged units.",
        "green": "Normal running; continue routine sampling.",
    }[payload["rating"]]
    return {"justification": justification, "operational_posture": posture}


def _reinspection(payload):
    notes = []
    for p in payload["patterns"]:
        loc = re.search(r"\(([^)]*)\)\s*$", p.get("title", ""))
        where = f" at {loc.group(1)}" if loc else ""
        label = _CATEGORY_LABEL.get(p["defect_category"], p["defect_category"])
        flag = f"; flag {p['measurement']} >= {p['threshold']}" if p["measurement"] else ""
        notes.append({
            "pattern_id": p["pattern_id"],
            "signature_to_watch": f"{label.capitalize()}{where} on {p['part_name']} ({', '.join(p['stations'])}), last values {p['observed_values']}{flag}.",
        })
    first = payload["patterns"][0] if payload["patterns"] else None
    summary = (
        f"Prioritise {first['part_name']} at {', '.join(first['stations'])} ({first['pattern_id']}), then the remaining patterns in order."
        if first else "No recurring patterns detected; continue routine sampling."
    )
    return {"plan_summary": summary, "watch_notes": notes}


def respond(stage, payload):
    if stage == "normalisation_translation":
        notes = []
        for n in payload["notes"]:
            lang = detect_language(n["operator_note"])
            if lang == "en":
                notes.append({"report_id": n["report_id"], "original_language": "en", "is_english": True, "english_text": n["operator_note"]})
                continue
            # Unsupported language: an empty translation, which normalisation records as untranslated.
            notes.append({
                "report_id": n["report_id"], "original_language": lang, "is_english": False,
                "english_text": translate(n["operator_note"], lang) or "",
            })
        return {"notes": notes}
    if stage == "report_interpretation":
        return {"interpretations": [_interpret(r) for r in payload["reports"]]}
    if stage == "pattern_grouping":
        return {"patterns": _patterns(payload["reports"])}
    if stage == "case_routing":
        patterns_by_id = {p["pattern_id"]: p for p in payload["patterns"]}
        return {"routes": [_route(c, patterns_by_id) for c in payload["flagged_cases"]]}
    if stage == "explanation_generation":
        return {"explanations": [_explain(c) for c in payload["flagged_cases"]]}
    if stage == "shift_trend_summary":
        return _trend_summary(payload)
    if stage == "disagreement_review":
        return {"items": [_disagreement(c) for c in payload["candidates"]]}
    if stage == "alert_justification":
        return _alert(payload)
    if stage == "reinspection_plan":
        return _reinspection(payload)
    raise ValueError(f"offline provider has no rules for stage {stage!r}")


def language_name(code):
    return _LANGUAGE_NAMES.get(code, code)
