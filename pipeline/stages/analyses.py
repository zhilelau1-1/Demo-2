"""Optional shift-level analyses, run as sub-steps of EXPLANATIONS_GENERATED.

For each artifact, code computes the facts, ratings and lists; one LLM call per artifact writes
only the prose. Each call is logged as its own stage in llm_calls.jsonl.
"""

from collections import Counter

from .. import prompts, vocab
from ..artifacts import utc_now, write_json, write_text
from ..context import (
    ALERT_FILE, DISAGREEMENTS_FILE, INTERPRETED_FILE, NORMALISED_FILE, PATTERNS_FILE, REINSPECTION_FILE,
    ROUTING_FILE, SCORES_FILE, TRENDS_FILE, build_cases,
)
from ..offline import keyword_category
from ..scoring import timestamp_key

TREND_EPSILON = 0.05  # change of at least 5% of the reference threshold counts as a trend


def _primary_category(categories):
    counts = Counter(categories)
    return max(vocab.DEFECT_CATEGORIES, key=lambda c: (counts.get(c, 0), -vocab.DEFECT_CATEGORIES.index(c)))


def _ratio(category, measurements):
    name = vocab.CATEGORY_MEASUREMENT.get(category)
    if not name:
        return None
    return float(measurements.get(name, 0) or 0) / vocab.MEASUREMENT_THRESHOLDS[name]


def pattern_trends(patterns, cases):
    trends = []
    for p in patterns:
        members = sorted((cases[r] for r in p["supporting_report_ids"]), key=lambda c: (timestamp_key(c["timestamp"]), c["report_id"]))
        category = _primary_category([m["defect_category"] for m in members])
        name = vocab.CATEGORY_MEASUREMENT.get(category)
        values = [{"report_id": m["report_id"], "timestamp": m["timestamp"], "value": m["measurements"].get(name, 0) if name else None, "severity": m["severity"]} for m in members]
        if not name or len(values) < 2:
            trend = "insufficient_data"
        else:
            delta = (values[-1]["value"] - values[0]["value"]) / vocab.MEASUREMENT_THRESHOLDS[name]
            trend = "worsening" if delta >= TREND_EPSILON else "improving" if delta <= -TREND_EPSILON else "stable"
        trends.append({
            "pattern_id": p["pattern_id"],
            "title": p["title"],
            "pattern_strength": p["pattern_strength"],
            "defect_category": category,
            "measurement": name,
            "threshold": vocab.MEASUREMENT_THRESHOLDS.get(name),
            "values": [v["value"] for v in values],
            "points": values,
            "trend": trend,
            "max_priority_score": max(m["priority_score"] for m in members),
        })
    return trends


def _activity(cases, key, label):
    groups = {}
    for c in cases.values():
        g = groups.setdefault(c[key], {label: c[key], "reports": 0, "defect_reports": 0, "holds": 0, "total_priority": 0.0, "max_priority": 0.0})
        if key == "part_id":
            g.update(part_name=c["part_name"], product_line=c["product_line"])
        g["reports"] += 1
        g["defect_reports"] += c["defect_category"] not in ("no_material_issue",)
        g["holds"] += c["disposition"] == "hold"
        g["total_priority"] = round(g["total_priority"] + c["priority_score"], 2)
        g["max_priority"] = max(g["max_priority"], c["priority_score"])
    return sorted(groups.values(), key=lambda g: (-g["defect_reports"], -g["total_priority"], g[label]))


def shift_facts(cases, patterns):
    ordered = sorted(cases.values(), key=lambda c: (timestamp_key(c["timestamp"]), c["report_id"]))
    hourly = {}
    for c in ordered:
        bucket = (c["timestamp"] or "unknown")[:13] + ":00"
        h = hourly.setdefault(bucket, {"hour": bucket, "reports": 0, "by_category": {}, "by_severity": {}, "report_ids": []})
        h["reports"] += 1
        h["by_category"][c["defect_category"]] = h["by_category"].get(c["defect_category"], 0) + 1
        h["by_severity"][c["severity"]] = h["by_severity"].get(c["severity"], 0) + 1
        h["report_ids"].append(c["report_id"])
    stations = _activity(cases, "station", "station")
    parts = _activity(cases, "part_id", "part_id")
    trends = pattern_trends(patterns, cases)
    return {
        "report_count": len(cases),
        "shift_window": {"start": ordered[0]["timestamp"], "end": ordered[-1]["timestamp"]},
        "category_distribution": dict(Counter(c["defect_category"] for c in ordered)),
        "severity_distribution": dict(Counter(c["severity"] for c in ordered)),
        "hourly_distribution": list(hourly.values()),
        "pattern_trends": trends,
        "worsening_patterns": [t["pattern_id"] for t in trends if t["trend"] == "worsening"],
        "station_activity": stations,
        "part_activity": parts,
        "most_affected": {
            "station": {"name": stations[0]["station"], **{k: stations[0][k] for k in ("defect_reports", "holds", "total_priority")}},
            "part": {"name": parts[0]["part_name"], "part_id": parts[0]["part_id"], "product_line": parts[0]["product_line"],
                     **{k: parts[0][k] for k in ("defect_reports", "holds", "total_priority")}},
            "method": "most non-'no_material_issue' reports, then highest summed priority score",
        },
    }


def disagreement_signals(case):
    signals = []
    m = case["measurements"]
    category = case["defect_category"]
    finding_category = keyword_category(case["model_finding"])
    if finding_category and finding_category != category:
        signals.append({"code": "finding_category_mismatch", "detail": f"Model finding '{case['model_finding']}' suggests {finding_category} but the interpretation is {category}."})
    ratio = _ratio(category, m)
    if ratio is not None:
        name = vocab.CATEGORY_MEASUREMENT[category]
        threshold = vocab.MEASUREMENT_THRESHOLDS[name]
        if case["severity"] in ("critical", "high") and ratio < 0.75:
            signals.append({"code": "measurement_contradicts_severity", "detail": f"Severity {case['severity']} but {name} {m.get(name)} is well below the {threshold} reference."})
        if (case["severity"] == "low" or case["disposition"] == "pass") and ratio >= 1.0:
            signals.append({"code": "measurement_contradicts_severity", "detail": f"Severity {case['severity']}/{case['disposition']} but {name} {m.get(name)} is at or above the {threshold} reference."})
        if 0.85 <= ratio < 1.0:
            signals.append({"code": "borderline_measurement", "detail": f"{name} {m.get(name)} is just below the {threshold} reference ({ratio:.0%})."})
    finding_ratio = _ratio(finding_category, m) if finding_category else None
    if finding_ratio is not None and finding_ratio < 0.5:
        name = vocab.CATEGORY_MEASUREMENT[finding_category]
        signals.append({"code": "finding_not_supported", "detail": f"Model reports '{case['model_finding']}' but {name} is only {m.get(name)} ({finding_ratio:.0%} of the {vocab.MEASUREMENT_THRESHOLDS[name]} reference)."})
    if case["confidence"] < 0.75:
        signals.append({"code": "low_confidence", "detail": f"Model confidence is low ({case['confidence']})."})
    if category == "uncertain":
        signals.append({"code": "uncertain_category", "detail": "The interpretation could not assign a definite category."})
    if signals and case["translated"]:
        signals.append({"code": "translated_note", "detail": f"The operator note was translated from '{case['original_language']}'."})
    return signals


def alert_rating(cases, patterns, trends, scores):
    trend_by_id = {t["pattern_id"]: t for t in trends}
    hold_parts = sorted({c["part_id"] for c in cases.values() if c["disposition"] == "hold"})
    holds = [c for c in cases.values() if c["disposition"] == "hold"]
    strong = [p for p in patterns if p["pattern_strength"] == "strong"]
    escalating = [
        p for p in strong
        if trend_by_id[p["pattern_id"]]["trend"] == "worsening"
        and trend_by_id[p["pattern_id"]]["defect_category"] in vocab.FUNCTIONAL_CATEGORIES
        and any(cases[r]["severity"] == "critical" for r in p["supporting_report_ids"])
    ]
    rules = {
        "escalating_functional_pattern": (bool(escalating), f"strong, worsening functional-defect pattern with a critical case: {[p['pattern_id'] for p in escalating]}"),
        "holds_span_multiple_parts": (len(hold_parts) >= 2, f"hold dispositions span {len(hold_parts)} parts: {hold_parts}"),
        "strong_pattern_present": (bool(strong), f"strong patterns: {[p['pattern_id'] for p in strong]}"),
        "multiple_holds": (len(holds) >= 2, f"{len(holds)} reports with hold disposition"),
        "any_pattern": (bool(patterns), f"{len(patterns)} recurring pattern(s)"),
        "any_hold_or_reinspect": (any(c["disposition"] in ("hold", "reinspect") for c in cases.values()), "at least one hold/reinspect disposition"),
    }
    hit = {k: v[0] for k, v in rules.items()}
    if hit["escalating_functional_pattern"] and hit["holds_span_multiple_parts"]:
        rating = "red"
    elif hit["strong_pattern_present"] or (hit["multiple_holds"] and hit["holds_span_multiple_parts"]):
        rating = "orange"
    elif hit["any_pattern"] or hit["any_hold_or_reinspect"]:
        rating = "yellow"
    else:
        rating = "green"
    defect_cases = [c for c in cases.values() if c["defect_category"] != "no_material_issue"]
    parts = sorted({c["part_id"] for c in defect_cases})
    stations = sorted({c["station"] for c in defect_cases})
    top = [{"report_id": s["report_id"], "rank": s["rank"], "priority_score": s["priority_score"]} for s in scores[:5]]
    evidence = {
        "report_ids": sorted({r for p in patterns for r in p["supporting_report_ids"]} | {t["report_id"] for t in top}),
        "patterns": [{
            "pattern_id": p["pattern_id"], "title": p["title"], "pattern_strength": p["pattern_strength"],
            "trend": trend_by_id[p["pattern_id"]]["trend"], "values": trend_by_id[p["pattern_id"]]["values"],
            "supporting_report_ids": p["supporting_report_ids"],
        } for p in patterns],
        "top_priority_scores": top,
        "parts_affected": parts,
        "stations_affected": stations,
        "multiple_parts": len(parts) > 1,
        "multiple_stations": len(stations) > 1,
    }
    return rating, [v[1] for k, v in rules.items() if v[0]], evidence


ALERT_RUBRIC = {
    "red": "escalating_functional_pattern AND holds_span_multiple_parts",
    "orange": "strong_pattern_present OR (multiple_holds AND holds_span_multiple_parts)",
    "yellow": "any_pattern OR any_hold_or_reinspect",
    "green": "none of the above",
}
_STRENGTH_ORDER = {s: i for i, s in enumerate(vocab.PATTERN_STRENGTHS)}
_SAMPLING = {"strong": "1 in 2 units", "moderate": "1 in 5 units", "weak": "1 in 10 units"}
_DEFAULT_OWNER = {"solder_voiding": "Process Engineering", "pin_misalignment": "Manufacturing", "surface_scratch": "Manufacturing"}


def reinspection_items(patterns, trends, cases, routing):
    trend_by_id = {t["pattern_id"]: t for t in trends}
    ordered = sorted(patterns, key=lambda p: (
        _STRENGTH_ORDER[p["pattern_strength"]],
        trend_by_id[p["pattern_id"]]["trend"] != "worsening",
        -trend_by_id[p["pattern_id"]]["max_priority_score"],
        p["pattern_id"],
    ))
    items = []
    for p in ordered:
        t = trend_by_id[p["pattern_id"]]
        member = cases[p["supporting_report_ids"][0]]
        routed = Counter(team for r in p["supporting_report_ids"] if r in routing for team in routing[r]["teams"])
        owner = routed.most_common(1)[0][0] if routed else _DEFAULT_OWNER.get(t["defect_category"], "Quality")
        sampling = _SAMPLING[p["pattern_strength"]]
        if p["pattern_strength"] == "strong" and t["trend"] == "worsening":
            sampling = "100% of the first 20 units"
        items.append({
            "pattern_id": p["pattern_id"],
            "title": p["title"],
            "part_ids": p["part_ids"],
            "part_name": member["part_name"],
            "stations": p["stations"],
            "defect_category": t["defect_category"],
            "measurement": t["measurement"],
            "threshold": t["threshold"],
            "trend": t["trend"],
            "observed_values": t["values"],
            "sampling": sampling,
            "owner_team": owner,
        })
    return items


def _stop_continue(item):
    if not item["measurement"]:
        return "Any recurrence of the signature", "Two consecutive clean shifts"
    thr = item["threshold"]
    stop = f"any unit with {item['measurement']} >= {thr}, or 2 units >= {round(thr * 0.9, 3)} in the first sample"
    cont = f"20 consecutive units with {item['measurement']} < {round(thr * 0.75, 3)}"
    return stop, cont


def render_reinspection(items, narrative, run_id):
    watch = {w["pattern_id"]: w["signature_to_watch"] for w in narrative.get("watch_notes", [])}
    parts = list(dict.fromkeys(f"{i['part_name']} ({', '.join(i['part_ids'])})" for i in items))
    stations = list(dict.fromkeys(s for i in items for s in i["stations"]))
    lines = [
        "# Reinspection Plan: Next Shift",
        "",
        f"Run `{run_id}` · generated {utc_now()}. Sampling order and stop/continue criteria are derived in code from the detected patterns.",
        "",
        "## Summary",
        "",
        narrative.get("plan_summary", "No recurring patterns detected; continue routine sampling."),
        "",
        "## Parts to sample first",
        "",
    ]
    lines += [f"{n}. {p}" for n, p in enumerate(parts, start=1)] or ["Routine sampling only."]
    lines += ["", "## Stations to prioritise", ""]
    lines += [f"{n}. {s}" for n, s in enumerate(stations, start=1)] or ["Routine sampling only."]
    lines += ["", "## Defect signatures to watch", ""]
    for i in items:
        lines.append(f"- **{i['pattern_id']} {i['title']}** ({i['trend']}, observed {i['observed_values']}): {watch.get(i['pattern_id'], '')}")
    lines += [
        "",
        "## Stop / continue criteria",
        "",
        "| Pattern | Part | Stations | Sampling | STOP and hold the lot if | CONTINUE routine sampling after | Owner team |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i in items:
        stop, cont = _stop_continue(i)
        lines.append(f"| {i['pattern_id']} | {i['part_name']} | {', '.join(i['stations'])} | {i['sampling']} | {stop} | {cont} | {i['owner_team']} |")
    owners = list(dict.fromkeys(i["owner_team"] for i in items))
    lines += [
        "",
        "## Owner teams",
        "",
        f"Plan owner: {owners[0] if owners else 'Quality'}. Supporting: {', '.join(owners[1:]) or 'none'}. Quality signs off any stop decision.",
        "",
        "## Human review gate",
        "",
        "A STOP trigger quarantines the lot pending human review; no release or scrap decision is taken by the pipeline.",
        "",
    ]
    return "\n".join(lines)


def run_all(ctx):
    normalised = ctx.load(NORMALISED_FILE)
    interpreted = ctx.load(INTERPRETED_FILE)
    patterns = ctx.load(PATTERNS_FILE)
    scores = ctx.load(SCORES_FILE)["scores"]
    routing = {r["report_id"]: r for r in ctx.load(ROUTING_FILE)}
    cases = build_cases(normalised, interpreted, scores)
    base_inputs = [ctx.out(NORMALISED_FILE), ctx.out(INTERPRETED_FILE), ctx.out(PATTERNS_FILE), ctx.out(SCORES_FILE)]
    outputs, stages = [], []

    def call(stage, builder_args, payload, validate, output, extra_inputs=()):
        instructions, schema = builder_args
        stages.append(stage)
        return ctx.llm.call(
            stage=stage, pipeline_state="EXPLANATIONS_GENERATED", instructions=instructions, payload=payload,
            schema=schema, validate=validate, input_artifacts=[*base_inputs, *extra_inputs], output_artifact=output,
        )

    def nonempty(*fields):
        return lambda d: [f"{f} is empty" for f in fields if not (d.get(f) or "").strip()]

    # 7. Shift-level trend summary
    facts = shift_facts(cases, patterns)
    narrative = call("shift_trend_summary", prompts.shift_trend_summary(), facts,
                     nonempty("summary", "worsening_assessment"), ctx.out(TRENDS_FILE))
    write_json(ctx.out(TRENDS_FILE), {**facts, "narrative": narrative})
    outputs.append(ctx.out(TRENDS_FILE))

    # 8. Low-confidence disagreement finder
    candidates = []
    for c in sorted(cases.values(), key=lambda c: c["rank"]):
        signals = disagreement_signals(c)
        if any(s["code"] != "translated_note" for s in signals):
            candidates.append({k: c[k] for k in ("report_id", "model_finding", "defect_category", "severity", "disposition", "confidence", "measurements", "note_for_interpretation")} | {"signals": signals})
    items = []
    if candidates:
        ids = [c["report_id"] for c in candidates]

        def validate(d):
            got = [i.get("report_id") for i in d.get("items", [])]
            errs = [] if sorted(got) == sorted(ids) else [f"expected one item per candidate {ids}, got {got}"]
            return errs + [f"{i.get('report_id')}: empty field" for i in d.get("items", []) if not (i.get("disagreement_reason") or "").strip() or not (i.get("recommended_follow_up") or "").strip()]

        reviewed = call("disagreement_review", prompts.disagreement_review(ids), {"candidates": candidates}, validate, ctx.out(DISAGREEMENTS_FILE))
        by_id = {i["report_id"]: i for i in reviewed["items"]}
        items = [{
            "report_id": c["report_id"],
            "disagreement_reason": by_id[c["report_id"]]["disagreement_reason"].strip(),
            "recommended_follow_up": by_id[c["report_id"]]["recommended_follow_up"].strip(),
            "signals": [s["code"] for s in c["signals"]],
        } for c in candidates]
    write_json(ctx.out(DISAGREEMENTS_FILE), items)
    outputs.append(ctx.out(DISAGREEMENTS_FILE))

    # 9. Lot-level alert
    rating, triggered, evidence = alert_rating(cases, patterns, facts["pattern_trends"], scores)
    payload = {"rating": rating, "rubric": ALERT_RUBRIC, "triggered_rules": triggered, "evidence": evidence}
    text = call("alert_justification", prompts.alert_justification(), payload,
                nonempty("justification", "operational_posture"), ctx.out(ALERT_FILE), [ctx.out(TRENDS_FILE)])
    write_json(ctx.out(ALERT_FILE), {**payload, "justification": text["justification"].strip(), "operational_posture": text["operational_posture"].strip()})
    outputs.append(ctx.out(ALERT_FILE))

    # 10. Reinspection plan
    plan_items = reinspection_items(patterns, facts["pattern_trends"], cases, routing)
    plan_narrative = {}
    if plan_items:
        pids = [i["pattern_id"] for i in plan_items]

        def validate_plan(d):
            got = [w.get("pattern_id") for w in d.get("watch_notes", [])]
            errs = nonempty("plan_summary")(d)
            return errs + ([] if sorted(got) == sorted(pids) else [f"expected one watch note per pattern {pids}, got {got}"])

        plan_narrative = call("reinspection_plan", prompts.reinspection_plan(pids), {"patterns": plan_items},
                              validate_plan, ctx.out(REINSPECTION_FILE), [ctx.out(TRENDS_FILE), ctx.out(ROUTING_FILE)])
    write_text(ctx.out(REINSPECTION_FILE), render_reinspection(plan_items, plan_narrative, ctx.state.run_id))
    outputs.append(ctx.out(REINSPECTION_FILE))
    return outputs, stages
