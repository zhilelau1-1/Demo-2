# Inspection Triage Pipeline

This is a replayable pipeline for machine-inspection findings on manufactured parts. It does the following:

1. Joins each report with the part catalog and translates non-English operator notes.
2. Interprets every finding against controlled vocabularies.
3. Groups related defects into recurring patterns.
4. Scores priority deterministically in code.
5. Routes the top 5 cases to owner teams.
6. Writes explainable reviewer summaries, each with a human review gate.

## Quick start

```bash
make run        # python -m pipeline.run: regenerates every artifact from the two input files
make validate   # python validate.py: 15 checks, exits non-zero on any failure
make test       # unit and end-to-end tests (offline provider, no network)
make clean      # delete generated artifacts
```

The pipeline needs only Python 3.10+ and the standard library. Live LLM calls also need `pip install -r requirements.txt` and an `ANTHROPIC_API_KEY` in the environment, for example `ANTHROPIC_API_KEY=sk-ant-... make run`. Live mode uses only that variable. Saved logins (`ant auth login` profiles, OAuth tokens) are ignored, so a run never bills whichever account happens to be logged in on the machine.

## LLM modes

| `--llm-mode` | Behaviour |
| --- | --- |
| `auto` (default) | Live when `ANTHROPIC_API_KEY` is set, otherwise offline |
| `live` | Anthropic Messages API with JSON-schema structured output and the server-side refusal fallback. Default model `claude-sonnet-5-5` (`--model`, `--effort`) |
| `offline` | Deterministic rule-based provider (`pipeline/offline.py`). It takes the same payloads, follows the same schemas, and is logged as `provider: "offline-rules"` |
| `replay` | Serves recorded live responses from `.llm_cache/` by prompt hash. A cache miss is an error |

Every live response is recorded to `.llm_cache/`, so `--llm-mode replay` reproduces a live run exactly without calling the API.

## Stages

The stage order is enforced by `pipeline/state.py`. A stage runs only when the previous state is current. Every transition is written to `run_manifest.json` with the sha256 of each artifact the stage read and wrote.

| State reached | What happens | LLM stage (logged) | Output |
| --- | --- | --- | --- |
| INPUTS_LOADED | Read and check both input files | none | none |
| REPORTS_NORMALISED | Join the catalog; detect language and translate notes. Code copies the original note verbatim | `normalisation_translation` | `normalised_reports.json` |
| REPORTS_INTERPRETED | **Stage 1:** one call over all reports, using notes and measurements | `report_interpretation` | `interpreted_reports.json` |
| PATTERNS_GROUPED | **Stage 2:** a separate call over the Stage 1 output joined with part, station, time, confidence, measurements and note. Pattern IDs are assigned in code | `pattern_grouping` | `patterns.json` |
| PRIORITIES_COMPUTED | Formula in code, with no LLM. Records the hashes of the patterns and interpretations it consumed | none | `priority_scores.json` |
| CASES_SELECTED | Top 5 by score; ties break on timestamp, then report_id | none | `flagged_cases.json` |
| ROUTING_COMPLETE | **Stage 3:** route the top 5, using all patterns and the team vocabulary | `case_routing` | `case_routing.json` |
| EXPLANATIONS_GENERATED | Code renders the facts, score breakdown and gate notes; the LLM writes the "why" paragraph. Also runs the optional analyses | `explanation_generation`, `shift_trend_summary`, `disagreement_review`, `alert_justification`, `reinspection_plan` | `review_summaries.md`, `shift_trends.json`, `disagreement_cases.json`, `alert_rating.json`, `reinspection_plan.md` |
| VALIDATION_COMPLETE | All checks in `pipeline/checks.py` | none | `validation_report.json` |
| RESULTS_FINALISED | Manifest marked complete | none | `run_manifest.json` |

The exact payload and the validated response of every LLM call are kept in `llm_io/`.

To resume from a stage, run `python -m pipeline.run --from-stage PRIORITIES_COMPUTED`. Earlier artifacts are reused only if their hashes still match the manifest.

## Division of work between code and the LLM

- **Vocabularies** live in `pipeline/vocab.py`. They appear in each prompt, are enforced as schema enums, and are re-validated in code after every call. Invalid output is retried once with the errors fed back. If the retry also fails, the run stops; values are never silently coerced.
- **Scoring** lives in `pipeline/scoring.py`. It implements the specified formula with Decimal arithmetic. The three measurement bonuses apply independently of the defect category, and the pattern bonus is 12 for each distinct pattern containing the report.
- **Optional analyses:** code computes the trend labels, disagreement signals, alert rating (a fixed rubric) and reinspection criteria. The LLM writes only the narrative text.
- **Human review gate notes** are built in code from a template per defect category, plus the LLM's case-specific verification focus. Every note therefore states what must be verified before any production decision.

## Validation

`validate.py` re-derives its expectations from the artifacts on disk rather than trusting the pipeline. It checks that:

- Every report was processed and joined with the catalog.
- Original notes are preserved verbatim.
- Non-English notes are translated.
- All vocabularies are respected.
- The Stage 2 input payload contains the Stage 1 labels.
- Recomputed scores match the stored ones, and were computed from the current `patterns.json` after grouping.
- The top 5 match the scores.
- Routing teams are valid.
- Every flagged case has a summary and a human review gate.
- Each LLM stage has exactly one successful log record.
- The manifest shows the 11 states in order.

## Limitations

- Offline translation is glossary-based and covers only Malay and Spanish. Indonesian notes are mostly handled, but are labelled as Malay. French, German and several non-Latin scripts are detected but not translated. Untranslated notes are recorded as such, and validation fails on them rather than passing silently. Use live mode for arbitrary languages.
- The live path has been checked up to the API call (request construction and error handling). A full live run needs an `ANTHROPIC_API_KEY`.
