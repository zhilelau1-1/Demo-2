PYTHON ?= python3
LLM_MODE ?= auto

GENERATED = normalised_reports.json interpreted_reports.json patterns.json priority_scores.json \
	flagged_cases.json case_routing.json review_summaries.md shift_trends.json disagreement_cases.json \
	alert_rating.json reinspection_plan.md llm_calls.jsonl run_manifest.json validation_report.json

.PHONY: all run validate test clean install

all: run validate

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m pipeline.run --llm-mode $(LLM_MODE)

validate:
	$(PYTHON) validate.py

test:
	$(PYTHON) -m unittest discover -s tests

clean:
	rm -rf $(GENERATED) llm_io
