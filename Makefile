PYTHON := .venv/bin/python3

.PHONY: eval eval-both eval-A eval-test install clean submit run run-fresh

# Strategy B (LangGraph) on labeled sample — default eval target
eval:
	$(PYTHON) code/evaluate.py --strategy B --dataset sample

# Strategy A (single-shot) on labeled sample
eval-A:
	$(PYTHON) code/evaluate.py --strategy A --dataset sample

# Both strategies on labeled sample → results/comparison.md
eval-both:
	$(PYTHON) code/evaluate.py --strategy both --dataset sample

# Strategy B on full test set → output.csv (no scoring, no ground truth)
eval-test:
	$(PYTHON) code/evaluate.py --strategy B --dataset test

# Install dependencies into .venv
install:
	python3 -m venv .venv
	.venv/bin/pip install -r code/requirements.txt -q

# Run main inference on claims.csv (production submission)
run:
	$(PYTHON) code/main.py --skip-onboarding

run-fresh:
	$(PYTHON) code/main.py --fresh --skip-onboarding

# Remove generated/temp files (checkpoints, caches, results) — keeps output.csv
clean:
	rm -f checkpoints.db
	rm -rf results
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '.DS_Store' -delete
	@echo "Cleaned: checkpoints.db, results/, __pycache__, *.pyc, .DS_Store"

# Full clean + fresh production run → prints the file to submit
submit: clean run-fresh
