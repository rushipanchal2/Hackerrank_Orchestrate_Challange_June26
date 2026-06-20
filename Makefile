PYTHON := .venv/bin/python3

.PHONY: eval eval-A eval-both test test10 run run-full run-sample install clean submit

# ── Default dev run: 5 rows, 2 threads ───────────────────────────────────────
test:
	$(PYTHON) code/main.py --test 5 --threads 2 --skip-onboarding

test10:
	$(PYTHON) code/main.py --test 10 --threads 2 --skip-onboarding

# ── Default run = 5 samples (safe for dev/iteration) ─────────────────────────
run:
	$(PYTHON) code/main.py --test 5 --threads 2 --skip-onboarding

# ── Full inference run on all 44 claims → output/ + output.csv ───────────────
run-full:
	$(PYTHON) code/main.py --threads 2 --skip-onboarding

# ── Resume an interrupted full run: only process rows missing from output.csv ─
resume:
	$(PYTHON) code/main.py --resume --threads 1 --skip-onboarding

# ── Run on sample_claims.csv (for manual inspection, 5 rows) ─────────────────
run-sample:
	$(PYTHON) code/main.py --sample --test 5 --threads 2 --skip-onboarding

# ── Evaluation (Strategy B LangGraph) on sample_claims.csv ───────────────────
eval:
	$(PYTHON) code/evaluation/main.py --strategy langgraph

# ── Evaluation (Strategy A single-shot) on sample_claims.csv ─────────────────
eval-A:
	$(PYTHON) code/evaluation/main.py --strategy single_shot

# ── Both strategies ───────────────────────────────────────────────────────────
eval-both: eval eval-A

# ── Install dependencies into .venv ──────────────────────────────────────────
install:
	python3 -m venv .venv
	.venv/bin/pip install -r code/requirements.txt -q

# ── Remove generated/temp files ───────────────────────────────────────────────
clean:
	rm -f checkpoints.db checkpoints.db-shm checkpoints.db-wal
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '.DS_Store' -delete
	@echo "Cleaned: checkpoints, __pycache__, *.pyc"

# ── Wipe all timestamped output runs (keeps root output.csv) ──────────────────
clean-output:
	rm -rf output/
	@echo "Cleaned: output/ runs"

# ── Full submission pipeline: clean → run → verify ───────────────────────────
submit: clean run-full
	@echo ""
	@echo "Submit these files to HackerRank:"
	@echo "  1. submission/output.csv"
	@echo "  2. code.zip (zip -r code.zip code/ dataset/ *.md Makefile .env.example)"
