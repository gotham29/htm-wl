# -------- HTM-WL Makefile --------
# Usage:
#   make help
#   make env          # create .venv and install requirements
#   make htmcore      # build/install htm.core into the active Python
#   make repro-fast   # quick paper repro (limited warmup/files)
#   make repro        # full paper repro
#   make score        # BYOD: warm + score runs -> results/scores/*.csv
#   make eval         # BYOD: compute metrics from scores + GT
#   make clean        # remove generated artifacts
# You can override variables, e.g.:
#   make score CONFIG=my_config.yaml OUTDIR=run1
# ----------------------------------

# ---- Tunables / defaults ----
PY            ?= python
CONFIG        ?= config.yaml
OUTDIR        ?= results
SCORES_DIR    ?= $(OUTDIR)/scores
GT            ?= data/subjects_testfiles_wltogglepoints.yaml
GT_FORMAT     ?= subjects_yaml     # csv | yaml | subjects_yaml
RATE          ?= 6.67              # Hz, for lag seconds
WARM_LIMIT    ?= 500               # repro-fast only
MAX_FILES     ?= 5                 # repro-fast only

# ----------------------------------

.PHONY: help env htmcore repro-fast repro score eval clean

help:
	@echo ""
	@echo "HTM-WL Make targets"
	@echo "  make env          - create .venv and install requirements"
	@echo "  make htmcore      - build/install htm.core into the active Python"
	@echo "  make repro-fast   - quick paper repro (limit warmup/files)"
	@echo "  make repro        - full paper repro"
	@echo "  make score        - BYOD: warm + score runs -> $(SCORES_DIR)"
	@echo "  make eval         - BYOD: metrics from scores + $(GT)"
	@echo "  make clean        - remove results and caches"
	@echo ""
	@echo "Variables (override like VAR=value):"
	@echo "  CONFIG=$(CONFIG)  OUTDIR=$(OUTDIR)  GT=$(GT)  GT_FORMAT=$(GT_FORMAT)  RATE=$(RATE)"
	@echo ""

env:
	@echo ">> creating .venv and installing requirements"
	@$(PY) -m venv .venv
	@. .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt
	@echo ">> done. activate with: source .venv/bin/activate"

htmcore:
	@echo ">> installing htm.core into active Python"
	@if [ ! -d htm.core ]; then git clone https://github.com/htm-community/htm.core.git; fi
	@cd htm.core && $(PY) htm_install.py
	@$(PY) - <<'PY'
import htm, sys
print("htm.core OK at:", htm.__file__)
print("python:", sys.executable)
PY

repro-fast:
	@echo ">> paper reproduction (FAST) with limits: WARM_LIMIT=$(WARM_LIMIT), MAX_FILES=$(MAX_FILES)"
	@$(PY) -m scripts.evaluate --config run_pipeline.yaml --data-root data --outdir $(OUTDIR) \
		--verbose --limit-train $(WARM_LIMIT) --max-files $(MAX_FILES)

repro:
	@echo ">> paper reproduction (FULL)"
	@$(PY) -m scripts.evaluate --config run_pipeline.yaml --data-root data --outdir $(OUTDIR) --verbose

score:
	@echo ">> scoring with config=$(CONFIG) -> $(SCORES_DIR)"
	@$(PY) -m scripts.score --config $(CONFIG) --outdir $(OUTDIR) --verbose
	@ls -1 $(SCORES_DIR) | wc -l | xargs -I{} echo ">> wrote {} score file(s) to $(SCORES_DIR)"

eval:
	@echo ">> evaluating scores in $(SCORES_DIR) with GT=$(GT) (format=$(GT_FORMAT))"
	@$(PY) -m scripts.eval_from_scores --scores-dir $(SCORES_DIR) \
		--gt $(GT) --gt-format $(GT_FORMAT) --rate-hz $(RATE) --out $(OUTDIR)/metrics.csv
	@echo ">> results -> $(OUTDIR)/metrics.csv"

clean:
	@echo ">> removing generated artifacts"
	@rm -rf $(OUTDIR) **/__pycache__ .pytest_cache .mypy_cache htm_wl.egg-info
	@echo ">> clean complete"
