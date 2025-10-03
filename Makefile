# -------- HTM-WL Makefile --------

PY            ?= python
CONFIG        ?= config.yaml
OUTDIR        ?= results
SCORES_DIR    ?= $(OUTDIR)/scores
GT            ?= data_demo/subjects_testfiles_wltogglepoints.yaml
GT_FORMAT     ?= subjects_yaml
RATE          ?= 6.67

.PHONY: help env htmcore score eval demo demo-clean clean test

help:
	@echo ""
	@echo "Targets:"
	@echo "  make env          - create .venv and install requirements"
	@echo "  make htmcore      - build/install htm.core into the active Python"
	@echo "  make score        - BYOD: warm + score (CONFIG=$(CONFIG)) -> $(SCORES_DIR)"
	@echo "  make eval         - BYOD: metrics from scores + $(GT)"
	@echo "  make demo         - generate synthetic data -> score -> eval"
	@echo "  make demo-clean   - remove demo artifacts"
	@echo "  make clean        - remove results and caches"
	@echo ""

env:
	@$(PY) -m venv .venv
	@. .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt
	@echo ">> activate with: source .venv/bin/activate"

htmcore:
	@if [ ! -d htm.core ]; then git clone https://github.com/htm-community/htm.core.git; fi
	@cd htm.core && $(PY) htm_install.py
	@$(PY) -c "import htm, sys; print('htm.core OK at:', htm.__file__); print('python:', sys.executable)"

score:
	@echo ">> scoring with $(CONFIG) -> $(SCORES_DIR)"
	@$(PY) -m scripts.score --config $(CONFIG) --outdir $(OUTDIR) --verbose
	@ls -1 $(SCORES_DIR) | wc -l | xargs -I{} echo ">> wrote {} score file(s) to $(SCORES_DIR)"

eval:
	@echo ">> evaluating scores in $(SCORES_DIR) with GT=$(GT) (format=$(GT_FORMAT))"
	@$(PY) -m scripts.eval_from_scores --scores-dir $(SCORES_DIR) \
		--gt $(GT) --gt-format $(GT_FORMAT) --rate-hz $(RATE) --out $(OUTDIR)/metrics.csv
	@echo ">> results -> $(OUTDIR)/metrics.csv"

demo:
	@echo ">> generating synthetic demo data..."
	@$(PY) -m scripts.make_demo --out data_demo --subjects 2 --rate-hz 6.67 \
		--len-A 1200 --len-B 300 --len-C 900 --len-D 300
	@echo ">> scoring demo data..."
	@$(PY) -m scripts.score --config config.demo.yaml --outdir results_demo --verbose
	@echo ">> evaluating demo results..."
	@$(PY) -m scripts.eval_from_scores --scores-dir results_demo/scores \
		--gt data_demo/subjects_testfiles_wltogglepoints.yaml --gt-format subjects_yaml \
		--rate-hz 6.67 --out results_demo/metrics.csv
	@echo ">> demo complete -> results_demo/metrics.csv"

demo-clean:
	@rm -rf data_demo results_demo

clean:
	@rm -rf results results_demo data_demo **/__pycache__ .pytest_cache .mypy_cache htm_wl.egg-info

.PHONY: plot-demo

plot-demo:
	@echo ">> plotting demo scores..."
	@$(PY) -m scripts.plot_scores \
		--scores-dir results_demo/scores \
		--gt data_demo/subjects_testfiles_wltogglepoints.yaml --gt-format subjects_yaml \
		--data-root data_demo --outdir results_demo/plots --show-raw
	@echo ">> plots -> results_demo/plots"

app:
	@streamlit run app/app.py

api:
	@uvicorn api.main:app --reload --port 8000
