PY ?= python
export PYTHONPATH := $(CURDIR)/src

.PHONY: all bench vqe bonus plots test

all: bench vqe bonus plots test

bench:
	$(PY) benchmarks/run_scaling.py

vqe:
	$(PY) experiments/h2_pes.py

bonus:
	$(PY) benchmarks/run_mps.py
	$(PY) experiments/lih_point.py

plots:
	$(PY) scripts/make_plots.py

test:
	$(PY) -m pytest -q
