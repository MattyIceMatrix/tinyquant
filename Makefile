# tinyquant -- everything in the README is produced by these targets.
#
#   make all     reproduce every number in the README from scratch
#   make verify  the one that matters: bit-exactness + claim checking

CC      ?= gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror
FAST    ?= -std=c99 -O3 -march=native -Wall -Wextra
PYTHON  ?= python3
REPS    ?= 150
FUZZ_TRIALS ?= 500
TRIALS  ?= 7

ART  := artifacts
BIN  := build
ENG  := engine/tinyq.c

.PHONY: all verify train export crossval bench sweep scaling charts test clean help portability audit fuzz opt-levels

help:
	@grep -E '^[a-z-]+:' Makefile | grep -v grep | cut -d: -f1 | sed 's/^/  make /'

all: train export crossval bench sweep scaling opt-levels portability audit fuzz charts test

$(BIN):
	mkdir -p $(BIN)

train:
	$(PYTHON) scripts/train.py --epochs 300

export:
	$(PYTHON) scripts/export.py

# Bit-exactness between the Python reference and the C kernel. Non-zero exit
# if a single output byte differs, so this fails the build rather than warning.
crossval: $(BIN)
	$(CC) $(CFLAGS) -DTQ_SKIP_ZEROS=0 $(ENG) engine/crossval.c -o $(BIN)/crossval
	$(BIN)/crossval $(ART)/model.tq8 $(ART)/test_vectors.tqv $(ART)/reference_out.bin \
		| tee $(ART)/crossval.json

# Timing is reported as a median of several trials. A single run on a shared
# vCPU swung the INT8/FP32 ratio between 0.48x and 0.71x during development --
# see FINDINGS.md section 8.
bench: $(BIN)
	$(CC) $(FAST) -DTQ_SKIP_ZEROS=0 $(ENG) engine/bench.c -o $(BIN)/bench
	$(CC) $(FAST) -DTQ_SKIP_ZEROS=1 $(ENG) engine/bench.c -o $(BIN)/bench_skipzeros
	$(PYTHON) scripts/latency_bench.py $(REPS) $(TRIALS)

sweep:
	$(PYTHON) scripts/sweep.py

scaling:
	$(PYTHON) scripts/scaling_study.py 50

opt-levels:
	$(PYTHON) scripts/opt_level_study.py

# 13 builds across x86-64, aarch64, riscv64 and arm32, every one checked for
# bit-exactness against the same Python-generated reference bytes.
portability:
	$(PYTHON) scripts/portability_matrix.py

audit:
	$(PYTHON) scripts/resource_audit.py

fuzz:
	$(PYTHON) scripts/fuzz_differential.py $(FUZZ_TRIALS)

charts:
	$(PYTHON) scripts/make_charts.py

test:
	$(PYTHON) -m pytest tests/ -q

verify: crossval test

clean:
	rm -rf $(BIN) __pycache__ */__pycache__ .pytest_cache
