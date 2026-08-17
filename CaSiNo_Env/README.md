# CaSiNo environment adapter

This package evaluates a focal LLM negotiator on the CaSiNo camping domain. It contains
the dialogue protocol, allocation utilities, direct/full-framework baselines, and the
continuous belief-usable planner. The directory was called `ASTRA_Env` in early private
experiments; the public name reflects the actual benchmark.

The reviewed CaSiNo split is included under `data/casino/` when permitted by Git ignore
rules, and can always be restored with `bash scripts/bootstrap_upstreams.sh`.

Run `python -m CaSiNo_Env.eval --help` from the repository root. Generated episodes go
under `runs/` and are ignored.
