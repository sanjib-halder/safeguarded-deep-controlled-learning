# Safeguarded deep controlled learning for multi-supplier sourcing under correlated disruptions

Code and data for the paper *Safeguarded deep controlled learning for multi-supplier sourcing under correlated disruptions* (submitted to the European Journal of Operational Research).

The repository contains the simulator and exact dynamic programme for periodic-review sourcing from N
suppliers with individual Markov failures and regional shocks; the generalised multi-index policy (GMI)
and the status-dependent index policy (SDI); safeguarded deep controlled learning (S-DCL) with its
confirmation test, base-action prior, ensemble and deployment gate; the benchmarks (original DCL,
rolling-horizon SAA, a pathwise-gradient policy); and the saved results behind every table and figure.
All instances are synthetic and are regenerated from fixed seeds.

## Structure

```
supplier_pilot/   the package: instances, simulator, exact DP, index policies, labelling, classifier
experiments/      scripts that produce the results, figures and diagnostics
tests/            test suite (simulator vs exact model, DP vs simulation, GMI = DI/CDI, feasibility, ...)
results/          saved results (*.json with all numbers, *.md summaries), trained networks (*.pt), figures/
```

## Installation

Python 3.10 or later.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_core.py          # or: python -m pytest tests
```

All experiments ran on the CPU of an Apple M4 laptop; no GPU is needed. Training one S-DCL network takes
about 25 minutes; the full set of experiments takes several days of CPU time. The tuned baselines of each
instance are cached in `results/cache/` on first use, so later runs are faster.

## Reproducing the paper

Every script writes a `.json` file (all numbers) and a `.md` summary to `results/`. The saved files are
included, so tables and figures can be checked without rerunning anything. The settings of every run are
stored in the `meta` field of its JSON file.

| Paper | Command | Result files |
|---|---|---|
| Table 2, Section S6 (two suppliers) | `python experiments/validate_dual_sourcing.py --full` | `validation_dual_sourcing.*` |
| Table 2 (40 random instances) | `python experiments/run_random_study.py --n 40 --jobs 4` | `random_study*` |
| Fig. 2 (factorial effects) | `python experiments/run_factorial.py --reps 16 --jobs 8` | `factorial_study*` |
| Table 3, Fig. 3 (H1) | `python experiments/train_dcl.py --fresh-train 48 --ensemble-seeds 101 202 303 --jobs 8` | `dcl_ensemble48_s101-202-303.*`, `dcl_member48_s*.pt` |
| Table 3, Fig. 3 (H2) | same, with `--reps 8 9 10 11 12 13 14 15` | `dcl_ensemble48_s101-202-303_r8-15.*` |
| Table 3 (original DCL) | `python experiments/train_dcl.py --only all --search sh --action-space abs --iterations 3 --confirm-M 0 --prior none --compare-ensemble 101 202 303 --variant orig --label "Original DCL (per instance)" --jobs 8` | `dcl_per-instance_orig.*` |
| Table 4, Fig. 5 (O1) | H1 command with `--test-target factorial-all` | `dcl_ensemble48_s101-202-303_test-factorial-all.*` |
| Table 4, Fig. 5 (O2) | H2 command with `--test-target factorial-all` | `dcl_ensemble48_s101-202-303_r8-15_test-factorial-all.*` |
| Table 5, Fig. 4 (N = 4) | `python experiments/train_dcl.py --target scale4 --fresh-train 48 --ensemble-seeds 101 202 303 --transfer-from factorial-hard --jobs 8` (and with `--reps 8 ... 15`) | `dcl_ensemble48_s101-202-303*_scale4.*` |
| Table 5, Fig. 4 (N = 5) | as for N = 4 with `--target scale5` | `dcl_ensemble48_s101-202-303*_scale5.*` |
| Table 6, Fig. 5 (ablation) | `python experiments/run_ablations.py --train-seed 101 --jobs 8` (repeat with 202 and 303) | `dcl_ablation48_s*.*`, `dcl_member48_s*_<variant>.pt` |
| Section 9, S6 (threshold z) | `python experiments/run_ablations.py --variants z05 z196 --train-seed 101 --jobs 8` | `dcl_ablation48_s101_z05-z196.*` |
| Figs. 1-5 | `python experiments/make_figures.py` (add `--grey` for greyscale) | `figures/` |
| Section S1, Table S2 (truncation) | `python experiments/check_truncation.py` | `truncation_check.*` |
| Section S3, Table S5 (prior) | `python experiments/print_prior.py` | printed to the screen |
| Section S4, Tables S6-S8 | `python experiments/correction_states.py --jobs 4` | `correction_states.*` |
| Section S5 (pathwise gradients) | `python experiments/train_neural.py --mode per-instance --only F00_lch F00_lcl F05_lcl F02_lcl`; `python experiments/train_neural.py`; `python experiments/train_neural.py --fresh-train 48 --iters 1500 --jobs 8`; `python experiments/check_gradients.py` | `neural_factorial-hard_*`, `gradient_check.*` |

On macOS, prefix long runs with `caffeinate -i` to keep the computer awake.

Seeds: test instances use factorial seed 7 (replicates 0-7 and 8-15); the three ensemble members use
training seeds 101, 202 and 303, each with its own training and validation instances; the random study
uses seed 2026. Simulated futures for tuning, labelling, validation and testing use separate seed ranges.

## Notation used in file names

`F05_lch` is replicate 5 of the factorial cell with low fixed costs (l), correlated regional shocks (c) and a
high price premium (h); `n` stands for no shocks. `dcl_member48_s202` is the network trained with seed 202
on 48 replicates of the two hard cells. `_scale4`/`_scale5` denote four and five suppliers.

## Licence

MIT; see `LICENSE`.

## Citation

If you use this code, please cite the paper (full reference to be added on acceptance).

Repository assembled on 2026-11-01.
