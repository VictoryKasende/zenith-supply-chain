"""Point d'entrée unique : régénère tout ``paper_results/`` depuis zéro.

Usage : ``python -m experiments.run_all [--jobs 7] [--keep-cache] [--skip-benchmark]``

Ordre : P0-1 rolling-origin → P0-2 métriques → P0-4 obsolescence →
P0-5 MILP vs LP → P1-6 politiques → P0-3 tests (H4/H5 utilisent P1-6) →
P1-7 service → P1-8 sensibilité → P2-9 environnement et temps →
``RESULTS_FOR_PAPER.md``.
"""
from __future__ import annotations

import argparse
import shutil
import time

from experiments.config import PAPER_DIR, ensure_dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--keep-cache", action="store_true", help="réutilise paper_results/_cache")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--runs", type=int, default=5, help="exécutions du benchmark")
    args = ap.parse_args()

    if PAPER_DIR.exists():
        if args.keep_cache:
            for p in PAPER_DIR.iterdir():
                if p.name != "_cache":
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
        else:
            shutil.rmtree(PAPER_DIR)
    ensure_dirs()

    from experiments import (benchmark_env, figures, metrics, milp_vs_lp, obsolescence_eval, paper_reproduction,
                             policy_baselines, report, rolling_origin, sensitivity, service_tradeoff, stats_tests)
    steps = [
        ("Reproduction des chiffres publiés", paper_reproduction.run),
        ("P0-1 rolling-origin", lambda: rolling_origin.run(jobs=args.jobs)),
        ("P0-2 métriques", metrics.run),
        ("P0-4 obsolescence", obsolescence_eval.run),
        ("P0-5 MILP vs LP", milp_vs_lp.run),
        ("P1-6 politiques", policy_baselines.run),
        ("P0-3 tests statistiques", stats_tests.run),
        ("P1-7 compromis coût/service", service_tradeoff.run),
        ("P1-8 sensibilité", sensitivity.run),
        ("figures", figures.run),
    ]
    if not args.skip_benchmark:
        steps.append(("P2-9 environnement et temps", lambda: benchmark_env.run(args.runs)))
    steps.append(("rapport", report.run))
    for name, fn in steps:
        tic = time.perf_counter()
        print(f"=== {name} ===", flush=True)
        fn()
        print(f"=== {name} : {time.perf_counter() - tic:.0f} s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
