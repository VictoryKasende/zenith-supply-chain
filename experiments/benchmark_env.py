"""P2-9 — Protocole de reproductibilité : environnement et temps d'exécution mesurés.

- ``paper_results/environment.json`` : CPU, cœurs, RAM, OS, Python, versions des
  dépendances directes (``requirements.txt`` + paquets importés mais non
  déclarés), version du solveur CBC, commit git.
- ``paper_results/runtime.csv`` : ``module, run_id, seconds, peak_memory_mb``.
  Chaque module du pipeline (``scripts/run_*.py``) est exécuté dans un
  processus Python neuf, 5 exécutions consécutives, sur une **copie
  temporaire** du dépôt (les sorties versionnées ne sont pas touchées).
  Temps = horloge murale du script (imports compris) ; mémoire = pic de
  l'ensemble de travail du processus (Windows ``peak_wset``, Unix
  ``ru_maxrss``), bibliothèques natives comprises.
- ``paper_results/runtime_summary.csv`` : moyenne, écart-type, min, max.

Périmètre : les six modules du pipeline (prétraitement, classification,
détection, prévision, optimisation, export Power BI). L'EDA (figures) et
l'évaluation H1–H6 sont mesurées à part ; **les tableaux de bord (Streamlit,
Power BI Desktop) sont exclus** : ce sont des applications interactives.

Usage : ``python -m experiments.benchmark_env [--runs 5]``
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from experiments.config import PAPER_DIR, ROOT, ensure_dirs

MODULES = [
    ("preprocessing", "run_preprocessing.py", True),
    ("classification", "run_classification.py", True),
    ("obsolescence_detection", "run_obsolescence.py", True),
    ("forecasting", "run_forecasting.py", True),
    ("optimization", "run_optimization.py", True),
    ("export_powerbi", "run_powerbi_export.py", True),
    ("eda_figures", "run_eda.py", False),
    ("evaluation_h1_h6", "run_evaluation.py", False),
]

RUNNER = r"""
import json, runpy, sys, time
script, out = sys.argv[1], sys.argv[2]
t = time.perf_counter()
try:
    runpy.run_path(script, run_name="__main__")
    code = 0
except SystemExit as e:
    code = int(e.code or 0)
elapsed = time.perf_counter() - t
try:
    import psutil
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
except Exception:
    peak = None
if peak is None:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
json.dump({"seconds": elapsed, "peak_bytes": peak, "exit_code": code}, open(out, "w"))
"""


def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").exists():
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        if m:
            return m.group(1).strip()
    return platform.processor()


def _cbc_version() -> str | None:
    import pulp
    path = pulp.PULP_CBC_CMD().path
    try:
        out = subprocess.run([path, "-quit"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"Version:\s*([\w.]+)", out)
        return m.group(1) if m else out.strip()[:200]
    except Exception as exc:  # pragma: no cover
        return f"indisponible ({exc})"


def _requirements() -> list[str]:
    names = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(re.split(r"[<>=!~\[ ]", line)[0])
    return names


def environment() -> dict:
    import psutil
    reqs = _requirements()
    versions = {}
    for n in reqs + ["shap", "psutil"]:
        try:
            versions[n] = md.version(n)
        except md.PackageNotFoundError:
            versions[n] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {
        "cpu_model": _cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "requirements_txt_direct_dependencies": len(reqs),
        "packages": versions,
        "undeclared_but_imported": [n for n in ("shap",) if n not in reqs],
        "cbc_version": _cbc_version(),
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def _copy_repo(dst: Path) -> None:
    for sub in ("src", "scripts", "data/raw"):
        shutil.copytree(ROOT / sub, dst / sub, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "requirements.txt", dst / "requirements.txt")


def benchmark(runs: int = 5) -> pd.DataFrame:
    rows = []
    with tempfile.TemporaryDirectory(prefix="zenith_bench_") as tmp:
        tmp = Path(tmp)
        _copy_repo(tmp)
        runner = tmp / "_runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        env["PYTHONIOENCODING"] = "utf-8"
        for run_id in range(1, runs + 1):
            for name, script, in_pipeline in MODULES:
                out = tmp / f"_res_{name}_{run_id}.json"
                proc = subprocess.run([sys.executable, str(runner), str(tmp / "scripts" / script), str(out)],
                                      cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace")
                res = json.loads(out.read_text()) if out.exists() else {"seconds": None, "peak_bytes": None,
                                                                        "exit_code": proc.returncode}
                rows.append({"module": name, "run_id": run_id, "seconds": res["seconds"],
                             "peak_memory_mb": (res["peak_bytes"] or 0) / 1024 ** 2 if res["peak_bytes"] else None,
                             "exit_code": res["exit_code"], "in_pipeline_total": in_pipeline,
                             "stderr_tail": proc.stderr[-300:] if res["exit_code"] else ""})
                print(f"run {run_id} {name}: {res['seconds']:.1f}s" if res["seconds"] else f"run {run_id} {name}: FAILED",
                      flush=True)
    return pd.DataFrame(rows)


def run(runs: int = 5) -> dict:
    ensure_dirs()
    env = environment()
    (PAPER_DIR / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    df = benchmark(runs)
    df[["module", "run_id", "seconds", "peak_memory_mb", "exit_code", "in_pipeline_total", "stderr_tail"]] \
        .to_csv(PAPER_DIR / "runtime.csv", index=False)
    tot = df[df["in_pipeline_total"]].groupby("run_id").agg(seconds=("seconds", "sum"),
                                                              peak_memory_mb=("peak_memory_mb", "max"))
    tot["module"] = "TOTAL_pipeline_6_modules"
    summ = df.groupby("module").agg(seconds_mean=("seconds", "mean"), seconds_sd=("seconds", "std"),
                                    seconds_min=("seconds", "min"), seconds_max=("seconds", "max"),
                                    peak_memory_mb_max=("peak_memory_mb", "max"),
                                    failures=("exit_code", lambda s: int((s != 0).sum()))).reset_index()
    summ = pd.concat([summ, pd.DataFrame([{
        "module": "TOTAL_pipeline_6_modules", "seconds_mean": tot["seconds"].mean(), "seconds_sd": tot["seconds"].std(),
        "seconds_min": tot["seconds"].min(), "seconds_max": tot["seconds"].max(),
        "peak_memory_mb_max": tot["peak_memory_mb"].max(), "failures": 0}])], ignore_index=True)
    summ.to_csv(PAPER_DIR / "runtime_summary.csv", index=False)
    return {"environment": env, "summary": summ}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    out = run(args.runs)
    print(json.dumps(out["environment"], indent=2, ensure_ascii=False))
    print(out["summary"].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
