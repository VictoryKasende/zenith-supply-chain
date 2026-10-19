"""Constantes partagées par toutes les expériences (seed, chemins, calendrier)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper_results"
CACHE_DIR = PAPER_DIR / "_cache"

SEED = 42

# --------------------------------------------------------------------- #
# Calendrier (données : 2022-08-01 → 2025-07-31, 36 mois)
# --------------------------------------------------------------------- #
DATA_START_MONTH = pd.Timestamp("2022-08-01")
DATA_END_MONTH = pd.Timestamp("2025-07-01")
MIN_HISTORY_MONTHS = 18
HORIZON = 3


def month_end(month_start: pd.Timestamp) -> pd.Timestamp:
    return month_start + pd.offsets.MonthEnd(0)


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS"))


ALL_MONTHS = _months(DATA_START_MONTH, DATA_END_MONTH)

#: Origines de prévision t0 (fins de mois) : ≥ 18 mois d'historique, jusqu'à
#: l'avant-dernier mois (la décision prise fin juin 2025 porte sur juillet).
FORECAST_ORIGINS = [month_end(m) for m in ALL_MONTHS[MIN_HISTORY_MONTHS - 1:-1]]

#: Origines dont les horizons 1..3 sont tous observés : panel équilibré
#: utilisé pour toutes les métriques de prévision.
EVAL_ORIGINS = [t for t in FORECAST_ORIGINS
                if t + pd.offsets.MonthBegin(HORIZON) <= DATA_END_MONTH + pd.offsets.MonthEnd(0)]

#: Mois simulés par le backtest de politique (février 2024 → juillet 2025).
SIM_MONTHS = [t + pd.offsets.MonthBegin(1) for t in FORECAST_ORIGINS]
#: Développement (réglage des paramètres des politiques classiques) / test.
DEV_END = pd.Timestamp("2024-12-01")
DEV_MONTHS = [m for m in SIM_MONTHS if m <= DEV_END]
TEST_MONTHS = [m for m in SIM_MONTHS if m > DEV_END]


def quiet_pipeline_logs() -> None:
    """Coupe les logs INFO des modules ``src`` (très bavards par origine)."""
    for name in ("preprocessing", "classification", "obsolescence", "forecasting",
                 "optimization", "evaluation"):
        logging.getLogger(name).setLevel(logging.WARNING)


def ensure_dirs() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
