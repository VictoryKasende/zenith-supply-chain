"""Politiques de réapprovisionnement comparées dans le backtest (P1-6 à P1-8).

Chaque politique ne voit que ce qui est connu à t0 : stock physique, commandes
en transit, prévisions rolling-origin faites à t0, classes et drapeaux
d'obsolescence recalculés à t0, ventes mensuelles ≤ t0.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from experiments.data import load_cat
from experiments.lp_model import Params, make_instance, round_like_code, solve_lp
from experiments.simulation import SimConfig, SimData, simulate, total_cost

Z_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.28, 1.5, 1.645, 2.0, 2.33, 2.5, 3.0)


class Policy:
    name = "policy"

    def reset(self, data: SimData, cfg: SimConfig) -> None:
        self.data, self.cfg = data, cfg

    def decide(self, k: int, on_hand: np.ndarray, future_arrivals: np.ndarray) -> tuple[np.ndarray, bool]:
        raise NotImplementedError


# --------------------------------------------------------------------- #
# 1. Politique empirique (incumbent) — simulate_baseline_policy, mois par mois
# --------------------------------------------------------------------- #
class IncumbentPolicy(Policy):
    """Commande ⌈2 × d̄⌉ quand le stock physique < d̄, d̄ = moyenne de la prévision sur
    l'horizon ; ignore les commandes en transit (« délai subi sans anticipation »)
    — règle de :func:`src.optimization.simulate_baseline_policy`. Comme dans
    l'article, d̄ vient des prévisions du pipeline."""

    def __init__(self, method: str = "pipeline", horizon: int = 3):
        self.method, self.horizon = method, horizon
        self.name = "incumbent" if method == "pipeline" else f"incumbent_{method}"

    def decide(self, k, on_hand, future_arrivals):
        avg = self.data.forecasts[(self.method, self.horizon)][k].mean(axis=1)
        q = np.where(on_hand < avg, np.ceil(2 * avg), 0.0)
        return q, True


# --------------------------------------------------------------------- #
# 2–4. Politiques classiques à paramètre z réglé sur la période de développement
# --------------------------------------------------------------------- #
def history_stats(data: SimData, k: int) -> dict[str, np.ndarray]:
    """μ, σ (12 derniers mois depuis le lancement), MA3 et RMSE de MA3 à 1 pas, ≤ t0."""
    cache = data._hist_cache
    if k in cache:
        return cache[k]
    hist = data.history(k)
    y = hist.to_numpy(dtype=float)                      # [mois, n]
    launched = np.maximum.accumulate(y > 0, axis=0)     # mois ≥ premier mois vendu
    last12 = np.where(launched[-12:], y[-12:], np.nan)
    with np.errstate(all="ignore"):
        mu = np.nanmean(last12, axis=0)
        sd = np.nanstd(last12, axis=0, ddof=1)
        last3 = np.where(launched[-3:], y[-3:], np.nan)
        ma3 = np.nanmean(last3, axis=0)
        errs = []
        for j in range(len(y) - 12, len(y)):
            if j < 3:
                continue
            ok = launched[j - 3]
            pred = y[j - 3:j].mean(axis=0)
            errs.append(np.where(ok, y[j] - pred, np.nan))
        rmse = np.sqrt(np.nanmean(np.asarray(errs) ** 2, axis=0))
    stats = {k_: np.nan_to_num(v, nan=0.0) for k_, v in
             {"mu": mu, "sd": sd, "ma3": ma3, "rmse_ma3": rmse}.items()}
    cache[k] = stats
    return stats


class StockLevelPolicy(Policy):
    """(s, S), order-up-to périodique, ou order-up-to sur prévision MA3.

    Révision R = 1 mois, délai L (connu du planificateur), position de stock
    IP = stock physique + en transit.
    - ``sS``  : s = μL + zσ√L, S = μ(L+R) + zσ√(L+R) ; commande S − IP si IP ≤ s.
    - ``OUT`` : commande max(0, S − IP) chaque mois.
    - ``MA3`` : OUT avec μ = MA3 et σ = RMSE à 1 pas de MA3 (12 derniers mois).
    z est fixé par classe ABC (classe à t0)."""

    def __init__(self, kind: str, z: dict[str, float]):
        assert kind in ("sS", "OUT", "MA3")
        self.kind, self.z = kind, dict(z)
        base = {"sS": "sS", "OUT": "order_up_to", "MA3": "ma3_order_up_to"}[kind]
        self.name = f"{base}_tuned"

    def decide(self, k, on_hand, future_arrivals):
        st = history_stats(self.data, k)
        mu, sd = (st["ma3"], st["rmse_ma3"]) if self.kind == "MA3" else (st["mu"], st["sd"])
        z = np.vectorize(lambda c: self.z.get(c, self.z.get("C", 1.0)))(self.data.abc(k))
        L = (self.data.lead + self.cfg.lead_extra).astype(float)
        ip = on_hand + future_arrivals.sum(axis=0)
        S = mu * (L + 1) + z * sd * np.sqrt(L + 1)
        if self.kind == "sS":
            s = mu * L + z * sd * np.sqrt(L)
            q = np.where(ip <= s, np.ceil(S - ip - 1e-9), 0.0)
        else:
            q = np.ceil(S - ip - 1e-9)
        return np.maximum(q, 0.0), True


def tune_z(kind: str, data: SimData, cfg: SimConfig, n_dev_months: int,
           grid: tuple[float, ...] = Z_GRID) -> tuple[dict[str, float], pd.DataFrame]:
    """Grille complète (z_A, z_B, z_C) minimisant le coût total sur les mois de développement."""
    mask = np.ones(n_dev_months, dtype=bool)
    rows = []
    for za, zb, zc in itertools.product(grid, repeat=3):
        pol = StockLevelPolicy(kind, {"A": za, "B": zb, "C": zc})
        res = simulate(pol, data, cfg, n_months=n_dev_months, record_lines=False)
        rows.append({"kind": kind, "z_A": za, "z_B": zb, "z_C": zc, "dev_total_cost": total_cost(res, mask)})
    grid_df = pd.DataFrame(rows)
    best = grid_df.loc[grid_df["dev_total_cost"].idxmin()]
    return {"A": float(best["z_A"]), "B": float(best["z_B"]), "C": float(best["z_C"])}, grid_df


# --------------------------------------------------------------------- #
# 5–8. Prévision + optimiseur LP du pipeline (horizon glissant)
# --------------------------------------------------------------------- #
class LPPolicy(Policy):
    """À chaque mois : LP de l'Étape 6 (coefficients identiques) sur l'horizon H avec
    les prévisions faites à t0, le stock physique et les commandes en transit ρ ;
    seule la décision du premier mois est appliquée, arrondie comme le code."""

    def __init__(self, method: str, name: str, params: Params | None = None,
                 use_obsolescence: bool = True):
        self.method, self.name = method, name
        self.params = params or Params()
        self.use_obsolescence = use_obsolescence

    def reset(self, data, cfg):
        super().reset(data, cfg)
        cat = pd.DataFrame({"produit_id": data.products, "cout_achat_unitaire": data.cost,
                            "prix_vente_unitaire": data.price * cfg.price_factor})
        cat = cat.merge(load_cat()[["produit_id", "origine_fournisseur"]], on="produit_id", how="left")
        self._base = cat
        self.lp_status: list[str] = []

    def decide(self, k, on_hand, future_arrivals):
        H = self.params.horizon
        F = self.data.forecasts[(self.method, H)][k]
        active = ~self.data.obsolete(k) if self.use_obsolescence else np.ones(self.data.n, bool)
        idx = np.flatnonzero(active)
        prod = self._base.iloc[idx].copy()
        prod["classe_abc"] = self.data.abc(k)[idx]
        pids = prod["produit_id"].tolist()
        fc = {p: F[i] for p, i in zip(pids, idx)}
        stock0 = {p: float(on_hand[i]) for p, i in zip(pids, idx)}
        rho = {p: np.array([future_arrivals[t, i] if t < len(future_arrivals) else 0.0 for t in range(H)])
               for p, i in zip(pids, idx)}
        inst = make_instance(prod, fc, stock0, self.params, in_transit=rho)
        lp = solve_lp(inst, self.params)
        self.lp_status.append(lp["status"])
        qr = round_like_code(lp["Q"])
        q = np.zeros(self.data.n)
        for p, i in zip(pids, idx):
            q[i] = qr[p][0]
        return q, lp["status"] == "Optimal"
