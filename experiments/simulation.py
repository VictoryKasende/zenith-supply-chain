"""Simulation d'inventaire mensuelle sur demande RÉALISÉE (backtest de politique).

Chronologie (identique pour toutes les politiques)
--------------------------------------------------
Pour chaque mois simulé m (février 2024 → juillet 2025), avec t0 = fin du
mois précédent :
1. la politique décide les quantités q[i] à partir de ce qu'elle sait à t0
   (stock physique, commandes en transit, prévisions et classes calculées
   sur ≤ t0) ;
2. contrôle budgétaire commun : si Σ c_i q_i > B, toutes les quantités sont
   réduites proportionnellement (arrondi inférieur) et l'événement est compté ;
3. la commande arrive au début du mois m + L_i (+ retard éventuel) ;
4. réception des arrivées du mois m, puis service des lignes de vente
   réelles du mois dans l'ordre chronologique : une ligne est servie
   entièrement si le stock disponible couvre le cumul des lignes jusqu'à
   elle ; la demande non servie est perdue (pas de backorder, comme le LP) ;
5. coûts : 50 $ par commande passée, stockage = stock fin de mois × coût ×
   0,1 %/jour × 30, rupture = unités perdues × (prix − coût) — exactement les
   définitions de :func:`src.optimization.financial_kpis`.

Hypothèses
----------
- Stock initial = stock central reconstruit au 31/01/2024 (Σ magasins du
  dernier stock observé) ; aucune commande en transit au départ.
- Univers : les 246 produits ayant au moins une vente au 31/01/2024.
- La demande réalisée est la vente observée : elle est elle-même censurée
  par les ruptures historiques (cf. V3), pour toutes les politiques.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from experiments.config import FORECAST_ORIGINS, PAPER_DIR, SEED, SIM_MONTHS
from experiments.data import full_clean, load_cat, monthly_matrix, realized_lines
from src.optimization import COUT_COMMANDE_FIXE, TAUX_STOCKAGE_JOURNALIER, lead_time_months


@dataclass
class SimConfig:
    holding_rate: float = TAUX_STOCKAGE_JOURNALIER
    order_cost: float = COUT_COMMANDE_FIXE
    budget: float = 500_000.0
    capacity: float = 5_000.0
    volume: float = 0.01
    lead_extra: int = 0
    #: retards aléatoires (mois) par (produit, mois de commande) ; None = aucun
    delays: np.ndarray | None = None
    #: multiplicateur des prix de vente en USD (sensibilité change USD/CDF)
    price_factor: float = 1.0


@dataclass
class SimData:
    products: list[str]
    cost: np.ndarray
    price: np.ndarray
    lead: np.ndarray
    demand: np.ndarray            # [n, K] demande réalisée
    line_cumsum: dict             # (i, k) → cumul chronologique des lignes
    stock0: np.ndarray
    months: list[pd.Timestamp]
    origins: list[pd.Timestamp]
    states: pd.DataFrame          # classes / drapeaux par origine (rolling_origin_states.csv)
    monthly_full: pd.DataFrame    # historique mensuel (lu uniquement ≤ t0 par les politiques)
    forecasts: dict = field(default_factory=dict)   # (method, H) → array [K, n, H]
    _states_by_k: dict = field(default_factory=dict)
    _hist_cache: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.products)

    def abc(self, k: int) -> np.ndarray:
        return self._state(k)["classe_abc"].to_numpy()

    def obsolete(self, k: int) -> np.ndarray:
        return self._state(k)["obsolete_flag"].to_numpy().astype(bool)

    def _state(self, k: int) -> pd.DataFrame:
        if k not in self._states_by_k:
            st = self.states[self.states["origin"] == str(self.origins[k].date())].set_index("produit_id")
            self._states_by_k[k] = st.reindex(self.products)
        return self._states_by_k[k]

    def history(self, k: int) -> pd.DataFrame:
        """Ventes mensuelles jusqu'au mois de t0 inclus (rien après)."""
        last = self.origins[k].to_period("M").to_timestamp()
        return self.monthly_full.loc[:last, self.products]


def load_sim_data(states_file: str = "rolling_origin_states.csv") -> SimData:
    states = pd.read_csv(PAPER_DIR / states_file, dtype={"origin": str})
    first = states[states["origin"] == str(FORECAST_ORIGINS[0].date())]
    products = sorted(first["produit_id"].tolist())
    cat = load_cat().set_index("produit_id").loc[products]
    clean = full_clean()
    monthly = monthly_matrix(clean, pd.Timestamp("2025-07-01"))
    months = list(SIM_MONTHS)
    demand = monthly.loc[months, products].to_numpy(dtype=float).T

    lines = realized_lines(clean)
    lines = lines[lines["produit_id"].isin(products) & lines["mois"].isin(months)]
    pidx = {p: i for i, p in enumerate(products)}
    midx = {m: k for k, m in enumerate(months)}
    cums = {}
    for (p, m), g in lines.groupby(["produit_id", "mois"], sort=False):
        cums[(pidx[p], midx[pd.Timestamp(m)])] = np.cumsum(g["quantite_vendue"].to_numpy(dtype=float))
    stock0 = first.set_index("produit_id")["central_stock"].reindex(products).fillna(0).to_numpy(dtype=float)
    return SimData(
        products=products,
        cost=cat["cout_achat_unitaire"].to_numpy(dtype=float),
        price=cat["prix_vente_unitaire"].to_numpy(dtype=float),
        lead=np.array([lead_time_months(o) for o in cat["origine_fournisseur"]], dtype=int),
        demand=demand, line_cumsum=cums, stock0=stock0, months=months,
        origins=list(FORECAST_ORIGINS), states=states, monthly_full=monthly,
    )


def attach_forecasts(data: SimData, errors_file: str, methods: list[str], horizon: int) -> None:
    """Charge les prévisions rolling-origin (méthode, origine, produit, h) en tableaux."""
    fc = pd.read_csv(PAPER_DIR / errors_file, usecols=["origin", "product_id", "method", "horizon", "y_pred"],
                     dtype={"origin": str})
    fc = fc[fc["method"].isin(methods) & (fc["horizon"] <= horizon)]
    origins = [str(o.date()) for o in data.origins]
    for method, g in fc.groupby("method"):
        arr = np.zeros((len(origins), data.n, horizon))
        piv = g.pivot_table(index=["origin", "product_id"], columns="horizon", values="y_pred")
        for k, o in enumerate(origins):
            sub = piv.loc[o].reindex(data.products)
            if sub.isna().any().any():
                raise ValueError(f"prévisions manquantes : {method} @ {o}")
            arr[k] = sub.to_numpy(dtype=float)
        data.forecasts[(method, horizon)] = arr


# --------------------------------------------------------------------- #
# Boucle de simulation
# --------------------------------------------------------------------- #
@dataclass
class SimResult:
    q: np.ndarray
    demand: np.ndarray
    served: np.ndarray
    lost: np.ndarray
    on_hand_end: np.ndarray
    available: np.ndarray
    lines_total: np.ndarray
    lines_full: np.ndarray
    budget_clips: int
    capacity_violations: int
    infeasible_decisions: int
    cfg: SimConfig
    data: SimData

    def frame(self, policy: str) -> pd.DataFrame:
        d, cfg = self.data, self.cfg
        n, K = self.q.shape
        price = d.price * cfg.price_factor
        hold = d.cost * cfg.holding_rate * 30
        rows = pd.DataFrame({
            "policy": policy,
            "month": np.repeat([m.date() for m in d.months[:K]], n),
            "product_id": np.tile(d.products, K),
            "q": self.q.T.ravel(), "demand": self.demand.T.ravel(), "served": self.served.T.ravel(),
            "lost": self.lost.T.ravel(), "on_hand_end": self.on_hand_end.T.ravel(),
            "lines_total": self.lines_total.T.ravel(), "lines_full": self.lines_full.T.ravel(),
        })
        cost = np.tile(d.cost, K); pr = np.tile(price, K); h = np.tile(hold, K)
        rows["ordering_cost"] = (rows["q"] > 0) * cfg.order_cost
        rows["holding_cost"] = rows["on_hand_end"] * h
        rows["stockout_cost"] = rows["lost"] * np.maximum(pr - cost, 0)
        rows["total_cost"] = rows["ordering_cost"] + rows["holding_cost"] + rows["stockout_cost"]
        rows["order_value"] = rows["q"] * cost
        rows["inventory_value"] = rows["on_hand_end"] * cost
        rows["revenue"] = rows["served"] * pr
        rows["gross_margin"] = rows["served"] * (pr - cost)
        return rows


def simulate(policy, data: SimData, cfg: SimConfig, n_months: int | None = None,
             record_lines: bool = True) -> SimResult:
    K = len(data.months) if n_months is None else n_months
    n = data.n
    lead = data.lead + cfg.lead_extra
    arrivals = np.zeros((K + 12, n))
    on_hand = data.stock0.copy()
    out = {k: np.zeros((n, K)) for k in ("q", "served", "lost", "on_hand_end", "available",
                                          "lines_total", "lines_full")}
    clips = cap_viol = infeasible = 0
    policy.reset(data, cfg)
    for k in range(K):
        q, ok = policy.decide(k, on_hand.copy(), arrivals[k:].copy())
        q = np.maximum(np.asarray(q, dtype=float), 0)
        infeasible += int(not ok)
        spend = float((q * data.cost).sum())
        if spend > cfg.budget + 1e-6:
            q = np.floor(q * cfg.budget / spend)
            clips += 1
        delay = cfg.delays[:, k] if cfg.delays is not None else 0
        arr_k = k + lead + delay
        placed = np.flatnonzero(q > 0)
        np.add.at(arrivals, (arr_k[placed], placed), q[placed])
        avail = on_hand + arrivals[k]
        dem = data.demand[:, k]
        served = np.minimum(avail, dem)
        lost = dem - served
        on_hand = avail - served
        if (on_hand * cfg.volume).sum() > cfg.capacity + 1e-9:
            cap_viol += 1
        out["q"][:, k] = q; out["served"][:, k] = served; out["lost"][:, k] = lost
        out["on_hand_end"][:, k] = on_hand; out["available"][:, k] = avail
        if record_lines:
            for i in np.flatnonzero(dem > 0):
                cs = data.line_cumsum.get((i, k))
                if cs is None:
                    continue
                out["lines_total"][i, k] = len(cs)
                out["lines_full"][i, k] = np.searchsorted(cs, avail[i] + 1e-9, side="right")
    return SimResult(q=out["q"], demand=data.demand[:, :K], served=out["served"], lost=out["lost"],
                     on_hand_end=out["on_hand_end"], available=out["available"],
                     lines_total=out["lines_total"], lines_full=out["lines_full"],
                     budget_clips=clips, capacity_violations=cap_viol, infeasible_decisions=infeasible,
                     cfg=cfg, data=data)


def total_cost(res: SimResult, months_mask: np.ndarray) -> float:
    d, cfg = res.data, res.cfg
    price = d.price * cfg.price_factor
    hold = (d.cost * cfg.holding_rate * 30)[:, None]
    oc = ((res.q > 0) * cfg.order_cost)[:, months_mask].sum()
    hc = (res.on_hand_end * hold)[:, months_mask].sum()
    sc = (res.lost * np.maximum(price - d.cost, 0)[:, None])[:, months_mask].sum()
    return float(oc + hc + sc)


def summarize(frame: pd.DataFrame, abc: pd.Series | None = None) -> dict:
    """KPI d'un sous-ensemble de lignes (produit × mois) d'une simulation."""
    dem = frame["demand"].sum()
    with_dem = frame[frame["demand"] > 0]
    out = {
        "n_orders": int((frame["q"] > 0).sum()),
        "total_units": float(frame["q"].sum()),
        "order_value": float(frame["order_value"].sum()),
        "avg_inventory_value": float(frame.groupby("month")["inventory_value"].sum().mean()),
        "ordering_cost": float(frame["ordering_cost"].sum()),
        "holding_cost": float(frame["holding_cost"].sum()),
        "stockout_cost": float(frame["stockout_cost"].sum()),
        "total_cost": float(frame["total_cost"].sum()),
        "revenue": float(frame["revenue"].sum()),
        "gross_margin": float(frame["gross_margin"].sum()),
        "unit_fill_rate": float(frame["served"].sum() / dem) if dem > 0 else np.nan,
        "order_fill_rate": float(frame["lines_full"].sum() / frame["lines_total"].sum())
        if frame["lines_total"].sum() > 0 else np.nan,
        "cycle_service_level": float((with_dem["lost"] <= 1e-9).mean()) if len(with_dem) else np.nan,
        "demand_units": float(dem),
        "lost_units": float(frame["lost"].sum()),
    }
    return out


def delay_matrix(n: int, K: int, p_delay: float, seed: int = SEED) -> np.ndarray:
    """Retards d'un mois tirés une fois pour toutes (nombres aléatoires communs)."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, K)) < p_delay).astype(int)
