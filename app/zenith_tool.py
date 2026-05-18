"""Zenith Supply Chain — application interactive (Streamlit).

Lance :
    streamlit run app/zenith_tool.py

Pages
-----
1. Tableau de bord global
2. Classification produits (ABC × XYZ × K-Means)
3. Alertes obsolescence
4. Prévisions de demande
5. Recommandations de commande
6. Simulation what-if (budget / niveau de service)

L'application lit uniquement les sorties produites par le pipeline
(``outputs/tables/``) — elle n'effectue **aucune ré-exécution coûteuse**
côté serveur, sauf pour la page 6 (what-if) qui rejoue ``optimize_orders``.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization import (  # noqa: E402
    compare_policies,
    optimize_orders,
    simulate_baseline_policy,
)

# --------------------------------------------------------------------- #
# Configuration thème
# --------------------------------------------------------------------- #
ZENITH_PRIMARY = "#1f4e79"
ZENITH_ACCENT = "#ff6b6b"
ZENITH_GREEN = "#2d6a4f"
ZENITH_ORANGE = "#ff9f1c"

st.set_page_config(
    page_title="Zenith Supply Chain",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------- #
# Chargement (avec cache)
# --------------------------------------------------------------------- #
@st.cache_data
def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/processed/zenith_clean.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_classification() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/classification_produits.csv")


@st.cache_data
def load_obsolescence() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/produits_obsoletes.csv")


@st.cache_data
def load_obsolescence_features() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/obsolescence_features.csv")


@st.cache_data
def load_previsions() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/previsions_complet.csv", parse_dates=["date"])


@st.cache_data
def load_commandes() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/commandes_recommandees.csv")


@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/comparaison_avant_apres.csv")


@st.cache_data
def load_product_features() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/features/product_features.csv")


@st.cache_data
def load_catalogue() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw/catalogue_produits_250.csv")


@st.cache_data
def build_products_input() -> pd.DataFrame:
    feats = load_product_features()
    classes = load_classification()[["produit_id", "classe_abc", "classe_xyz"]]
    obs = set(load_obsolescence()["produit_id"].tolist())
    cat = load_catalogue()[["produit_id", "origine_fournisseur", "cout_achat_unitaire", "prix_vente_unitaire"]]
    products = feats.merge(classes, on="produit_id", how="left").merge(cat, on="produit_id", how="left")
    products["a_risque_obsolescence"] = products["produit_id"].isin(obs).astype(int)
    products["prix_vente_unitaire"] = products["prix_vente_unitaire"].fillna(
        products["prix_vente_unitaire_moyen"]
    )
    if "stock_courant" not in products.columns:
        products["stock_courant"] = 0.0
    return products


# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #
st.sidebar.markdown(
    f"<h1 style='color:{ZENITH_PRIMARY}'>Zenith</h1>"
    "<p style='margin-top:-15px;color:#666'>Supply Chain Intelligence</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
PAGES = [
    "📊 Tableau de bord",
    "📦 Classification produits",
    "⚠️ Alertes obsolescence",
    "🔮 Prévisions de demande",
    "🛒 Recommandations de commande",
    "🧪 Simulation what-if",
]
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Mémoire UDBL Data Science · Supply Chain · 2026")


# ====================================================================== #
# Page 1 — Tableau de bord global
# ====================================================================== #
def page_dashboard():
    st.markdown(f"<h1 style='color:{ZENITH_PRIMARY}'>Tableau de bord global</h1>", unsafe_allow_html=True)
    transactions = load_transactions()
    obs = load_obsolescence()
    classes = load_classification()

    # Filtre période
    min_d, max_d = transactions["date"].min().date(), transactions["date"].max().date()
    start, end = st.slider(
        "Période d'analyse",
        min_value=min_d, max_value=max_d, value=(min_d, max_d),
        format="YYYY-MM-DD",
    )
    df = transactions[(transactions["date"].dt.date >= start) & (transactions["date"].dt.date <= end)]

    # KPIs
    ca = df["montant_total"].sum()
    n_tx = len(df)
    n_ruptures = int(df["rupture_signalee"].sum()) if "rupture_signalee" in df.columns else 0
    n_obs = len(obs)
    panier_moyen = ca / max(n_tx, 1)
    val_stock = (df.sort_values("date").groupby("produit_id")["stock_apres_vente"].last()
                 * df.groupby("produit_id")["cout_achat_unitaire"].last()).sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CA total", f"{ca:,.0f} $".replace(",", " "))
    c2.metric("Transactions", f"{n_tx:,}".replace(",", " "))
    c3.metric("Panier moyen", f"{panier_moyen:,.1f} $".replace(",", " "))
    c4.metric("Stock valorisé", f"{val_stock:,.0f} $".replace(",", " "))
    c5.metric("Produits à risque", f"{n_obs}", delta=f"{100 * n_obs / len(classes):.1f} %", delta_color="inverse")

    st.markdown("### Évolution mensuelle du CA par magasin")
    monthly = (
        df.assign(mois=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["mois", "magasin"])["montant_total"].sum().reset_index()
    )
    chart = (
        alt.Chart(monthly).mark_line(strokeWidth=2)
        .encode(x="mois:T", y="montant_total:Q", color="magasin:N",
                tooltip=["mois:T", "magasin:N", "montant_total:Q"])
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top 10 produits du mois")
        recent = df.sort_values("date").tail(10000)
        top = (
            recent.groupby(["produit_id", "produit_nom"])["montant_total"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        st.dataframe(top.style.background_gradient(subset=["montant_total"], cmap="Blues"))

    with col2:
        st.markdown("### Performance par magasin (CA)")
        perf = df.groupby(["ville", "magasin"])["montant_total"].sum().reset_index()
        perf = perf.sort_values("montant_total", ascending=False)
        bar = (
            alt.Chart(perf).mark_bar()
            .encode(x="montant_total:Q", y=alt.Y("magasin:N", sort="-x"),
                    color=alt.Color("ville:N", scale=alt.Scale(scheme="set2")),
                    tooltip=["magasin:N", "ville:N", "montant_total:Q"])
            .properties(height=350)
        )
        st.altair_chart(bar, use_container_width=True)


# ====================================================================== #
# Page 2 — Classification produits
# ====================================================================== #
def page_classification():
    st.markdown(f"<h1 style='color:{ZENITH_PRIMARY}'>Classification produits</h1>", unsafe_allow_html=True)
    df = load_classification()
    cat = load_catalogue()[["produit_id", "produit_nom", "famille", "marque", "origine_fournisseur"]]
    df = df.merge(cat, on="produit_id", how="left")

    c1, c2, c3, c4 = st.columns(4)
    classes_abc = ["Toutes"] + sorted(df["classe_abc"].dropna().unique().tolist())
    classes_xyz = ["Toutes"] + sorted(df["classe_xyz"].dropna().unique().tolist())
    clusters = ["Tous"] + sorted(df["libelle_cluster"].dropna().unique().tolist())
    familles = ["Toutes"] + sorted(df["famille"].dropna().unique().tolist())
    f_abc = c1.selectbox("Classe ABC", classes_abc)
    f_xyz = c2.selectbox("Classe XYZ", classes_xyz)
    f_cluster = c3.selectbox("Cluster K-Means", clusters)
    f_famille = c4.selectbox("Famille", familles)

    out = df.copy()
    if f_abc != "Toutes":
        out = out[out["classe_abc"] == f_abc]
    if f_xyz != "Toutes":
        out = out[out["classe_xyz"] == f_xyz]
    if f_cluster != "Tous":
        out = out[out["libelle_cluster"] == f_cluster]
    if f_famille != "Toutes":
        out = out[out["famille"] == f_famille]

    st.markdown(f"**{len(out)} produits** correspondent au filtre")
    show_cols = [
        "produit_id", "produit_nom", "famille", "marque", "classe_abc", "classe_xyz",
        "libelle_cluster", "ca_total_36mois", "ventes_totales_36mois",
        "coefficient_variation", "tendance_3_mois", "jours_depuis_derniere_vente",
    ]
    st.dataframe(out[show_cols].sort_values("ca_total_36mois", ascending=False), use_container_width=True)

    csv = out[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger la sélection (CSV)", data=csv,
                       file_name=f"classification_{datetime.now():%Y%m%d}.csv", mime="text/csv")


# ====================================================================== #
# Page 3 — Alertes obsolescence
# ====================================================================== #
def page_obsolescence():
    st.markdown(f"<h1 style='color:{ZENITH_PRIMARY}'>Alertes obsolescence</h1>", unsafe_allow_html=True)
    flagged = load_obsolescence()
    cat = load_catalogue()[["produit_id", "produit_nom", "famille"]]
    flagged = flagged.merge(cat, on="produit_id", how="left")
    transactions = load_transactions()

    total_val = flagged["valeur_stock_dormant"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Produits à risque", f"{len(flagged)}")
    c2.metric("Stock dormant total", f"{total_val:,.0f} $".replace(",", " "))
    c3.metric("% du catalogue", f"{100 * len(flagged) / 250:.1f} %")

    st.markdown("### Liste des produits à risque (triés par stock immobilisé)")
    show_cols = [
        "produit_id", "produit_nom", "famille", "classe_abc",
        "jours_depuis_derniere_vente", "nombre_mois_consecutifs_sans_vente",
        "valeur_stock_dormant", "score_obsolescence",
    ]
    sorted_df = flagged.sort_values("valeur_stock_dormant", ascending=False)
    st.dataframe(sorted_df[show_cols], use_container_width=True)

    st.markdown("### 🔍 Inspection d'un produit")
    pid = st.selectbox("Choisir un produit flagué", sorted_df["produit_id"])
    sub = transactions[transactions["produit_id"] == pid].copy()
    if not sub.empty:
        sub["mois"] = sub["date"].dt.to_period("M").dt.to_timestamp()
        monthly = sub.groupby("mois")["quantite_vendue"].sum().reset_index()
        chart = (
            alt.Chart(monthly).mark_area(opacity=0.4, color=ZENITH_ACCENT)
            .encode(x="mois:T", y="quantite_vendue:Q",
                    tooltip=["mois:T", "quantite_vendue:Q"])
            .properties(height=300, title=f"Historique des ventes — {pid}")
        )
        line = alt.Chart(monthly).mark_line(color=ZENITH_ACCENT, strokeWidth=2).encode(x="mois:T", y="quantite_vendue:Q")
        st.altair_chart(chart + line, use_container_width=True)

    st.info(
        "💡 **Recommandation d'action** : pour les produits avec stock dormant > 200 USD, "
        "lancer une opération de déstockage (promotion, retour fournisseur). "
        "Pour les autres, simplement les exclure du prochain réapprovisionnement."
    )

    csv = sorted_df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Exporter la liste des alertes", data=csv,
                       file_name=f"alertes_obsolescence_{datetime.now():%Y%m%d}.csv", mime="text/csv")


# ====================================================================== #
# Page 4 — Prévisions de demande
# ====================================================================== #
def page_previsions():
    st.markdown(f"<h1 style='color:{ZENITH_PRIMARY}'>Prévisions de demande</h1>", unsafe_allow_html=True)
    previsions = load_previsions()
    transactions = load_transactions()
    cat = load_catalogue()[["produit_id", "produit_nom", "famille"]]

    col1, col2 = st.columns([2, 1])
    pid = col1.selectbox("Produit", sorted(previsions["produit_id"].unique()))
    horizon = col2.selectbox("Horizon (mois)", [1, 3, 6], index=1)

    nom = cat[cat["produit_id"] == pid]["produit_nom"].iloc[0] if not cat[cat["produit_id"] == pid].empty else ""
    pred = previsions[previsions["produit_id"] == pid].head(horizon).copy()
    modele = pred["modele_utilise"].iloc[0] if not pred.empty else "—"

    st.markdown(f"**Produit** : {pid} — *{nom}*   ·   **Modèle** : `{modele}`   ·   **Horizon** : {horizon} mois")

    hist = (
        transactions[transactions["produit_id"] == pid]
        .assign(mois=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("mois")["quantite_vendue"].sum().reset_index()
        .rename(columns={"quantite_vendue": "valeur"})
        .assign(type="historique")
    )
    fc_df = pd.DataFrame({
        "mois": pd.to_datetime(pred["date"]),
        "valeur": pred["prevision"],
        "type": "prévision",
        "lower": pred["intervalle_confiance_bas"],
        "upper": pred["intervalle_confiance_haut"],
    })

    base = alt.Chart(hist).mark_line(color=ZENITH_PRIMARY, strokeWidth=2).encode(x="mois:T", y="valeur:Q")
    fc_line = alt.Chart(fc_df).mark_line(color=ZENITH_ACCENT, strokeWidth=3).encode(x="mois:T", y="valeur:Q")
    band = alt.Chart(fc_df).mark_area(opacity=0.2, color=ZENITH_ACCENT).encode(x="mois:T", y="lower:Q", y2="upper:Q")
    st.altair_chart((band + base + fc_line).properties(height=400), use_container_width=True)

    st.dataframe(fc_df.drop(columns="type"), use_container_width=True)


# ====================================================================== #
# Page 5 — Recommandations de commande
# ====================================================================== #
def page_commandes():
    st.markdown(f"<h1 style='color:{ZENITH_PRIMARY}'>Recommandations de commande</h1>", unsafe_allow_html=True)
    cmd = load_commandes()
    cat = load_catalogue()[["produit_id", "produit_nom"]]
    cmd = cmd.merge(cat, on="produit_id", how="left")

    c1, c2, c3, c4 = st.columns(4)
    magasins = ["Tous"] + sorted(cmd["magasin"].dropna().unique().tolist())
    fournisseurs = ["Tous"] + sorted(cmd["fournisseur"].dropna().unique().tolist())
    classes = ["Toutes"] + sorted(cmd["classe_abc"].dropna().unique().tolist())
    mois = ["Tous"] + sorted(cmd["mois_offset"].unique().astype(str).tolist())
    f_mag = c1.selectbox("Magasin", magasins)
    f_four = c2.selectbox("Fournisseur", fournisseurs)
    f_cls = c3.selectbox("Classe ABC", classes)
    f_mois = c4.selectbox("Mois", mois)

    out = cmd.copy()
    if f_mag != "Tous":
        out = out[out["magasin"] == f_mag]
    if f_four != "Tous":
        out = out[out["fournisseur"] == f_four]
    if f_cls != "Toutes":
        out = out[out["classe_abc"] == f_cls]
    if f_mois != "Tous":
        out = out[out["mois_offset"].astype(str) == f_mois]

    total_qte = int(out["quantite_commandee"].sum())
    total_montant = float(out["montant_total"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Commandes sélectionnées", f"{len(out)}")
    c2.metric("Quantité totale", f"{total_qte:,}".replace(",", " "))
    c3.metric("Budget nécessaire", f"{total_montant:,.0f} $".replace(",", " "))

    cols = [
        "produit_id", "produit_nom", "magasin", "fournisseur", "mois_offset",
        "classe_abc", "quantite_commandee", "cout_achat", "montant_total",
        "demande_prevue", "lead_time_mois",
    ]
    st.dataframe(out[cols].sort_values("montant_total", ascending=False), use_container_width=True)

    csv = out[cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger bon de commande (CSV)", data=csv,
                       file_name=f"bon_commande_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")


# ====================================================================== #
# Page 6 — Simulation what-if
# ====================================================================== #
def page_simulation():
    st.markdown(f"<h1 style='color:{ZENITH_PRIMARY}'>Simulation what-if</h1>", unsafe_allow_html=True)
    st.info(
        "Ajustez les leviers ci-dessous pour ré-optimiser le plan de commande "
        "et voir l'impact financier en temps réel."
    )
    products = build_products_input()
    previsions = load_previsions()

    c1, c2, c3 = st.columns(3)
    budget = c1.slider("Budget mensuel (USD)", 100_000, 1_000_000, 500_000, step=50_000)
    capacite = c2.slider("Capacité de stockage (m³)", 1_000, 20_000, 5_000, step=500)
    pondération_a = c3.slider("Pondération service classe A", 1.0, 10.0, 4.0, step=0.5)

    if st.button("🚀 Lancer la simulation", type="primary"):
        with st.spinner("Optimisation en cours…"):
            from src.optimization import (
                BUDGET_MENSUEL_DEFAUT, CAPACITE_STOCKAGE_DEFAUT,
                NIVEAU_SERVICE_A, optimize_orders,
            )
            plan_lp, kpis = optimize_orders(
                products, previsions,
                budget_mensuel=budget,
                capacite_stockage=capacite,
                niveau_service_a=NIVEAU_SERVICE_A,
            )
            plan_emp = simulate_baseline_policy(products, previsions)
            compare = compare_policies(plan_lp, plan_emp)

        st.success(f"Solveur : **{kpis['statut']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Commandes", kpis["nb_commandes_passees"])
        c2.metric("Quantité totale", f"{kpis['quantite_totale']:,}".replace(",", " "))
        c3.metric("Valeur (USD)", f"{kpis['valeur_commandes_usd']:,.0f}".replace(",", " "))
        c4.metric("Stock immo moyen", f"{kpis['stock_moyen_immo_usd']:.0f}")

        st.markdown("### KPI comparatifs : empirique vs simulation")
        st.dataframe(compare, use_container_width=True)
    else:
        st.caption("Ajustez les paramètres puis cliquez sur **Lancer la simulation**.")


# --------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------- #
if page == PAGES[0]:
    page_dashboard()
elif page == PAGES[1]:
    page_classification()
elif page == PAGES[2]:
    page_obsolescence()
elif page == PAGES[3]:
    page_previsions()
elif page == PAGES[4]:
    page_commandes()
elif page == PAGES[5]:
    page_simulation()
