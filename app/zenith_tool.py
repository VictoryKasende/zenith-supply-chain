"""Zenith Supply Chain — application interactive (Streamlit).

Tableau de bord opérationnel pour les gestionnaires de Zenith Informatique
et Bureautique : pilotage commercial, classification ABC×XYZ, détection
d'obsolescence, prévisions de demande, recommandations de commande,
simulation what-if.

Lancement :
    streamlit run app/zenith_tool.py

Charte graphique : couleurs du logo Zenith (bleu #1D3B8A, rouge #E63946,
blanc). Les icônes utilisent la police web Material Symbols (chargée via
CSS) dans les contextes HTML, et le paramètre ``icon=`` dans les widgets
natifs (boutons, alertes).

Logo : placez le logo officiel dans ``app/assets/zenith_logo.png`` — il
s'affichera automatiquement dans la sidebar (sinon un branding
typographique de repli est utilisé).
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
# Charte Zenith (extraite du logo officiel)
# --------------------------------------------------------------------- #
ZN_BLUE = "#1D3B8A"        # bleu principal Zenith
ZN_BLUE_DARK = "#152B66"   # bleu foncé pour hover / titres
ZN_RED = "#E63946"         # rouge logo Zenith
ZN_RED_DARK = "#B22633"    # rouge foncé
ZN_WHITE = "#FFFFFF"
ZN_BG = "#F8FAFC"          # gris très clair pour fond carte
ZN_BORDER = "#E2E8F0"
ZN_TEXT = "#0F172A"
ZN_MUTED = "#64748B"
ZN_SUCCESS = "#16A34A"
ZN_WARNING = "#EA580C"

PALETTE_CHARTS = [ZN_BLUE, ZN_RED, "#0EA5E9", "#F59E0B", "#10B981", "#8B5CF6", "#EC4899"]

# --------------------------------------------------------------------- #
# Configuration page
# --------------------------------------------------------------------- #
st.set_page_config(
    page_title="Zenith Supply Chain",
    page_icon=":material/inventory_2:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------- #
# CSS injection (cartes, métriques, sidebar, navigation, tables)
# --------------------------------------------------------------------- #
CUSTOM_CSS = f"""
<style>
  /* Police d'icônes Material Symbols (Google Fonts) */
  @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
  .material-symbols-outlined {{
      font-family: 'Material Symbols Outlined';
      font-weight: normal;
      font-style: normal;
      line-height: 1;
      letter-spacing: normal;
      text-transform: none;
      display: inline-block;
      white-space: nowrap;
      word-wrap: normal;
      direction: ltr;
      vertical-align: middle;
      -webkit-font-feature-settings: 'liga';
      -webkit-font-smoothing: antialiased;
  }}

  /* Reset general */
  .main .block-container {{
      padding-top: 1.5rem;
      padding-bottom: 3rem;
      max-width: 1400px;
  }}

  /* En-tête bandeau */
  .zn-header {{
      background: linear-gradient(135deg, {ZN_BLUE} 0%, {ZN_BLUE_DARK} 100%);
      color: {ZN_WHITE};
      padding: 1.5rem 2rem;
      border-radius: 12px;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 16px rgba(29, 59, 138, 0.18);
  }}
  .zn-header h1 {{
      color: {ZN_WHITE};
      font-size: 1.6rem;
      margin: 0 0 0.3rem 0;
      font-weight: 700;
      letter-spacing: -0.3px;
  }}
  .zn-header p {{
      color: rgba(255,255,255,0.85);
      margin: 0;
      font-size: 0.95rem;
  }}
  .zn-header .accent {{ color: {ZN_RED}; }}

  /* Cartes KPI */
  .zn-kpi {{
      background: {ZN_WHITE};
      border: 1px solid {ZN_BORDER};
      border-left: 4px solid {ZN_BLUE};
      border-radius: 10px;
      padding: 1rem 1.2rem;
      margin-bottom: 0.6rem;
      box-shadow: 0 1px 3px rgba(15,23,42,0.05);
  }}
  .zn-kpi.accent {{ border-left-color: {ZN_RED}; }}
  .zn-kpi .label {{
      color: {ZN_MUTED};
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 600;
      margin-bottom: 0.3rem;
  }}
  .zn-kpi .value {{
      color: {ZN_TEXT};
      font-size: 1.6rem;
      font-weight: 700;
      line-height: 1.1;
  }}
  .zn-kpi .delta-up {{ color: {ZN_SUCCESS}; font-size: 0.85rem; font-weight: 600; }}
  .zn-kpi .delta-down {{ color: {ZN_RED}; font-size: 0.85rem; font-weight: 600; }}
  .zn-kpi .delta-neutral {{ color: {ZN_MUTED}; font-size: 0.85rem; font-weight: 600; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background: {ZN_WHITE};
      border-right: 1px solid {ZN_BORDER};
  }}
  section[data-testid="stSidebar"] .stRadio > div {{ gap: 0.2rem; }}
  section[data-testid="stSidebar"] label[data-baseweb="radio"] {{
      padding: 0.6rem 0.8rem;
      border-radius: 8px;
      margin: 0;
      transition: background 0.15s ease;
      font-weight: 500;
      color: {ZN_TEXT};
  }}
  section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{
      background: {ZN_BG};
  }}

  /* Titres de section */
  .zn-section-title {{
      color: {ZN_BLUE_DARK};
      font-size: 1.1rem;
      font-weight: 700;
      margin: 1.3rem 0 0.6rem 0;
      padding-bottom: 0.4rem;
      border-bottom: 2px solid {ZN_BG};
      display: flex;
      align-items: center;
      gap: 0.5rem;
  }}

  /* Tables */
  div[data-testid="stDataFrame"] {{
      border: 1px solid {ZN_BORDER};
      border-radius: 8px;
      overflow: hidden;
  }}

  /* Boutons primaires */
  div.stButton > button[kind="primary"] {{
      background: {ZN_BLUE};
      border: none;
      color: {ZN_WHITE};
      font-weight: 600;
      border-radius: 8px;
      padding: 0.5rem 1.2rem;
      transition: background 0.15s ease;
  }}
  div.stButton > button[kind="primary"]:hover {{
      background: {ZN_BLUE_DARK};
  }}

  /* Download buttons */
  div[data-testid="stDownloadButton"] > button {{
      background: {ZN_WHITE};
      color: {ZN_BLUE};
      border: 1.5px solid {ZN_BLUE};
      border-radius: 8px;
      font-weight: 600;
      transition: all 0.15s ease;
  }}
  div[data-testid="stDownloadButton"] > button:hover {{
      background: {ZN_BLUE};
      color: {ZN_WHITE};
  }}

  /* Badges */
  .zn-badge {{
      display: inline-block;
      padding: 0.2rem 0.65rem;
      border-radius: 14px;
      font-size: 0.75rem;
      font-weight: 600;
  }}
  .zn-badge.blue {{ background: rgba(29,59,138,0.1); color: {ZN_BLUE}; }}
  .zn-badge.red  {{ background: rgba(230,57,70,0.1); color: {ZN_RED}; }}
  .zn-badge.green{{ background: rgba(22,163,74,0.1); color: {ZN_SUCCESS}; }}

  /* Info box override */
  div[data-testid="stAlert"] {{
      border-radius: 10px;
      border-left-width: 4px;
  }}

  /* Footer cleanup */
  footer {{ visibility: hidden; }}
  #MainMenu {{ visibility: hidden; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# Helpers UI
# --------------------------------------------------------------------- #
def mi(name: str, size: int = 22, color: str = "currentColor") -> str:
    """Renvoie le HTML d'une icône Material Symbols (police web).

    Utilisable dans tout contexte HTML (``st.markdown(..., unsafe_allow_html=True)``).
    Pour les widgets natifs (boutons, alertes), préférer le paramètre ``icon=``.
    """
    return (
        f'<span class="material-symbols-outlined" '
        f'style="font-size:{size}px;color:{color};">{name}</span>'
    )


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    """Bandeau d'en-tête uniforme pour chaque page."""
    st.markdown(
        f"""
        <div class="zn-header">
            <h1>{mi(icon, size=30, color=ZN_WHITE)} <span style="vertical-align:middle;">{title}</span></h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None,
             delta_type: str = "neutral", accent: bool = False) -> str:
    cls = "zn-kpi accent" if accent else "zn-kpi"
    delta_html = ""
    if delta:
        delta_cls = {"up": "delta-up", "down": "delta-down"}.get(delta_type, "delta-neutral")
        arrow = {"up": "▲", "down": "▼"}.get(delta_type, "•")
        delta_html = f'<div class="{delta_cls}">{arrow} {delta}</div>'
    return f"""
    <div class="{cls}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """


def section_title(icon: str, text: str) -> None:
    st.markdown(
        f'<div class="zn-section-title">{mi(icon, size=20, color=ZN_BLUE)}'
        f'<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def fmt_usd(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f} $".replace(",", " ")


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}".replace(",", " ")


# --------------------------------------------------------------------- #
# Chargement (avec cache)
# --------------------------------------------------------------------- #
@st.cache_data
def load_transactions() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/processed/zenith_clean.csv", parse_dates=["date"])


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
# Sidebar — branding (logo) + navigation
# --------------------------------------------------------------------- #
LOGO_CANDIDATES = [
    ROOT / "app" / "assets" / "zenith_logo.jpg",
    ROOT / "app" / "zenith_logo.jpg",
    ROOT / "assets" / "zenith_logo.jpg",
]
logo_path = next((p for p in LOGO_CANDIDATES if p.exists()), None)

if logo_path is not None:
    c1, c2, c3 = st.sidebar.columns([1, 4, 1])
    with c2:
        st.image(str(logo_path), use_container_width=True)
    st.sidebar.markdown(
        f"""
        <div style="text-align:center; margin-top:-0.3rem;">
            <div style="color:{ZN_MUTED}; font-size:0.72rem; font-style:italic;">
                le choix de la qualité c'est nous
            </div>
        </div>
        <hr style="margin:0.8rem 0; border-color:{ZN_BORDER};"/>
        """,
        unsafe_allow_html=True,
    )
else:
    # Repli typographique si le logo n'est pas présent dans app/assets/
    st.sidebar.markdown(
        f"""
        <div style="text-align:center; padding: 1.4rem 0 0.6rem 0;">
            <div style="color:{ZN_BLUE}; font-size:1.9rem; font-weight:800;
                        letter-spacing:-0.5px; line-height:1;">
                ZENITH<span style="color:{ZN_RED};">.</span>
            </div>
            <div style="color:{ZN_RED}; font-size:0.72rem; font-weight:700;
                        letter-spacing:3px; margin-top:0.2rem;">
                INFORMATIQUE
            </div>
            <div style="color:{ZN_MUTED}; font-size:0.72rem; margin-top:0.5rem; font-style:italic;">
                le choix de la qualité c'est nous
            </div>
        </div>
        <hr style="margin:0.8rem 0; border-color:{ZN_BORDER};"/>
        """,
        unsafe_allow_html=True,
    )

PAGES = {
    "Tableau de bord":           ("dashboard", "page_dashboard"),
    "Classification produits":   ("inventory_2", "page_classification"),
    "Alertes obsolescence":      ("warning_amber", "page_obsolescence"),
    "Prévisions de demande":     ("insights", "page_previsions"),
    "Recommandations":           ("shopping_cart", "page_commandes"),
    "Simulation":        ("science", "page_simulation"),
}

st.sidebar.markdown(
    f'<div style="color:{ZN_MUTED}; font-size:0.7rem; font-weight:700; '
    f'text-transform:uppercase; letter-spacing:1px; padding:0 0.4rem 0.4rem;">'
    f'Navigation</div>',
    unsafe_allow_html=True,
)
page_name = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    label_visibility="collapsed",
)

st.sidebar.markdown(
    f"""
    <hr style="margin:1rem 0; border-color:{ZN_BORDER};"/>
    <div style="color:{ZN_MUTED}; font-size:0.7rem; line-height:1.5; padding: 0 0.4rem;">
        <strong style="color:{ZN_TEXT};">Mémoire UDBL</strong><br/>
        Data Science · Supply Chain<br/>
        KASENDE NGELEKA Victoire — 2026
    </div>
    """,
    unsafe_allow_html=True,
)


# ====================================================================== #
# Page 1 — Tableau de bord global
# ====================================================================== #
def page_dashboard():
    page_header(
        "dashboard", "Tableau de bord global",
        "Vue d'ensemble de l'activité commerciale Zenith Informatique &amp; Bureautique",
    )
    transactions = load_transactions()
    obs = load_obsolescence()
    classes = load_classification()

    min_d, max_d = transactions["date"].min().date(), transactions["date"].max().date()
    with st.container():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            start, end = st.slider(
                "Période d'analyse",
                min_value=min_d, max_value=max_d, value=(min_d, max_d),
                format="YYYY-MM-DD",
            )
        with col_b:
            st.metric("Période sélectionnée",
                      f"{(end - start).days} jours",
                      label_visibility="visible")

    df = transactions[(transactions["date"].dt.date >= start) & (transactions["date"].dt.date <= end)]

    ca = df["montant_total"].sum()
    n_tx = len(df)
    n_obs = len(obs)
    panier_moyen = ca / max(n_tx, 1)
    val_stock = (
        df.sort_values("date").groupby("produit_id")["stock_apres_vente"].last()
        * df.groupby("produit_id")["cout_achat_unitaire"].last()
    ).sum()
    n_clients = df["client_id"].nunique()

    cols = st.columns(5)
    cols[0].markdown(kpi_card("Chiffre d'affaires", fmt_usd(ca)), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Transactions", fmt_int(n_tx)), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Panier moyen", fmt_usd(panier_moyen, 1)), unsafe_allow_html=True)
    cols[3].markdown(kpi_card("Stock valorisé", fmt_usd(val_stock)), unsafe_allow_html=True)
    cols[4].markdown(
        kpi_card("Produits à risque",
                 f"{n_obs}",
                 delta=f"{100 * n_obs / len(classes):.1f} % du catalogue",
                 delta_type="down", accent=True),
        unsafe_allow_html=True,
    )

    section_title("trending_up", "Évolution mensuelle du CA par magasin")
    monthly = (
        df.assign(mois=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["mois", "magasin"])["montant_total"].sum().reset_index()
    )
    chart = (
        alt.Chart(monthly).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=40))
        .encode(
            x=alt.X("mois:T", title=None),
            y=alt.Y("montant_total:Q", title="CA (USD)"),
            color=alt.Color("magasin:N", scale=alt.Scale(range=PALETTE_CHARTS),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["mois:T", "magasin:N",
                     alt.Tooltip("montant_total:Q", format=",.0f", title="CA USD")],
        )
        .properties(height=360)
        .configure_axis(grid=True, gridColor=ZN_BORDER, labelColor=ZN_MUTED,
                        titleColor=ZN_TEXT, labelFontSize=11, titleFontSize=12)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        section_title("emoji_events", "Top 10 produits (CA)")
        top = (
            df.groupby(["produit_id", "produit_nom"])["montant_total"].sum()
            .sort_values(ascending=False).head(10).reset_index()
            .rename(columns={"montant_total": "CA (USD)"})
        )
        top["CA (USD)"] = top["CA (USD)"].round(0)
        st.dataframe(top, use_container_width=True, hide_index=True,
                     column_config={"CA (USD)": st.column_config.NumberColumn(format="%.0f $")})

    with col2:
        section_title("store", "Performance par magasin")
        perf = df.groupby(["ville", "magasin"])["montant_total"].sum().reset_index()
        perf = perf.sort_values("montant_total", ascending=False)
        bar = (
            alt.Chart(perf).mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("montant_total:Q", title="CA (USD)"),
                y=alt.Y("magasin:N", sort="-x", title=None),
                color=alt.Color("ville:N", scale=alt.Scale(range=PALETTE_CHARTS),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["magasin:N", "ville:N",
                         alt.Tooltip("montant_total:Q", format=",.0f", title="CA USD")],
            )
            .properties(height=320)
            .configure_axis(grid=True, gridColor=ZN_BORDER, labelColor=ZN_MUTED,
                            titleColor=ZN_TEXT, labelFontSize=11, titleFontSize=12)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(bar, use_container_width=True)


# ====================================================================== #
# Page 2 — Classification produits
# ====================================================================== #
def page_classification():
    page_header(
        "inventory_2", "Classification produits",
        "Segmentation ABC × XYZ et clustering K-Means du catalogue des 250 références",
    )
    df = load_classification()
    cat = load_catalogue()[["produit_id", "produit_nom", "famille", "marque", "origine_fournisseur"]]
    df = df.merge(cat, on="produit_id", how="left")

    cols = st.columns(4)
    cols[0].markdown(kpi_card("Total produits", fmt_int(len(df))), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Classe A", fmt_int((df["classe_abc"] == "A").sum()),
                              delta="bestsellers", delta_type="up"), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Classe B", fmt_int((df["classe_abc"] == "B").sum())), unsafe_allow_html=True)
    cols[3].markdown(kpi_card("Classe C", fmt_int((df["classe_abc"] == "C").sum()),
                              delta="queue de catalogue", delta_type="neutral", accent=True),
                     unsafe_allow_html=True)

    section_title("filter_alt", "Filtres")
    c1, c2, c3, c4 = st.columns(4)
    f_abc = c1.selectbox("Classe ABC", ["Toutes"] + sorted(df["classe_abc"].dropna().unique().tolist()))
    f_xyz = c2.selectbox("Classe XYZ", ["Toutes"] + sorted(df["classe_xyz"].dropna().unique().tolist()))
    f_cluster = c3.selectbox("Cluster K-Means", ["Tous"] + sorted(df["libelle_cluster"].dropna().unique().tolist()))
    f_famille = c4.selectbox("Famille", ["Toutes"] + sorted(df["famille"].dropna().unique().tolist()))

    out = df.copy()
    if f_abc != "Toutes":
        out = out[out["classe_abc"] == f_abc]
    if f_xyz != "Toutes":
        out = out[out["classe_xyz"] == f_xyz]
    if f_cluster != "Tous":
        out = out[out["libelle_cluster"] == f_cluster]
    if f_famille != "Toutes":
        out = out[out["famille"] == f_famille]

    st.markdown(
        f'<span class="zn-badge blue">{len(out)} produits</span> correspondent à votre sélection',
        unsafe_allow_html=True,
    )

    show_cols = [
        "produit_id", "produit_nom", "famille", "marque", "classe_abc", "classe_xyz",
        "libelle_cluster", "ca_total_36mois", "ventes_totales_36mois",
        "coefficient_variation", "tendance_3_mois", "jours_depuis_derniere_vente",
    ]
    st.dataframe(
        out[show_cols].sort_values("ca_total_36mois", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "ca_total_36mois": st.column_config.NumberColumn("CA 36 mois", format="%.0f $"),
            "ventes_totales_36mois": st.column_config.NumberColumn("Ventes 36 mois", format="%.0f"),
            "coefficient_variation": st.column_config.NumberColumn("CV", format="%.2f"),
            "tendance_3_mois": st.column_config.NumberColumn("Tend. 3 mois", format="%.2f"),
        },
    )

    csv = out[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger la sélection (CSV)",
        data=csv, file_name=f"classification_{datetime.now():%Y%m%d}.csv",
        mime="text/csv", icon=":material/download:",
    )


# ====================================================================== #
# Page 3 — Alertes obsolescence
# ====================================================================== #
def page_obsolescence():
    page_header(
        "warning_amber", "Alertes obsolescence",
        "Produits à risque détectés par Isolation Forest et règles métier",
    )
    flagged = load_obsolescence()
    cat = load_catalogue()[["produit_id", "produit_nom", "famille"]]
    flagged = flagged.merge(cat, on="produit_id", how="left")
    transactions = load_transactions()

    total_val = flagged["valeur_stock_dormant"].sum()
    cols = st.columns(3)
    cols[0].markdown(kpi_card("Produits à risque", fmt_int(len(flagged)), accent=True),
                     unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Stock dormant total", fmt_usd(total_val),
                              delta="trésorerie immobilisée", delta_type="down", accent=True),
                     unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Part du catalogue",
                              f"{100 * len(flagged) / 250:.1f} %"), unsafe_allow_html=True)

    section_title("list_alt", "Produits à risque (triés par stock immobilisé)")
    show_cols = [
        "produit_id", "produit_nom", "famille", "classe_abc",
        "jours_depuis_derniere_vente", "nombre_mois_consecutifs_sans_vente",
        "valeur_stock_dormant", "score_obsolescence",
    ]
    sorted_df = flagged.sort_values("valeur_stock_dormant", ascending=False)
    st.dataframe(
        sorted_df[show_cols], use_container_width=True, hide_index=True,
        column_config={
            "valeur_stock_dormant": st.column_config.NumberColumn("Stock dormant", format="%.0f $"),
            "score_obsolescence": st.column_config.NumberColumn("Score", format="%.4f"),
            "jours_depuis_derniere_vente": st.column_config.NumberColumn("Jours sans vente"),
            "nombre_mois_consecutifs_sans_vente": st.column_config.NumberColumn("Mois consécutifs"),
        },
    )

    section_title("query_stats", "Inspection d'un produit")
    pid = st.selectbox("Produit flagué", sorted_df["produit_id"])
    sub = transactions[transactions["produit_id"] == pid].copy()
    if not sub.empty:
        sub["mois"] = sub["date"].dt.to_period("M").dt.to_timestamp()
        monthly = sub.groupby("mois")["quantite_vendue"].sum().reset_index()
        area = (
            alt.Chart(monthly).mark_area(opacity=0.25, color=ZN_RED)
            .encode(x=alt.X("mois:T", title=None),
                    y=alt.Y("quantite_vendue:Q", title="Quantité vendue"),
                    tooltip=["mois:T", "quantite_vendue:Q"])
            .properties(height=300)
        )
        line = alt.Chart(monthly).mark_line(color=ZN_RED, strokeWidth=2.5).encode(
            x="mois:T", y="quantite_vendue:Q"
        )
        st.altair_chart(
            (area + line)
            .configure_axis(grid=True, gridColor=ZN_BORDER, labelColor=ZN_MUTED)
            .configure_view(strokeWidth=0),
            use_container_width=True,
        )

    st.info(
        "**Recommandation d'action** — pour les produits avec stock "
        "dormant > 200 USD, lancer une opération de déstockage (promotion, retour fournisseur). "
        "Pour les autres, simplement les exclure du prochain réapprovisionnement.",
        icon=":material/lightbulb:",
    )

    csv = sorted_df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Exporter la liste des alertes",
        data=csv, file_name=f"alertes_obsolescence_{datetime.now():%Y%m%d}.csv",
        mime="text/csv", icon=":material/download:",
    )


# ====================================================================== #
# Page 4 — Prévisions de demande
# ====================================================================== #
def page_previsions():
    page_header(
        "insights", "Prévisions de demande",
        "Horizon 6 mois — modèles SARIMA / LightGBM / LSTM-like / Croston SBA",
    )
    previsions = load_previsions()
    transactions = load_transactions()
    cat = load_catalogue()[["produit_id", "produit_nom", "famille"]]

    col1, col2 = st.columns([2, 1])
    pid = col1.selectbox("Produit", sorted(previsions["produit_id"].unique()))
    horizon = col2.selectbox("Horizon (mois)", [1, 3, 6], index=1)

    nom = cat[cat["produit_id"] == pid]["produit_nom"].iloc[0] if not cat[cat["produit_id"] == pid].empty else ""
    pred = previsions[previsions["produit_id"] == pid].head(horizon).copy()
    modele = pred["modele_utilise"].iloc[0] if not pred.empty else "—"

    cols = st.columns(3)
    cols[0].markdown(kpi_card("Produit", f"{pid}"), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Modèle utilisé", modele), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Horizon", f"{horizon} mois"), unsafe_allow_html=True)

    st.markdown(f"<div style='color:{ZN_MUTED}; margin: 0.5rem 0 1rem 0;'>{nom}</div>",
                unsafe_allow_html=True)

    section_title("show_chart", "Historique et prévision")
    hist = (
        transactions[transactions["produit_id"] == pid]
        .assign(mois=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("mois")["quantite_vendue"].sum().reset_index()
        .rename(columns={"quantite_vendue": "valeur"})
    )
    fc_df = pd.DataFrame({
        "mois": pd.to_datetime(pred["date"]),
        "valeur": pred["prevision"],
        "lower": pred["intervalle_confiance_bas"],
        "upper": pred["intervalle_confiance_haut"],
    })

    base = alt.Chart(hist).mark_line(color=ZN_BLUE, strokeWidth=2.5).encode(
        x=alt.X("mois:T", title=None),
        y=alt.Y("valeur:Q", title="Quantité"),
        tooltip=["mois:T", "valeur:Q"],
    )
    band = alt.Chart(fc_df).mark_area(opacity=0.18, color=ZN_RED).encode(
        x="mois:T", y="lower:Q", y2="upper:Q",
    )
    fc_line = alt.Chart(fc_df).mark_line(color=ZN_RED, strokeWidth=3,
                                          strokeDash=[6, 3]).encode(x="mois:T", y="valeur:Q")
    chart = (band + base + fc_line).properties(height=380).configure_axis(
        grid=True, gridColor=ZN_BORDER, labelColor=ZN_MUTED,
    ).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)

    section_title("table_chart", "Détail des prévisions")
    fc_display = fc_df.copy()
    fc_display["mois"] = fc_display["mois"].dt.strftime("%Y-%m")
    st.dataframe(
        fc_display, use_container_width=True, hide_index=True,
        column_config={
            "valeur": st.column_config.NumberColumn("Prévision", format="%.1f"),
            "lower": st.column_config.NumberColumn("IC bas", format="%.1f"),
            "upper": st.column_config.NumberColumn("IC haut", format="%.1f"),
        },
    )


# ====================================================================== #
# Page 5 — Recommandations de commande
# ====================================================================== #
def page_commandes():
    page_header(
        "shopping_cart", "Recommandations de commande",
        "Plan de réapprovisionnement optimisé par programme linéaire (PuLP/CBC)",
    )
    cmd = load_commandes()
    cat = load_catalogue()[["produit_id", "produit_nom"]]
    cmd = cmd.merge(cat, on="produit_id", how="left")

    section_title("filter_alt", "Filtres")
    c1, c2, c3, c4 = st.columns(4)
    f_mag = c1.selectbox("Magasin", ["Tous"] + sorted(cmd["magasin"].dropna().unique().tolist()))
    f_four = c2.selectbox("Fournisseur", ["Tous"] + sorted(cmd["fournisseur"].dropna().unique().tolist()))
    f_cls = c3.selectbox("Classe ABC", ["Toutes"] + sorted(cmd["classe_abc"].dropna().unique().tolist()))
    f_mois = c4.selectbox("Mois", ["Tous"] + sorted(cmd["mois_offset"].unique().astype(str).tolist()))

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
    nb_lignes = len(out)
    avg_lead = float(out["lead_time_mois"].mean()) if not out.empty else 0

    cols = st.columns(4)
    cols[0].markdown(kpi_card("Commandes sélectionnées", fmt_int(nb_lignes)), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Quantité totale", fmt_int(total_qte)), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Budget nécessaire", fmt_usd(total_montant), accent=True),
                     unsafe_allow_html=True)
    cols[3].markdown(kpi_card("Délai moyen", f"{avg_lead:.1f} mois"), unsafe_allow_html=True)

    section_title("receipt_long", "Détail des commandes recommandées")
    cols_show = [
        "produit_id", "produit_nom", "magasin", "fournisseur", "mois_offset",
        "classe_abc", "quantite_commandee", "cout_achat", "montant_total",
        "demande_prevue", "lead_time_mois",
    ]
    st.dataframe(
        out[cols_show].sort_values("montant_total", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "cout_achat": st.column_config.NumberColumn("Coût unitaire", format="%.2f $"),
            "montant_total": st.column_config.NumberColumn("Montant", format="%.0f $"),
            "demande_prevue": st.column_config.NumberColumn("Demande prévue", format="%.0f"),
        },
    )

    csv = out[cols_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger le bon de commande (CSV)",
        data=csv, file_name=f"bon_commande_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv", icon=":material/file_download:",
    )


# ====================================================================== #
# Page 6 — Simulation what-if
# ====================================================================== #
def page_simulation():
    page_header(
        "science", "Simulation",
        "Ajustez les leviers pour ré-optimiser le plan de commandes en temps réel",
    )
    products = build_products_input()
    previsions = load_previsions()

    section_title("tune", "Leviers d'optimisation")
    c1, c2, c3 = st.columns(3)
    budget = c1.slider("Budget mensuel (USD)", 100_000, 1_000_000, 500_000, step=50_000,
                       format="%d $")
    capacite = c2.slider("Capacité de stockage (m³)", 1_000, 20_000, 5_000, step=500)
    ponderation_a = c3.slider("Pondération service classe A", 1.0, 10.0, 4.0, step=0.5)

    launch = st.button("Lancer la simulation", type="primary", icon=":material/play_arrow:")

    if launch:
        with st.spinner("Optimisation en cours…"):
            from src.optimization import NIVEAU_SERVICE_A
            plan_lp, kpis = optimize_orders(
                products, previsions,
                budget_mensuel=budget,
                capacite_stockage=capacite,
                niveau_service_a=NIVEAU_SERVICE_A,
            )
            plan_emp = simulate_baseline_policy(products, previsions)
            compare = compare_policies(plan_lp, plan_emp)

        status_msg = kpis['statut']
        if status_msg == "Optimal":
            st.success(f"Solveur : **{status_msg}**", icon=":material/check_circle:")
        else:
            st.warning(f"Solveur : **{status_msg}**", icon=":material/error:")

        section_title("speed", "Indicateurs simulés")
        cols = st.columns(4)
        cols[0].markdown(kpi_card("Commandes", fmt_int(kpis["nb_commandes_passees"])), unsafe_allow_html=True)
        cols[1].markdown(kpi_card("Quantité totale", fmt_int(kpis["quantite_totale"])), unsafe_allow_html=True)
        cols[2].markdown(kpi_card("Valeur commandes", fmt_usd(kpis["valeur_commandes_usd"])),
                         unsafe_allow_html=True)
        cols[3].markdown(kpi_card("Stock immo moyen", fmt_usd(kpis["stock_moyen_immo_usd"]),
                                  accent=True), unsafe_allow_html=True)

        section_title("compare_arrows", "Politique simulée vs politique empirique")
        st.dataframe(
            compare, use_container_width=True, hide_index=True,
            column_config={
                "politique_empirique": st.column_config.NumberColumn("Empirique", format="%.2f"),
                "politique_optimisee": st.column_config.NumberColumn("Optimisée", format="%.2f"),
                "delta": st.column_config.NumberColumn("Δ", format="%.2f"),
                "delta_pct": st.column_config.NumberColumn("Δ %", format="%.1f %%"),
            },
        )
    else:
        st.info("Ajustez les paramètres ci-dessus puis cliquez sur "
                "**Lancer la simulation** pour ré-optimiser le plan.",
                icon=":material/info:")


# --------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------- #
PAGE_FUNCS = {
    "Tableau de bord": page_dashboard,
    "Classification produits": page_classification,
    "Alertes obsolescence": page_obsolescence,
    "Prévisions de demande": page_previsions,
    "Recommandations": page_commandes,
    "Simulation": page_simulation,
}
PAGE_FUNCS[page_name]()
