"""
Agriculture Yield Analytics Dashboard
=======================================
A complete Data Analytics + AI application for agricultural crop yield
analysis and prediction using the yield_df.csv dataset.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:8050 in your browser.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":        "#F7F9F7",        # page background — very light sage
    "surface":   "#FFFFFF",        # card surface
    "sidebar":   "#1B3A2D",        # deep forest green sidebar
    "accent":    "#2D6A4F",        # primary green
    "accent2":   "#52B788",        # lighter green for highlights
    "accent3":   "#74C69D",        # mint for secondary accents
    "text":      "#1A2E22",        # near-black green text
    "muted":     "#6B7C6E",        # muted text
    "border":    "#D8E8DC",        # subtle border
    "amber":     "#D97706",        # warm amber for warnings/max
    "blue":      "#2563EB",        # blue for model metrics
    "danger":    "#DC2626",
}

CHART_COLORS = [
    "#2D6A4F","#52B788","#74C69D","#B7E4C7",
    "#D97706","#F59E0B","#2563EB","#7C3AED",
    "#EC4899","#0891B2",
]

FONT = "Inter, 'Segoe UI', system-ui, sans-serif"

CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_family=FONT,
    font_color=C["text"],
    title_font_size=14,
    title_font_color=C["text"],
    margin=dict(l=16, r=16, t=44, b=16),
    legend=dict(font_size=11, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#F0F0F0", linecolor="#E0E0E0"),
    yaxis=dict(gridcolor="#F0F0F0", linecolor="#E0E0E0"),
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(path="yield_df.csv"):
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.strip() == ""]
    df.drop(columns=unnamed, inplace=True)
    df.rename(columns={
        "hg/ha_yield":                   "yield_hg_ha",
        "average_rain_fall_mm_per_year": "rainfall_mm",
        "pesticides_tonnes":             "pesticides_t",
        "avg_temp":                      "avg_temp_c",
    }, inplace=True)
    for col in ["yield_hg_ha", "rainfall_mm", "pesticides_t", "avg_temp_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Year"] = df["Year"].astype(int)
    df = (df.groupby(["Area", "Item", "Year"], as_index=False)
            .agg({"yield_hg_ha": "first", "rainfall_mm": "first",
                  "pesticides_t": "first", "avg_temp_c": "mean"}))
    df.dropna(subset=["yield_hg_ha","rainfall_mm","pesticides_t","avg_temp_c"], inplace=True)
    df["log_yield"]      = np.log1p(df["yield_hg_ha"])
    df["log_pesticides"] = np.log1p(df["pesticides_t"])
    df.reset_index(drop=True, inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MACHINE LEARNING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_model(df):
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    df = df.copy()
    df["area_enc"] = le_area.fit_transform(df["Area"])
    df["item_enc"] = le_item.fit_transform(df["Item"])
    feature_cols = ["area_enc","item_enc","Year","rainfall_mm","log_pesticides","avg_temp_c"]
    X = df[feature_cols].values
    y = df["log_yield"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                   min_samples_leaf=2, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = mean_squared_error(y_true, y_pred) ** 0.5
    r2    = r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    cv    = cross_val_score(model, X, y, cv=5, scoring="r2")
    metrics = {
        "R² Score":     f"{r2:.4f}",
        "MAE":          f"{mae:,.0f} hg/ha",
        "RMSE":         f"{rmse:,.0f} hg/ha",
        "MAPE":         f"{mape:.2f}%",
        "CV R² (mean)": f"{cv.mean():.4f}",
        "CV R² (std)":  f"{cv.std():.4f}",
    }
    feat_imp = pd.DataFrame({
        "Feature":    ["Country","Crop Type","Year","Rainfall","Pesticides (log)","Avg Temp"],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    return model, feature_cols, metrics, feat_imp, X_test, y_true, y_pred, le_area, le_item


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def apply_theme(fig):
    fig.update_layout(**CHART_LAYOUT)
    return fig

def fig_yield_by_crop(df):
    d = df.groupby("Item")["yield_hg_ha"].mean().reset_index().sort_values("yield_hg_ha")
    fig = px.bar(d, x="yield_hg_ha", y="Item", orientation="h",
                 labels={"yield_hg_ha": "Mean Yield (hg/ha)", "Item": ""},
                 color="yield_hg_ha", color_continuous_scale=["#B7E4C7","#2D6A4F"])
    fig.update_layout(coloraxis_showscale=False, title="Mean Yield by Crop")
    return apply_theme(fig)

def fig_yield_trend(df):
    d = df.groupby(["Year","Item"])["yield_hg_ha"].mean().reset_index()
    fig = px.line(d, x="Year", y="yield_hg_ha", color="Item",
                  labels={"yield_hg_ha": "Mean Yield (hg/ha)"},
                  color_discrete_sequence=CHART_COLORS)
    fig.update_traces(mode="lines+markers", marker_size=4, line_width=2)
    fig.update_layout(title="Yield Trend by Crop (1990–2013)")
    return apply_theme(fig)

def fig_top_countries(df):
    d = (df.groupby("Area")["yield_hg_ha"].mean().reset_index()
           .sort_values("yield_hg_ha", ascending=False).head(15))
    fig = px.bar(d, x="Area", y="yield_hg_ha",
                 labels={"yield_hg_ha": "Mean Yield (hg/ha)", "Area": ""},
                 color="yield_hg_ha", color_continuous_scale=["#B7E4C7","#1B4332"])
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35,
                      title="Top 15 Countries by Mean Yield")
    return apply_theme(fig)

def fig_yield_distribution(df):
    fig = px.box(df, x="Item", y="yield_hg_ha", color="Item",
                 labels={"yield_hg_ha": "Yield (hg/ha)", "Item": ""},
                 color_discrete_sequence=CHART_COLORS)
    fig.update_layout(showlegend=False, xaxis_tickangle=-20,
                      title="Yield Distribution per Crop")
    return apply_theme(fig)

def fig_rainfall_vs_yield(df):
    s = df.sample(min(3000, len(df)), random_state=1)
    fig = px.scatter(s, x="rainfall_mm", y="yield_hg_ha", color="Item", opacity=0.5,
                     labels={"rainfall_mm": "Rainfall (mm/year)", "yield_hg_ha": "Yield (hg/ha)"},
                     color_discrete_sequence=CHART_COLORS)
    fig.update_traces(marker_size=5)
    fig.update_layout(title="Rainfall vs Yield")
    return apply_theme(fig)

def fig_temp_vs_yield(df):
    s = df.sample(min(3000, len(df)), random_state=2)
    fig = px.scatter(s, x="avg_temp_c", y="yield_hg_ha", color="Item", opacity=0.5,
                     labels={"avg_temp_c": "Avg Temperature (°C)", "yield_hg_ha": "Yield (hg/ha)"},
                     color_discrete_sequence=CHART_COLORS)
    fig.update_traces(marker_size=5)
    fig.update_layout(title="Temperature vs Yield")
    return apply_theme(fig)

def fig_pesticides_vs_yield(df):
    s = df.sample(min(3000, len(df)), random_state=3)
    fig = px.scatter(s, x="log_pesticides", y="log_yield", color="Item", opacity=0.5,
                     labels={"log_pesticides": "Pesticides — log scale",
                             "log_yield": "Yield — log scale"},
                     color_discrete_sequence=CHART_COLORS)
    fig.update_traces(marker_size=5)
    fig.update_layout(title="Pesticides vs Yield (log scale)")
    return apply_theme(fig)

def fig_heatmap(df):
    pivot = df.groupby(["Area","Item"])["yield_hg_ha"].mean().unstack(fill_value=0)
    top30 = pivot.sum(axis=1).nlargest(25).index
    pivot = pivot.loc[top30]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#F0FFF4"],[0.5,"#52B788"],[1,"#1B4332"]],
        colorbar=dict(title="hg/ha", thickness=12),
    ))
    fig.update_layout(title="Yield Heatmap — Top 25 Countries × Crop",
                      xaxis_tickangle=-30, height=520)
    return apply_theme(fig)

def fig_rain_map(df):
    rain = df.drop_duplicates("Area")[["Area","rainfall_mm"]]
    fig = px.choropleth(rain, locations="Area", locationmode="country names",
                        color="rainfall_mm", color_continuous_scale="Blues",
                        labels={"rainfall_mm": "Rainfall (mm)"})
    fig.update_layout(title="Annual Rainfall by Country", geo=dict(showframe=False))
    return apply_theme(fig)

def fig_yield_map(df):
    cy = df.groupby("Area")["yield_hg_ha"].mean().reset_index()
    fig = px.choropleth(cy, locations="Area", locationmode="country names",
                        color="yield_hg_ha", color_continuous_scale="YlGn",
                        labels={"yield_hg_ha": "Mean Yield (hg/ha)"})
    fig.update_layout(title="Mean Crop Yield by Country", geo=dict(showframe=False))
    return apply_theme(fig)

def fig_feature_importance(feat_imp):
    fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=["#B7E4C7","#2D6A4F"])
    fig.update_layout(coloraxis_showscale=False,
                      yaxis=dict(categoryorder="total ascending"),
                      title="Feature Importance (Random Forest)")
    return apply_theme(fig)

def fig_actual_vs_predicted(y_true, y_pred):
    idx = np.random.choice(len(y_true), min(4000, len(y_true)), replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true[idx], y=y_pred[idx], mode="markers",
                             marker=dict(color=C["accent2"], opacity=0.45, size=5),
                             name="Predictions"))
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                             line=dict(color=C["danger"], dash="dash", width=2),
                             name="Perfect Fit"))
    fig.update_layout(title="Actual vs Predicted Yield",
                      xaxis_title="Actual (hg/ha)", yaxis_title="Predicted (hg/ha)")
    return apply_theme(fig)

def fig_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.histogram(x=residuals, nbins=60,
                       labels={"x": "Residual (Actual − Predicted)"},
                       color_discrete_sequence=[C["accent"]])
    fig.update_layout(title="Residuals Distribution")
    return apply_theme(fig)

def fig_yearly_growth(df):
    d = df.groupby("Year")["yield_hg_ha"].mean().reset_index()
    fig = px.area(d, x="Year", y="yield_hg_ha",
                  labels={"yield_hg_ha": "Mean Yield (hg/ha)"},
                  color_discrete_sequence=[C["accent2"]])
    fig.update_traces(line_color=C["accent"], fillcolor="rgba(82,183,136,0.2)")
    fig.update_layout(title="Global Mean Yield Growth Over Time")
    return apply_theme(fig)

def fig_correlation(df):
    corr = df[["yield_hg_ha","rainfall_mm","pesticides_t","avg_temp_c"]].corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0, text=corr.values.round(2),
        texttemplate="%{text}", colorbar=dict(thickness=12),
    ))
    fig.update_layout(title="Feature Correlation Matrix")
    return apply_theme(fig)

def fig_crop_share(df):
    share = df.groupby("Item")["yield_hg_ha"].sum().reset_index()
    fig = px.pie(share, names="Item", values="yield_hg_ha",
                 color_discrete_sequence=CHART_COLORS, hole=0.4)
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      marker=dict(line=dict(color="#fff", width=2)))
    fig.update_layout(showlegend=False, title="Yield Share by Crop Type")
    return apply_theme(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DASH APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def create_app(df, metrics, feat_imp, y_true, y_pred, model, le_area, le_item, df_raw):

    # ── Global CSS injected via index_string ─────────────────────────────────
    CUSTOM_CSS = f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: {FONT};
        background: {C['bg']};
        color: {C['text']};
        font-size: 14px;
        line-height: 1.6;
    }}
    /* ── Sidebar ── */
    #sidebar {{
        position: fixed; top: 0; left: 0; bottom: 0; width: 220px;
        background: {C['sidebar']};
        padding: 0;
        z-index: 100;
        display: flex; flex-direction: column;
        box-shadow: 3px 0 20px rgba(0,0,0,0.15);
        overflow: hidden;
    }}
    #sidebar-logo {{
        padding: 28px 24px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    #sidebar-logo .logo-icon {{ font-size: 2rem; display:block; margin-bottom:6px; }}
    #sidebar-logo .logo-title {{
        font-size: 1.05rem; font-weight: 700; color: #fff;
        letter-spacing: -0.3px; line-height: 1.2;
    }}
    #sidebar-logo .logo-sub {{
        font-size: 0.72rem; color: rgba(255,255,255,0.5);
        margin-top: 3px; text-transform: uppercase; letter-spacing: 0.8px;
    }}
    /* ── Nav links ── */
    .nav-section-label {{
        font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1.2px; color: rgba(255,255,255,0.35);
        padding: 20px 24px 6px;
    }}
    .nav-item {{
        display: flex; align-items: center; gap: 10px;
        padding: 10px 24px; cursor: pointer;
        color: rgba(255,255,255,0.65); font-size: 0.88rem; font-weight: 500;
        border-left: 3px solid transparent;
        transition: all 0.18s ease;
        text-decoration: none !important;
        user-select: none;
    }}
    .nav-item:hover {{
        background: rgba(255,255,255,0.07);
        color: #fff;
    }}
    .nav-item.active {{
        background: rgba(82,183,136,0.18);
        color: {C['accent3']};
        border-left: 3px solid {C['accent3']};
        font-weight: 600;
    }}
    .nav-item .nav-icon {{ font-size: 1.1rem; width: 22px; text-align: center; }}
    /* ── Main content ── */
    #main-content {{
        margin-left: 220px;
        min-height: 100vh;
        padding: 28px 32px 40px;
    }}
    /* ── Page header ── */
    .page-header {{
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid {C['border']};
    }}
    .page-header h2 {{
        font-size: 1.4rem; font-weight: 700; color: {C['text']};
        letter-spacing: -0.4px; margin-bottom: 3px;
    }}
    .page-header p {{
        font-size: 0.85rem; color: {C['muted']}; margin: 0;
    }}
    /* ── KPI Cards ── */
    .kpi-card {{
        background: {C['surface']};
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid {C['border']};
        transition: box-shadow 0.2s;
    }}
    .kpi-card:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
    .kpi-icon {{ font-size: 1.6rem; margin-bottom: 8px; }}
    .kpi-value {{
        font-size: 1.55rem; font-weight: 700;
        letter-spacing: -0.5px; line-height: 1;
        margin-bottom: 4px;
    }}
    .kpi-label {{ font-size: 0.75rem; color: {C['muted']}; font-weight: 500; }}
    /* ── Chart cards ── */
    .chart-card {{
        background: {C['surface']};
        border-radius: 14px;
        border: 1px solid {C['border']};
        padding: 4px;
        transition: box-shadow 0.2s;
    }}
    .chart-card:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,0.07); }}
    /* ── Section label ── */
    .section-label {{
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1px; color: {C['accent']};
        margin: 28px 0 12px;
        display: flex; align-items: center; gap: 8px;
    }}
    .section-label::after {{
        content: ''; flex: 1; height: 1px; background: {C['border']};
    }}
    /* ── Filter panel ── */
    .filter-panel {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }}
    .filter-label {{
        font-size: 0.78rem; font-weight: 600; color: {C['muted']};
        text-transform: uppercase; letter-spacing: 0.7px;
        margin-bottom: 8px;
    }}
    /* ── Predict card ── */
    .predict-result-card {{
        background: linear-gradient(135deg, #F0FFF4 0%, #fff 100%);
        border: 1.5px solid {C['accent3']};
        border-radius: 16px;
        padding: 28px 32px;
    }}
    .predict-big-number {{
        font-size: 2.8rem; font-weight: 800; color: {C['accent']};
        letter-spacing: -1px; line-height: 1;
    }}
    .predict-sub {{
        font-size: 0.9rem; color: {C['muted']}; margin-top: 4px;
    }}
    /* ── Data table ── */
    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {{
        background: {C['sidebar']} !important;
        color: #fff !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }}
    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 99px; }}
    /* ── Slider ── */
    .rc-slider-track {{ background-color: {C['accent2']} !important; }}
    .rc-slider-handle {{ border-color: {C['accent']} !important; }}
    """

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        ],
        title="AgriYield Analytics",
        suppress_callback_exceptions=True,
    )

    # inject custom CSS
    app.index_string = f"""<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{CUSTOM_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""

    # ── Pre-compute all figures ───────────────────────────────────────────────
    F = {
        "yield_crop":   fig_yield_by_crop(df),
        "trend":        fig_yield_trend(df),
        "top_cnt":      fig_top_countries(df),
        "box":          fig_yield_distribution(df),
        "rain":         fig_rainfall_vs_yield(df),
        "temp":         fig_temp_vs_yield(df),
        "pest":         fig_pesticides_vs_yield(df),
        "heatmap":      fig_heatmap(df),
        "rain_map":     fig_rain_map(df),
        "yield_map":    fig_yield_map(df),
        "feat_imp":     fig_feature_importance(feat_imp),
        "actual_pred":  fig_actual_vs_predicted(y_true, y_pred),
        "residuals":    fig_residuals(y_true, y_pred),
        "growth":       fig_yearly_growth(df),
        "corr":         fig_correlation(df),
        "pie":          fig_crop_share(df),
    }

    # ── KPI data ─────────────────────────────────────────────────────────────
    kpis = [
        ("🗃️", f"{len(df):,}",              "Total Records",   C["accent"]),
        ("🌍", str(df["Area"].nunique()),    "Countries",       C["accent"]),
        ("🌾", str(df["Item"].nunique()),    "Crop Types",      C["accent"]),
        ("📅", f"{df['Year'].min()}–{df['Year'].max()}", "Year Range", C["accent"]),
        ("🏆", f"{df['yield_hg_ha'].max():,.0f}", "Max Yield (hg/ha)", C["amber"]),
        ("📈", f"{df['yield_hg_ha'].mean():,.0f}", "Mean Yield (hg/ha)", C["blue"]),
    ]

    ml_kpis = [
        ("🎯", metrics["R² Score"],     "R² Score",     C["blue"]),
        ("📉", metrics["MAE"],          "MAE",          C["accent"]),
        ("📐", metrics["RMSE"],         "RMSE",         C["accent"]),
        ("📊", metrics["MAPE"],         "MAPE",         C["amber"]),
        ("🔁", metrics["CV R² (mean)"], "CV R² Mean",   C["blue"]),
        ("📏", metrics["CV R² (std)"],  "CV R² Std",    C["muted"]),
    ]

    def kpi_card(icon, value, label, color):
        return dbc.Col(
            html.Div([
                html.Div(icon,  className="kpi-icon"),
                html.Div(value, className="kpi-value", style={"color": color}),
                html.Div(label, className="kpi-label"),
            ], className="kpi-card"),
            xs=6, sm=4, md=2, className="mb-3",
        )

    def chart_card(fig, span=6):
        return dbc.Col(
            html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False},
                               style={"height": "360px"}),
                     className="chart-card"),
            md=span, className="mb-4",
        )

    def section(label):
        return html.Div(label, className="section-label")

    def page_header(title, subtitle=""):
        return html.Div([
            html.H2(title),
            html.P(subtitle) if subtitle else None,
        ], className="page-header")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    NAV_ITEMS = [
        ("tab-eda",     "📊", "EDA Overview"),
        ("tab-geo",     "🗺️", "Geo Maps"),
        ("tab-ml",      "🤖", "ML & Predictions"),
        ("tab-explore", "🔍", "Explore Data"),
        ("tab-predict", "🎯", "Yield Predictor"),
    ]

    sidebar = html.Div(id="sidebar", children=[
        html.Div(id="sidebar-logo", children=[
            html.Span("🌾", className="logo-icon"),
            html.Div("AgriYield", className="logo-title"),
            html.Div("Analytics Dashboard", className="logo-sub"),
        ]),
        html.Div("Navigation", className="nav-section-label"),
        *[html.Div([
            html.Span(icon, className="nav-icon"),
            html.Span(label),
          ], className="nav-item", id=f"nav-{tab}",
             **{"data-tab": tab})
          for tab, icon, label in NAV_ITEMS],
        # footer
        html.Div(style={"flex": 1}),
        html.Div([
            html.Div("Keerti · Agriculture Analytics",
                     style={"fontSize": "0.7rem", "color": "rgba(255,255,255,0.25)",
                            "padding": "16px 24px"}),
        ]),
    ])

    # ── App layout ────────────────────────────────────────────────────────────
    app.layout = html.Div([
        dcc.Store(id="active-tab", data="tab-eda"),
        sidebar,
        html.Div(id="main-content", children=[
            dbc.Row([kpi_card(*k) for k in kpis], className="mb-2"),
            html.Div(id="page-content"),
        ]),
    ])

    # ── Tab switching via clientside callback ─────────────────────────────────
    for tab, icon, label in NAV_ITEMS:
        app.clientside_callback(
            f"""function(n) {{
                if (n) return '{tab}';
                return window.dash_clientside.no_update;
            }}""",
            Output("active-tab", "data", allow_duplicate=True),
            Input(f"nav-{tab}", "n_clicks"),
            prevent_initial_call=True,
        )

    # Highlight active nav item
    app.clientside_callback(
        """function(tab) {
            const tabs = ['tab-eda','tab-geo','tab-ml','tab-explore','tab-predict'];
            const res = {};
            tabs.forEach(t => {
                res['nav-' + t] = tab === t ? 'nav-item active' : 'nav-item';
            });
            return [res['nav-tab-eda'], res['nav-tab-geo'], res['nav-tab-ml'],
                    res['nav-tab-explore'], res['nav-tab-predict']];
        }""",
        [Output(f"nav-{t}", "className") for t, _, __ in NAV_ITEMS],
        Input("active-tab", "data"),
    )

    # ── Page content callback ─────────────────────────────────────────────────
    @app.callback(Output("page-content", "children"), Input("active-tab", "data"))
    def render_page(tab):

        # ── EDA ───────────────────────────────────────────────────────────────
        if tab == "tab-eda":
            return html.Div([
                page_header("Exploratory Data Analysis",
                            "Patterns, distributions, and relationships across crops, countries, and years"),
                section("CROP YIELD OVERVIEW"),
                dbc.Row([chart_card(F["yield_crop"]), chart_card(F["pie"])]),
                section("YIELD TRENDS OVER TIME"),
                dbc.Row([chart_card(F["trend"], 12)]),
                section("DISTRIBUTIONS & RANKINGS"),
                dbc.Row([chart_card(F["box"], 12)]),
                dbc.Row([chart_card(F["top_cnt"], 12)]),
                section("CLIMATE & INPUT RELATIONSHIPS"),
                dbc.Row([chart_card(F["rain"]), chart_card(F["temp"])]),
                dbc.Row([chart_card(F["pest"]), chart_card(F["corr"])]),
                section("COUNTRY × CROP HEATMAP"),
                dbc.Row([chart_card(F["heatmap"], 12)]),
                section("GLOBAL PRODUCTIVITY GROWTH"),
                dbc.Row([chart_card(F["growth"], 12)]),
            ])

        # ── Geo Maps ──────────────────────────────────────────────────────────
        elif tab == "tab-geo":
            return html.Div([
                page_header("Geospatial Maps",
                            "World-level distribution of yield and rainfall"),
                dbc.Row([
                    dbc.Col(html.Div(
                        dcc.Graph(figure=F["yield_map"],
                                  config={"displayModeBar": False},
                                  style={"height": "420px"}),
                        className="chart-card"), md=12, className="mb-4"),
                ]),
                dbc.Row([
                    dbc.Col(html.Div(
                        dcc.Graph(figure=F["rain_map"],
                                  config={"displayModeBar": False},
                                  style={"height": "420px"}),
                        className="chart-card"), md=12, className="mb-4"),
                ]),
            ])

        # ── ML & Predictions ──────────────────────────────────────────────────
        elif tab == "tab-ml":
            return html.Div([
                page_header("Machine Learning & Model Evaluation",
                            "Random Forest Regressor — 200 trees, log-transformed target, 80/20 split"),
                section("MODEL PERFORMANCE"),
                dbc.Row([kpi_card(*k) for k in ml_kpis], className="mb-2"),
                html.Div([
                    html.Span("ℹ️  "),
                    html.Span("Model trained on 80% of data · Evaluated on 20% held-out test set · "
                              "5-fold cross-validation for robustness", style={"color": C["muted"]}),
                ], style={"fontSize": "0.82rem", "background": "#F0FFF4",
                          "border": f"1px solid {C['border']}", "borderRadius": "10px",
                          "padding": "12px 16px", "marginBottom": "20px"}),
                section("FEATURE IMPORTANCE & MODEL FIT"),
                dbc.Row([chart_card(F["feat_imp"]), chart_card(F["actual_pred"])]),
                section("RESIDUAL ANALYSIS"),
                dbc.Row([chart_card(F["residuals"], 6)]),
            ])

        # ── Explore Data ──────────────────────────────────────────────────────
        elif tab == "tab-explore":
            crops     = [{"label": c, "value": c} for c in sorted(df["Item"].unique())]
            countries = [{"label": c, "value": c} for c in sorted(df["Area"].unique())]
            return html.Div([
                page_header("Explore Data",
                            "Interactively filter by crop, country, and year range"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div("Crop Type", className="filter-label"),
                            dcc.Dropdown(id="dd-crop", options=crops, multi=True,
                                         placeholder="All crops…",
                                         style={"fontSize": "13px"}),
                        ], md=4),
                        dbc.Col([
                            html.Div("Country", className="filter-label"),
                            dcc.Dropdown(id="dd-country", options=countries, multi=True,
                                         placeholder="All countries…",
                                         style={"fontSize": "13px"}),
                        ], md=4),
                        dbc.Col([
                            html.Div("Year Range", className="filter-label"),
                            dcc.RangeSlider(
                                id="sl-year",
                                min=df["Year"].min(), max=df["Year"].max(), step=1,
                                marks={y: str(y) for y in range(
                                    df["Year"].min(), df["Year"].max()+1, 4)},
                                value=[df["Year"].min(), df["Year"].max()],
                            ),
                        ], md=4, style={"paddingTop": "8px"}),
                    ]),
                ], className="filter-panel"),
                html.Div(id="explore-charts"),
                html.Div(id="explore-table"),
            ])

        # ── Yield Predictor ───────────────────────────────────────────────────
        elif tab == "tab-predict":
            country_opts = [{"label": c, "value": c} for c in sorted(df["Area"].unique())]
            crop_opts    = [{"label": c, "value": c} for c in sorted(df["Item"].unique())]
            return html.Div([
                page_header("Yield Predictor",
                            "Enter crop and climate parameters to predict yield using the trained model"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div("Country", className="filter-label"),
                            dcc.Dropdown(id="p-country", options=country_opts,
                                         value="India", clearable=False,
                                         style={"fontSize": "13px"}),
                        ], md=3),
                        dbc.Col([
                            html.Div("Crop Type", className="filter-label"),
                            dcc.Dropdown(id="p-crop", options=crop_opts,
                                         value="Wheat", clearable=False,
                                         style={"fontSize": "13px"}),
                        ], md=3),
                        dbc.Col([
                            html.Div("Year", className="filter-label"),
                            dcc.Input(id="p-year", type="number", value=2023,
                                      min=1990, max=2100, className="form-control",
                                      style={"fontSize": "13px"}),
                        ], md=2),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Rainfall (mm / year)", className="filter-label"),
                            dcc.Slider(id="p-rain", min=50, max=3250, step=10, value=1000,
                                       marks={50:"50",1000:"1000",2000:"2000",3250:"3250"},
                                       tooltip={"placement":"bottom","always_visible":True}),
                        ], md=6),
                        dbc.Col([
                            html.Div("Avg Temperature (°C)", className="filter-label"),
                            dcc.Slider(id="p-temp", min=1, max=31, step=0.5, value=20,
                                       marks={1:"1°C",10:"10°C",20:"20°C",31:"31°C"},
                                       tooltip={"placement":"bottom","always_visible":True}),
                        ], md=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Pesticides (tonnes)", className="filter-label"),
                            dcc.Slider(id="p-pest", min=0, max=400000, step=500, value=20000,
                                       marks={0:"0",100000:"100k",200000:"200k",400000:"400k"},
                                       tooltip={"placement":"bottom","always_visible":True}),
                        ], md=8),
                        dbc.Col([
                            dbc.Button("Predict Yield →", id="btn-predict",
                                       style={"background": C["accent"],
                                              "border": "none",
                                              "borderRadius": "10px",
                                              "fontWeight": "600",
                                              "fontSize": "0.95rem",
                                              "padding": "12px 28px",
                                              "marginTop": "22px",
                                              "width": "100%",
                                              "letterSpacing": "0.3px"}),
                        ], md=4),
                    ]),
                ], className="filter-panel"),
                html.Div(id="predict-result"),
            ])

        return html.Div()

    # ── Explore callback ──────────────────────────────────────────────────────
    @app.callback(
        Output("explore-charts", "children"),
        Output("explore-table",  "children"),
        Input("dd-crop",    "value"),
        Input("dd-country", "value"),
        Input("sl-year",    "value"),
    )
    def update_explore(crops_sel, countries_sel, yr):
        dff = df.copy()
        if crops_sel:     dff = dff[dff["Item"].isin(crops_sel)]
        if countries_sel: dff = dff[dff["Area"].isin(countries_sel)]
        dff = dff[(dff["Year"] >= yr[0]) & (dff["Year"] <= yr[1])]
        if dff.empty:
            return dbc.Alert("No data matches the current filters.", color="warning"), ""

        t_fig = px.line(
            dff.groupby(["Year","Item"])["yield_hg_ha"].mean().reset_index(),
            x="Year", y="yield_hg_ha", color="Item",
            labels={"yield_hg_ha": "Mean Yield (hg/ha)"},
            color_discrete_sequence=CHART_COLORS)
        t_fig.update_traces(mode="lines+markers", marker_size=4)
        t_fig.update_layout(title="Filtered: Yield Trend")
        apply_theme(t_fig)

        b_fig = px.box(dff, x="Item", y="yield_hg_ha", color="Item",
                       labels={"yield_hg_ha": "Yield (hg/ha)", "Item": ""},
                       color_discrete_sequence=CHART_COLORS)
        b_fig.update_layout(showlegend=False, xaxis_tickangle=-20,
                            title="Filtered: Yield Distribution")
        apply_theme(b_fig)

        charts = dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=t_fig,
                                       config={"displayModeBar": False},
                                       style={"height": "340px"}),
                             className="chart-card"), md=6, className="mb-4"),
            dbc.Col(html.Div(dcc.Graph(figure=b_fig,
                                       config={"displayModeBar": False},
                                       style={"height": "340px"}),
                             className="chart-card"), md=6, className="mb-4"),
        ])

        show_cols = ["Area","Item","Year","yield_hg_ha","rainfall_mm","pesticides_t","avg_temp_c"]
        tbl = dash_table.DataTable(
            data=dff[show_cols].head(200).to_dict("records"),
            columns=[{"name": c, "id": c} for c in show_cols],
            page_size=12,
            style_table={"overflowX": "auto", "borderRadius": "10px",
                         "border": f"1px solid {C['border']}"},
            style_header={"backgroundColor": C["sidebar"], "color": "#fff",
                          "fontWeight": "600", "fontSize": "12px",
                          "textTransform": "uppercase", "letterSpacing": "0.5px"},
            style_cell={"textAlign": "center", "padding": "9px 12px",
                        "fontSize": "13px", "fontFamily": FONT,
                        "border": f"1px solid {C['border']}"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#F9FBF9"},
            ],
            filter_action="native",
            sort_action="native",
        )
        table_section = html.Div([
            html.Div(f"Showing first 200 of {len(dff):,} records",
                     style={"fontSize": "0.8rem", "color": C["muted"],
                            "marginBottom": "10px"}),
            tbl,
        ], className="mt-2")
        return charts, table_section

    # ── Predict callback ──────────────────────────────────────────────────────
    @app.callback(
        Output("predict-result", "children"),
        Input("btn-predict", "n_clicks"),
        Input("p-country", "value"),
        Input("p-crop",    "value"),
        Input("p-year",    "value"),
        Input("p-rain",    "value"),
        Input("p-pest",    "value"),
        Input("p-temp",    "value"),
        prevent_initial_call=True,
    )
    def predict_yield(n, country, crop, year, rain, pest, temp):
        if not n: return ""
        try:    area_enc = le_area.transform([country])[0]
        except: area_enc = 0
        try:    item_enc = le_item.transform([crop])[0]
        except: item_enc = 0
        feat = np.array([[area_enc, item_enc, year, rain, np.log1p(pest), temp]])
        pred = np.expm1(model.predict(feat)[0])
        hist = df[(df["Area"]==country) & (df["Item"]==crop)]
        hist_mean = hist["yield_hg_ha"].mean() if not hist.empty else None
        delta = ((pred/hist_mean - 1)*100) if hist_mean else None
        delta_color = C["accent"] if delta and delta >= 0 else C["danger"]
        delta_text  = (f"+{delta:.1f}%" if delta and delta >= 0
                       else f"{delta:.1f}%" if delta else "N/A")

        result = html.Div([
            html.Div(className="predict-result-card", children=[
                dbc.Row([
                    dbc.Col([
                        html.Div("Predicted Yield",
                                 style={"fontSize":"0.75rem","fontWeight":"600",
                                        "color":C["muted"],"textTransform":"uppercase",
                                        "letterSpacing":"0.8px","marginBottom":"6px"}),
                        html.Div(f"{pred:,.0f}", className="predict-big-number"),
                        html.Div("hectograms per hectare", className="predict-sub"),
                        html.Div(f"≈ {pred/10000:.2f} tonnes / hectare",
                                 style={"fontSize":"0.82rem","color":C["muted"],
                                        "marginTop":"4px"}),
                    ], md=4),
                    dbc.Col([
                        html.Div("Input Parameters", style={"fontSize":"0.75rem",
                                  "fontWeight":"600","color":C["muted"],
                                  "textTransform":"uppercase","letterSpacing":"0.8px",
                                  "marginBottom":"12px"}),
                        *[html.Div([
                            html.Span(lbl, style={"color":C["muted"],"fontSize":"0.82rem",
                                                  "width":"140px","display":"inline-block"}),
                            html.Span(val, style={"fontWeight":"600","fontSize":"0.88rem"}),
                          ], style={"marginBottom":"6px"})
                          for lbl, val in [
                              ("Country:",     country),
                              ("Crop:",        crop),
                              ("Year:",        str(year)),
                              ("Rainfall:",    f"{rain} mm/yr"),
                              ("Pesticides:",  f"{pest:,} t"),
                              ("Temperature:", f"{temp}°C"),
                          ]],
                    ], md=4),
                    dbc.Col([
                        html.Div("vs Historical Average",
                                 style={"fontSize":"0.75rem","fontWeight":"600",
                                        "color":C["muted"],"textTransform":"uppercase",
                                        "letterSpacing":"0.8px","marginBottom":"12px"}),
                        html.Div(
                            f"{hist_mean:,.0f} hg/ha" if hist_mean else "No history",
                            style={"fontSize":"1.1rem","fontWeight":"600","color":C["text"]}),
                        html.Div("historical mean", className="predict-sub"),
                        html.Div(delta_text,
                                 style={"fontSize":"1.6rem","fontWeight":"800",
                                        "color": delta_color,"marginTop":"8px"}),
                        html.Div("vs historical", className="predict-sub"),
                    ], md=4),
                ]),
            ]),
        ], style={"marginTop": "20px"})
        return result

    return app


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  AgriYield Analytics  —  Loading & Preprocessing…")
    print("=" * 60)
    df = load_and_preprocess("yield_df.csv")
    print(f"  Clean dataset: {len(df):,} rows x {df.shape[1]} columns")
    print(f"  Countries: {df['Area'].nunique()}  |  Crops: {df['Item'].nunique()}  "
          f"|  Years: {df['Year'].min()}-{df['Year'].max()}")
    print("\n  Training Random Forest model...")
    model, fcols, metrics, feat_imp, X_test, y_true, y_pred, le_area, le_item = build_model(df)
    print("  Model trained.")
    print("\n  Model Performance:")
    for k, v in metrics.items():
        print(f"    {k:20s}: {v}")
    df_raw = df.copy()
    print("\n  Launching dashboard at http://127.0.0.1:8050")
    print("  Press CTRL+C to stop.\n")
    app = create_app(df, metrics, feat_imp, y_true, y_pred, model, le_area, le_item, df_raw)
    app.run(debug=False, host="127.0.0.1", port=8050)
