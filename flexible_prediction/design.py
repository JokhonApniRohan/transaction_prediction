# -*- coding: utf-8 -*-
"""
app_combined_enhanced.py -- Monthly Report Dashboard (v4)

Design principles (v4):
- Both tabs share an IDENTICAL visual language:
    filter_card → kpi_strip row → chart grid → section_card tables
- Customer Activity quad panel (dark bg + sparklines) replaced with
  plain kpi_strip_card tiles — same as Transaction KPI strip
- KPI Head filter removed from Transaction KPI tab
- KPI HEAD DETAIL table removed from Transaction KPI tab
- All time-series x-axes show month labels (tickangle=-30)
- Consistent chart height (300 px), margin, font everywhere
"""

import os
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, dash_table, ctx
from dotenv import load_dotenv
from sqlalchemy import text
from db import get_dept_df, get_engine

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
NAVY    = "#1a3a6b"
TEAL    = "#0d7a6b"
GOLD    = "#c8a84b"
GOLD_LT = "#fffbf0"
CREAM   = "#fdf8ee"
DARK    = "#0f1923"
DARK2   = "#1c2b3a"
WHITE   = "#ffffff"
BG      = "#f0f2f5"
MUTED   = "#6b7a90"
GREEN   = "#27ae60"
RED     = "#e53e3e"
BORDER  = "#dde1ea"
BLUE    = "#3a9bd5"
PURPLE  = "#7c5cbf"

DEPT_COLORS = {
    "B2B Sales":       "#1a3a6b",
    "Banking Sales":   "#2176ae",
    "Everyones KPI":   "#1d9e75",
    "Govt. Sales":     "#9b59b6",
    "Merchant":        "#e07b39",
    "Mobile Recharge": "#c0392b",
    "Retail Sales":    "#16a085",
}

CHART_H     = 300
SECTION_GAP = "16px"
DD          = {"fontSize": "13px"}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _xmonth():
    return dict(
        showgrid=False, zeroline=False, showline=True, linecolor=BORDER,
        tickfont=dict(size=9, color=MUTED, family="DM Sans"),
        tickangle=-30, automargin=True,
    )

def _yclean(fmt=",.0f", suffix=""):
    return dict(
        showgrid=True, gridcolor="#f0f2f5", zeroline=False,
        tickfont=dict(size=9, color=MUTED, family="DM Sans"),
        tickformat=fmt, ticksuffix=suffix,
    )

def _ctitle(text):
    return dict(text=text,
                font=dict(size=12, color=DARK2, family="Barlow Condensed"),
                x=0, xanchor="left", pad=dict(l=4))

# Base layout applied to every chart — no margin here to avoid duplicate-kwarg errors
_CBASE = dict(
    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
    height=CHART_H, autosize=True, showlegend=False,
    font=dict(family="DM Sans"),
)
# Standard margins per chart type
_M_DEFAULT = dict(l=10, r=16,  t=44, b=56)
_M_BAR_H   = dict(l=10, r=80,  t=44, b=10)
_M_PIE     = dict(l=10, r=10,  t=44, b=10)
_M_TREND   = dict(l=10, r=16,  t=56, b=56)
_M_MOM     = dict(l=10, r=16,  t=44, b=40)

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800"
        "&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,400&display=swap",
    ],
    title="Monthly Report Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
USER_SQL = """
    SELECT
        month, month_label, year, month_num,
        active_unique_customers,
        active_mom_change, active_mom_growth_pct,
        recurring_customers, recurring_rate,
        recurring_mom_change, recurring_mom_growth_pct,
        churn_back_customers, churn_back_rate,
        churn_back_mom_change, churn_back_mom_growth_pct,
        first_time_customers, first_time_rate,
        first_time_mom_change, first_time_mom_growth_pct,
        churn_customers, churn_rate,
        churn_mom_change, churn_mom_growth_pct,
        total_txn_count, total_txn_amount,
        avg_txn_per_customer, avg_txn_amount_per_customer,
        avg_services_per_customer_money_in,
        avg_services_per_customer_money_out,
        channel_dependency_customers, bank_dependency_customers,
        utility_dependency_customers, education_dependency_customers,
        cable_internet_dependency_customers,
        mrt_refresh_date, mart_refreshed_at
    FROM public.mrt_rp_monthly_kpi_user
    ORDER BY month
"""

def load_user_df():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(USER_SQL), conn)
        df["month"] = df["month"].astype(str)
        return df
    except Exception as e:
        print(f"[USER] ERROR: {e}")
        return pd.DataFrame()

def load_dept_df():
    try:
        df = get_dept_df()
        df["month"] = df["month"].astype(str)
        return df
    except Exception as e:
        print(f"[DEPT] ERROR: {e}")
        return pd.DataFrame()

def month_maps(df):
    if df.empty:
        return [], {}
    m = df[["month", "month_label"]].drop_duplicates().sort_values("month", ascending=False)
    return m["month"].tolist(), dict(zip(m["month"], m["month_label"]))

def get_ts(df, col="mart_refreshed_at"):
    try:
        if not df.empty and col in df.columns:
            v = df[col].max()
            if pd.notna(v):
                return pd.Timestamp(v).strftime("Refreshed: %d %b %Y  %H:%M")
    except Exception:
        pass
    return "Refreshed: --"

# initial loads
DEPT_DF = load_dept_df()
USER_DF = load_user_df()
DEPT_MONTHS, DEPT_ML = month_maps(DEPT_DF)
USER_MONTHS, USER_ML = month_maps(USER_DF)
ALL_DEPTS   = sorted(DEPT_DF["department_name"].dropna().unique().tolist()) if not DEPT_DF.empty else []
USER_LATEST = USER_MONTHS[0] if USER_MONTHS else None
DEPT_LATEST = DEPT_MONTHS[0] if DEPT_MONTHS else None

# ─────────────────────────────────────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────
CRORE = 1e7

def fmt_bdt(n):
    if pd.isna(n): return "--"
    if abs(n) >= CRORE: return f"Tk {n/CRORE:.2f} Cr"
    if abs(n) >= 1e5:   return f"Tk {n/1e5:.2f} Lac"
    if abs(n) >= 1e3:   return f"Tk {n/1e3:.1f}K"
    return f"Tk {n:,.0f}"

def fmt_num(n):
    if pd.isna(n): return "--"
    return f"{int(round(n)):,}"

def fmt_k(n):
    if pd.isna(n): return "--"
    if abs(n) >= 1e6: return f"{n/1e6:.1f}M"
    if abs(n) >= 1e3: return f"{n/1e3:.0f}K"
    return f"{n:,.0f}"

def fmt_dec(n, digits=2):
    if pd.isna(n): return "--"
    return f"{n:,.{digits}f}"

def fmt_pct(v, sign=True):
    if pd.isna(v): return "--"
    s = "+" if (v >= 0 and sign) else ""
    return f"{s}{v:.1f}%"

def pct_or_dash(part, total):
    if pd.isna(part) or pd.isna(total) or total == 0: return "--"
    return f"{(part/total)*100:.1f}%"

def pc(v):
    if pd.isna(v): return MUTED
    return GREEN if v >= 0 else RED

def dc(d):
    return DEPT_COLORS.get(d, "#888")

def first_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UI PRIMITIVES  (identical in both tabs)
# ─────────────────────────────────────────────────────────────────────────────
def make_table(df):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_header={
            "backgroundColor": NAVY, "color": WHITE,
            "fontWeight": "600", "fontSize": "11px", "border": "none",
            "letterSpacing": "0.8px", "fontFamily": "'Barlow Condensed'",
            "padding": "9px 14px", "textTransform": "uppercase",
        },
        style_cell={
            "fontSize": "12px", "padding": "8px 14px",
            "border": "1px solid #f0f2f5", "textAlign": "left",
            "color": DARK2, "whiteSpace": "normal", "height": "auto",
            "fontFamily": "'DM Sans', sans-serif",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}],
        style_table={"overflowX": "auto"},
        page_size=15, sort_action="native",
    )

def section_card(title, child):
    """Navy-header card wrapping any child — used for every table in both tabs."""
    return dbc.Card([
        html.Div(title, style={
            "fontFamily": "'Barlow Condensed'", "fontWeight": "700",
            "fontSize": "11px", "letterSpacing": "2px", "color": WHITE,
            "background": NAVY, "padding": "10px 16px", "textTransform": "uppercase",
        }),
        html.Div(child),
    ], style={
        "borderRadius": "10px", "border": "none",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
        "overflow": "hidden", "height": "100%",
    })

def chart_card(graph_id):
    """White bordered card wrapping a dcc.Graph — used for every chart in both tabs."""
    return dbc.Card(
        dcc.Graph(id=graph_id,
                  config={"displayModeBar": False, "responsive": True},
                  style={"height": f"{CHART_H}px"}),
        style={
            "border": f"1px solid {BORDER}", "borderRadius": "10px",
            "padding": "2px", "background": WHITE,
            "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "height": "100%",
        },
    )

def filter_card(row_children):
    """Filter bar card — identical wrapper in both tabs."""
    return dbc.Card(
        dbc.CardBody(dbc.Row(row_children, align="end", className="g-2"),
                     style={"padding": "12px 16px"}),
        style={
            "border": f"1px solid {BORDER}", "borderRadius": "10px",
            "marginBottom": SECTION_GAP, "background": WHITE,
            "boxShadow": "0 1px 4px rgba(0,0,0,0.05)",
        },
    )

def label_wrap(text, control):
    return html.Div([
        html.Label(text, style={
            "fontSize": "8px", "fontWeight": "700", "color": MUTED,
            "letterSpacing": "1.5px", "marginBottom": "4px",
            "display": "block", "textTransform": "uppercase",
        }),
        control,
    ])

# ─────────────────────────────────────────────────────────────────────────────
# KPI STRIP CARD  (the single card component used in BOTH tabs)
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub_value=None, mom=None, sub_lbl="MoM",
             accent=NAVY, wide=False):
    """
    Compact metric tile with:
      - label (small caps)
      - large value
      - optional sub_value line (e.g. count alongside a rate)
      - optional MoM badge
    accent colour drives the left border.
    wide=True stretches to md=4 instead of md=3.
    """
    mom_badge = html.Span(
        [
            html.Span(fmt_pct(mom),
                      style={"color": pc(mom), "fontWeight": "700", "fontSize": "11px"}),
            html.Span(f" {sub_lbl}", style={"color": MUTED, "fontSize": "10px"}),
        ],
        style={"display": "inline-flex", "alignItems": "center", "gap": "2px",
               "marginTop": "4px"},
    ) if (mom is not None and not pd.isna(mom)) else html.Span()

    sub_line = html.Div(sub_value,
                        style={"fontSize": "11px", "color": MUTED, "marginTop": "2px"}
                        ) if sub_value else html.Div()

    return dbc.Card(
        dbc.CardBody([
            html.P(label, style={
                "fontSize": "9px", "color": MUTED, "marginBottom": "4px",
                "textTransform": "uppercase", "letterSpacing": "1px", "fontWeight": "600",
            }),
            html.Div(value, style={
                "fontSize": "22px", "fontWeight": "800", "color": DARK2,
                "fontFamily": "'Barlow Condensed'", "lineHeight": "1.1",
            }),
            sub_line,
            mom_badge,
        ], style={"padding": "14px 16px"}),
        style={
            "borderLeft": f"4px solid {accent}",
            "borderTop": "none", "borderRight": "none", "borderBottom": "none",
            "borderRadius": "10px",
            "background": WHITE,
            "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
            "height": "100%",
        },
    )

# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT CARD  (gold panel — Customer Activity only)
# ─────────────────────────────────────────────────────────────────────────────
def insight_card(text_id):
    return dbc.Card(
        dbc.CardBody(
            html.P(id=text_id, style={
                "fontSize": "12px", "color": DARK2, "lineHeight": "1.9",
                "margin": "0", "fontStyle": "italic",
            }),
            style={"padding": "16px 18px"},
        ),
        style={
            "borderRadius": "10px", "background": GOLD_LT,
            "border": f"1.5px solid {GOLD}",
            "boxShadow": "0 3px 12px rgba(200,168,75,0.10)", "height": "100%",
        },
    )

# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _bar_text_style(vals, threshold_pct=0.22):
    """
    Universal helper used by every bar chart.

    Rule: abs(bar) >= 22% of the largest abs bar  →  INSIDE  (white text)
          smaller bars                             →  OUTSIDE (dark text)

    Works for horizontal bars (all positive) AND vertical MoM bars
    (mixed +/-) because we compare abs(val) against the threshold.
    """
    if not vals:
        return [], []
    max_abs   = max(abs(v) for v in vals) or 1
    threshold = max_abs * threshold_pct
    positions = ["inside"  if abs(v) >= threshold else "outside" for v in vals]
    text_col  = [WHITE     if p == "inside"       else DARK2     for p in positions]
    return positions, text_col


def empty_fig(msg="No data available"):
    fig = go.Figure()
    fig.update_layout(
        height=CHART_H, plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False,
                          font=dict(color=MUTED, size=13))],
    )
    return fig

def build_active_trend(df_s):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_s["month_label"],
        y=df_s["active_unique_customers"],
        mode="lines+markers+text",
        line=dict(color=TEAL, width=2.5),
        marker=dict(color=WHITE, size=8, line=dict(color=TEAL, width=2.5)),
        text=[fmt_k(v) for v in df_s["active_unique_customers"]],
        textposition="top center",
        textfont=dict(size=9, color=DARK2, family="Barlow Condensed"),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **_CBASE, margin=_M_DEFAULT,
        title=_ctitle("Active Customers — Monthly Trend"),
        xaxis=_xmonth(),
        yaxis=dict(visible=False),
    )
    return fig

def build_churn_trend(df_s):
    """
    Dual-axis chart:
      Left  Y → customer composition bars: Recurring / First-time / Churn-back
      Right Y → Churn Rate line (different denominator — % of prior-month customers lost)
    """
    fig = go.Figure()

    x = df_s["month_label"].tolist()

    # ── Composition bars (left axis) ──────────────────────────────────────
    composition = [
        ("recurring_rate",  "Recurring",  TEAL,  "y1"),
        ("first_time_rate", "First-time", BLUE,  "y1"),
        ("churn_back_rate", "Churn-back", GOLD,  "y1"),
    ]
    for col, name, color in [c[:3] for c in composition]:
        if col not in df_s.columns:
            continue
        y_vals = (df_s[col].fillna(0) * 100).tolist()
        fig.add_trace(go.Bar(
            x=x, y=y_vals, name=name,
            marker_color=color,
            opacity=0.82,
            yaxis="y1",
            hovertemplate=f"{name}: %{{y:.1f}}%<extra></extra>",
        ))

    # ── Churn Rate line (right axis, dashed red) ──────────────────────────
    if "churn_rate" in df_s.columns:
        churn_y = (df_s["churn_rate"].fillna(0) * 100).tolist()
        fig.add_trace(go.Scatter(
            x=x, y=churn_y, name="Churn Rate",
            mode="lines+markers",
            line=dict(color=RED, width=2.5, dash="dot"),
            marker=dict(size=6, color=RED, symbol="diamond"),
            yaxis="y2",
            hovertemplate="Churn Rate: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **{**_CBASE, "showlegend": True},
        barmode="stack",
        margin=dict(l=10, r=52, t=56, b=56),
        title=_ctitle("Customer Segment Composition & Churn Rate"),
        xaxis=_xmonth(),
        yaxis=dict(
            title=None,
            ticksuffix="%",
            tickformat=".0f",
            showgrid=True,
            gridcolor="#f0f2f5",
            zeroline=False,
            tickfont=dict(size=9, color=MUTED, family="DM Sans"),
            range=[0, 105],
        ),
        yaxis2=dict(
            title=None,
            ticksuffix="%",
            tickformat=".0f",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=9, color=RED, family="DM Sans"),
            range=[0, 105],
        ),
        legend=dict(
            font=dict(size=9, family="DM Sans"),
            orientation="h",
            yanchor="bottom", y=1.02, x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig

def build_dept_trend(tdf, dept, metric, yl):
    fig = go.Figure()
    if dept == "ALL":
        for d in sorted(tdf["department_name"].dropna().unique()):
            td = (tdf[tdf["department_name"] == d]
                  .groupby(["month", "month_label"], dropna=False)[metric]
                  .sum().reset_index().sort_values("month"))
            fig.add_trace(go.Scatter(
                x=td["month_label"], y=td[metric], name=d,
                mode="lines+markers",
                line=dict(color=dc(d), width=2),
                marker=dict(size=5, color=dc(d)),
                hovertemplate="%{x}: %{y:,}<extra>" + d + "</extra>",
            ))
    else:
        if "kpi_head" in tdf.columns:
            td = (tdf.groupby(["month", "month_label", "kpi_head"], dropna=False)[metric]
                  .sum().reset_index().sort_values("month"))
            for k in td["kpi_head"].dropna().unique():
                kd = td[td["kpi_head"] == k]
                fig.add_trace(go.Scatter(
                    x=kd["month_label"], y=kd[metric], name=str(k),
                    mode="lines+markers", line=dict(width=2), marker=dict(size=5),
                    hovertemplate="%{x}: %{y:,}<extra>" + str(k) + "</extra>",
                ))
        else:
            td = (tdf.groupby(["month", "month_label"], dropna=False)[metric]
                  .sum().reset_index().sort_values("month"))
            fig.add_trace(go.Scatter(
                x=td["month_label"], y=td[metric], name=dept,
                mode="lines+markers", line=dict(width=2), marker=dict(size=5),
            ))
    fig.update_layout(
        **{**_CBASE, "showlegend": True}, margin=_M_TREND,
        title=_ctitle(f"{yl} Trend — Last 12 Months"),
        xaxis=_xmonth(), yaxis=_yclean(),
        legend=dict(font=dict(size=9), orientation="h",
                    yanchor="bottom", y=1.02, x=0),
    )
    return fig

def build_dept_bar(bd, metric, fn, yl, ml):
    vals = bd[metric].tolist() if metric in bd.columns else [0] * len(bd)
    text_positions, text_colors = _bar_text_style(vals)

    fig = go.Figure(go.Bar(
        x=vals,
        y=bd["label"], orientation="h",
        marker_color=bd["_color"],
        text=[fn(v) for v in vals],
        textposition=text_positions,
        textfont=dict(size=11, family="Barlow Condensed", color=text_colors),
        hovertemplate="%{y}: %{x:,}<extra></extra>",
        cliponaxis=False,
    ))
    fig.update_layout(
        **_CBASE, margin=dict(l=10, r=100, t=44, b=10),
        title=_ctitle(f"{yl} by KPI Head  |  {ml}"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=9, family="DM Sans")),
    )
    return fig

def build_dept_pie(pie_grp, ml):
    fig = go.Figure(go.Pie(
        labels=pie_grp["department_name"],
        values=pie_grp["txn_amount"],
        hole=0.52,
        marker_colors=[dc(d) for d in pie_grp["department_name"]],
        textinfo="label+percent",
        textfont=dict(size=10, family="DM Sans"),
        hovertemplate="%{label}: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_CBASE, margin=_M_PIE,
        title=_ctitle(f"Txn Amount Contribution  |  {ml}"),
    )
    return fig

def build_mom_bar(mom_grp, mom_col, ml):
    fig = go.Figure()
    if not mom_grp.empty and mom_col and mom_col in mom_grp.columns:
        vals   = mom_grp[mom_col].tolist()
        colors = [GREEN if v >= 0 else RED for v in vals]
        texts  = [fmt_pct(v) for v in vals]

        # Same threshold rule as every other bar chart
        text_positions, text_colors = _bar_text_style(vals)

        fig.add_trace(go.Bar(
            x=mom_grp["department_name"],
            y=vals,
            marker_color=colors,
            text=texts,
            textposition=text_positions,
            textfont=dict(size=11, family="Barlow Condensed", color=text_colors),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            cliponaxis=False,
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)

        # Pad y-axis so no label is ever clipped
        ymin    = min(vals)
        ymax    = max(vals)
        pad     = max(abs(ymin), abs(ymax)) * 0.14
        y_range = [ymin - pad, ymax + pad if ymax > 0 else pad]
    else:
        y_range = [-10, 10]

    fig.update_layout(
        **_CBASE, margin=dict(l=10, r=16, t=44, b=54),
        title=_ctitle(f"MoM Growth %  |  {ml}"),
        xaxis=dict(showgrid=False,
                   tickfont=dict(size=9, color=MUTED, family="DM Sans"),
                   tickangle=-30, automargin=True),
        yaxis=dict(
            range=y_range,
            ticksuffix="%", tickformat=".0f",
            showgrid=True, gridcolor="#f0f2f5", zeroline=False,
            tickfont=dict(size=9, color=MUTED, family="DM Sans"),
        ),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    fluid=True,
    style={"background": BG, "minHeight": "100vh", "padding": "0",
           "fontFamily": "'DM Sans', sans-serif"},
    children=[

        # ── TOP NAV ──────────────────────────────────────────────────────────
        html.Div(style={"background": DARK}, children=[
            dbc.Container(fluid=True, style={"padding": "11px 28px"}, children=[
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Span("Monthly Report Dashboard", style={
                            "color": WHITE, "fontFamily": "'Barlow Condensed'",
                            "fontWeight": "700", "fontSize": "22px",
                        }),
                        html.Span("  |  Analytics", style={
                            "color": GOLD, "fontSize": "11px", "marginLeft": "10px",
                        }),
                    ])),
                    dbc.Col(html.Div(id="hdr-ts", style={
                        "color": "rgba(255,255,255,.4)", "fontSize": "10px",
                        "textAlign": "right", "fontStyle": "italic",
                    }), width="auto", style={"display": "flex", "alignItems": "center"}),
                ], align="center"),
            ]),
            # Tab strip
            html.Div(style={
                "background": DARK2, "borderTop": "1px solid #1e2e3e",
                "padding": "0 28px", "display": "flex",
            }, children=[
                html.Button("Customer Activity", id="btn-user", style={
                    "fontFamily": "'Barlow Condensed'", "fontWeight": "700",
                    "fontSize": "12px", "letterSpacing": "1.5px", "color": TEAL,
                    "background": "transparent", "border": "none",
                    "borderBottom": f"3px solid {TEAL}",
                    "padding": "10px 22px", "textTransform": "uppercase", "cursor": "pointer",
                }),
                html.Button("Transaction KPI", id="btn-dept", style={
                    "fontFamily": "'Barlow Condensed'", "fontWeight": "700",
                    "fontSize": "12px", "letterSpacing": "1.5px", "color": "#6688aa",
                    "background": "transparent", "border": "none",
                    "borderBottom": "3px solid transparent",
                    "padding": "10px 22px", "textTransform": "uppercase", "cursor": "pointer",
                }),
            ]),
        ]),

        # ── PAGE BODY ─────────────────────────────────────────────────────────
        dbc.Container(fluid=True, style={"padding": "20px 28px"}, children=[

            # ==================================================================
            # TAB 1 — CUSTOMER ACTIVITY
            # ==================================================================
            html.Div(id="tab-user", children=[

                # Filter row
                filter_card([
                    dbc.Col(label_wrap("Month", dcc.Dropdown(
                        id="u-month",
                        options=[{"label": USER_ML.get(m, m), "value": m} for m in USER_MONTHS],
                        value=USER_LATEST, clearable=False, style=DD,
                    )), md=2, xs=12),
                    dbc.Col(html.Div(id="u-title", style={
                        "fontFamily": "'Barlow Condensed'", "fontSize": "17px",
                        "fontWeight": "700", "color": DARK2,
                        "borderLeft": f"4px solid {TEAL}", "paddingLeft": "14px",
                        "display": "flex", "alignItems": "center",
                    }), md=7, xs=12),
                    dbc.Col(html.Div(id="u-ts", style={
                        "fontSize": "10px", "color": MUTED, "textAlign": "right",
                        "fontStyle": "italic", "display": "flex",
                        "alignItems": "center", "justifyContent": "flex-end",
                    }), md=3, xs=12),
                ]),

                # Row 1: 2 hero KPI cards + insight
                dbc.Row([
                    dbc.Col(html.Div(id="u-kpi-act"),  md=3, xs=6),
                    dbc.Col(html.Div(id="u-kpi-ft"),   md=3, xs=6),
                    dbc.Col(insight_card("u-ins"),      md=6, xs=12),
                ], className="g-3", style={"marginBottom": SECTION_GAP}),

                # Row 2: 4 segment rate KPI cards
                dbc.Row(id="u-seg-row", className="g-3",
                        style={"marginBottom": SECTION_GAP}),

                # Row 3: Active trend + Segment rate trend
                dbc.Row([
                    dbc.Col(chart_card("u-tr"),   md=6, xs=12),
                    dbc.Col(chart_card("u-seg"),  md=6, xs=12),
                ], className="g-3", style={"marginBottom": SECTION_GAP}),

                # Row 4: 2 detail tables
                dbc.Row([
                    dbc.Col(html.Div(id="u-tbl1"), md=6, xs=12),
                    dbc.Col(html.Div(id="u-tbl2"), md=6, xs=12),
                ], className="g-3", style={"marginBottom": "24px"}),
            ]),

            # ==================================================================
            # TAB 2 — TRANSACTION KPI
            # ==================================================================
            html.Div(id="tab-dept", style={"display": "none"}, children=[

                # Filter row  (no KPI Head filter)
                filter_card([
                    dbc.Col(label_wrap("Month", dcc.Dropdown(
                        id="d-month",
                        options=[{"label": DEPT_ML.get(m, m), "value": m} for m in DEPT_MONTHS],
                        value=DEPT_LATEST, clearable=False, style=DD,
                    )), md=2, xs=12),
                    dbc.Col(label_wrap("Department", dcc.Dropdown(
                        id="d-dept",
                        options=[{"label": "All Departments", "value": "ALL"}] +
                                [{"label": d, "value": d} for d in ALL_DEPTS],
                        value="ALL", clearable=False, style=DD,
                    )), md=3, xs=12),
                    dbc.Col(label_wrap("Metric", dcc.Dropdown(
                        id="d-metric",
                        options=[
                            {"label": "Txn Amount (BDT)", "value": "txn_amount"},
                            {"label": "Txn Count",        "value": "txn_count"},
                            {"label": "Customer Count",   "value": "customer_count"},
                        ],
                        value="txn_amount", clearable=False, style=DD,
                    )), md=2, xs=12),
                    dbc.Col(html.Div(id="d-ts", style={
                        "fontSize": "10px", "color": MUTED, "textAlign": "right",
                        "fontStyle": "italic", "display": "flex",
                        "alignItems": "flex-end", "justifyContent": "flex-end",
                        "paddingBottom": "4px",
                    }), md=5, xs=12),
                ]),

                # Row 1: 4 KPI cards
                html.Div(id="d-kpi", style={"marginBottom": SECTION_GAP}),

                # Row 2: bar + pie
                dbc.Row([
                    dbc.Col(chart_card("d-bar"), md=6, xs=12),
                    dbc.Col(chart_card("d-pie"), md=6, xs=12),
                ], className="g-3", style={"marginBottom": SECTION_GAP}),

                # Row 3: MoM bar + trend
                dbc.Row([
                    dbc.Col(chart_card("d-mom"), md=6, xs=12),
                    dbc.Col(chart_card("d-tr"),  md=6, xs=12),
                ], className="g-3", style={"marginBottom": SECTION_GAP}),

                # Row 4: Department summary table (full width) + YTD table
                dbc.Row([
                    dbc.Col(html.Div(id="d-tbl1"), md=8, xs=12),
                    dbc.Col(html.Div(id="d-tbl2"), md=4, xs=12),
                ], className="g-3", style={"marginBottom": "24px"}),
            ]),
        ]),

        dcc.Store(id="store"),
        dcc.Interval(id="interval", interval=60 * 60 * 1000, n_intervals=0),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-user", "style"), Output("tab-dept", "style"),
    Output("btn-user", "style"), Output("btn-dept", "style"),
    Input("btn-user",  "n_clicks"), Input("btn-dept", "n_clicks"),
)
def switch_tab(_, __):
    triggered = ctx.triggered_id or "btn-user"
    base = {
        "fontFamily": "'Barlow Condensed'", "fontWeight": "700",
        "fontSize": "12px", "letterSpacing": "1.5px",
        "background": "transparent", "border": "none",
        "padding": "10px 22px", "textTransform": "uppercase", "cursor": "pointer",
    }
    if triggered == "btn-dept":
        return (
            {"display": "none"}, {"display": "block"},
            {**base, "color": "#6688aa", "borderBottom": "3px solid transparent"},
            {**base, "color": GOLD,      "borderBottom": f"3px solid {GOLD}"},
        )
    return (
        {"display": "block"}, {"display": "none"},
        {**base, "color": TEAL, "borderBottom": f"3px solid {TEAL}"},
        {**base, "color": "#6688aa", "borderBottom": "3px solid transparent"},
    )


@app.callback(
    Output("store",  "data"), Output("hdr-ts", "children"),
    Input("interval", "n_intervals"),
)
def reload(_):
    global DEPT_DF, USER_DF, DEPT_MONTHS, DEPT_ML
    global USER_MONTHS, USER_ML, ALL_DEPTS, USER_LATEST, DEPT_LATEST
    try:
        DEPT_DF = load_dept_df();  USER_DF = load_user_df()
        DEPT_MONTHS, DEPT_ML = month_maps(DEPT_DF)
        USER_MONTHS, USER_ML = month_maps(USER_DF)
        ALL_DEPTS   = sorted(DEPT_DF["department_name"].dropna().unique().tolist()) if not DEPT_DF.empty else []
        USER_LATEST = USER_MONTHS[0] if USER_MONTHS else None
        DEPT_LATEST = DEPT_MONTHS[0] if DEPT_MONTHS else None
        return {}, get_ts(USER_DF)
    except Exception as e:
        return {}, f"Reload error: {e}"


# ── CUSTOMER ACTIVITY ─────────────────────────────────────────────────────────
@app.callback(
    Output("u-title",   "children"),
    Output("u-ts",      "children"),
    Output("u-kpi-act", "children"),
    Output("u-kpi-ft",  "children"),
    Output("u-ins",     "children"),
    Output("u-seg-row", "children"),
    Output("u-tr",      "figure"),
    Output("u-seg",     "figure"),
    Output("u-tbl1",    "children"),
    Output("u-tbl2",    "children"),
    Input("u-month",    "value"),
    Input("store",      "data"),
)
def upd_user(month, _):
    ef  = empty_fig()
    emp = html.P("No data available. Check DB connection.",
                 style={"color": MUTED, "padding": "16px", "fontStyle": "italic"})

    if USER_DF.empty:
        return "--", "DB error", emp, emp, "No data.", [], ef, ef, emp, emp
    if not month:
        return "--", "--", emp, emp, "Select a month.", [], ef, ef, emp, emp

    df  = USER_DF.copy()
    row = df[df["month"] == month]
    if row.empty:
        return (f"No data for {month}", "--", emp, emp,
                f"No data for {month}.", [], ef, ef, emp, emp)

    r      = row.iloc[0]
    mlabel = USER_ML.get(month, month)
    ts     = get_ts(df)

    prev = df[df["month"] < month].sort_values("month")
    pl   = USER_ML.get(prev.iloc[-1]["month"], "prev month") if not prev.empty else "prev month"
    pa   = r["active_unique_customers"] - (0 if pd.isna(r["active_mom_change"]) else r["active_mom_change"])
    pf   = r["first_time_customers"]    - (0 if pd.isna(r["first_time_mom_change"]) else r["first_time_mom_change"])

    def _r(col):
        v = r.get(col)
        return 0 if (v is None or pd.isna(v)) else v

    rec_p = _r("recurring_rate")   * 100
    ft_p  = _r("first_time_rate")  * 100
    cb_p  = _r("churn_back_rate")  * 100
    ch_p  = _r("churn_rate")       * 100
    cd    = "reduced"   if _r("churn_mom_growth_pct")      < 0 else "increased"
    cbd   = "increased" if _r("churn_back_mom_growth_pct") > 0 else "decreased"

    insight = (
        f"Active customer base at {fmt_k(r['active_unique_customers'])} "
        f"({fmt_pct(r['active_mom_growth_pct'])} MoM). "
        f"Recurring loyalty at {rec_p:.0f}% — "
        f"{'strong retention signal' if rec_p >= 60 else 'retention needs attention'}. "
        f"First-time customers at {ft_p:.0f}% — "
        f"{'healthy acquisition' if ft_p >= 20 else 'acquisition pressure'}. "
        f"Churn-back {cbd} to {cb_p:.0f}%, churn rate {cd} to {ch_p:.0f}%."
    )

    # Hero KPI cards (same kpi_card component as Transaction KPI)
    card_act = kpi_card(
        "Monthly Active Customers",
        fmt_k(r["active_unique_customers"]),
        sub_value=f"Vs {fmt_k(pa)} in {pl}",
        mom=r["active_mom_growth_pct"],
        accent=TEAL,
    )
    card_ft = kpi_card(
        "Onboard (First-time) Customers",
        fmt_k(r["first_time_customers"]),
        sub_value=f"Vs {fmt_k(pf)} in {pl}",
        mom=r["first_time_mom_growth_pct"],
        accent=GOLD,
    )

    # Segment rate KPI strip (4 cards — same style as Transaction KPI strip)
    seg_row = [
        dbc.Col(kpi_card(
            "Recurring Rate",
            f"{rec_p:.0f}%",
            sub_value=fmt_num(r["recurring_customers"]) + " customers",
            mom=r["recurring_mom_growth_pct"],
            accent=TEAL,
        ), md=3, xs=6),
        dbc.Col(kpi_card(
            "First-time Rate",
            f"{ft_p:.0f}%",
            sub_value=fmt_num(r["first_time_customers"]) + " customers",
            mom=r["first_time_mom_growth_pct"],
            accent=BLUE,
        ), md=3, xs=6),
        dbc.Col(kpi_card(
            "Churn-back Rate",
            f"{cb_p:.0f}%",
            sub_value=fmt_num(r["churn_back_customers"]) + " customers",
            mom=r["churn_back_mom_growth_pct"],
            accent=GOLD,
        ), md=3, xs=6),
        dbc.Col(kpi_card(
            "Churn Rate",
            f"{ch_p:.0f}%",
            sub_value=fmt_num(r["churn_customers"]) + " customers",
            mom=r["churn_mom_growth_pct"],
            accent=RED,
        ), md=3, xs=6),
    ]

    # Charts
    df_t      = df[df["month"] <= month].sort_values("month").tail(15)
    fig_trend = build_active_trend(df_t)
    fig_seg   = build_churn_trend(df_t)

    # Table 1 — Dependency Breakdown
    dep_rows = []
    for lbl, col in [
        ("Channel Dependency", "channel_dependency_customers"),
        ("Bank Dependency",    "bank_dependency_customers"),
        ("Utility",            "utility_dependency_customers"),
        ("Education",          "education_dependency_customers"),
        ("Cable / Internet",   "cable_internet_dependency_customers"),
    ]:
        val = r.get(col)
        dep_rows.append({
            "Category":    lbl,
            "Customers":   fmt_k(val),
            "% of Active": pct_or_dash(val, r["active_unique_customers"]),
        })
    tbl1 = section_card("CUSTOMER DEPENDENCY BREAKDOWN",
                        make_table(pd.DataFrame(dep_rows)))

    # Table 2 — Performance Detail
    perf_rows = [
        ("Active Unique Customers",             fmt_num(r["active_unique_customers"]),                        fmt_pct(r["active_mom_growth_pct"])),
        ("Recurring Customers",                 fmt_num(r["recurring_customers"]),                            fmt_pct(r["recurring_mom_growth_pct"])),
        ("First-time Customers",                fmt_num(r["first_time_customers"]),                           fmt_pct(r["first_time_mom_growth_pct"])),
        ("Churn-back Customers",                fmt_num(r["churn_back_customers"]),                           fmt_pct(r["churn_back_mom_growth_pct"])),
        ("Churn Customers",                     fmt_num(r["churn_customers"]),                                fmt_pct(r["churn_mom_growth_pct"])),
        ("Total Txn Count",                     fmt_num(r["total_txn_count"]),                                "--"),
        ("Total Txn Amount",                    fmt_bdt(r["total_txn_amount"]),                               "--"),
        ("Avg Txn per Customer",                fmt_dec(r["avg_txn_per_customer"]),                           "--"),
        ("Avg Txn Amount per Customer",         fmt_bdt(r["avg_txn_amount_per_customer"]),                    "--"),
        ("Avg Services / Customer (Money In)",  fmt_dec(r["avg_services_per_customer_money_in"],  4),         "--"),
        ("Avg Services / Customer (Money Out)", fmt_dec(r["avg_services_per_customer_money_out"], 4),         "--"),
    ]
    tbl2 = section_card("CUSTOMER PERFORMANCE DETAIL",
                        make_table(pd.DataFrame(perf_rows,
                                                columns=["Metric", "Value", "MoM Growth"])))

    return (
        f"Customer Activity  |  {mlabel}", ts,
        card_act, card_ft, insight,
        seg_row,
        fig_trend, fig_seg,
        tbl1, tbl2,
    )


# ── TRANSACTION KPI ───────────────────────────────────────────────────────────
@app.callback(
    Output("d-kpi",  "children"),
    Output("d-bar",  "figure"),
    Output("d-pie",  "figure"),
    Output("d-mom",  "figure"),
    Output("d-tr",   "figure"),
    Output("d-tbl1", "children"),
    Output("d-tbl2", "children"),
    Output("d-ts",   "children"),
    Input("d-month",  "value"),
    Input("d-dept",   "value"),
    Input("d-metric", "value"),
    Input("store",    "data"),
)
def upd_dept(month, dept, metric, _):
    ef  = empty_fig()
    emp = html.P("No data. Check DB connection.",
                 style={"color": RED, "padding": "16px"})

    try:
        if DEPT_DF.empty:
            return emp, ef, ef, ef, ef, emp, emp, "--"

        df = DEPT_DF.copy()
        df["month"] = df["month"].astype(str)
        ml     = DEPT_ML.get(month, month)
        ts_str = get_ts(DEPT_DF)

        df_f = df if dept == "ALL" else df[df["department_name"] == dept]
        df_m = df_f[df_f["month"] == month] if month else df_f

        if df_m.empty:
            return emp, ef, ef, ef, ef, emp, emp, ts_str

        yl = {"txn_amount": "Txn Amount",
              "txn_count":  "Txn Count",
              "customer_count": "Customers"}.get(metric, metric)

        amount_mom_col   = first_col(df_m, ["txn_amount_mom_pct",     "txn_amount_mom_growth_pct"])
        count_mom_col    = first_col(df_m, ["txn_count_mom_pct",      "txn_count_mom_growth_pct"])
        customer_mom_col = first_col(df_m, ["customer_count_mom_pct", "customer_mom_growth_pct",
                                             "customer_count_mom_growth_pct"])
        ytd_yoy_col      = first_col(df_m, ["ytd_txn_amount_yoy_pct", "ytd_txn_amount_yoy_growth_pct"])
        sel_mom_col      = first_col(df_m, [f"{metric}_mom_pct",      f"{metric}_mom_growth_pct"])

        ta  = df_m["txn_amount"].sum()     if "txn_amount"     in df_m.columns else None
        tc  = df_m["txn_count"].sum()      if "txn_count"      in df_m.columns else None
        tcu = df_m["customer_count"].sum() if "customer_count" in df_m.columns else None
        yta = df_m["ytd_txn_amount"].sum() if "ytd_txn_amount" in df_m.columns else None
        ma  = df_m[amount_mom_col].mean()   if amount_mom_col   else None
        mc_ = df_m[count_mom_col].mean()    if count_mom_col    else None
        mcu = df_m[customer_mom_col].mean() if customer_mom_col else None
        yoy = df_m[ytd_yoy_col].mean()      if ytd_yoy_col      else None

        # KPI strip — same kpi_card component as Customer Activity
        kpi_strip = dbc.Row([
            dbc.Col(kpi_card("Txn Amount",  fmt_bdt(ta),  mom=ma,  accent=NAVY),   md=3, xs=6),
            dbc.Col(kpi_card("Txn Count",   fmt_num(tc),  mom=mc_, accent=TEAL),   md=3, xs=6),
            dbc.Col(kpi_card("Customers",   fmt_num(tcu), mom=mcu, accent=BLUE),   md=3, xs=6),
            dbc.Col(kpi_card("YTD Amount",  fmt_bdt(yta), mom=yoy, sub_lbl="YoY",
                              accent=GOLD), md=3, xs=6),
        ], className="g-3")

        fn = fmt_bdt if metric == "txn_amount" else fmt_num

        # Bar grouping
        grp_keys = (["kpi_head"] if dept != "ALL" else ["department_name", "kpi_head"]) \
                   if "kpi_head" in df_m.columns else ["department_name"]
        num_cols = [c for c in ["customer_count", "txn_count", "txn_amount"] if c in df_m.columns]
        agg = df_m.groupby(grp_keys, dropna=False)[num_cols].sum().reset_index()

        if dept == "ALL" and "kpi_head" in agg.columns:
            agg["label"]  = agg["department_name"].fillna("") + " | " + agg["kpi_head"].fillna("")
            agg["_color"] = agg["department_name"].apply(dc)
        elif "kpi_head" in agg.columns:
            agg["label"]  = agg["kpi_head"].fillna("Unknown")
            agg["_color"] = dc(dept)
        else:
            agg["label"]  = agg["department_name"].fillna("Unknown")
            agg["_color"] = agg["department_name"].apply(dc)

        bd   = agg.sort_values(metric, ascending=True).tail(15) if metric in agg.columns else agg.tail(15)
        fbar = build_dept_bar(bd, metric, fn, yl, ml)

        # Pie
        if "txn_amount" in df_m.columns:
            pig = df_m.groupby("department_name", dropna=False)["txn_amount"].sum().reset_index()
            pig = pig[pig["txn_amount"] > 0]
        else:
            pig = pd.DataFrame(columns=["department_name", "txn_amount"])
        fpie = build_dept_pie(pig, ml)

        # MoM
        mom_col = sel_mom_col or amount_mom_col
        if mom_col and mom_col in df_m.columns:
            mg = (df_m.groupby("department_name", dropna=False)[mom_col]
                  .mean().reset_index().dropna(subset=[mom_col]).sort_values(mom_col))
        else:
            mg = pd.DataFrame(columns=["department_name"])
        fmom = build_mom_bar(mg, mom_col, ml)

        # Trend
        m_avail = df_f[["month", "month_label"]].drop_duplicates().sort_values("month")
        last12  = m_avail["month"].astype(str).tolist()[-12:]
        tdf     = df_f[df_f["month"].isin(last12)]
        ftr     = build_dept_trend(tdf, dept, metric, yl)

        # Table 1 — Department Summary (wider column)
        g_cols = [c for c in ["customer_count", "txn_count", "txn_amount", "ytd_txn_amount"]
                  if c in df_m.columns]
        dt = df_m.groupby("department_name", dropna=False)[g_cols].sum(min_count=1).reset_index()
        if amount_mom_col:
            mj = (df_m.groupby("department_name", dropna=False)[amount_mom_col].mean()
                  .reset_index().rename(columns={amount_mom_col: "MoM %"}))
            dt = dt.merge(mj, on="department_name", how="left")
        if ytd_yoy_col:
            yj = (df_m.groupby("department_name", dropna=False)[ytd_yoy_col].mean()
                  .reset_index().rename(columns={ytd_yoy_col: "YTD YoY %"}))
            dt = dt.merge(yj, on="department_name", how="left")
        if "txn_amount" in dt.columns:
            tot = dt["txn_amount"].sum()
            dt["Contrib %"] = dt["txn_amount"] / tot * 100 if tot else None
        if {"txn_amount", "txn_count"}.issubset(dt.columns):
            dt["Avg Ticket"] = dt["txn_amount"] / dt["txn_count"].replace(0, pd.NA)
        dt["_s"] = dt.get("txn_amount", 0)
        dt = dt.rename(columns={
            "department_name": "Department", "customer_count": "Customers",
            "txn_count": "Txn Count", "txn_amount": "Txn Amount",
            "ytd_txn_amount": "YTD Amount",
        }).sort_values("_s", ascending=False)
        for c in ["Customers", "Txn Count"]:
            if c in dt.columns: dt[c] = dt[c].apply(fmt_num)
        for c in ["Txn Amount", "YTD Amount", "Avg Ticket"]:
            if c in dt.columns: dt[c] = dt[c].apply(fmt_bdt)
        for c in ["Contrib %", "MoM %", "YTD YoY %"]:
            if c in dt.columns: dt[c] = dt[c].apply(fmt_pct)
        dt = dt.drop(columns=["_s"], errors="ignore")
        tbl1 = section_card("DEPARTMENT SUMMARY", make_table(dt))

        # Table 2 — Avg Metrics snapshot (narrower column)
        snap_rows = [
            ("Avg Ticket Size",       fmt_bdt(df_m["avg_ticket_size"].mean())     if "avg_ticket_size"         in df_m.columns else "--"),
            ("Avg Txn / Customer",    fmt_dec(df_m["avg_txn_per_customer"].mean()) if "avg_txn_per_customer"    in df_m.columns else "--"),
            ("Avg Amt / Customer",    fmt_bdt(df_m["avg_amount_per_customer"].mean()) if "avg_amount_per_customer" in df_m.columns else "--"),
            ("Total Customers",       fmt_num(tcu)),
            ("Total Txn Count",       fmt_num(tc)),
            ("Total Txn Amount",      fmt_bdt(ta)),
            ("YTD Txn Amount",        fmt_bdt(yta)),
        ]
        snap_df = pd.DataFrame(snap_rows, columns=["Metric", "Value"])
        tbl2 = section_card("SNAPSHOT", make_table(snap_df))

        return kpi_strip, fbar, fpie, fmom, ftr, tbl1, tbl2, ts_str

    except Exception as e:
        err = html.Div([
            html.Div("Error in Transaction KPI",
                     style={"fontWeight": "700", "color": RED, "marginBottom": "6px"}),
            html.Pre(str(e), style={"whiteSpace": "pre-wrap", "fontSize": "11px", "margin": 0}),
        ], style={"padding": "16px"})
        return err, empty_fig(), empty_fig(), empty_fig(), empty_fig(), err, err, "--"


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT_COMBINED", 8053)),
        debug=False,
    )