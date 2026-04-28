import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd

from backend_prophet import predict_transaction

app = dash.Dash(__name__)

# =====================================================
# THEME CONSTANTS  (mirrored from reference design)
# =====================================================
NAVY   = "#1a3a6b"
TEAL   = "#0d7a6b"
GOLD   = "#c8a84b"
DARK   = "#0f1923"
DARK2  = "#1c2b3a"
WHITE  = "#ffffff"
BG     = "#f0f2f5"
MUTED  = "#6b7a90"
GREEN  = "#27ae60"
RED    = "#e53e3e"
BORDER = "#dde1ea"
BLUE   = "#3a9bd5"

# =====================================================
# GLOBAL CSS
# =====================================================
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Transaction Forecast</title>
    {%favicon%}
    {%css%}

    <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,400&display=swap" rel="stylesheet">

    <style>

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --navy:   #1a3a6b;
        --teal:   #0d7a6b;
        --gold:   #c8a84b;
        --dark:   #0f1923;
        --dark2:  #1c2b3a;
        --white:  #ffffff;
        --bg:     #f0f2f5;
        --muted:  #6b7a90;
        --green:  #27ae60;
        --red:    #e53e3e;
        --border: #dde1ea;
        --blue:   #3a9bd5;
        --sans:   'DM Sans', sans-serif;
        --cond:   'Barlow Condensed', sans-serif;
    }

    html, body {
        font-family: var(--sans);
        background: var(--bg);
        color: var(--dark2);
        min-height: 100vh;
    }

    /* ── SCROLL INDICATOR ── */
    #scroll-indicator {
        position: fixed;
        bottom: 28px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 500;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px;
        opacity: 1;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }

    #scroll-indicator.hidden { opacity: 0; }

    .scroll-label {
        font-family: var(--sans);
        font-size: 9px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .scroll-chevron {
        width: 26px; height: 26px;
        border-radius: 50%;
        background: var(--white);
        border: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: bounce 1.8s ease-in-out infinite;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .scroll-chevron svg {
        width: 11px; height: 11px;
        stroke: var(--muted);
        fill: none;
        stroke-width: 2.5;
        stroke-linecap: round;
        stroke-linejoin: round;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(4px); }
    }

    /* ── TOP NAV BAR ── */
    #top-nav {
        background: var(--dark);
        padding: 11px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .nav-title {
        font-family: var(--cond);
        font-weight: 700;
        font-size: 22px;
        color: var(--white);
        letter-spacing: 0.5px;
    }

    .nav-subtitle {
        color: var(--gold);
        font-size: 11px;
        margin-left: 10px;
        font-family: var(--sans);
    }

    .nav-ts {
        font-size: 10px;
        color: rgba(255,255,255,0.4);
        font-style: italic;
        font-family: var(--sans);
    }

    /* ── LAYOUT SHELL ── */
    #root-wrap {
        max-width: 1280px;
        margin: 0 auto;
        padding: 24px 28px 80px;
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    /* ── FILTER CARD ── */
    .filter-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        display: flex;
        align-items: flex-end;
        gap: 20px;
    }

    .ctrl-group {
        display: flex;
        flex-direction: column;
        gap: 4px;
        position: relative;
        z-index: 200;
    }

    .ctrl-label {
        font-family: var(--sans);
        font-size: 8px;
        font-weight: 700;
        color: var(--muted);
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* Date picker overrides */
    .SingleDatePickerInput {
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        background: var(--white) !important;
    }

    .DateInput_input {
        font-family: var(--sans) !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        color: var(--dark2) !important;
        background: transparent !important;
        border-bottom: none !important;
        padding: 7px 10px !important;
    }

    .SingleDatePicker_picker,
    .DatePickerSingle_picker,
    .DateRangePicker_picker {
        position: fixed !important;
        z-index: 99999 !important;
    }

    .ctrl-sep {
        width: 1px;
        height: 32px;
        background: var(--border);
        align-self: flex-end;
        margin-bottom: 2px;
    }

    .run-btn {
        font-family: var(--cond);
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        background: var(--navy);
        color: var(--white);
        border: none;
        padding: 9px 22px;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s;
        white-space: nowrap;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(26,58,107,0.18);
    }

    .run-btn:hover  { background: #142e58; transform: translateY(-1px); }
    .run-btn:active { transform: scale(0.97); }

    /* ── KPI ROW ── */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
    }

    /* ── KPI CARD  (left-border tile, matches reference exactly) ── */
    .kpi-card {
        background: var(--white);
        border-radius: 10px;
        border-left: 4px solid var(--navy);
        border-top: none;
        border-right: none;
        border-bottom: none;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        position: relative;
    }

    .kpi-card.kpi-max { border-left-color: var(--teal); }
    .kpi-card.kpi-min { border-left-color: var(--red);  }
    .kpi-card.kpi-avg { border-left-color: var(--gold); }

    .kpi-label {
        font-family: var(--sans);
        font-size: 9px;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-family: var(--cond);
        font-size: 28px;
        font-weight: 800;
        color: var(--dark2);
        line-height: 1.1;
        letter-spacing: 0.2px;
    }

    .kpi-sub {
        font-family: var(--sans);
        font-size: 11px;
        color: var(--muted);
        margin-top: 2px;
    }

    /* ── CHART PANEL ── */
    .chart-panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .panel-title {
        font-family: var(--cond);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--dark2);
        padding: 12px 16px 0;
    }

    /* ── BOTTOM ROW ── */
    .bottom-row {
        display: grid;
        grid-template-columns: 1fr 340px;
        gap: 14px;
        align-items: start;
    }

    /* ── TABLE PANEL  (navy-header section_card) ── */
    .table-panel {
        background: var(--white);
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        overflow: hidden;
    }

    .table-header-row {
        background: var(--navy);
        padding: 10px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .table-header-row .panel-title {
        font-family: var(--cond);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--white);
        padding: 0;
        margin: 0;
    }

    .table-count {
        font-family: var(--sans);
        font-size: 10px;
        color: rgba(255,255,255,0.55);
    }

    .scroll-box {
        max-height: 280px;
        overflow-y: auto;
    }

    .scroll-box::-webkit-scrollbar { width: 4px; }
    .scroll-box::-webkit-scrollbar-track { background: #fafbfc; }
    .scroll-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    table {
        width: 100%;
        border-collapse: collapse;
        font-family: var(--sans);
        font-size: 12px;
    }

    thead tr { background: var(--navy); }

    th {
        padding: 9px 14px;
        text-align: left;
        font-size: 11px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: var(--white);
        font-family: var(--cond);
        font-weight: 600;
        position: sticky;
        top: 0;
        background: var(--navy);
        z-index: 2;
        border: none;
    }

    td {
        padding: 8px 14px;
        border-bottom: 1px solid #f0f2f5;
        color: var(--dark2);
        font-weight: 400;
        font-size: 12px;
    }

    tbody tr:nth-child(odd)  { background: #fafbfc; }
    tbody tr:nth-child(even) { background: var(--white); }
    tbody tr:hover           { background: #f0f5ff; }
    tbody tr:last-child td   { border-bottom: none; }

    td:nth-child(2) { font-weight: 500; text-align: right; }
    th:nth-child(2) { text-align: right; }

    /* ── PIE PANEL ── */
    .pie-panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    </style>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var indicator = document.getElementById('scroll-indicator');
        function checkScroll() {
            var scrolled  = window.scrollY || document.documentElement.scrollTop;
            var maxScroll = document.documentElement.scrollHeight - window.innerHeight;
            if (maxScroll <= 0 || scrolled >= maxScroll - 10) {
                indicator.classList.add('hidden');
            } else {
                indicator.classList.remove('hidden');
            }
        }
        window.addEventListener('scroll', checkScroll);
        window.addEventListener('resize', checkScroll);
        setTimeout(checkScroll, 600);
    });
    </script>
</head>
<body>

    <!-- SCROLL INDICATOR -->
    <div id="scroll-indicator">
        <span class="scroll-label">Scroll</span>
        <div class="scroll-chevron">
            <svg viewBox="0 0 12 8">
                <polyline points="1,1 6,7 11,1"/>
            </svg>
        </div>
    </div>

    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""


# =====================================================
# LAYOUT
# =====================================================
app.layout = html.Div(children=[

    # TOP NAV BAR
    html.Div(id="top-nav", children=[
        html.Div([
            html.Span("Transaction Forecast", className="nav-title"),
            html.Span("|  Analytics", className="nav-subtitle"),
        ]),
        html.Div("Prophet-based forecasting engine", className="nav-ts"),
    ]),

    # PAGE BODY
    html.Div(id="root-wrap", children=[

        # FILTER CARD
        html.Div(className="filter-card", children=[

            html.Div(className="ctrl-group", children=[
                html.Label("Start Date", className="ctrl-label"),
                dcc.DatePickerSingle(id="start-date", display_format="YYYY-MM-DD"),
            ]),

            html.Div(className="ctrl-sep"),

            html.Div(className="ctrl-group", children=[
                html.Label("End Date", className="ctrl-label"),
                dcc.DatePickerSingle(id="end-date", display_format="YYYY-MM-DD"),
            ]),

            html.Button("Run Forecast →", id="predict-btn", className="run-btn"),
        ]),

        # KPI CARDS
        html.Div(className="kpi-row", children=[
            html.Div(id="total-output", className="kpi-card"),
            html.Div(id="max-output",   className="kpi-card kpi-max"),
            html.Div(id="min-output",   className="kpi-card kpi-min"),
            html.Div(id="avg-output",   className="kpi-card kpi-avg"),
        ]),

        # MAIN CHART
        html.Div(className="chart-panel", children=[
            html.Div("Daily Transaction Forecast", className="panel-title"),
            dcc.Graph(
                id="forecast-graph",
                style={"height": "380px"},
                config={"displayModeBar": False, "responsive": True},
            ),
        ]),

        # BOTTOM ROW: TABLE + PIE
        html.Div(className="bottom-row", children=[

            # TABLE
            html.Div(className="table-panel", children=[
                html.Div(className="table-header-row", children=[
                    html.Div("Daily Breakdown", className="panel-title"),
                    html.Div(id="table-count", className="table-count"),
                ]),
                html.Div(className="scroll-box", children=[
                    html.Table(id="forecast-table"),
                ]),
            ]),

            # PIE CHART
            html.Div(className="pie-panel", children=[
                html.Div("Transaction Share by Day", className="panel-title"),
                dcc.Graph(
                    id="pie-chart",
                    style={"height": "300px"},
                    config={"displayModeBar": False},
                ),
            ]),

        ]),
    ]),
])


# =====================================================
# CALLBACK
# =====================================================
@app.callback(
    Output("forecast-graph",  "figure"),
    Output("total-output",    "children"),
    Output("max-output",      "children"),
    Output("min-output",      "children"),
    Output("avg-output",      "children"),
    Output("forecast-table",  "children"),
    Output("table-count",     "children"),
    Output("pie-chart",       "figure"),
    Input("predict-btn",      "n_clicks"),
    State("start-date",       "date"),
    State("end-date",         "date"),
)
def update(n_clicks, start, end):

    EMPTY = go.Figure(layout=go.Layout(
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(
            text="Select a date range and click Run Forecast",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=MUTED, size=13, family="DM Sans"),
        )],
    ))

    if not n_clicks or not start or not end:
        return EMPTY, "", "", "", "", "", "", go.Figure(layout=go.Layout(
            paper_bgcolor=WHITE, plot_bgcolor=WHITE,
            margin=dict(l=0, r=0, t=0, b=0),
        ))

    df = predict_transaction(start, end)

    total   = df["y"].sum()
    avg     = df["y"].mean()
    max_row = df.loc[df["y"].idxmax()]
    min_row = df.loc[df["y"].idxmin()]
    n       = len(df)

    SANS = "DM Sans, sans-serif"
    COND = "Barlow Condensed, sans-serif"

    # ── MAIN CHART ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Bar (volume backdrop)
    fig.add_trace(go.Bar(
        x=df["ds"], y=df["y"],
        name="Volume",
        marker=dict(
            color="rgba(26,58,107,0.10)",
            line=dict(color="rgba(26,58,107,0.22)", width=1),
        ),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Transactions: %{y:,.0f}<extra></extra>",
    ))

    # Trend line — teal to match reference accent
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines+markers",
        name="Trend",
        line=dict(color=TEAL, width=2.5),
        marker=dict(color=WHITE, size=7, line=dict(color=TEAL, width=2.5)),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        height=380,
        margin=dict(l=10, r=16, t=44, b=56),
        font=dict(family=SANS, size=9, color=MUTED),
        legend=dict(
            orientation="h", y=1.10, x=0,
            font=dict(size=9, family=SANS, color=DARK2),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickformat="%b %d",
            tickfont=dict(size=9, color=MUTED, family=SANS),
            tickangle=-30,
            linecolor=BORDER, linewidth=1,
            ticks="outside", ticklen=4,
            title=None,
        ),
        yaxis=dict(
            showgrid=True, zeroline=False,
            gridcolor="#f0f2f5", gridwidth=1,
            tickformat=",",
            tickfont=dict(size=9, color=MUTED, family=SANS),
            title=None,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=WHITE,
            bordercolor=BORDER,
            font=dict(family=SANS, size=11, color=DARK2),
        ),
        bargap=0.30,
        showlegend=True,
    )

    # ── PIE CHART ───────────────────────────────────────────────────────────
    labels = [str(r["ds"].date()) for _, r in df.iterrows()]
    values = df["y"].tolist()

    palette = [
        NAVY,    # #1a3a6b
        TEAL,    # #0d7a6b
        "#9b59b6",
        "#e07b39",
        BLUE,    # #3a9bd5
        "#c0392b",
        "#16a085",
        "#2176ae",
        GOLD,    # #c8a84b
        "#1d9e75",
        "#7c5cbf",
        "#d35400",
    ]
    colors = [palette[i % len(palette)] for i in range(n)]

    pie_fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(
            colors=colors,
            line=dict(color=WHITE, width=2.5),
        ),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} txns  ·  %{percent}<extra></extra>",
    ))

    pie_fig.update_layout(
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=WHITE, bordercolor=BORDER,
            font=dict(family=SANS, size=11, color=DARK2),
        ),
    )

    # ── KPI CARDS ────────────────────────────────────────────────────────────
    def kpi(label, value, sub):
        return [
            html.P(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub"),
        ]

    total_content = kpi("Total Transactions", f"{int(total):,}", f"{n} day period")
    max_content   = kpi("Peak Day",           f"{int(max_row['y']):,}", str(max_row["ds"].date()))
    min_content   = kpi("Lowest Day",         f"{int(min_row['y']):,}", str(min_row["ds"].date()))
    avg_content   = kpi("Daily Average",      f"{int(avg):,}", "per day avg")

    # ── TABLE ─────────────────────────────────────────────────────────────────
    thead = html.Thead(html.Tr([
        html.Th("Date"),
        html.Th("Predicted Transactions"),
    ]))
    tbody = html.Tbody([
        html.Tr([
            html.Td(str(row["ds"].date())),
            html.Td(f"{int(row['y']):,}"),
        ])
        for _, row in df.iterrows()
    ])

    return (
        fig,
        total_content, max_content, min_content, avg_content,
        [thead, tbody], f"{n} rows",
        pie_fig,
    )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)