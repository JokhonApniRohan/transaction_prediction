import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd

from backend_prophet import predict_transaction

app = dash.Dash(__name__)

# =====================================================
# GLOBAL CSS
# =====================================================
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Txn Forecast</title>
    {%favicon%}
    {%css%}

    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">

    <style>

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --glass:     rgba(255,255,255,0.32);
        --glass-hvy: rgba(255,255,255,0.58);
        --glass-bdr: rgba(255,255,255,0.62);
        --blur:      20px;

        /* Accent colours — saturated enough to pop on the soft bg */
        --accent:    #1a56db;
        --accent-lt: rgba(26,86,219,0.11);
        --green:     #047857;
        --green-lt:  rgba(4,120,87,0.10);
        --danger:    #b91c1c;
        --danger-lt: rgba(185,28,28,0.10);
        --purple:    #6d28d9;
        --purple-lt: rgba(109,40,217,0.10);

        /* Text — darker than before so it pops on glass */
        --text:      #0c1420;
        --text-sec:  #374151;
        --muted:     #6b7280;

        --mono:      'JetBrains Mono', monospace;
        --sans:      'Plus Jakarta Sans', sans-serif;
        --radius:    18px;
    }

    html, body {
        font-family: var(--sans);
        /* Slightly deeper gradient so glass panels read clearly */
        background:
            linear-gradient(160deg,
                #dde5f4 0%,
                #cfdaf0 22%,
                #dbd5f0 50%,
                #cce8de 78%,
                #d8ecda 100%);
        background-attachment: fixed;
        color: var(--text);
        min-height: 100vh;
    }

    /* ---- SCROLL INDICATOR ---- */
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
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .scroll-chevron {
        width: 26px; height: 26px;
        border-radius: 50%;
        background: rgba(255,255,255,0.50);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.65);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: bounce 1.8s ease-in-out infinite;
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

    /* ---- LAYOUT SHELL ---- */
    #root-wrap {
        max-width: 1200px;
        margin: 0 auto;
        padding: 44px 28px 100px;
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    /* ---- GLASS BASE ---- */
    .glass {
        background: var(--glass);
        backdrop-filter: blur(var(--blur));
        -webkit-backdrop-filter: blur(var(--blur));
        border: 1px solid var(--glass-bdr);
        border-radius: var(--radius);
        box-shadow:
            0 2px 8px rgba(12,20,32,0.06),
            0 8px 28px rgba(12,20,32,0.07),
            inset 0 1px 0 rgba(255,255,255,0.70);
    }

    .glass-heavy {
        background: var(--glass-hvy);
        backdrop-filter: blur(var(--blur));
        -webkit-backdrop-filter: blur(var(--blur));
        border: 1px solid var(--glass-bdr);
        border-radius: var(--radius);
        box-shadow:
            0 2px 8px rgba(12,20,32,0.07),
            0 8px 28px rgba(12,20,32,0.09),
            inset 0 1px 0 rgba(255,255,255,0.80);
    }

    /* ---- HEADER ---- */
    .header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 22px 28px;
    }

    .header-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: var(--text);
        line-height: 1;
    }

    .header-sub {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 5px;
        font-weight: 400;
    }

    .header-badge {
        font-family: var(--mono);
        font-size: 9px;
        background: rgba(12,20,32,0.08);
        color: var(--text-sec);
        padding: 5px 14px;
        border-radius: 50px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 500;
        border: 1px solid rgba(12,20,32,0.12);
    }

    /* ---- CONTROLS STRIP ---- */
    .controls-strip {
        display: flex;
        align-items: flex-end;
        gap: 20px;
        padding: 18px 22px;
        position: relative;
        z-index: 200;
    }

    .ctrl-group {
        display: flex;
        flex-direction: column;
        gap: 5px;
        position: relative;
        z-index: 200;
    }

    .ctrl-label {
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-sec);
        font-weight: 500;
    }

    /* Date picker overrides */
    .SingleDatePickerInput {
        border: 1px solid rgba(12,20,32,0.14) !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.55) !important;
        backdrop-filter: blur(8px) !important;
    }

    .DateInput_input {
        font-family: var(--mono) !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: var(--text) !important;
        background: transparent !important;
        border-bottom: none !important;
        padding: 7px 11px !important;
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
        background: rgba(12,20,32,0.12);
        align-self: flex-end;
        margin-bottom: 2px;
    }

    .run-btn {
        font-family: var(--sans);
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.02em;
        background: var(--text);
        color: #f0f4ff;
        border: 1px solid rgba(255,255,255,0.10);
        padding: 9px 22px;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.18s;
        white-space: nowrap;
        margin-left: auto;
        box-shadow: 0 2px 14px rgba(12,20,32,0.22);
    }

    .run-btn:hover  { opacity: 0.85; transform: translateY(-1px); }
    .run-btn:active { transform: scale(0.97); }

    /* ---- KPI ROW ---- */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        position: relative;
        z-index: 1;
    }

    .kpi-card {
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }

    /* Coloured left border stripe */
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 16px; bottom: 16px; left: 0;
        width: 3px;
        border-radius: 0 2px 2px 0;
        background: var(--accent);
    }

    .kpi-card.kpi-max::before { background: var(--green); }
    .kpi-card.kpi-min::before { background: var(--danger); }
    .kpi-card.kpi-avg::before { background: var(--purple); }

    .kpi-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--accent);
        margin-bottom: 10px;
        box-shadow: 0 0 0 3px var(--accent-lt);
    }

    .kpi-card.kpi-max .kpi-dot { background: var(--green);  box-shadow: 0 0 0 3px var(--green-lt); }
    .kpi-card.kpi-min .kpi-dot { background: var(--danger); box-shadow: 0 0 0 3px var(--danger-lt); }
    .kpi-card.kpi-avg .kpi-dot { background: var(--purple); box-shadow: 0 0 0 3px var(--purple-lt); }

    .kpi-label {
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
        font-weight: 500;
    }

    .kpi-value {
        font-family: var(--sans);
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1;
        color: var(--text);
    }

    .kpi-sub {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--muted);
        margin-top: 5px;
        font-weight: 400;
    }

    /* ---- CHART PANEL ---- */
    .chart-panel {
        padding: 22px;
        position: relative;
        z-index: 1;
    }

    .panel-title {
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-sec);
        margin-bottom: 14px;
        font-weight: 500;
    }

    /* ---- BOTTOM ROW ---- */
    .bottom-row {
        display: grid;
        grid-template-columns: 1fr 360px;
        gap: 14px;
        align-items: start;
        position: relative;
        z-index: 1;
    }

    /* ---- TABLE PANEL ---- */
    .table-panel { overflow: hidden; }

    .table-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 18px;
        border-bottom: 1px solid rgba(12,20,32,0.09);
    }

    .table-header-row .panel-title { margin: 0; }

    .table-count {
        font-family: var(--mono);
        font-size: 9px;
        color: var(--muted);
        font-weight: 500;
    }

    .scroll-box {
        max-height: 280px;
        overflow-y: auto;
    }

    .scroll-box::-webkit-scrollbar { width: 4px; }
    .scroll-box::-webkit-scrollbar-track { background: transparent; }
    .scroll-box::-webkit-scrollbar-thumb { background: rgba(12,20,32,0.14); border-radius: 2px; }

    table {
        width: 100%;
        border-collapse: collapse;
        font-family: var(--mono);
        font-size: 12px;
    }

    thead tr { border-bottom: 1px solid rgba(12,20,32,0.09); }

    th {
        padding: 9px 18px;
        text-align: left;
        font-size: 9px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-sec);
        font-weight: 600;
        position: sticky;
        top: 0;
        background: rgba(255,255,255,0.62);
        backdrop-filter: blur(10px);
        z-index: 2;
    }

    td {
        padding: 9px 18px;
        border-bottom: 1px solid rgba(12,20,32,0.05);
        color: var(--text);
        font-weight: 400;
    }

    tbody tr:last-child td { border-bottom: none; }
    tbody tr:hover { background: rgba(255,255,255,0.35); }

    td:nth-child(2) { font-weight: 600; text-align: right; color: var(--text); }
    th:nth-child(2) { text-align: right; }

    /* ---- PIE PANEL ---- */
    .pie-panel { padding: 18px; }

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
app.layout = html.Div(id="root-wrap", children=[

    # HEADER
    html.Div(className="glass-heavy header-row", children=[
        html.Div([
            html.Div("Transaction Forecast", className="header-title"),
            html.Div("Prophet-based forecasting engine", className="header-sub"),
        ]),
        html.Div("Prophet Model", className="header-badge")
    ]),

    # CONTROLS STRIP
    html.Div(className="glass controls-strip", children=[

        html.Div(className="ctrl-group", children=[
            html.Label("Start Date", className="ctrl-label"),
            dcc.DatePickerSingle(id="start-date", display_format="YYYY-MM-DD")
        ]),

        html.Div(className="ctrl-sep"),

        html.Div(className="ctrl-group", children=[
            html.Label("End Date", className="ctrl-label"),
            dcc.DatePickerSingle(id="end-date", display_format="YYYY-MM-DD")
        ]),

        html.Button("Run Forecast →", id="predict-btn", className="run-btn")

    ]),

    # KPI CARDS
    html.Div(className="kpi-row", children=[
        html.Div(id="total-output", className="glass kpi-card"),
        html.Div(id="max-output",   className="glass kpi-card kpi-max"),
        html.Div(id="min-output",   className="glass kpi-card kpi-min"),
        html.Div(id="avg-output",   className="glass kpi-card kpi-avg"),
    ]),

    # CHART
    html.Div(className="glass chart-panel", children=[
        html.Div("Daily Transaction Forecast", className="panel-title"),
        dcc.Graph(id="forecast-graph", style={"height": "400px"},
                  config={"displayModeBar": False})
    ]),

    # BOTTOM ROW: TABLE + PIE
    html.Div(className="bottom-row", children=[

        html.Div(className="glass table-panel", children=[
            html.Div(className="table-header-row", children=[
                html.Div("Daily Breakdown", className="panel-title"),
                html.Div(id="table-count", className="table-count")
            ]),
            html.Div(className="scroll-box", children=[
                html.Table(id="forecast-table")
            ])
        ]),

        html.Div(className="glass pie-panel", children=[
            html.Div("Transaction Share by Day", className="panel-title"),
            dcc.Graph(id="pie-chart", style={"height": "300px"},
                      config={"displayModeBar": False})
        ])

    ])

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
    Input("predict-btn",  "n_clicks"),
    State("start-date",   "date"),
    State("end-date",     "date")
)
def update(n_clicks, start, end):

    T = "rgba(0,0,0,0)"

    EMPTY = go.Figure(layout=go.Layout(
        paper_bgcolor=T, plot_bgcolor=T,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    ))

    if not n_clicks or not start or not end:
        return EMPTY, "", "", "", "", "", "", EMPTY

    df = predict_transaction(start, end)

    total   = df["y"].sum()
    avg     = df["y"].mean()
    max_row = df.loc[df["y"].idxmax()]
    min_row = df.loc[df["y"].idxmin()]

    MONO  = "JetBrains Mono, monospace"
    SANS  = "Plus Jakarta Sans, sans-serif"
    TEXT  = "#0c1420"
    SEC   = "#374151"
    MUTED = "#6b7280"

    # ── LINE + BAR CHART ────────────────────────────────
    fig = go.Figure()

    # Bar: accent fill, visible but not overwhelming
    fig.add_trace(go.Bar(
        x=df["ds"], y=df["y"],
        name="Volume",
        marker=dict(
            color="rgba(26,86,219,0.13)",
            line=dict(color="rgba(26,86,219,0.30)", width=1)
        ),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Transactions: %{y:,.0f}<extra></extra>"
    ))

    # Trend line: deep navy so it reads clearly
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines+markers",
        name="Trend",
        line=dict(color="#1a3a8f", width=2.5),
        marker=dict(
            color="#1a56db", size=6,
            line=dict(color="#ffffff", width=2)
        ),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        paper_bgcolor=T,
        plot_bgcolor=T,
        margin=dict(l=4, r=4, t=10, b=4),
        font=dict(family=MONO, size=10, color=SEC),
        legend=dict(
            orientation="h", y=1.10, x=0,
            font=dict(size=10, family=MONO, color=SEC),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickformat="%b %d",
            tickfont=dict(size=10, color=MUTED),
            linecolor="rgba(12,20,32,0.12)", linewidth=1,
            ticks="outside", ticklen=4,
            title=None
        ),
        yaxis=dict(
            showgrid=True, zeroline=False,
            gridcolor="rgba(12,20,32,0.07)", gridwidth=1,
            tickformat=",",
            tickfont=dict(size=10, color=MUTED),
            title=None
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="rgba(12,20,32,0.14)",
            font=dict(family=MONO, size=11, color=TEXT)
        ),
        bargap=0.30
    )

    # ── PIE CHART ───────────────────────────────────────
    n = len(df)
    labels = [str(r["ds"].date()) for _, r in df.iterrows()]
    values = df["y"].tolist()

    # Well-separated, saturated palette — distinct on glass bg
    palette = [
        "#1a56db",  # blue
        "#047857",  # emerald
        "#6d28d9",  # violet
        "#b45309",  # amber
        "#0e7490",  # cyan
        "#be185d",  # rose
        "#1e40af",  # indigo
        "#065f46",  # dark green
        "#92400e",  # brown
        "#7c3aed",  # purple
        "#0f766e",  # teal
        "#9f1239",  # crimson
    ]
    colors = [palette[i % len(palette)] for i in range(n)]

    pie_fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.54,
        marker=dict(
            colors=colors,
            line=dict(color="rgba(255,255,255,0.85)", width=2.5)
        ),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} txns  ·  %{percent}<extra></extra>"
    ))

    pie_fig.update_layout(
        paper_bgcolor=T,
        plot_bgcolor=T,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="rgba(12,20,32,0.14)",
            font=dict(family=MONO, size=11, color=TEXT)
        )
    )

    # ── KPI CONTENT ─────────────────────────────────────
    dot = lambda: html.Div(className="kpi-dot")

    total_content = [
        dot(),
        html.Div("Total Transactions", className="kpi-label"),
        html.Div(f"{int(total):,}", className="kpi-value"),
        html.Div(f"{n} day period", className="kpi-sub"),
    ]
    max_content = [
        dot(),
        html.Div("Peak Day", className="kpi-label"),
        html.Div(f"{int(max_row['y']):,}", className="kpi-value"),
        html.Div(str(max_row["ds"].date()), className="kpi-sub"),
    ]
    min_content = [
        dot(),
        html.Div("Lowest Day", className="kpi-label"),
        html.Div(f"{int(min_row['y']):,}", className="kpi-value"),
        html.Div(str(min_row["ds"].date()), className="kpi-sub"),
    ]
    avg_content = [
        dot(),
        html.Div("Daily Average", className="kpi-label"),
        html.Div(f"{int(avg):,}", className="kpi-value"),
        html.Div("per day avg", className="kpi-sub"),
    ]

    # ── TABLE ────────────────────────────────────────────
    thead = html.Thead(html.Tr([html.Th("Date"), html.Th("Predicted")]))
    tbody = html.Tbody([
        html.Tr([
            html.Td(str(row["ds"].date())),
            html.Td(f"{int(row['y']):,}")
        ])
        for _, row in df.iterrows()
    ])

    return (fig, total_content, max_content, min_content,
            avg_content, [thead, tbody], f"{n} rows", pie_fig)


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)