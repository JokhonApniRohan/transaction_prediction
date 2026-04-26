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
    <title>Transaction Forecast</title>
    {%favicon%}
    {%css%}

    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <style>

    body {
        margin: 0;
        font-family: Inter, system-ui, sans-serif;

        background:
            radial-gradient(circle at 20% 20%, #eef6ff 0%, transparent 45%),
            radial-gradient(circle at 80% 0%, #f5f9ff 0%, transparent 40%),
            linear-gradient(180deg, #ffffff 0%, #f6f9ff 100%);
    }

    .glass {
        background: rgba(255,255,255,0.55);
        border-radius: 18px;
        padding: 18px;

        box-shadow:
            0 8px 20px rgba(0,0,0,0.05),
            0 18px 50px rgba(37, 99, 235, 0.08);

        position: static !important;
    }

    .btn {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        padding: 12px 22px;
        border-radius: 14px;
        font-weight: 600;
        cursor: pointer;
    }

    .DatePickerSingle,
    .DateRangePicker {
        z-index: 2147483647 !important;
        position: relative !important;
    }

    .DatePickerSingle_picker,
    .DateRangePicker_picker {
        position: fixed !important;
        z-index: 2147483647 !important;
    }

    .DayPicker,
    .CalendarMonth,
    .CalendarDay {
        z-index: 2147483647 !important;
    }

    .scroll-box {
        max-height: 260px;
        overflow-y: auto;
        padding-right: 10px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }

    th, td {
        padding: 8px;
        border-bottom: 1px solid #eee;
        text-align: left;
    }

    th {
        position: sticky;
        top: 0;
        background: white;
        z-index: 2;
    }

    </style>
</head>
<body>
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
app.layout = html.Div(style={
    "padding": "40px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "20px"
}, children=[

    html.Div(className="glass", children=[
        html.H1("Transaction Forecast", style={"margin": 0}),
        html.P("Prophet-based forecasting dashboard")
    ]),

    html.Div(className="glass", style={
        "display": "flex",
        "justifyContent": "space-between",
        "gap": "20px",
        "alignItems": "center"
    }, children=[

        html.Div([
            html.Label("Start Date"),
            dcc.DatePickerSingle(id="start-date")
        ]),

        html.Div([
            html.Label("End Date"),
            dcc.DatePickerSingle(id="end-date")
        ]),

        html.Button("Run Prediction", id="predict-btn", className="btn")

    ]),

    # KPI CARDS (UNCHANGED)
    html.Div(style={"display": "flex", "gap": "15px"}, children=[

        html.Div(id="total-output", className="glass", style={"flex": 1}),
        html.Div(id="max-output", className="glass", style={"flex": 1}),
        html.Div(id="min-output", className="glass", style={"flex": 1}),

    ]),

    html.Div(className="glass", style={"height": "520px"}, children=[
        dcc.Graph(id="forecast-graph", style={"height": "100%"})
    ]),

    html.Div(className="glass", children=[

        html.H3("Daily Forecast Breakdown"),

        html.Div(className="scroll-box", children=[
            html.Table(id="forecast-table")
        ])

    ])

])


# =====================================================
# CALLBACK
# =====================================================
@app.callback(
    Output("forecast-graph", "figure"),
    Output("total-output", "children"),
    Output("max-output", "children"),
    Output("min-output", "children"),
    Output("forecast-table", "children"),
    Input("predict-btn", "n_clicks"),
    State("start-date", "date"),
    State("end-date", "date")
)
def update(n_clicks, start, end):

    if not n_clicks or not start or not end:
        return go.Figure(), "", "", "", ""

    df = predict_transaction(start, end)

    # KPIs
    total = df["y"].sum()

    max_row = df.loc[df["y"].idxmax()]
    min_row = df.loc[df["y"].idxmin()]

    max_text = f"{max_row['ds'].date()} → {int(max_row['y']):,}"
    min_text = f"{min_row['ds'].date()} → {int(min_row['y']):,}"

    # PLOT
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["ds"],
        y=df["y"],
        name="Daily Transactions",
        marker_color="rgba(37,99,235,0.4)"
    ))

    fig.add_trace(go.Scatter(
        x=df["ds"],
        y=df["y"],
        mode="lines+markers+text",
        name="Trend",
        line=dict(color="#1d4ed8", width=3),
        text=df["y"].round(0),
        textposition="top center"
    ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # TABLE
    table_header = html.Tr([html.Th("Date"), html.Th("Prediction")])

    table_rows = [
        html.Tr([
            html.Td(row["ds"].date()),
            html.Td(f"{int(row['y']):,}")
        ])
        for _, row in df.iterrows()
    ]

    table = [table_header] + table_rows

    # ✅ FIXED KPI OUTPUTS
    return (
        fig,

        [html.H4("Total"), html.H2(f"{int(total):,}")],

        [html.H4("Max"), html.P(max_text)],

        [html.H4("Min"), html.P(min_text)],

        table
    )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)