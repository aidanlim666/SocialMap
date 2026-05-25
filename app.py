"""
SocialMap — interactive Facebook SCI choropleth for US + Canada counties.

Run:
  python app.py

Prerequisites:
  python data/setup.py [path/to/county_county.tsv]
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import ALL, Input, Output, Patch, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from data.db import (
    db_ready,
    get_county_info,
    get_sci_scores,
    normalise_scores,
    search_counties,
)

# ---------------------------------------------------------------------------
# Load static assets at startup
# ---------------------------------------------------------------------------

GEOJSON_PATH = Path("data/counties.geojson")

if not GEOJSON_PATH.exists():
    raise FileNotFoundError(
        "data/counties.geojson not found. Run: python data/setup.py"
    )

with open(GEOJSON_PATH) as f:
    COUNTIES_GEOJSON = json.load(f)

# Ordered list of all county metadata — the z-array must keep this order.
COUNTIES_DF = pd.DataFrame(
    [
        {
            "fips": feat["properties"]["fips"],
            "name": feat["properties"]["name"],
            "state": feat["properties"]["state"],
            "population": feat["properties"]["population"],
            "country": feat["properties"]["country"],
        }
        for feat in COUNTIES_GEOJSON["features"]
    ]
)

ALL_FIPS: list[str] = COUNTIES_DF["fips"].tolist()
FIPS_INDEX: dict[str, int] = {f: i for i, f in enumerate(ALL_FIPS)}
N = len(ALL_FIPS)

SCI_AVAILABLE = db_ready()


# ---------------------------------------------------------------------------
# Plotly figure helpers
# ---------------------------------------------------------------------------

def _color_from_score(score: float) -> str:
    """Map 0-100 → white (#ffffff) to dark green (#006400)."""
    t = score / 100.0
    r = round(255 * (1 - t))
    g = round(100 + 155 * (1 - t))
    b = round(255 * (1 - t))
    return f"rgb({r},{g},{b})"


# Pre-build the discrete colorscale Plotly needs: list of [fraction, color] pairs.
_N_STOPS = 101
COLORSCALE = [
    [i / (_N_STOPS - 1), _color_from_score(i / (_N_STOPS - 1) * 100)]
    for i in range(_N_STOPS)
]


def create_base_figure() -> go.Figure:
    z_init = [0.0] * N
    custom = COUNTIES_DF[["name", "state", "population"]].values

    trace = go.Choroplethmap(
        geojson=COUNTIES_GEOJSON,
        featureidkey="properties.fips",
        locations=ALL_FIPS,
        z=z_init,
        colorscale=COLORSCALE,
        zmin=0,
        zmax=100,
        marker_opacity=0.75,
        marker_line_width=0.4,
        marker_line_color="#555",
        showscale=True,
        colorbar=dict(
            title=dict(text="SCI", font=dict(color="#ccc")),
            thickness=14,
            tickfont=dict(color="#ccc"),
            x=1.0,
        ),
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Population: %{customdata[2]:,}<br>"
            "SCI: %{z:.1f}"
            "<extra></extra>"
        ),
    )

    fig = go.Figure(trace)
    fig.update_layout(
        map_style="carto-darkmatter",
        map_zoom=3.2,
        map_center={"lat": 47.5, "lon": -96},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        uirevision="static",  # keeps zoom/pan when figure data updates
    )
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _info_placeholder():
    return html.P(
        "Search for a county to begin.",
        className="text-muted small mt-2",
    )


def _sci_unavailable_banner():
    if SCI_AVAILABLE:
        return html.Div()
    return dbc.Alert(
        [
            html.Strong("SCI data not loaded. "),
            "Run ",
            html.Code("python data/setup.py path/to/county_county.tsv"),
            " to enable connectivity scores.",
        ],
        color="warning",
        className="small py-2 px-3 mb-3",
    )


SIDEBAR = dbc.Col(
    [
        html.Div(
            [
                html.H4("Social Map", className="text-white fw-bold mb-1"),
                html.P(
                    "Facebook Social Connectedness Index",
                    className="text-muted small mb-3",
                ),
            ]
        ),
        _sci_unavailable_banner(),
        dbc.Input(
            id="search-input",
            placeholder="Search county or province…",
            type="text",
            debounce=True,
            className="mb-2",
            style={"backgroundColor": "#1c2128", "color": "#e6edf3", "border": "1px solid #30363d"},
        ),
        html.Div(
            id="search-results",
            style={"maxHeight": "240px", "overflowY": "auto"},
            className="mb-3",
        ),
        html.Hr(style={"borderColor": "#30363d"}),
        html.Div(id="selected-county-info", children=_info_placeholder()),
        html.Hr(style={"borderColor": "#30363d"}),
        html.Div(id="top-bottom-panel"),
    ],
    width=3,
    style={
        "backgroundColor": "#0d1117",
        "height": "100vh",
        "overflowY": "auto",
        "padding": "20px 16px",
        "borderRight": "1px solid #30363d",
    },
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Social Map",
)
server = app.server  # expose for gunicorn / deployment

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                SIDEBAR,
                dbc.Col(
                    dcc.Graph(
                        id="map",
                        figure=create_base_figure(),
                        style={"height": "100vh"},
                        config={"scrollZoom": True, "displayModeBar": False},
                    ),
                    width=9,
                    className="p-0",
                ),
            ],
            className="g-0",
        ),
        dcc.Store(id="selected-fips"),
    ],
    fluid=True,
    className="p-0",
    style={"backgroundColor": "#0d1117"},
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("search-results", "children"),
    Input("search-input", "value"),
)
def update_search_results(query: str):
    if not query or len(query.strip()) < 2:
        return []

    results = search_counties(query.strip(), limit=10)
    if not results:
        return html.P("No counties found.", className="text-muted small mt-1")

    return [
        dbc.Button(
            f"{r['name']}, {r['state']}",
            id={"type": "county-btn", "index": r["fips"]},
            color="secondary",
            size="sm",
            outline=True,
            className="d-block w-100 text-start mb-1 text-light",
            style={"fontSize": "0.78rem", "border": "1px solid #30363d"},
        )
        for r in results
    ]


@callback(
    Output("selected-fips", "data"),
    Input({"type": "county-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def capture_county_selection(n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered or all(c is None for c in n_clicks):
        return no_update
    prop_id = ctx.triggered[0]["prop_id"]
    fips = json.loads(prop_id.rsplit(".", 1)[0])["index"]
    return fips


@callback(
    Output("map", "figure"),
    Output("selected-county-info", "children"),
    Output("top-bottom-panel", "children"),
    Input("selected-fips", "data"),
    prevent_initial_call=True,
)
def update_on_county_select(fips: str):
    if not fips:
        return no_update, no_update, no_update

    county = get_county_info(fips)
    if county is None:
        return no_update, no_update, no_update

    # --- Info panel ---
    info_panel = html.Div(
        [
            html.H5(county["name"], className="text-white mb-0"),
            html.P(
                f"{county['state']} · {county['country']}",
                className="text-muted small mb-1",
            ),
            html.P(
                f"Population: {county['population']:,}",
                className="text-info small mb-0",
            ),
        ]
    )

    # --- SCI scores ---
    def _clear_map():
        p = Patch()
        p["data"][0]["z"] = [0.0] * N
        return p

    if not SCI_AVAILABLE:
        return _clear_map(), info_panel, html.P(
            "SCI data not loaded.", className="text-muted small"
        )

    raw_scores = get_sci_scores(fips)
    if not raw_scores:
        country = county.get("country", "US")
        msg = (
            "SCI data is only available for US counties."
            if country == "CA"
            else "No SCI data found for this county."
        )
        return _clear_map(), info_panel, html.P(msg, className="text-muted small")

    scores = normalise_scores(raw_scores)

    # --- Update map colours via Patch (only z array is re-sent) ---
    z = [scores.get(f, 0.0) for f in ALL_FIPS]
    patched = Patch()
    patched["data"][0]["z"] = z

    # --- Top / bottom 5 ---
    sorted_pairs = sorted(
        [(f, s) for f, s in scores.items() if s > 0 and f != fips],
        key=lambda x: x[1],
        reverse=True,
    )
    top5 = sorted_pairs[:5]
    bottom5 = sorted_pairs[-5:] if len(sorted_pairs) >= 5 else sorted_pairs
    bottom5 = list(reversed(bottom5))

    fips_to_meta = COUNTIES_DF.set_index("fips")[["name", "state"]].to_dict("index")

    def make_row(f: str, s: float, color: str):
        meta = fips_to_meta.get(f, {"name": f, "state": ""})
        label = f"{meta['name']}, {meta['state']}"
        return dbc.Row(
            [
                dbc.Col(
                    html.Span(label, style={"fontSize": "0.75rem", "color": "#e6edf3"}),
                    width=9,
                ),
                dbc.Col(
                    html.Span(
                        f"{s:.1f}",
                        style={"fontSize": "0.75rem", "color": color, "fontWeight": "bold"},
                    ),
                    width=3,
                    className="text-end",
                ),
            ],
            className="mb-1 align-items-center",
        )

    top_bottom = html.Div(
        [
            html.P("Top 5 Connected", className="text-success small fw-bold mb-1"),
            *[make_row(f, s, "#3fb950") for f, s in top5],
            html.Hr(style={"borderColor": "#30363d", "margin": "8px 0"}),
            html.P("Bottom 5 Connected", className="text-danger small fw-bold mb-1"),
            *[make_row(f, s, "#f85149") for f, s in bottom5],
        ]
    )

    return patched, info_panel, top_bottom


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
