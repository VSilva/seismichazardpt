from __future__ import annotations

from pathlib import Path
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dash import Dash, html, dcc, Input, Output, no_update
import plotly.graph_objects as go
from PIL import Image


# ---------------- CONFIG ----------------
MAPS_DIR = Path("./maps")  # folder containing the PNGs
# Expected filename format:
#   <GMM>_<maptype>_<IM>_<rate>.png
# Examples:
#   AbrahamsonEtAl2014_hazard_map-mean_PGA_0.002105.png
#   BooreEtAl2014_quantile_map-0.05_SA(1.0)_0.000404.png   (would work too)
#
FILENAME_RE = re.compile(
    r"^(?P<gmm>.+?)_"                       # non-greedy up to first _ before maptype
    r"(?P<maptype>hazard_map-mean|quantile_map-0\.05|quantile_map-0\.95)_"
    r"(?P<im>.+?)_"                         # IM can contain parentheses/dots/etc.
    r"(?P<rate>\d+(?:\.\d+)?)"
    r"\.png$"
)
# ----------------------------------------


MAPTYPE_LABEL = {
    "hazard_map-mean": "Mean",
    "quantile_map-0.05": "Quantile 5%",
    "quantile_map-0.95": "Quantile 95%",
}


@dataclass(frozen=True)
class Key:
    gmm: str
    maptype: str
    im: str
    rate: str


def discover_maps(folder: Path) -> Dict[Key, Path]:
    idx: Dict[Key, Path] = {}
    for p in folder.glob("*.png"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        key = Key(
            gmm=m.group("gmm"),
            maptype=m.group("maptype"),
            im=m.group("im"),
            rate=m.group("rate"),
        )
        idx[key] = p
    return idx


MAP_INDEX = discover_maps(MAPS_DIR)


def unique_sorted(values: List[str]) -> List[str]:
    return sorted(set(values))


def options_from_index(
    idx: Dict[Key, Path],
    gmm: str | None = None,
    maptype: str | None = None,
    im: str | None = None,
    rate: str | None = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return available values for each dimension given optional filters."""
    keys = list(idx.keys())

    def ok(k: Key) -> bool:
        return ((gmm is None or k.gmm == gmm) and
                (maptype is None or k.maptype == maptype) and
                (im is None or k.im == im) and
                (rate is None or k.rate == rate))

    filt = [k for k in keys if ok(k)]
    gmms = unique_sorted([k.gmm for k in filt])
    maptypes = unique_sorted([k.maptype for k in filt])
    ims = unique_sorted([k.im for k in filt])
    rates = sorted(set([k.rate for k in filt]), key=lambda x: float(x))  # numeric sort
    return gmms, maptypes, ims, rates


def pick_first(available: List[str], current: str | None) -> str | None:
    if not available:
        return None
    if current in available:
        return current
    return available[0]


def make_image_figure(img_path: Path, title: str) -> go.Figure:
    img = Image.open(img_path)

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=img,
            x=0, y=1, xref="paper", yref="paper",
            sizex=1, sizey=1,
            xanchor="left", yanchor="top",
            layer="below",
        )
    )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1], scaleanchor="x")
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=55, b=10),
        template="plotly_white",
    )
    return fig


# ---------------- DASH APP ----------------
app = Dash(__name__)
app.title = "Portugal Seismic Hazard Dashboard"

# Initial available sets (no filters)
ALL_GMMS, ALL_MAPTYPES, ALL_IMS, ALL_RATES = options_from_index(MAP_INDEX)

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "18px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Seismic Hazard Maps — Portugal"),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(220px, 1fr))", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("Ground motion model"),
                    dcc.Dropdown(
                        id="dd-gmm",
                        options=[{"label": x, "value": x} for x in ALL_GMMS],
                        value=ALL_GMMS[0] if ALL_GMMS else None,
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Map type"),
                    dcc.Dropdown(
                        id="dd-maptype",
                        options=[{"label": MAPTYPE_LABEL.get(x, x), "value": x} for x in ALL_MAPTYPES],
                        value=ALL_MAPTYPES[0] if ALL_MAPTYPES else None,
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Intensity measure"),
                    dcc.Dropdown(
                        id="dd-im",
                        options=[{"label": x, "value": x} for x in ALL_IMS],
                        value=ALL_IMS[0] if ALL_IMS else None,
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Rate"),
                    dcc.Dropdown(
                        id="dd-rate",
                        options=[{"label": x, "value": x} for x in ALL_RATES],
                        value=ALL_RATES[0] if ALL_RATES else None,
                        clearable=False,
                    ),
                ]),
            ],
        ),

        html.Div(style={"height": "10px"}),

        dcc.Graph(id="map-graph", style={"height": "78vh"}),

        html.Div(id="status", style={"marginTop": "6px", "color": "#555"}),
    ],
)


# --- Cascading dropdown options + values ---
@app.callback(
    Output("dd-gmm", "options"), Output("dd-gmm", "value"),
    Output("dd-maptype", "options"), Output("dd-maptype", "value"),
    Output("dd-im", "options"), Output("dd-im", "value"),
    Output("dd-rate", "options"), Output("dd-rate", "value"),
    Input("dd-gmm", "value"),
    Input("dd-maptype", "value"),
    Input("dd-im", "value"),
    Input("dd-rate", "value"),
)
def sync_dropdowns(gmm, maptype, im, rate):
    if not MAP_INDEX:
        return [], None, [], None, [], None, [], None

    # Available options for each dropdown given the OTHER three selections.
    gmms, _, _, _ = options_from_index(MAP_INDEX, maptype=maptype, im=im, rate=rate)
    _, maptypes, _, _ = options_from_index(MAP_INDEX, gmm=gmm, im=im, rate=rate)
    _, _, ims, _ = options_from_index(MAP_INDEX, gmm=gmm, maptype=maptype, rate=rate)
    _, _, _, rates = options_from_index(MAP_INDEX, gmm=gmm, maptype=maptype, im=im)

    gmm = pick_first(gmms, gmm)
    maptype = pick_first(maptypes, maptype)
    im = pick_first(ims, im)
    rate = pick_first(rates, rate)

    gmm_opts = [{"label": x, "value": x} for x in gmms]
    maptype_opts = [{"label": MAPTYPE_LABEL.get(x, x), "value": x} for x in maptypes]
    im_opts = [{"label": x, "value": x} for x in ims]
    rate_opts = [{"label": x, "value": x} for x in rates]

    return gmm_opts, gmm, maptype_opts, maptype, im_opts, im, rate_opts, rate


# --- Figure update ---
@app.callback(
    Output("map-graph", "figure"),
    Output("status", "children"),
    Input("dd-gmm", "value"),
    Input("dd-maptype", "value"),
    Input("dd-im", "value"),
    Input("dd-rate", "value"),
)
def update_map(gmm, maptype, im, rate):
    if not MAP_INDEX or None in (gmm, maptype, im, rate):
        fig = go.Figure()
        fig.update_layout(template="plotly_white", annotations=[dict(
            text="No maps found. Check MAPS_DIR and filename pattern.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )])
        return fig, "No maps available."

    key = Key(gmm=gmm, maptype=maptype, im=im, rate=rate)
    path = MAP_INDEX.get(key)

    if not path:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", annotations=[dict(
            text="Selected combination not available.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )])
        return fig, f"Missing file for: {key}"

    title = f"{gmm} — {MAPTYPE_LABEL.get(maptype, maptype)} — {im} — rate {rate}"
    fig = make_image_figure(path, title)
    return fig, f"Showing: {path.name}"


if __name__ == "__main__":
    app.run(debug=True)