from __future__ import annotations

from pathlib import Path
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go


# -------------------- Cities --------------------
# (lon, lat, name)
CITIES = [
    (-9.14, 38.72, "Lisbon"),
    (-8.61099, 41.14961, "Porto"),
    (-8.41955, 40.20564, "Coimbra"),
    (-8.64554, 40.64427, "Aveiro"),
    (-7.93223, 37.01937, "Faro"),
    (-6.75719, 41.80582, "Braganca"),
    (-7.86323, 38.01506, "Beja"),
    (-8.9117, 37.0833, "Vila do Bispo"),
]
CITY_BY_NAME = {name: (lon, lat) for lon, lat, name in CITIES}


# -------------------- Paths (works in .py AND notebooks) --------------------
try:
    THIS_FILE = Path(__file__).resolve()
    ROOT = THIS_FILE.parents[2]          # dashboard -> Mapping -> ROOT
except NameError:
    ROOT = Path.cwd().parents[1]         # dashboard (cwd) -> Mapping -> ROOT

SEISMIC_HAZARD_DIR = ROOT / "seismic_hazard"


# -------------------- Constants for periods --------------------
PGA_PERIOD = 0.01         # place PGA at 0.01 s (instead of 0.0)
PERIOD_MIN = 0.01         # minimum period shown / selectable


# -------------------- Kind labels --------------------
KIND_LABEL = {
    "mean": "Mean",
    "q05": "Quantile 5%",
    "q95": "Quantile 95%",
}


# -------------------- Filename parsing --------------------
TRAILING_CALCID_RE = re.compile(r"^(?P<base>.+?)_(?P<id>\d+)\.csv$", re.IGNORECASE)

def strip_calcid(name: str) -> str:
    m = TRAILING_CALCID_RE.match(name)
    if m:
        return m.group("base")
    m2 = re.match(r"^(?P<base>.+?)_(?P<id>\d+)$", name)
    if m2:
        return m2.group("base")
    return name.replace(".csv", "")

def parse_kind_from_uhs_filename(csv_path: Path) -> Optional[str]:
    base = strip_calcid(csv_path.name).replace(".csv", "").lower()
    if "uhs" not in base:
        return None
    if "mean" in base:
        return "mean"
    if "0.05" in base:
        return "q05"
    if "0.95" in base:
        return "q95"
    return None


# -------------------- Index discovery --------------------
@dataclass(frozen=True)
class Key:
    model: str
    kind: str  # mean/q05/q95

def discover_uhs_files() -> Dict[Key, Path]:
    idx: Dict[Key, Path] = {}
    if not SEISMIC_HAZARD_DIR.exists():
        return idx

    for model_dir in sorted(SEISMIC_HAZARD_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        outputs = model_dir / "outputs"
        if not outputs.exists():
            continue

        for csv_path in outputs.glob("*.csv"):
            kind = parse_kind_from_uhs_filename(csv_path)
            if kind is None:
                continue
            idx[Key(model=model_dir.name, kind=kind)] = csv_path

    return idx

UHS_INDEX = discover_uhs_files()


# -------------------- Read OpenQuake CSV --------------------
def read_openquake_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    return df.dropna(axis=1, how="all")


# -------------------- UHS parsing --------------------
UHS_COL_RE = re.compile(r"^(?P<rate>\d+(?:\.\d+)?)~(?P<imt>.+)$")
SA_RE = re.compile(r"^SA\((?P<T>\d+(?:\.\d+)?)\)$", re.IGNORECASE)

def find_lon_lat_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols_lower = [c.lower() for c in df.columns]
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]

    lon_col = None
    lat_col = None
    for cand in lon_candidates:
        if cand in cols_lower:
            lon_col = df.columns[cols_lower.index(cand)]
            break
    for cand in lat_candidates:
        if cand in cols_lower:
            lat_col = df.columns[cols_lower.index(cand)]
            break

    if lon_col is None or lat_col is None:
        raise ValueError(f"Could not find lon/lat columns. Columns: {list(df.columns)}")
    return lon_col, lat_col

def parse_uhs_structure(df: pd.DataFrame):
    lon_col, lat_col = find_lon_lat_columns(df)
    lons = df[lon_col].to_numpy(dtype=float)
    lats = df[lat_col].to_numpy(dtype=float)

    cols_by_rate: Dict[str, List[Tuple[float, str]]] = {}

    for c in df.columns:
        m = UHS_COL_RE.match(str(c))
        if not m:
            continue
        rate = m.group("rate")
        imt = m.group("imt").strip()

        if imt.upper() == "PGA":
            T = PGA_PERIOD
        else:
            ms = SA_RE.match(imt)
            if not ms:
                continue
            T = float(ms.group("T"))
            if T < PERIOD_MIN:
                # enforce minimum period (drop smaller periods if any)
                continue

        cols_by_rate.setdefault(rate, []).append((T, c))

    if not cols_by_rate:
        raise ValueError("No '<rate>~PGA' or '<rate>~SA(T)' columns found in UHS CSV.")

    rates = sorted(cols_by_rate.keys(), key=lambda x: float(x))
    spec_by_rate = {}
    for rate in rates:
        pairs = sorted(cols_by_rate[rate], key=lambda t: t[0])  # sort by period
        periods = np.array([p for p, _ in pairs], dtype=float)
        cols = [col for _, col in pairs]
        values = df[cols].to_numpy(dtype=float)
        spec_by_rate[rate] = (periods, values)

    return lons, lats, rates, spec_by_rate

def nearest_site_index(lons: np.ndarray, lats: np.ndarray, lon: float, lat: float) -> int:
    d2 = (lons - lon) ** 2 + (lats - lat) ** 2
    return int(np.argmin(d2))


@lru_cache(maxsize=48)
def load_uhs_file(path_str: str):
    df = read_openquake_csv(Path(path_str))
    return parse_uhs_structure(df)


# -------------------- Option utilities --------------------
def available_models() -> List[str]:
    return sorted({k.model for k in UHS_INDEX})

def available_kinds_for_model(model: str) -> List[str]:
    kinds = sorted({k.kind for k in UHS_INDEX if k.model == model})
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]
    return kinds

def available_kinds_for_models(models: List[str]) -> List[str]:
    kinds = sorted({k.kind for k in UHS_INDEX if k.model in set(models)})
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]
    return kinds


# -------------------- Plot helpers --------------------
def apply_axes(fig: go.Figure, logx: bool, logy: bool, x_range: List[float] | None):
    if logx:
        fig.update_xaxes(type="log")
    else:
        fig.update_xaxes(type="linear")

    if logy:
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(type="linear")

    if x_range and len(x_range) == 2:
        xmin, xmax = float(x_range[0]), float(x_range[1])
        xmin = max(xmin, PERIOD_MIN)
        if logx:
            fig.update_xaxes(range=[np.log10(xmin), np.log10(xmax)])
        else:
            fig.update_xaxes(range=[xmin, xmax])


# -------------------- Dash app --------------------
app = Dash(__name__)
app.title = "Portugal Uniform Hazard Spectra (UHS)"

MODELS = available_models()

app.layout = html.Div(
    style={"maxWidth": "1300px", "margin": "18px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Uniform Hazard Spectra (UHS)"),

        # ===== Section A: compare cities =====
        html.H3("A) Compare cities (same GMM + curve type + rate)", style={"marginTop": "10px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("Ground motion model (GMM)"),
                    dcc.Dropdown(
                        id="a-dd-model",
                        options=[{"label": m, "value": m} for m in MODELS],
                        value=MODELS[0] if MODELS else None,
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("UHS type"),
                    dcc.Dropdown(id="a-dd-kind", options=[], value=None, clearable=False),
                ]),
                html.Div([
                    html.Label("Rate"),
                    dcc.Dropdown(id="a-dd-rate", options=[], value=None, clearable=False),
                ]),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2.2fr", "gap": "14px", "marginTop": "10px"},
            children=[
                html.Div(children=[
                    html.H4("Cities", style={"marginTop": "0px"}),
                    dcc.Checklist(
                        id="a-ck-cities",
                        options=[{"label": name, "value": name} for _, _, name in CITIES],
                        value=["Lisbon"],
                        labelStyle={"display": "block"},
                    ),
                    html.Div(style={"height": "12px"}),
                    dcc.Checklist(
                        id="a-ck-axes",
                        options=[
                            {"label": "Log scale (X)", "value": "logx"},
                            {"label": "Log scale (Y)", "value": "logy"},
                        ],
                        value=["logx"],
                        labelStyle={"display": "block"},
                    ),
                    dcc.Markdown("**Period range (s)**"),
                    dcc.RangeSlider(
                        id="a-xrange",
                        min=PERIOD_MIN,
                        max=5.0,
                        step=0.001,
                        value=[PERIOD_MIN, 3.0],
                        tooltip={"placement": "bottom", "always_visible": False},
                        allowCross=False,
                    ),
                    html.Div(id="a-status", style={"marginTop": "12px", "color": "#555"}),
                ]),
                html.Div(children=[
                    dcc.Graph(id="a-graph", style={"height": "72vh"}),
                ]),
            ],
        ),

        html.Hr(style={"marginTop": "22px", "marginBottom": "18px"}),

        # ===== Section B: compare GMMs =====
        html.H3("B) Compare GMMs (same city + curve type + rate)", style={"marginTop": "0px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("City"),
                    dcc.Dropdown(
                        id="b-dd-city",
                        options=[{"label": name, "value": name} for _, _, name in CITIES],
                        value="Lisbon",
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("UHS type"),
                    dcc.Dropdown(id="b-dd-kind", options=[], value=None, clearable=False),
                ]),
                html.Div([
                    html.Label("Rate"),
                    dcc.Dropdown(id="b-dd-rate", options=[], value=None, clearable=False),
                ]),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2.2fr", "gap": "14px", "marginTop": "10px"},
            children=[
                html.Div(children=[
                    html.H4("GMMs", style={"marginTop": "0px"}),
                    dcc.Checklist(
                        id="b-ck-models",
                        options=[{"label": m, "value": m} for m in MODELS],
                        value=MODELS[:2] if len(MODELS) >= 2 else MODELS,
                        labelStyle={"display": "block"},
                    ),
                    html.Div(style={"height": "12px"}),
                    dcc.Checklist(
                        id="b-ck-axes",
                        options=[
                            {"label": "Log scale (X)", "value": "logx"},
                            {"label": "Log scale (Y)", "value": "logy"},
                        ],
                        value=["logx"],
                        labelStyle={"display": "block"},
                    ),
                    dcc.Markdown("**Period range (s)**"),
                    dcc.RangeSlider(
                        id="b-xrange",
                        min=PERIOD_MIN,
                        max=5.0,
                        step=0.001,
                        value=[PERIOD_MIN, 3.0],
                        tooltip={"placement": "bottom", "always_visible": False},
                        allowCross=False,
                    ),
                    html.Div(id="b-status", style={"marginTop": "12px", "color": "#555"}),
                ]),
                html.Div(children=[
                    dcc.Graph(id="b-graph", style={"height": "72vh"}),
                ]),
            ],
        ),
    ],
)


# -------------------- Callbacks: Section A --------------------
@app.callback(
    Output("a-dd-kind", "options"),
    Output("a-dd-kind", "value"),
    Input("a-dd-model", "value"),
)
def a_sync_kind(model):
    if not model:
        return [], None
    kinds = available_kinds_for_model(model)
    opts = [{"label": KIND_LABEL.get(k, k), "value": k} for k in kinds]
    return opts, (kinds[0] if kinds else None)


@app.callback(
    Output("a-dd-rate", "options"),
    Output("a-dd-rate", "value"),
    Input("a-dd-model", "value"),
    Input("a-dd-kind", "value"),
)
def a_sync_rate(model, kind):
    if not model or not kind:
        return [], None
    path = UHS_INDEX.get(Key(model=model, kind=kind))
    if not path:
        return [], None
    _, _, rates, _ = load_uhs_file(str(path))
    opts = [{"label": r, "value": r} for r in rates]
    return opts, (rates[0] if rates else None)


@app.callback(
    Output("a-graph", "figure"),
    Output("a-status", "children"),
    Input("a-dd-model", "value"),
    Input("a-dd-kind", "value"),
    Input("a-dd-rate", "value"),
    Input("a-ck-cities", "value"),
    Input("a-ck-axes", "value"),
    Input("a-xrange", "value"),
)
def a_update(model, kind, rate, cities, axes_vals, xrange_vals):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=25, t=55, b=60))

    if not UHS_INDEX:
        fig.add_annotation(text="No UHS CSVs found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"No files discovered under: {SEISMIC_HAZARD_DIR}"

    if not model or not kind or not rate or not cities:
        fig.add_annotation(text="Select model, type, rate and at least one city.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select model, type, rate and at least one city."

    path = UHS_INDEX.get(Key(model=model, kind=kind))
    if not path:
        fig.add_annotation(text="Selected UHS file not found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Missing selected UHS file."

    lons, lats, _, spec_by_rate = load_uhs_file(str(path))
    if rate not in spec_by_rate:
        fig.add_annotation(text="Selected rate not present in this file.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Selected rate not present."

    periods, vals = spec_by_rate[rate]

    nearest_info = []
    for city in cities:
        lon_c, lat_c = CITY_BY_NAME[city]
        si = nearest_site_index(lons, lats, lon_c, lat_c)
        y = vals[si, :]

        fig.add_trace(go.Scatter(x=periods, y=y, mode="lines+markers", name=city))
        nearest_info.append(f"{city}->{lats[si]:.3f},{lons[si]:.3f}")

    fig.update_layout(title=f"{model} — {KIND_LABEL.get(kind, kind)} — rate {rate} (Compare cities)")
    fig.update_xaxes(title="Period (s)")
    fig.update_yaxes(title="UHS (g)")

    logx = "logx" in (axes_vals or [])
    logy = "logy" in (axes_vals or [])
    apply_axes(fig, logx=logx, logy=logy, x_range=xrange_vals)

    return fig, "Nearest sites: " + " | ".join(nearest_info)


# -------------------- Callbacks: Section B --------------------
@app.callback(
    Output("b-dd-kind", "options"),
    Output("b-dd-kind", "value"),
    Input("b-ck-models", "value"),
)
def b_sync_kind(models):
    if not models:
        return [], None
    kinds = available_kinds_for_models(models)
    opts = [{"label": KIND_LABEL.get(k, k), "value": k} for k in kinds]
    return opts, (kinds[0] if kinds else None)


@app.callback(
    Output("b-dd-rate", "options"),
    Output("b-dd-rate", "value"),
    Input("b-dd-kind", "value"),
    Input("b-ck-models", "value"),
)
def b_sync_rate(kind, models):
    if not kind or not models:
        return [], None
    for m in models:
        path = UHS_INDEX.get(Key(model=m, kind=kind))
        if path:
            _, _, rates, _ = load_uhs_file(str(path))
            opts = [{"label": r, "value": r} for r in rates]
            return opts, (rates[0] if rates else None)
    return [], None


@app.callback(
    Output("b-graph", "figure"),
    Output("b-status", "children"),
    Input("b-dd-city", "value"),
    Input("b-dd-kind", "value"),
    Input("b-dd-rate", "value"),
    Input("b-ck-models", "value"),
    Input("b-ck-axes", "value"),
    Input("b-xrange", "value"),
)
def b_update(city, kind, rate, models, axes_vals, xrange_vals):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=25, t=55, b=60))

    if not UHS_INDEX:
        fig.add_annotation(text="No UHS CSVs found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"No files discovered under: {SEISMIC_HAZARD_DIR}"

    if not city or not kind or not rate or not models:
        fig.add_annotation(text="Select city, type, rate and at least one GMM.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select city, type, rate and at least one GMM."

    lon_c, lat_c = CITY_BY_NAME[city]
    missing = []
    nearest_info = []

    for model in models:
        path = UHS_INDEX.get(Key(model=model, kind=kind))
        if not path:
            missing.append(model)
            continue

        lons, lats, _, spec_by_rate = load_uhs_file(str(path))
        if rate not in spec_by_rate:
            missing.append(f"{model}(no rate)")
            continue

        periods, vals = spec_by_rate[rate]
        si = nearest_site_index(lons, lats, lon_c, lat_c)
        y = vals[si, :]

        fig.add_trace(go.Scatter(x=periods, y=y, mode="lines+markers", name=model))
        nearest_info.append(f"{model}->{lats[si]:.3f},{lons[si]:.3f}")

    fig.update_layout(title=f"{city} — {KIND_LABEL.get(kind, kind)} — rate {rate} (Compare GMMs)")
    fig.update_xaxes(title="Period (s)")
    fig.update_yaxes(title="UHS (g)")

    logx = "logx" in (axes_vals or [])
    logy = "logy" in (axes_vals or [])
    apply_axes(fig, logx=logx, logy=logy, x_range=xrange_vals)

    status_bits = ["Nearest sites: " + " | ".join(nearest_info)]
    if missing:
        status_bits.append(f"Missing: {', '.join(missing[:6])}" + (" ..." if len(missing) > 6 else ""))
    return fig, " — ".join(status_bits)


if __name__ == "__main__":
    app.run(debug=True, port=8052)