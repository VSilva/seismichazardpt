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


# -------------------- Constants --------------------
RATES_LINES = [0.002105, 0.000404]  # horizontal reference lines on Y axis
RATES_MIN = 1 / 10000              # y-axis minimum
IML_MAX_DEFAULT = 10.0             # default x-axis maximum
IML_MIN_DEFAULT = 1e-3             # default x-axis minimum (slider start)


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
    # Notebook / IPython: assume current working dir is Mapping/dashboard
    ROOT = Path.cwd().parents[1]         # dashboard (cwd) -> Mapping -> ROOT

SEISMIC_HAZARD_DIR = ROOT / "seismic_hazard"


# -------------------- Labels --------------------
KIND_LABEL = {
    "mean": "Mean",
    "q05": "Quantile 5%",
    "q95": "Quantile 95%",
}


# -------------------- Filename parsing --------------------
# Example: hazard_curve-mean-PGA_3.csv -> kind=mean, im=PGA, ignore _3

TRAILING_CALCID_RE = re.compile(r"^(?P<base>.+?)_(?P<id>\d+)\.csv$", re.IGNORECASE)

MEAN_RE = re.compile(r"(?i).*mean[-_](?P<im>.+)$")
Q05_RE = re.compile(r"(?i).*(0\.05)[-_](?P<im>.+)$")
Q95_RE = re.compile(r"(?i).*(0\.95)[-_](?P<im>.+)$")


def strip_calcid(stem_or_name: str) -> str:
    m = TRAILING_CALCID_RE.match(stem_or_name)
    if m:
        return m.group("base")
    m2 = re.match(r"^(?P<base>.+?)_(?P<id>\d+)$", stem_or_name)
    if m2:
        return m2.group("base")
    return stem_or_name.replace(".csv", "")


def parse_kind_and_im_from_filename(csv_path: Path) -> Tuple[Optional[str], Optional[str]]:
    base = strip_calcid(csv_path.name).replace(".csv", "")
    low = base.lower()

    kind = None
    if "mean" in low:
        kind = "mean"
    elif "0.05" in low:
        kind = "q05"
    elif "0.95" in low:
        kind = "q95"
    if not kind:
        return None, None

    if kind == "mean":
        m = MEAN_RE.match(base)
    elif kind == "q05":
        m = Q05_RE.match(base)
    else:
        m = Q95_RE.match(base)

    if not m:
        return kind, None

    im = m.group("im").strip()
    return kind, im


# -------------------- OpenQuake CSV reading --------------------
META_IMT_RE = re.compile(r"imt='([^']+)'", re.IGNORECASE)

def read_openquake_csv(path: Path) -> pd.DataFrame:
    # OQ: first line is metadata starting with '#'
    df = pd.read_csv(path, comment="#")
    return df.dropna(axis=1, how="all")


def infer_im_from_metadata(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
    except Exception:
        return None
    m = META_IMT_RE.search(first)
    return m.group(1) if m else None


# -------------------- Index discovery --------------------
@dataclass(frozen=True)
class Key:
    model: str
    im: str
    kind: str  # mean/q05/q95


def discover_curve_files() -> Dict[Key, Path]:
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
            kind, im = parse_kind_and_im_from_filename(csv_path)
            if im is None:
                im = infer_im_from_metadata(csv_path)

            if kind is None or im is None:
                continue

            idx[Key(model=model_dir.name, im=im, kind=kind)] = csv_path

    return idx


CURVE_INDEX = discover_curve_files()


# -------------------- Curve parsing: lon/lat + poe-IML columns --------------------
POE_COL_RE = re.compile(r"^(poe|afe)-(?P<iml>\d+(?:\.\d+)?)$", re.IGNORECASE)

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


def extract_imls_and_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_col, lat_col = find_lon_lat_columns(df)
    lons = df[lon_col].to_numpy(dtype=float)
    lats = df[lat_col].to_numpy(dtype=float)

    iml_cols = []
    imls = []
    for c in df.columns:
        m = POE_COL_RE.match(str(c))
        if not m:
            continue
        iml_cols.append(c)
        imls.append(float(m.group("iml")))

    if not iml_cols:
        raise ValueError("No poe-<IML> (or afe-<IML>) columns found.")

    imls = np.array(imls, dtype=float)
    values = df[iml_cols].to_numpy(dtype=float)

    order = np.argsort(imls)
    return imls[order], values[:, order], lons, lats


def nearest_site_index(lons: np.ndarray, lats: np.ndarray, lon: float, lat: float) -> int:
    d2 = (lons - lon) ** 2 + (lats - lat) ** 2
    return int(np.argmin(d2))


@lru_cache(maxsize=96)
def load_curve_file(path_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = read_openquake_csv(Path(path_str))
    return extract_imls_and_values(df)


# -------------------- Option utilities --------------------
def available_models() -> List[str]:
    return sorted({k.model for k in CURVE_INDEX})

def available_ims_for_model(model: str) -> List[str]:
    return sorted({k.im for k in CURVE_INDEX if k.model == model})

def available_kinds_for_model_im(model: str, im: str) -> List[str]:
    kinds = sorted({k.kind for k in CURVE_INDEX if k.model == model and k.im == im})
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]
    return kinds

def available_ims_global() -> List[str]:
    return sorted({k.im for k in CURVE_INDEX})

def available_kinds_for_im_any_model(im: str, models: List[str]) -> List[str]:
    kinds = sorted({k.kind for k in CURVE_INDEX if k.im == im and k.model in set(models)})
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]
    return kinds


# -------------------- Plot helpers --------------------
def set_axis_limits(fig: go.Figure, x_min: float, x_max: float, y_min: float, y_max: float, logx: bool, logy: bool):
    # X
    if logx:
        x_min = max(x_min, 1e-12)  # avoid log(0)
        fig.update_xaxes(type="log", range=[np.log10(x_min), np.log10(x_max)])
    else:
        fig.update_xaxes(type="linear", range=[x_min, x_max])

    # Y
    if logy:
        y_min = max(y_min, 1e-12)
        fig.update_yaxes(type="log", range=[np.log10(y_min), np.log10(y_max)])
    else:
        fig.update_yaxes(type="linear", range=[y_min, y_max])


def add_rate_lines(fig: go.Figure, x_min: float, x_max: float):
    for r in RATES_LINES:
        fig.add_shape(
            type="line",
            x0=x_min, x1=x_max,
            y0=r, y1=r,
            xref="x", yref="y",
            line=dict(width=1, dash="dot"),
        )
        fig.add_annotation(
            x=x_max, y=r,
            xref="x", yref="y",
            text=f"{r}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11),
        )


# -------------------- Dash app --------------------
app = Dash(__name__)
app.title = "Portugal Seismic Hazard Curves"

MODELS = available_models()

app.layout = html.Div(
    style={"maxWidth": "1300px", "margin": "18px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Seismic Hazard Curves"),

        # ===== Section 1: compare cities (same model + IM) =====
        html.H3("A) Compare cities (same GMM + IM)", style={"marginTop": "10px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("Ground motion model (GMM)"),
                    dcc.Dropdown(
                        id="s1-dd-model",
                        options=[{"label": m, "value": m} for m in MODELS],
                        value=MODELS[0] if MODELS else None,
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Intensity measure (IM)"),
                    dcc.Dropdown(id="s1-dd-im", options=[], value=None, clearable=False),
                ]),
                html.Div([
                    html.Label("Curve types"),
                    dcc.Checklist(id="s1-ck-kinds", options=[], value=[], labelStyle={"display": "block"}),
                    html.Div("Mean is always present; quantiles appear only if available.",
                             style={"fontSize": "0.85rem", "color": "#666", "marginTop": "4px"}),
                ]),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2.2fr", "gap": "14px", "marginTop": "10px"},
            children=[
                html.Div(children=[
                    html.H4("Cities", style={"marginTop": "0px"}),
                    dcc.Checklist(
                        id="s1-ck-cities",
                        options=[{"label": name, "value": name} for _, _, name in CITIES],
                        value=["Lisbon"],
                        labelStyle={"display": "block"},
                    ),
                    html.Div(style={"height": "12px"}),
                    dcc.Checklist(
                        id="s1-ck-axes",
                        options=[
                            {"label": "Log scale (X)", "value": "logx"},
                            {"label": "Log scale (Y)", "value": "logy"},
                        ],
                        value=["logx", "logy"],   # log X active by default
                        labelStyle={"display": "block"},
                    ),
                    dcc.Markdown("**X-axis range (IML)**"),
                    dcc.RangeSlider(
                        id="s1-xrange",
                        min=1e-4,
                        max=IML_MAX_DEFAULT,
                        step=1e-4,
                        value=[IML_MIN_DEFAULT, IML_MAX_DEFAULT],
                        tooltip={"placement": "bottom", "always_visible": False},
                        allowCross=False,
                    ),
                    html.Div(id="s1-status", style={"marginTop": "12px", "color": "#555"}),
                ]),
                html.Div(children=[
                    dcc.Graph(id="s1-graph", style={"height": "72vh"}),
                ]),
            ],
        ),

        html.Hr(style={"marginTop": "22px", "marginBottom": "18px"}),

        # ===== Section 2: compare models (same city + IM) =====
        html.H3("B) Compare GMMs (same city + IM)", style={"marginTop": "0px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("City"),
                    dcc.Dropdown(
                        id="s2-dd-city",
                        options=[{"label": name, "value": name} for _, _, name in CITIES],
                        value="Lisbon",
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Intensity measure (IM)"),
                    dcc.Dropdown(
                        id="s2-dd-im",
                        options=[{"label": im, "value": im} for im in available_ims_global()],
                        value=(available_ims_global()[0] if available_ims_global() else None),
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Curve types"),
                    dcc.Checklist(id="s2-ck-kinds", options=[], value=[], labelStyle={"display": "block"}),
                ]),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2.2fr", "gap": "14px", "marginTop": "10px"},
            children=[
                html.Div(children=[
                    html.H4("GMMs", style={"marginTop": "0px"}),
                    dcc.Checklist(
                        id="s2-ck-models",
                        options=[{"label": m, "value": m} for m in MODELS],
                        value=MODELS[:2] if len(MODELS) >= 2 else MODELS,
                        labelStyle={"display": "block"},
                    ),
                    html.Div(style={"height": "12px"}),
                    dcc.Checklist(
                        id="s2-ck-axes",
                        options=[
                            {"label": "Log scale (X)", "value": "logx"},
                            {"label": "Log scale (Y)", "value": "logy"},
                        ],
                        value=["logx", "logy"],   # log X active by default
                        labelStyle={"display": "block"},
                    ),
                    dcc.Markdown("**X-axis range (IML)**"),
                    dcc.RangeSlider(
                        id="s2-xrange",
                        min=1e-4,
                        max=IML_MAX_DEFAULT,
                        step=1e-4,
                        value=[IML_MIN_DEFAULT, IML_MAX_DEFAULT],
                        tooltip={"placement": "bottom", "always_visible": False},
                        allowCross=False,
                    ),
                    html.Div(id="s2-status", style={"marginTop": "12px", "color": "#555"}),
                ]),
                html.Div(children=[
                    dcc.Graph(id="s2-graph", style={"height": "72vh"}),
                ]),
            ],
        ),
    ],
)


# -------------------- Section 1 callbacks --------------------
@app.callback(
    Output("s1-dd-im", "options"),
    Output("s1-dd-im", "value"),
    Input("s1-dd-model", "value"),
)
def s1_sync_im(model):
    if not model:
        return [], None
    ims = available_ims_for_model(model)
    return [{"label": im, "value": im} for im in ims], (ims[0] if ims else None)


@app.callback(
    Output("s1-ck-kinds", "options"),
    Output("s1-ck-kinds", "value"),
    Input("s1-dd-model", "value"),
    Input("s1-dd-im", "value"),
)
def s1_sync_kinds(model, im):
    if not model or not im:
        return [], []
    kinds = available_kinds_for_model_im(model, im)
    opts = [{"label": KIND_LABEL.get(k, k), "value": k} for k in kinds]
    default = ["mean"] if "mean" in kinds else (kinds[:1] if kinds else [])
    return opts, default


@app.callback(
    Output("s1-graph", "figure"),
    Output("s1-status", "children"),
    Input("s1-dd-model", "value"),
    Input("s1-dd-im", "value"),
    Input("s1-ck-kinds", "value"),
    Input("s1-ck-cities", "value"),
    Input("s1-ck-axes", "value"),
    Input("s1-xrange", "value"),
)
def s1_update_plot(model, im, kinds, cities, axes_vals, xrange_vals):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=25, t=55, b=60))

    if not CURVE_INDEX:
        fig.add_annotation(text="No curve CSVs found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"No files discovered under: {SEISMIC_HAZARD_DIR}"

    if not model or not im or not cities or not kinds:
        fig.add_annotation(text="Select model, IM, curve type(s) and at least one city.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select model, IM, curve type(s) and at least one city."

    kinds = list(dict.fromkeys(kinds))
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]

    missing = []
    total_traces = 0
    x_min_data = None

    for kind in kinds:
        path = CURVE_INDEX.get(Key(model=model, im=im, kind=kind))
        if not path:
            missing.append(kind)
            continue

        imls, values, lons, lats = load_curve_file(str(path))
        x_min_data = float(imls[0]) if x_min_data is None else min(x_min_data, float(imls[0]))

        for city in cities:
            lon_c, lat_c = CITY_BY_NAME[city]
            si = nearest_site_index(lons, lats, lon_c, lat_c)
            y = values[si, :]

            name = f"{city} — {KIND_LABEL.get(kind, kind)}"
            line_dash = "solid" if kind == "mean" else "dash"
            fig.add_trace(go.Scatter(x=imls, y=y, mode="lines", name=name, line=dict(dash=line_dash)))
            total_traces += 1

    fig.update_layout(title=f"{model} — {im} (Compare cities)")
    fig.update_xaxes(title=f"IML ({im})")
    fig.update_yaxes(title="PoE")

    logx = "logx" in (axes_vals or [])
    logy = "logy" in (axes_vals or [])

    # Slider overrides
    x_min = float(xrange_vals[0]) if xrange_vals else (x_min_data if x_min_data is not None else 1e-6)
    x_max = float(xrange_vals[1]) if xrange_vals else IML_MAX_DEFAULT

    y_min = RATES_MIN
    y_max = 1.0

    set_axis_limits(fig, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, logx=logx, logy=logy)
    add_rate_lines(fig, x_min=x_min, x_max=x_max)

    status_bits = [f"Traces: {total_traces}"]
    if missing:
        status_bits.append("Missing types: " + ", ".join(KIND_LABEL.get(k, k) for k in missing))
    return fig, " — ".join(status_bits)


# -------------------- Section 2 callbacks --------------------
@app.callback(
    Output("s2-ck-kinds", "options"),
    Output("s2-ck-kinds", "value"),
    Input("s2-dd-im", "value"),
    Input("s2-ck-models", "value"),
)
def s2_sync_kinds(im, models):
    if not im or not models:
        return [], []
    kinds = available_kinds_for_im_any_model(im, models)
    opts = [{"label": KIND_LABEL.get(k, k), "value": k} for k in kinds]
    default = ["mean"] if "mean" in kinds else (kinds[:1] if kinds else [])
    return opts, default


@app.callback(
    Output("s2-graph", "figure"),
    Output("s2-status", "children"),
    Input("s2-dd-city", "value"),
    Input("s2-dd-im", "value"),
    Input("s2-ck-kinds", "value"),
    Input("s2-ck-models", "value"),
    Input("s2-ck-axes", "value"),
    Input("s2-xrange", "value"),
)
def s2_update_plot(city, im, kinds, models, axes_vals, xrange_vals):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=25, t=55, b=60))

    if not CURVE_INDEX:
        fig.add_annotation(text="No curve CSVs found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"No files discovered under: {SEISMIC_HAZARD_DIR}"

    if not city or not im or not models or not kinds:
        fig.add_annotation(text="Select city, IM, curve type(s) and at least one GMM.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select city, IM, curve type(s) and at least one GMM."

    kinds = list(dict.fromkeys(kinds))
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]

    lon_c, lat_c = CITY_BY_NAME[city]
    missing = []
    total_traces = 0
    x_min_data = None

    for model in models:
        for kind in kinds:
            path = CURVE_INDEX.get(Key(model=model, im=im, kind=kind))
            if not path:
                missing.append(f"{model}:{kind}")
                continue

            imls, values, lons, lats = load_curve_file(str(path))
            x_min_data = float(imls[0]) if x_min_data is None else min(x_min_data, float(imls[0]))

            si = nearest_site_index(lons, lats, lon_c, lat_c)
            y = values[si, :]

            name = f"{model} — {KIND_LABEL.get(kind, kind)}"
            line_dash = "solid" if kind == "mean" else "dash"
            fig.add_trace(go.Scatter(x=imls, y=y, mode="lines", name=name, line=dict(dash=line_dash)))
            total_traces += 1

    fig.update_layout(title=f"{city} — {im} (Compare GMMs)")
    fig.update_xaxes(title=f"IML ({im})")
    fig.update_yaxes(title="PoE")

    logx = "logx" in (axes_vals or [])
    logy = "logy" in (axes_vals or [])

    # Slider overrides
    x_min = float(xrange_vals[0]) if xrange_vals else (x_min_data if x_min_data is not None else 1e-6)
    x_max = float(xrange_vals[1]) if xrange_vals else IML_MAX_DEFAULT

    y_min = RATES_MIN
    y_max = 1.0

    set_axis_limits(fig, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, logx=logx, logy=logy)
    add_rate_lines(fig, x_min=x_min, x_max=x_max)

    status_bits = [f"Traces: {total_traces}"]
    if missing:
        status_bits.append(f"Missing combos: {len(missing)}")
    return fig, " — ".join(status_bits)


if __name__ == "__main__":
    app.run(debug=True, port=8051)