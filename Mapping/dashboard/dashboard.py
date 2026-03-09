from __future__ import annotations

from pathlib import Path
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go


# ============================================================
# Paths (works in .py AND notebooks)
# Project layout (as you described):
# <ROOT>/
#   seismic_hazard/<GMM>/outputs/*.csv
#   Mapping/maps/*.png
#   Mapping/dashboard/app_all.py   (this file)
# ============================================================
try:
    THIS_FILE = Path(__file__).resolve()
    ROOT = THIS_FILE.parents[2]          # dashboard -> Mapping -> ROOT
    DASHBOARD_DIR = THIS_FILE.parent
except NameError:
    # Notebook / IPython: assume cwd is Mapping/dashboard
    DASHBOARD_DIR = Path.cwd()
    ROOT = DASHBOARD_DIR.parents[1]

SEISMIC_HAZARD_DIR = ROOT / "seismic_hazard"
MAPS_DIR = ROOT / "Mapping" / "maps"

# Direct-download link (dl=1)
OQ_HAZARD_MODEL_ZIP_URL = (
    "https://www.dropbox.com/scl/fi/n4mudne13tgc7g6oeqjw2/GMMLogicTree.zip"
    "?rlkey=bst4xmpcyijm8z7q7t3x2f58d&dl=1"
)


# ============================================================
# Shared: Cities
# ============================================================
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


# ============================================================
# Shared helpers
# ============================================================
def pick_first(avail: List[str], cur: Optional[str]) -> Optional[str]:
    if not avail:
        return None
    return cur if cur in avail else avail[0]


def nearest_site_index(lons: np.ndarray, lats: np.ndarray, lon: float, lat: float) -> int:
    d2 = (lons - lon) ** 2 + (lats - lat) ** 2
    return int(np.argmin(d2))


def _filter_by_range(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return (x >= xmin) & (x <= xmax)


# ============================================================
# ========================  MAPS  ============================
# ============================================================
MAPTYPE_LABEL = {
    "hazard_map-mean": "Mean",
    "quantile_map-0.05": "Quantile 5%",
    "quantile_map-0.95": "Quantile 95%",
}

MAP_FILENAME_RE = re.compile(
    r"^(?P<gmm>.+?)_"
    r"(?P<maptype>hazard_map-mean|quantile_map-0\.05|quantile_map-0\.95)_"
    r"(?P<im>.+?)_"
    r"(?P<rate>\d+(?:\.\d+)?)"
    r"\.png$"
)

@dataclass(frozen=True)
class MapKey:
    gmm: str
    maptype: str
    im: str
    rate: str


def discover_maps(folder: Path) -> Dict[MapKey, Path]:
    idx: Dict[MapKey, Path] = {}
    if not folder.exists():
        return idx
    for p in folder.glob("*.png"):
        m = MAP_FILENAME_RE.match(p.name)
        if not m:
            continue
        idx[MapKey(
            gmm=m.group("gmm"),
            maptype=m.group("maptype"),
            im=m.group("im"),
            rate=m.group("rate"),
        )] = p
    return idx


MAP_INDEX = discover_maps(MAPS_DIR)


def map_options_from_index(
    gmm: str | None = None,
    maptype: str | None = None,
    im: str | None = None,
    rate: str | None = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    keys = list(MAP_INDEX.keys())

    def ok(k: MapKey) -> bool:
        return ((gmm is None or k.gmm == gmm) and
                (maptype is None or k.maptype == maptype) and
                (im is None or k.im == im) and
                (rate is None or k.rate == rate))

    filt = [k for k in keys if ok(k)]
    gmms = sorted({k.gmm for k in filt})
    maptypes = sorted({k.maptype for k in filt})
    ims = sorted({k.im for k in filt})
    rates = sorted({k.rate for k in filt}, key=lambda x: float(x))
    return gmms, maptypes, ims, rates


def make_image_figure(img_path: Path, title: str) -> go.Figure:
    img = Image.open(img_path)
    fig = go.Figure()
    # Center the image in the plot area
    fig.add_layout_image(dict(
        source=img,
        x=0.5, y=0.5, xref="paper", yref="paper",
        sizex=1, sizey=1,
        xanchor="center", yanchor="middle",
        layer="below",
    ))
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1], scaleanchor="x")
    fig.update_layout(template="plotly_white", title=title, margin=dict(l=10, r=10, t=55, b=10))
    return fig


# ============================================================
# =====================  HAZARD CURVES  ======================
# ============================================================
RATES_LINES = [0.002105, 0.000404]
RATES_MIN = 1 / 10000
IML_MAX_DEFAULT = 10.0
IML_MIN_DEFAULT = 1e-3

KIND_LABEL = {"mean": "Mean", "q05": "Quantile 5%", "q95": "Quantile 95%"}

TRAILING_CALCID_RE = re.compile(r"^(?P<base>.+?)_(?P<id>\d+)\.csv$", re.IGNORECASE)
MEAN_RE = re.compile(r"(?i).*mean[-_](?P<im>.+)$")
Q05_RE = re.compile(r"(?i).*(0\.05)[-_](?P<im>.+)$")
Q95_RE = re.compile(r"(?i).*(0\.95)[-_](?P<im>.+)$")
META_IMT_RE = re.compile(r"imt='([^']+)'", re.IGNORECASE)

POE_COL_RE = re.compile(r"^(poe|afe)-(?P<iml>\d+(?:\.\d+)?)$", re.IGNORECASE)

@dataclass(frozen=True)
class CurveKey:
    model: str
    im: str
    kind: str  # mean/q05/q95


def strip_calcid(stem_or_name: str) -> str:
    m = TRAILING_CALCID_RE.match(stem_or_name)
    if m:
        return m.group("base")
    m2 = re.match(r"^(?P<base>.+?)_(?P<id>\d+)$", stem_or_name)
    if m2:
        return m2.group("base")
    return stem_or_name.replace(".csv", "")


def parse_curve_kind_and_im(csv_path: Path) -> Tuple[Optional[str], Optional[str]]:
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

    im = m.group("im").strip() if m else None
    return kind, im


def infer_im_from_metadata(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
    except Exception:
        return None
    m = META_IMT_RE.search(first)
    return m.group(1) if m else None


def read_openquake_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#").dropna(axis=1, how="all")


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


def extract_curve_imls_and_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_col, lat_col = find_lon_lat_columns(df)
    lons = df[lon_col].to_numpy(dtype=float)
    lats = df[lat_col].to_numpy(dtype=float)

    iml_cols, imls = [], []
    for c in df.columns:
        m = POE_COL_RE.match(str(c))
        if not m:
            continue
        iml_cols.append(c)
        imls.append(float(m.group("iml")))

    if not iml_cols:
        raise ValueError("No poe-<IML> (or afe-<IML>) columns found in hazard curve CSV.")

    imls = np.array(imls, dtype=float)
    values = df[iml_cols].to_numpy(dtype=float)

    order = np.argsort(imls)
    return imls[order], values[:, order], lons, lats


def discover_curve_files() -> Dict[CurveKey, Path]:
    idx: Dict[CurveKey, Path] = {}
    if not SEISMIC_HAZARD_DIR.exists():
        return idx

    for model_dir in sorted(SEISMIC_HAZARD_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        outputs = model_dir / "outputs"
        if not outputs.exists():
            continue

        for csv_path in outputs.glob("*.csv"):
            name_low = csv_path.name.lower()
            # Keep only hazard curve files
            if "curve" not in name_low or "uhs" in name_low:
                continue

            kind, im = parse_curve_kind_and_im(csv_path)
            if im is None:
                im = infer_im_from_metadata(csv_path)

            if kind is None or im is None:
                continue

            idx[CurveKey(model=model_dir.name, im=im, kind=kind)] = csv_path

    return idx


CURVE_INDEX = discover_curve_files()


@lru_cache(maxsize=96)
def load_curve_file(path_str: str):
    df = read_openquake_csv(Path(path_str))
    return extract_curve_imls_and_values(df)


def curves_available_models() -> List[str]:
    return sorted({k.model for k in CURVE_INDEX})


def curves_available_ims_for_model(model: str) -> List[str]:
    return sorted({k.im for k in CURVE_INDEX if k.model == model})


def curves_available_kinds_for_model_im(model: str, im: str) -> List[str]:
    kinds = sorted({k.kind for k in CURVE_INDEX if k.model == model and k.im == im})
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]
    return kinds


def curves_available_ims_global() -> List[str]:
    return sorted({k.im for k in CURVE_INDEX})


def curves_available_kinds_for_im_any_model(im: str, models: List[str]) -> List[str]:
    kinds = sorted({k.kind for k in CURVE_INDEX if k.im == im and k.model in set(models)})
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]
    return kinds


def set_axis_limits(fig: go.Figure, x_min: float, x_max: float, y_min: float, y_max: float, logx: bool, logy: bool):
    if logx:
        x_min = max(x_min, 1e-12)
        fig.update_xaxes(type="log", range=[np.log10(x_min), np.log10(x_max)])
    else:
        fig.update_xaxes(type="linear", range=[x_min, x_max])

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


# ============================================================
# ===========================  UHS  ==========================
# ============================================================
PGA_PERIOD = 0.01
PERIOD_MIN = 0.01

UHS_COL_RE = re.compile(r"^(?P<rate>\d+(?:\.\d+)?)~(?P<imt>.+)$")
SA_RE = re.compile(r"^SA\((?P<T>\d+(?:\.\d+)?)\)$", re.IGNORECASE)

@dataclass(frozen=True)
class UHSKey:
    model: str
    kind: str  # mean/q05/q95


def parse_uhs_kind_from_filename(csv_path: Path) -> Optional[str]:
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


def discover_uhs_files() -> Dict[UHSKey, Path]:
    idx: Dict[UHSKey, Path] = {}
    if not SEISMIC_HAZARD_DIR.exists():
        return idx

    for model_dir in sorted(SEISMIC_HAZARD_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        outputs = model_dir / "outputs"
        if not outputs.exists():
            continue

        for csv_path in outputs.glob("*.csv"):
            name_low = csv_path.name.lower()
            if "uhs" not in name_low:
                continue
            kind = parse_uhs_kind_from_filename(csv_path)
            if kind is None:
                continue
            idx[UHSKey(model=model_dir.name, kind=kind)] = csv_path

    return idx


UHS_INDEX = discover_uhs_files()


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
                continue

        cols_by_rate.setdefault(rate, []).append((T, c))

    if not cols_by_rate:
        raise ValueError("No '<rate>~PGA' or '<rate>~SA(T)' columns found in UHS CSV.")

    rates = sorted(cols_by_rate.keys(), key=lambda x: float(x))
    spec_by_rate = {}
    for rate in rates:
        pairs = sorted(cols_by_rate[rate], key=lambda t: t[0])
        periods = np.array([p for p, _ in pairs], dtype=float)
        cols = [col for _, col in pairs]
        values = df[cols].to_numpy(dtype=float)  # n_sites x n_periods
        spec_by_rate[rate] = (periods, values)

    return lons, lats, rates, spec_by_rate


@lru_cache(maxsize=48)
def load_uhs_file(path_str: str):
    df = read_openquake_csv(Path(path_str))
    return parse_uhs_structure(df)


def uhs_available_models() -> List[str]:
    return sorted({k.model for k in UHS_INDEX})


def uhs_available_kinds_for_model(model: str) -> List[str]:
    kinds = sorted({k.kind for k in UHS_INDEX if k.model == model})
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]
    return kinds


def uhs_available_kinds_for_models(models: List[str]) -> List[str]:
    kinds = sorted({k.kind for k in UHS_INDEX if k.model in set(models)})
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]
    return kinds


def uhs_apply_axes(fig: go.Figure, logx: bool, logy: bool, x_range: List[float] | None):
    fig.update_xaxes(type="log" if logx else "linear")
    fig.update_yaxes(type="log" if logy else "linear")

    if x_range and len(x_range) == 2:
        xmin, xmax = float(x_range[0]), float(x_range[1])
        xmin = max(xmin, PERIOD_MIN)
        if logx:
            fig.update_xaxes(range=[np.log10(xmin), np.log10(xmax)])
        else:
            fig.update_xaxes(range=[xmin, xmax])


# ============================================================
# =========================== DASH ===========================
# ============================================================
app = Dash(__name__)
app.title = "Portugal Seismic Hazard Dashboard"



# ---------- Initial values ----------
MAP_GMMS, MAP_MAPTYPES, MAP_IMS, MAP_RATES = map_options_from_index()
CURVE_MODELS = curves_available_models()
CURVE_IMS_GLOBAL = curves_available_ims_global()
UHS_MODELS = uhs_available_models()


def layout_maps_tab():
    return html.Div(
        style={"maxWidth": "1250px", "margin": "10px auto", "fontFamily": "Arial"},
        children=[
            html.H3("Hazard Maps"),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(220px, 1fr))", "gap": "12px"},
                children=[
                    html.Div([
                        html.Label("Ground motion model"),
                        dcc.Dropdown(id="map-gmm",
                                     options=[{"label": x, "value": x} for x in MAP_GMMS],
                                     value=(MAP_GMMS[0] if MAP_GMMS else None),
                                     clearable=False),
                    ]),
                    html.Div([
                        html.Label("Map type"),
                        dcc.Dropdown(id="map-type",
                                     options=[{"label": MAPTYPE_LABEL.get(x, x), "value": x} for x in MAP_MAPTYPES],
                                     value=(MAP_MAPTYPES[0] if MAP_MAPTYPES else None),
                                     clearable=False),
                    ]),
                    html.Div([
                        html.Label("Intensity measure"),
                        dcc.Dropdown(id="map-im",
                                     options=[{"label": x, "value": x} for x in MAP_IMS],
                                     value=(MAP_IMS[0] if MAP_IMS else None),
                                     clearable=False),
                    ]),
                    html.Div([
                        html.Label("Rate"),
                        dcc.Dropdown(id="map-rate",
                                     options=[{"label": x, "value": x} for x in MAP_RATES],
                                     value=(MAP_RATES[0] if MAP_RATES else None),
                                     clearable=False),
                    ]),
                ],
            ),

            html.Div(style={"height": "10px"}),
            dcc.Graph(id="map-graph", style={"height": "78vh"}),
            html.Div(id="map-status", style={"marginTop": "6px", "color": "#555"}),

            html.Div(
                style={"marginTop": "10px"},
                children=[
                    html.A(
                        "Download OQ Hazard Model for Portugal",
                        href=OQ_HAZARD_MODEL_ZIP_URL,
                        target="_blank",
                        style={
                            "display": "inline-block",
                            "padding": "10px 14px",
                            "border": "1px solid #ccc",
                            "borderRadius": "6px",
                            "textDecoration": "none",
                        },
                    )
                ],
            ),

            html.Div(
                style={"marginTop": "10px", "color": "#777", "fontSize": "0.9rem"},
                children=[
                    html.Div(f"Maps folder: {MAPS_DIR} (exists={MAPS_DIR.exists()})"),
                ],
            ),
        ],
    )


def layout_curves_tab():
    return html.Div(
        style={"maxWidth": "1300px", "margin": "10px auto", "fontFamily": "Arial"},
        children=[
            html.H3("Hazard Curves"),

            html.H4("A) Compare cities (same GMM + IM)", style={"marginTop": "10px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
                children=[
                    html.Div([
                        html.Label("Ground motion model (GMM)"),
                        dcc.Dropdown(id="c1-model",
                                     options=[{"label": m, "value": m} for m in CURVE_MODELS],
                                     value=(CURVE_MODELS[0] if CURVE_MODELS else None),
                                     clearable=False),
                    ]),
                    html.Div([
                        html.Label("Intensity measure (IM)"),
                        dcc.Dropdown(id="c1-im", options=[], value=None, clearable=False),
                    ]),
                    html.Div([
                        html.Label("Curve types"),
                        dcc.Checklist(id="c1-kinds", options=[], value=[], labelStyle={"display": "block"}),
                    ]),
                ],
            ),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2.2fr", "gap": "14px", "marginTop": "10px"},
                children=[
                    html.Div(children=[
                        html.H5("Cities", style={"marginTop": "0px"}),
                        dcc.Checklist(
                            id="c1-cities",
                            options=[{"label": name, "value": name} for _, _, name in CITIES],
                            value=["Lisbon"],
                            labelStyle={"display": "block"},
                        ),
                        html.Div(style={"height": "10px"}),
                        dcc.Checklist(
                            id="c1-axes",
                            options=[
                                {"label": "Log scale (X)", "value": "logx"},
                                {"label": "Log scale (Y)", "value": "logy"},
                            ],
                            value=["logx", "logy"],
                            labelStyle={"display": "block"},
                        ),
                        dcc.Markdown("**X-axis range (IML)**"),
                        dcc.RangeSlider(
                            id="c1-xrange",
                            min=1e-4,
                            max=IML_MAX_DEFAULT,
                            step=1e-4,
                            value=[IML_MIN_DEFAULT, IML_MAX_DEFAULT],
                            tooltip={"placement": "bottom", "always_visible": False},
                            allowCross=False,
                        ),
                        html.Div(id="c1-status", style={"marginTop": "12px", "color": "#555"}),
                    ]),
                    html.Div(children=[
                        dcc.Graph(id="c1-graph", style={"height": "70vh"}),
                        html.Div(style={"marginTop": "8px"}, children=[
                            html.Button("Download CSV (shown curves)", id="c1-download-btn", n_clicks=0),
                            dcc.Download(id="c1-download"),
                        ]),
                    ]),
                ],
            ),

            html.Hr(style={"marginTop": "18px", "marginBottom": "14px"}),

            html.H4("B) Compare GMMs (same city + IM)"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
                children=[
                    html.Div([
                        html.Label("City"),
                        dcc.Dropdown(
                            id="c2-city",
                            options=[{"label": name, "value": name} for _, _, name in CITIES],
                            value="Lisbon",
                            clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("Intensity measure (IM)"),
                        dcc.Dropdown(
                            id="c2-im",
                            options=[{"label": im, "value": im} for im in CURVE_IMS_GLOBAL],
                            value=(CURVE_IMS_GLOBAL[0] if CURVE_IMS_GLOBAL else None),
                            clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("Curve types"),
                        dcc.Checklist(id="c2-kinds", options=[], value=[], labelStyle={"display": "block"}),
                    ]),
                ],
            ),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2.2fr", "gap": "14px", "marginTop": "10px"},
                children=[
                    html.Div(children=[
                        html.H5("GMMs", style={"marginTop": "0px"}),
                        dcc.Checklist(
                            id="c2-models",
                            options=[{"label": m, "value": m} for m in CURVE_MODELS],
                            value=(CURVE_MODELS[:2] if len(CURVE_MODELS) >= 2 else CURVE_MODELS),
                            labelStyle={"display": "block"},
                        ),
                        html.Div(style={"height": "10px"}),
                        dcc.Checklist(
                            id="c2-axes",
                            options=[
                                {"label": "Log scale (X)", "value": "logx"},
                                {"label": "Log scale (Y)", "value": "logy"},
                            ],
                            value=["logx", "logy"],
                            labelStyle={"display": "block"},
                        ),
                        dcc.Markdown("**X-axis range (IML)**"),
                        dcc.RangeSlider(
                            id="c2-xrange",
                            min=1e-4,
                            max=IML_MAX_DEFAULT,
                            step=1e-4,
                            value=[IML_MIN_DEFAULT, IML_MAX_DEFAULT],
                            tooltip={"placement": "bottom", "always_visible": False},
                            allowCross=False,
                        ),
                        html.Div(id="c2-status", style={"marginTop": "12px", "color": "#555"}),
                    ]),
                    html.Div(children=[
                        dcc.Graph(id="c2-graph", style={"height": "70vh"}),
                        html.Div(style={"marginTop": "8px"}, children=[
                            html.Button("Download CSV (shown curves)", id="c2-download-btn", n_clicks=0),
                            dcc.Download(id="c2-download"),
                        ]),
                    ]),
                ],
            ),
        ],
    )


def layout_uhs_tab():
    return html.Div(
        style={"maxWidth": "1300px", "margin": "10px auto", "fontFamily": "Arial"},
        children=[
            html.H3("Uniform Hazard Spectra (UHS)"),

            html.H4("A) Compare cities (same GMM + type + rate)", style={"marginTop": "10px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
                children=[
                    html.Div([
                        html.Label("Ground motion model (GMM)"),
                        dcc.Dropdown(
                            id="u1-model",
                            options=[{"label": m, "value": m} for m in UHS_MODELS],
                            value=(UHS_MODELS[0] if UHS_MODELS else None),
                            clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("UHS type"),
                        dcc.Dropdown(id="u1-kind", options=[], value=None, clearable=False),
                    ]),
                    html.Div([
                        html.Label("Rate"),
                        dcc.Dropdown(id="u1-rate", options=[], value=None, clearable=False),
                    ]),
                ],
            ),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2.2fr", "gap": "14px", "marginTop": "10px"},
                children=[
                    html.Div(children=[
                        html.H5("Cities", style={"marginTop": "0px"}),
                        dcc.Checklist(
                            id="u1-cities",
                            options=[{"label": name, "value": name} for _, _, name in CITIES],
                            value=["Lisbon"],
                            labelStyle={"display": "block"},
                        ),
                        html.Div(style={"height": "10px"}),
                        dcc.Checklist(
                            id="u1-axes",
                            options=[
                                {"label": "Log scale (X)", "value": "logx"},
                                {"label": "Log scale (Y)", "value": "logy"},
                            ],
                            value=["logx"],
                            labelStyle={"display": "block"},
                        ),
                        dcc.Markdown("**Period range (s)**"),
                        dcc.RangeSlider(
                            id="u1-xrange",
                            min=PERIOD_MIN,
                            max=5.0,
                            step=0.001,
                            value=[PERIOD_MIN, 3.0],
                            tooltip={"placement": "bottom", "always_visible": False},
                            allowCross=False,
                        ),
                        html.Div(id="u1-status", style={"marginTop": "12px", "color": "#555"}),
                    ]),
                    html.Div(children=[
                        dcc.Graph(id="u1-graph", style={"height": "70vh"}),
                        html.Div(style={"marginTop": "8px"}, children=[
                            html.Button("Download CSV (shown UHS)", id="u1-download-btn", n_clicks=0),
                            dcc.Download(id="u1-download"),
                        ]),
                    ]),
                ],
            ),

            html.Hr(style={"marginTop": "18px", "marginBottom": "14px"}),

            html.H4("B) Compare GMMs (same city + type + rate)"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
                children=[
                    html.Div([
                        html.Label("City"),
                        dcc.Dropdown(
                            id="u2-city",
                            options=[{"label": name, "value": name} for _, _, name in CITIES],
                            value="Lisbon",
                            clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("UHS type"),
                        dcc.Dropdown(id="u2-kind", options=[], value=None, clearable=False),
                    ]),
                    html.Div([
                        html.Label("Rate"),
                        dcc.Dropdown(id="u2-rate", options=[], value=None, clearable=False),
                    ]),
                ],
            ),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "minmax(330px, 1fr) 2.2fr", "gap": "14px", "marginTop": "10px"},
                children=[
                    html.Div(children=[
                        html.H5("GMMs", style={"marginTop": "0px"}),
                        dcc.Checklist(
                            id="u2-models",
                            options=[{"label": m, "value": m} for m in UHS_MODELS],
                            value=(UHS_MODELS[:2] if len(UHS_MODELS) >= 2 else UHS_MODELS),
                            labelStyle={"display": "block"},
                        ),
                        html.Div(style={"height": "10px"}),
                        dcc.Checklist(
                            id="u2-axes",
                            options=[
                                {"label": "Log scale (X)", "value": "logx"},
                                {"label": "Log scale (Y)", "value": "logy"},
                            ],
                            value=["logx"],
                            labelStyle={"display": "block"},
                        ),
                        dcc.Markdown("**Period range (s)**"),
                        dcc.RangeSlider(
                            id="u2-xrange",
                            min=PERIOD_MIN,
                            max=5.0,
                            step=0.001,
                            value=[PERIOD_MIN, 3.0],
                            tooltip={"placement": "bottom", "always_visible": False},
                            allowCross=False,
                        ),
                        html.Div(id="u2-status", style={"marginTop": "12px", "color": "#555"}),
                    ]),
                    html.Div(children=[
                        dcc.Graph(id="u2-graph", style={"height": "70vh"}),
                        html.Div(style={"marginTop": "8px"}, children=[
                            html.Button("Download CSV (shown UHS)", id="u2-download-btn", n_clicks=0),
                            dcc.Download(id="u2-download"),
                        ]),
                    ]),
                ],
            ),
        ],
    )


app.layout = html.Div(
    style={"fontFamily": "Arial"},
    children=[
        html.H2("Portugal Seismic Hazard Dashboard",
                style={
                    "maxWidth": "1300px",
                    "margin": "18px auto 8px auto",
                    "fontSize": "50px",      # <- increase this
                    "fontWeight": "700",
                    "textAlign": "center",   # <- center it
                        },
),        dcc.Tabs(
            id="tabs",
            value="tab-maps",
            children=[
                dcc.Tab(label="Hazard Maps", value="tab-maps", children=layout_maps_tab()),
                dcc.Tab(label="Hazard Curves", value="tab-curves", children=layout_curves_tab()),
                dcc.Tab(label="Uniform Hazard Spectra", value="tab-uhs", children=layout_uhs_tab()),
            ],
        ),
    ],
)

# ============================================================
# ===================== MAPS callbacks =======================
# ============================================================
@app.callback(
    Output("map-gmm", "options"), Output("map-gmm", "value"),
    Output("map-type", "options"), Output("map-type", "value"),
    Output("map-im", "options"), Output("map-im", "value"),
    Output("map-rate", "options"), Output("map-rate", "value"),
    Input("map-gmm", "value"),
    Input("map-type", "value"),
    Input("map-im", "value"),
    Input("map-rate", "value"),
)
def sync_map_dropdowns(gmm, maptype, im, rate):
    if not MAP_INDEX:
        return [], None, [], None, [], None, [], None

    gmms, _, _, _ = map_options_from_index(maptype=maptype, im=im, rate=rate)
    _, maptypes, _, _ = map_options_from_index(gmm=gmm, im=im, rate=rate)
    _, _, ims, _ = map_options_from_index(gmm=gmm, maptype=maptype, rate=rate)
    _, _, _, rates = map_options_from_index(gmm=gmm, maptype=maptype, im=im)

    gmm = pick_first(gmms, gmm)
    maptype = pick_first(maptypes, maptype)
    im = pick_first(ims, im)
    rate = pick_first(rates, rate)

    return (
        [{"label": x, "value": x} for x in gmms], gmm,
        [{"label": MAPTYPE_LABEL.get(x, x), "value": x} for x in maptypes], maptype,
        [{"label": x, "value": x} for x in ims], im,
        [{"label": x, "value": x} for x in rates], rate
    )


@app.callback(
    Output("map-graph", "figure"),
    Output("map-status", "children"),
    Input("map-gmm", "value"),
    Input("map-type", "value"),
    Input("map-im", "value"),
    Input("map-rate", "value"),
)
def update_map(gmm, maptype, im, rate):
    fig = go.Figure()
    fig.update_layout(template="plotly_white")
    if not MAP_INDEX:
        fig.add_annotation(
            text=f"No PNG maps found in: {MAPS_DIR}",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        return fig, f"No maps discovered in {MAPS_DIR}"

    if None in (gmm, maptype, im, rate):
        fig.add_annotation(text="Select map options.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select map options."

    key = MapKey(gmm=gmm, maptype=maptype, im=im, rate=rate)
    path = MAP_INDEX.get(key)
    if not path:
        fig.add_annotation(text="Selected map not found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"Missing file for: {gmm}_{maptype}_{im}_{rate}.png"

    title = f"{gmm} — {MAPTYPE_LABEL.get(maptype, maptype)} — {im} — rate {rate}"
    return make_image_figure(path, title), f"Showing: {path.name}"


# ============================================================
# ==================== CURVES callbacks ======================
# ============================================================
@app.callback(
    Output("c1-im", "options"),
    Output("c1-im", "value"),
    Input("c1-model", "value"),
)
def c1_sync_im(model):
    if not model:
        return [], None
    ims = curves_available_ims_for_model(model)
    return [{"label": im, "value": im} for im in ims], (ims[0] if ims else None)


@app.callback(
    Output("c1-kinds", "options"),
    Output("c1-kinds", "value"),
    Input("c1-model", "value"),
    Input("c1-im", "value"),
)
def c1_sync_kinds(model, im):
    if not model or not im:
        return [], []
    kinds = curves_available_kinds_for_model_im(model, im)
    opts = [{"label": KIND_LABEL.get(k, k), "value": k} for k in kinds]
    default = ["mean"] if "mean" in kinds else (kinds[:1] if kinds else [])
    return opts, default


@app.callback(
    Output("c1-graph", "figure"),
    Output("c1-status", "children"),
    Input("c1-model", "value"),
    Input("c1-im", "value"),
    Input("c1-kinds", "value"),
    Input("c1-cities", "value"),
    Input("c1-axes", "value"),
    Input("c1-xrange", "value"),
)
def c1_update(model, im, kinds, cities, axes_vals, xrange_vals):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=25, t=55, b=60))

    if not CURVE_INDEX:
        fig.add_annotation(text="No hazard curve CSVs found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"No curve files discovered under {SEISMIC_HAZARD_DIR}"

    if not model or not im or not cities or not kinds:
        fig.add_annotation(text="Select model, IM, curve type(s) and at least one city.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select model, IM, curve type(s) and at least one city."

    kinds = list(dict.fromkeys(kinds))
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]

    x_min_data = None
    total_traces = 0
    missing = []

    for kind in kinds:
        path = CURVE_INDEX.get(CurveKey(model=model, im=im, kind=kind))
        if not path:
            missing.append(kind)
            continue

        imls, values, lons, lats = load_curve_file(str(path))
        x_min_data = float(imls[0]) if x_min_data is None else min(x_min_data, float(imls[0]))

        for city in cities:
            lon_c, lat_c = CITY_BY_NAME[city]
            si = nearest_site_index(lons, lats, lon_c, lat_c)
            y = values[si, :]
            fig.add_trace(go.Scatter(
                x=imls, y=y, mode="lines",
                name=f"{city} — {KIND_LABEL.get(kind, kind)}",
                line=dict(dash=("solid" if kind == "mean" else "dash")),
            ))
            total_traces += 1

    fig.update_layout(title=f"{model} — {im} (Compare cities)")
    fig.update_xaxes(title=f"IML ({im})")
    fig.update_yaxes(title="PoE")

    logx = "logx" in (axes_vals or [])
    logy = "logy" in (axes_vals or [])

    x_min = float(xrange_vals[0]) if xrange_vals else (x_min_data if x_min_data is not None else 1e-6)
    x_max = float(xrange_vals[1]) if xrange_vals else IML_MAX_DEFAULT

    set_axis_limits(fig, x_min=x_min, x_max=x_max, y_min=RATES_MIN, y_max=1.0, logx=logx, logy=logy)
    add_rate_lines(fig, x_min=x_min, x_max=x_max)

    status = f"Traces: {total_traces}"
    if missing:
        status += " — Missing types: " + ", ".join(KIND_LABEL.get(k, k) for k in missing)
    return fig, status


@app.callback(
    Output("c2-kinds", "options"),
    Output("c2-kinds", "value"),
    Input("c2-im", "value"),
    Input("c2-models", "value"),
)
def c2_sync_kinds(im, models):
    if not im or not models:
        return [], []
    kinds = curves_available_kinds_for_im_any_model(im, models)
    opts = [{"label": KIND_LABEL.get(k, k), "value": k} for k in kinds]
    default = ["mean"] if "mean" in kinds else (kinds[:1] if kinds else [])
    return opts, default


@app.callback(
    Output("c2-graph", "figure"),
    Output("c2-status", "children"),
    Input("c2-city", "value"),
    Input("c2-im", "value"),
    Input("c2-kinds", "value"),
    Input("c2-models", "value"),
    Input("c2-axes", "value"),
    Input("c2-xrange", "value"),
)
def c2_update(city, im, kinds, models, axes_vals, xrange_vals):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=25, t=55, b=60))

    if not CURVE_INDEX:
        fig.add_annotation(text="No hazard curve CSVs found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"No curve files discovered under {SEISMIC_HAZARD_DIR}"

    if not city or not im or not models or not kinds:
        fig.add_annotation(text="Select city, IM, curve type(s) and at least one GMM.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select city, IM, curve type(s) and at least one GMM."

    kinds = list(dict.fromkeys(kinds))
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]

    lon_c, lat_c = CITY_BY_NAME[city]
    x_min_data = None
    total_traces = 0
    missing = 0

    for model in models:
        for kind in kinds:
            path = CURVE_INDEX.get(CurveKey(model=model, im=im, kind=kind))
            if not path:
                missing += 1
                continue

            imls, values, lons, lats = load_curve_file(str(path))
            x_min_data = float(imls[0]) if x_min_data is None else min(x_min_data, float(imls[0]))

            si = nearest_site_index(lons, lats, lon_c, lat_c)
            y = values[si, :]

            fig.add_trace(go.Scatter(
                x=imls, y=y, mode="lines",
                name=f"{model} — {KIND_LABEL.get(kind, kind)}",
                line=dict(dash=("solid" if kind == "mean" else "dash")),
            ))
            total_traces += 1

    fig.update_layout(title=f"{city} — {im} (Compare GMMs)")
    fig.update_xaxes(title=f"IML ({im})")
    fig.update_yaxes(title="PoE")

    logx = "logx" in (axes_vals or [])
    logy = "logy" in (axes_vals or [])

    x_min = float(xrange_vals[0]) if xrange_vals else (x_min_data if x_min_data is not None else 1e-6)
    x_max = float(xrange_vals[1]) if xrange_vals else IML_MAX_DEFAULT

    set_axis_limits(fig, x_min=x_min, x_max=x_max, y_min=RATES_MIN, y_max=1.0, logx=logx, logy=logy)
    add_rate_lines(fig, x_min=x_min, x_max=x_max)

    return fig, f"Traces: {total_traces} — Missing combos: {missing}"


# ============================================================
# ===================== UHS callbacks ========================
# ============================================================
@app.callback(
    Output("u1-kind", "options"),
    Output("u1-kind", "value"),
    Input("u1-model", "value"),
)
def u1_sync_kind(model):
    if not model:
        return [], None
    kinds = uhs_available_kinds_for_model(model)
    opts = [{"label": KIND_LABEL.get(k, k), "value": k} for k in kinds]
    return opts, (kinds[0] if kinds else None)


@app.callback(
    Output("u1-rate", "options"),
    Output("u1-rate", "value"),
    Input("u1-model", "value"),
    Input("u1-kind", "value"),
)
def u1_sync_rate(model, kind):
    if not model or not kind:
        return [], None
    path = UHS_INDEX.get(UHSKey(model=model, kind=kind))
    if not path:
        return [], None
    _, _, rates, _ = load_uhs_file(str(path))
    return [{"label": r, "value": r} for r in rates], (rates[0] if rates else None)


@app.callback(
    Output("u1-graph", "figure"),
    Output("u1-status", "children"),
    Input("u1-model", "value"),
    Input("u1-kind", "value"),
    Input("u1-rate", "value"),
    Input("u1-cities", "value"),
    Input("u1-axes", "value"),
    Input("u1-xrange", "value"),
)
def u1_update(model, kind, rate, cities, axes_vals, xrange_vals):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=25, t=55, b=60))

    if not UHS_INDEX:
        fig.add_annotation(text="No UHS CSVs found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"No UHS files discovered under {SEISMIC_HAZARD_DIR}"

    if not model or not kind or not rate or not cities:
        fig.add_annotation(text="Select model, type, rate and at least one city.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select model, type, rate and at least one city."

    path = UHS_INDEX.get(UHSKey(model=model, kind=kind))
    if not path:
        fig.add_annotation(text="Selected UHS file not found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Missing selected UHS file."

    lons, lats, _rates, spec_by_rate = load_uhs_file(str(path))
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

    uhs_apply_axes(fig,
                   logx=("logx" in (axes_vals or [])),
                   logy=("logy" in (axes_vals or [])),
                   x_range=xrange_vals)

    return fig, "Nearest sites: " + " | ".join(nearest_info)


@app.callback(
    Output("u2-kind", "options"),
    Output("u2-kind", "value"),
    Input("u2-models", "value"),
)
def u2_sync_kind(models):
    if not models:
        return [], None
    kinds = uhs_available_kinds_for_models(models)
    opts = [{"label": KIND_LABEL.get(k, k), "value": k} for k in kinds]
    return opts, (kinds[0] if kinds else None)


@app.callback(
    Output("u2-rate", "options"),
    Output("u2-rate", "value"),
    Input("u2-kind", "value"),
    Input("u2-models", "value"),
)
def u2_sync_rate(kind, models):
    if not kind or not models:
        return [], None
    for m in models:
        path = UHS_INDEX.get(UHSKey(model=m, kind=kind))
        if path:
            _, _, rates, _ = load_uhs_file(str(path))
            return [{"label": r, "value": r} for r in rates], (rates[0] if rates else None)
    return [], None


@app.callback(
    Output("u2-graph", "figure"),
    Output("u2-status", "children"),
    Input("u2-city", "value"),
    Input("u2-kind", "value"),
    Input("u2-rate", "value"),
    Input("u2-models", "value"),
    Input("u2-axes", "value"),
    Input("u2-xrange", "value"),
)
def u2_update(city, kind, rate, models, axes_vals, xrange_vals):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=70, r=25, t=55, b=60))

    if not UHS_INDEX:
        fig.add_annotation(text="No UHS CSVs found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, f"No UHS files discovered under {SEISMIC_HAZARD_DIR}"

    if not city or not kind or not rate or not models:
        fig.add_annotation(text="Select city, type, rate and at least one GMM.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, "Select city, type, rate and at least one GMM."

    lon_c, lat_c = CITY_BY_NAME[city]
    missing = []
    nearest_info = []

    for model in models:
        path = UHS_INDEX.get(UHSKey(model=model, kind=kind))
        if not path:
            missing.append(model)
            continue

        lons, lats, _rates, spec_by_rate = load_uhs_file(str(path))
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

    uhs_apply_axes(fig,
                   logx=("logx" in (axes_vals or [])),
                   logy=("logy" in (axes_vals or [])),
                   x_range=xrange_vals)

    status_bits = ["Nearest sites: " + " | ".join(nearest_info)]
    if missing:
        status_bits.append(f"Missing: {', '.join(missing[:6])}" + (" ..." if len(missing) > 6 else ""))
    return fig, " — ".join(status_bits)


# ============================================================
# ===================== DOWNLOAD callbacks ===================
# ============================================================
@app.callback(
    Output("c1-download", "data"),
    Input("c1-download-btn", "n_clicks"),
    State("c1-model", "value"),
    State("c1-im", "value"),
    State("c1-kinds", "value"),
    State("c1-cities", "value"),
    State("c1-xrange", "value"),
    prevent_initial_call=True,
)
def download_curves_cities(_n, model, im, kinds, cities, xrange_vals):
    if not model or not im or not kinds or not cities or not xrange_vals:
        return None

    xmin, xmax = float(xrange_vals[0]), float(xrange_vals[1])

    kinds = list(dict.fromkeys(kinds))
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]

    out = None

    for kind in kinds:
        path = CURVE_INDEX.get(CurveKey(model=model, im=im, kind=kind))
        if not path:
            continue

        imls, values, lons, lats = load_curve_file(str(path))
        mask = _filter_by_range(imls, xmin, xmax)
        x = imls[mask]

        if out is None:
            out = pd.DataFrame({"IML": x})

        for city in cities:
            lon_c, lat_c = CITY_BY_NAME[city]
            si = nearest_site_index(lons, lats, lon_c, lat_c)
            y = values[si, :][mask]
            col = f"{city}__{KIND_LABEL.get(kind, kind)}"
            out[col] = y

    if out is None or out.shape[1] == 1:
        return None

    filename = f"hazard_curves_cities_{model}_{im}.csv".replace(" ", "_")
    return dcc.send_data_frame(out.to_csv, filename, index=False)


@app.callback(
    Output("c2-download", "data"),
    Input("c2-download-btn", "n_clicks"),
    State("c2-city", "value"),
    State("c2-im", "value"),
    State("c2-kinds", "value"),
    State("c2-models", "value"),
    State("c2-xrange", "value"),
    prevent_initial_call=True,
)
def download_curves_models(_n, city, im, kinds, models, xrange_vals):
    if not city or not im or not kinds or not models or not xrange_vals:
        return None

    xmin, xmax = float(xrange_vals[0]), float(xrange_vals[1])
    lon_c, lat_c = CITY_BY_NAME[city]

    kinds = list(dict.fromkeys(kinds))
    if "mean" in kinds:
        kinds = ["mean"] + [k for k in kinds if k != "mean"]

    out = None

    for model in models:
        for kind in kinds:
            path = CURVE_INDEX.get(CurveKey(model=model, im=im, kind=kind))
            if not path:
                continue

            imls, values, lons, lats = load_curve_file(str(path))
            mask = _filter_by_range(imls, xmin, xmax)
            x = imls[mask]

            if out is None:
                out = pd.DataFrame({"IML": x})

            si = nearest_site_index(lons, lats, lon_c, lat_c)
            y = values[si, :][mask]
            col = f"{model}__{KIND_LABEL.get(kind, kind)}"
            out[col] = y

    if out is None or out.shape[1] == 1:
        return None

    filename = f"hazard_curves_models_{city}_{im}.csv".replace(" ", "_")
    return dcc.send_data_frame(out.to_csv, filename, index=False)


@app.callback(
    Output("u1-download", "data"),
    Input("u1-download-btn", "n_clicks"),
    State("u1-model", "value"),
    State("u1-kind", "value"),
    State("u1-rate", "value"),
    State("u1-cities", "value"),
    State("u1-xrange", "value"),
    prevent_initial_call=True,
)
def download_uhs_cities(_n, model, kind, rate, cities, xrange_vals):
    if not model or not kind or not rate or not cities or not xrange_vals:
        return None

    xmin, xmax = float(xrange_vals[0]), float(xrange_vals[1])

    path = UHS_INDEX.get(UHSKey(model=model, kind=kind))
    if not path:
        return None

    lons, lats, _rates, spec_by_rate = load_uhs_file(str(path))
    if rate not in spec_by_rate:
        return None

    periods, vals = spec_by_rate[rate]
    mask = _filter_by_range(periods, xmin, xmax)
    p = periods[mask]

    out = pd.DataFrame({"Period_s": p})
    for city in cities:
        lon_c, lat_c = CITY_BY_NAME[city]
        si = nearest_site_index(lons, lats, lon_c, lat_c)
        out[city] = vals[si, :][mask]

    filename = f"uhs_cities_{model}_{KIND_LABEL.get(kind,kind)}_{rate}.csv".replace(" ", "_")
    return dcc.send_data_frame(out.to_csv, filename, index=False)


@app.callback(
    Output("u2-download", "data"),
    Input("u2-download-btn", "n_clicks"),
    State("u2-city", "value"),
    State("u2-kind", "value"),
    State("u2-rate", "value"),
    State("u2-models", "value"),
    State("u2-xrange", "value"),
    prevent_initial_call=True,
)
def download_uhs_models(_n, city, kind, rate, models, xrange_vals):
    if not city or not kind or not rate or not models or not xrange_vals:
        return None

    xmin, xmax = float(xrange_vals[0]), float(xrange_vals[1])
    lon_c, lat_c = CITY_BY_NAME[city]

    out = None
    periods_ref = None

    for model in models:
        path = UHS_INDEX.get(UHSKey(model=model, kind=kind))
        if not path:
            continue

        lons, lats, _rates, spec_by_rate = load_uhs_file(str(path))
        if rate not in spec_by_rate:
            continue

        periods, vals = spec_by_rate[rate]
        mask = _filter_by_range(periods, xmin, xmax)
        p = periods[mask]

        if out is None:
            periods_ref = p
            out = pd.DataFrame({"Period_s": periods_ref})
        else:
            # If periods differ, interpolate onto the reference (rare, but safe)
            if not np.allclose(p, periods_ref):
                si = nearest_site_index(lons, lats, lon_c, lat_c)
                y_full = vals[si, :]
                y_interp = np.interp(periods_ref, periods, y_full)
                out[model] = y_interp
                continue

        si = nearest_site_index(lons, lats, lon_c, lat_c)
        out[model] = vals[si, :][mask]

    if out is None or out.shape[1] == 1:
        return None

    filename = f"uhs_models_{city}_{KIND_LABEL.get(kind,kind)}_{rate}.csv".replace(" ", "_")
    return dcc.send_data_frame(out.to_csv, filename, index=False)


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    # If 8050 is in use, change port=8051
    app.run(debug=True, port=8053)