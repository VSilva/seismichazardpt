import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# CSV is in the same directory as this script/notebook
csv_file = "Mag_Dist-mean-2_156.csv"
df = pd.read_csv(csv_file, skiprows=1)

# Clean columns
df.columns = df.columns.str.strip()
df["imt"] = df["imt"].astype(str).str.strip()

# Numeric columns
for c in ["poe", "mag", "dist", "mean"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# POE values present in the file
poe1 = 4.04e-04
poe2 = 2.105e-03

# Data filters + requested subplot titles (in order)
plot_specs = [
    ("PGA", poe1, "PGA for the 475-year RP"),
    ("PGA", poe2, "PGA for the 2450-year RP"),
    ("SA(1.0)", poe1, "SA(1.0s) for the 475-year RP"),
    ("SA(1.0)", poe2, "SA(1.0s) for the 2450-y RP"),
]

# One figure with 4 rows x 1 column of 3D subplots
fig = plt.figure(figsize=(12, 22))

for i, (imt_val, poe_val, plot_title) in enumerate(plot_specs, start=1):
    ax = fig.add_subplot(4, 1, i, projection="3d")

    # Filter rows for this plot
    sub = df[
        (df["imt"] == imt_val) &
        (np.isclose(df["poe"], poe_val, rtol=1e-6, atol=1e-12))
    ].copy()

    sub = sub.dropna(subset=["dist", "mag", "mean"]).sort_values(["mag", "dist"])

    if sub.empty:
        ax.text2D(0.05, 0.5, f"No data for IMT={imt_val}, POE={poe_val:.3E}", transform=ax.transAxes)
        ax.set_axis_off()
        continue

    # Normalize mean PER PLOT so sum = 1
    sub_total = sub["mean"].sum(skipna=True)
    if np.isclose(sub_total, 0.0):
        ax.text2D(0.05, 0.5, f"sum(mean)=0 for IMT={imt_val}, POE={poe_val:.3E}", transform=ax.transAxes)
        ax.set_axis_off()
        continue

    sub["mean"] = sub["mean"] / sub_total

    # Magnitude intervals
    mag_vals = np.sort(sub["mag"].unique())
    if len(mag_vals) < 2:
        ax.text2D(0.05, 0.5, f"Need at least 2 magnitude values ({imt_val}, {poe_val:.3E})", transform=ax.transAxes)
        ax.set_axis_off()
        continue

    # Make number of distance bins equal to number of magnitude intervals
    n_mag_intervals = len(mag_vals) - 1
    n_dist_bins = max(1, n_mag_intervals)

    # Distance binning (equal-width)
    dmin, dmax = sub["dist"].min(), sub["dist"].max()

    if np.isclose(dmin, dmax):
        sub["dist_bin"] = 0
        dist_edges = np.array([dmin - 0.5, dmax + 0.5], dtype=float)
        dist_centers = np.array([dmin], dtype=float)
    else:
        dist_edges = np.linspace(dmin, dmax, n_dist_bins + 1)
        sub["dist_bin"] = pd.cut(
            sub["dist"],
            bins=dist_edges,
            labels=False,
            include_lowest=True,
            duplicates="drop"
        )

        if sub["dist_bin"].dropna().empty:
            ax.text2D(0.05, 0.5, f"No valid distance bins ({imt_val}, {poe_val:.3E})", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])

    # Aggregate contributions (SUM preserves normalization)
    agg = (
        sub.dropna(subset=["dist_bin"])
           .groupby(["mag", "dist_bin"], as_index=False)["mean"]
           .sum()
    )

    if agg.empty:
        ax.text2D(0.05, 0.5, f"Aggregation produced no data ({imt_val}, {poe_val:.3E})", transform=ax.transAxes)
        ax.set_axis_off()
        continue

    agg["dist_bin"] = agg["dist_bin"].astype(int)

    # Coordinates
    x = dist_centers[agg["dist_bin"].to_numpy()]
    y = agg["mag"].to_numpy()
    z0 = np.zeros(len(agg), dtype=float)
    dz = agg["mean"].to_numpy() * 100.0  # %

    # Bar dimensions
    dx = 0.8 * (dist_edges[1] - dist_edges[0]) if len(dist_edges) > 1 else 1.0
    mag_step = np.min(np.diff(mag_vals)) if len(mag_vals) > 1 else 0.5
    dy = 0.8 * mag_step

    # Colors: light blue -> dark blue (within each plot)
    vmin = np.nanmin(dz)
    vmax = np.nanmax(dz)
    if np.isclose(vmin, vmax):
        norm = colors.Normalize(vmin=vmin - 1, vmax=vmax + 1)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.get_cmap("Greens")
    bar_colors = cmap(norm(dz))

    # Draw bars
    ax.bar3d(
        x - dx / 2,
        y - dy / 2,
        z0,
        dx * np.ones_like(x),
        dy * np.ones_like(y),
        dz,
        color=bar_colors,
        shade=True
    )

    # Axis labels
    ax.set_xlabel("Distance (km)", labelpad=8)
    ax.set_ylabel("Magnitude", labelpad=8)

    # Avoid clipped 3D z-label: use 2D text instead and move it right
    ax.set_zlabel("")
    ax.text2D(
        1.08, 0.5,
        "Contribution to seismic hazard (%)",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="left"
    )

    # Title closer to the plot
    ax.set_title(plot_title, pad=-12)

    ax.tick_params(axis="z", pad=2)
    ax.view_init(elev=25, azim=-60)

# Layout for stacked 3D axes
fig.subplots_adjust(
    left=0.06,
    right=0.84,
    top=0.98,
    bottom=0.03,
    hspace=0.10
)

plt.show()