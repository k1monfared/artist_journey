"""Static matplotlib figures for the README.

Seven figures are produced:

  journey_single.png       one artist trajectory with release markers.
  comparison_calendar.png  several artists overlaid on calendar dates.
  comparison_career.png    the same artists aligned on career age.
  projection.png           one artist with a projected trajectory band.
  development_quadrant.png the fixed plane quadrant with per year trajectories.
  investment_board.png     the investment scoreboard as a ranked bar chart.
  breakthrough_model.png   the breakthrough probability field with outcomes.

The visual style is intentionally plain and readable for a business audience.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from . import config, metrics, predict

# A colorblind friendly qualitative palette.
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})


def _series(timeseries: pd.DataFrame, artist_id: str) -> pd.DataFrame:
    g = timeseries[timeseries["artist_id"] == artist_id].sort_values("week_index")
    return g


def _millions(x, _pos):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def figure_journey(tables: dict, artist_id: str, out_path: str) -> None:
    """Single artist journey with release markers."""
    g = _series(tables["timeseries"], artist_id)
    artists = tables["artists"].set_index("artist_id")
    releases = tables["releases"]
    rel = releases[releases["artist_id"] == artist_id]
    name = artists.loc[artist_id, "name"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    dates = pd.to_datetime(g["date"])
    ax.plot(dates, g["streams"], color=PALETTE[0], linewidth=2, label="Weekly streams")

    ymax = g["streams"].max()
    marker_style = {"Single": "^", "EP": "s", "Album": "o"}
    seen = set()
    for _, r in rel.iterrows():
        rd = pd.Timestamp(r["release_date"])
        rtype = r["release_type"]
        color = {"Single": "#888888", "EP": "#D55E00", "Album": "#009E73"}[rtype]
        ax.axvline(rd, color=color, alpha=0.35, linewidth=1)
        label = rtype if rtype not in seen else None
        seen.add(rtype)
        ax.scatter([rd], [ymax * 1.03], marker=marker_style[rtype],
                   color=color, s=60, zorder=5, label=label)

    ax.set_title(f"Artist journey: {name}")
    ax.set_xlabel("Calendar date")
    ax.set_ylabel("Weekly streams")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Legend outside, to the right, so it never sits on top of the series.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _overlay(tables: dict, artist_ids: list[str], x_field: str,
             xlabel: str, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    artists = tables["artists"].set_index("artist_id")
    for i, aid in enumerate(artist_ids):
        g = _series(tables["timeseries"], aid)
        color = PALETTE[i % len(PALETTE)]
        x = pd.to_datetime(g["date"]) if x_field == "date" else g[x_field]
        ax.plot(x, g["streams"], color=color, linewidth=1.8,
                label=artists.loc[aid, "name"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Weekly streams")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
    if x_field == "date":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Legend outside, to the right, so it never overlaps the plotted series.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_comparison_calendar(tables: dict, artist_ids: list[str], out_path: str) -> None:
    _overlay(tables, artist_ids, "date", "Calendar date",
             "Comparison on calendar dates", out_path)


def figure_comparison_career(tables: dict, artist_ids: list[str], out_path: str) -> None:
    _overlay(tables, artist_ids, "career_month", "Career age (months since debut)",
             "Comparison aligned on career age", out_path)


def figure_projection(forecast: dict, artists: pd.DataFrame, artist_id: str,
                      out_path: str) -> None:
    """One artist's yearly streams, observed and continued by the recursive forecast.

    The forward path is the recursive vector autoregression used on the
    development plane and the yearly time series, fit in log space and stepped
    forward year by year, not the earlier log linear weekly extrapolation. The
    two totals co evolve, so this shows the streams path the same model produces
    for the quadrant, read here as a raw yearly trajectory for one artist.
    """
    name_by_id = artists.set_index("artist_id")["name"].to_dict()
    name = name_by_id.get(artist_id, artist_id)
    a = forecast["artists"][artist_id]
    obs, pred = a["obs"], a["pred"]
    oy, py = obs["years"], pred["years"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(oy, obs["streams"], color=PALETTE[0], linewidth=2, marker="o",
            label="Observed yearly streams")
    ax.plot([oy[-1]] + py, [obs["streams"][-1]] + pred["streams"],
            color=PALETTE[1], linewidth=2, linestyle="--", marker="o",
            markerfacecolor="none", label="Recursive forecast")

    ax.set_title(f"Projected trajectory: {name}")
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Total streams per year")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
    # One integer year per tick, angled so the labels never crowd or clip.
    all_years = list(oy) + list(py)
    ax.set_xticks(all_years)
    ax.set_xticklabels([str(y) for y in all_years], rotation=45, ha="right")
    ax.margins(x=0.02)
    # Legend outside, to the right, so it never overlaps the series.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Positional, non prescriptive quadrant labels. They name where a point sits,
# not whether that is good or bad, since the same corner can mean either.
QUADRANT_CORNERS = {
    "High reach, high streams": (0.99, 0.99, "right", "top"),
    "Low reach, high streams": (-0.99, 0.99, "left", "top"),
    "High reach, low streams": (0.99, -0.99, "right", "bottom"),
    "Low reach, low streams": (-0.99, -0.99, "left", "bottom"),
}


def _draw_quadrant_base(ax, quadrant: dict) -> None:
    """Draw the shared cloud, cross hair, corner labels, and axes framing."""
    cloud = quadrant["cloud"]
    ax.scatter(cloud["x"], cloud["y"], s=8, color="#c4ccd8", alpha=0.4,
               edgecolor="none", zorder=1, label="Field, all years")
    ax.axvline(0.0, color="#5a6b8c", linewidth=1.1, linestyle="--", zorder=2)
    ax.axhline(0.0, color="#5a6b8c", linewidth=1.1, linestyle="--", zorder=2)
    for label, (tx, ty, ha, va) in QUADRANT_CORNERS.items():
        ax.text(tx, ty, label, fontsize=9, color="#8a91a1", ha=ha, va=va,
                style="italic", zorder=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Total placements per year (reach), below to above median",
                  fontsize=10, labelpad=8)
    ax.set_ylabel("Total streams per year, below to above median", fontsize=10)
    ax.grid(False)


# On the development plane, the predicted trajectory is drawn only at these year
# ahead horizons rather than every year, matching the dashboard quadrant. The
# full year by year forecast still drives the yearly time series panels.
QUADRANT_FORECAST_HORIZONS = [1, 2, 5, 10]


def _draw_artist_path(ax, obs: dict, pred: dict | None, color: str,
                      name: str) -> None:
    """Draw one artist's observed path and its dashed forecast continuation."""
    x, y, hy = obs["x"], obs["y"], obs["years"]
    ax.plot(x, y, color=color, linewidth=1.7, alpha=0.85, zorder=3, label=name)
    ax.scatter(x, y, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=4)
    ax.scatter([x[0]], [y[0]], s=60, facecolor="white", edgecolor=color,
               linewidth=1.6, zorder=5)
    ax.scatter([x[-1]], [y[-1]], s=150, color=color, edgecolor="white",
               linewidth=1.2, zorder=6)
    ax.annotate(name, (x[-1], y[-1]), textcoords="offset points",
                xytext=(9, 6), fontsize=8.5, color=color, fontweight="bold")
    ax.annotate(str(hy[0]), (x[0], y[0]), textcoords="offset points",
                xytext=(6, -11), fontsize=7, color=color, alpha=0.85)
    if len(hy) > 1:
        ax.annotate(str(hy[-1]), (x[-1], y[-1]), textcoords="offset points",
                    xytext=(9, -9), fontsize=7, color=color, alpha=0.85)
    if pred:
        # Show only the 1, 2, 5, and 10 year ahead horizons on the plane.
        idxs = [hz - 1 for hz in QUADRANT_FORECAST_HORIZONS
                if 0 <= hz - 1 < len(pred["x"])]
        fpx = [pred["x"][i] for i in idxs]
        fpy = [pred["y"][i] for i in idxs]
        # Dashed continuation from the last observed point through the horizons.
        fx = [x[-1]] + fpx
        fy = [y[-1]] + fpy
        ax.plot(fx, fy, color=color, linewidth=1.5, linestyle=(0, (4, 3)),
                alpha=0.9, zorder=3)
        ax.scatter(fpx, fpy, s=26, facecolor="none", edgecolor=color,
                   linewidth=1.1, alpha=0.9, zorder=4)
        ax.scatter([fpx[-1]], [fpy[-1]], s=90, marker="X",
                   color=color, edgecolor="white", linewidth=0.8, zorder=6)
        ax.annotate(str(pred["years"][idxs[-1]]), (fpx[-1], fpy[-1]),
                    textcoords="offset points", xytext=(8, -10), fontsize=7,
                    color=color, alpha=0.85)


def figure_development_quadrant(quadrant: dict, forecast: dict,
                                artists: pd.DataFrame, artist_ids: list[str],
                                out_path: str,
                                title: str | None = None) -> None:
    """Fixed zero centered quadrant with a stable cloud, paths, and forecasts.

    Each year is centered on its median and scaled into -1 to 1, so the cross
    hair sits at the origin every year and the field looks the same across years.
    One stable pooled cloud of all artist years is drawn in the background. Each
    highlighted artist is a solid trajectory across observed years, continued by
    a dashed recursive forecast of where the same normalization places them next.
    """
    name_by_id = artists.set_index("artist_id")["name"].to_dict()
    fc_artists = forecast["artists"]

    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.19)

    _draw_quadrant_base(ax, quadrant)

    traj = {h["id"]: h for h in quadrant["highlight"]}
    for i, aid in enumerate(artist_ids):
        h = traj.get(aid)
        if not h:
            continue
        color = PALETTE[i % len(PALETTE)]
        fc = fc_artists.get(aid)
        _draw_artist_path(ax, h, fc["pred"] if fc else None, color,
                          name_by_id.get(aid, aid))

    ax.set_title(title or "Development quadrant: observed paths and forecast")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8.5)
    cap = ("Each axis is " + quadrant["transform"] + " transformed, then centered "
           "on the year's median and scaled into -1 to 1, so the median is the "
           "origin.\nSolid path with filled dots is observed history, hollow marker "
           "is the first year.\nThe dashed line and open dots are the recursive "
           "forecast at the 1, 2, 5, and 10 year horizons, X marks year "
           + str(forecast["predict_years"]) + ".")
    fig.text(0.43, 0.02, cap, ha="center", va="bottom", fontsize=7.8,
             color="#555555")
    fig.savefig(out_path)
    plt.close(fig)


def figure_scenario(quadrant: dict, forecast: dict, artists: pd.DataFrame,
                    artist_id: str, out_path: str, headline: str) -> None:
    """Two panel scenario chart: the plane on the left, yearly totals on the right.

    The left panel places the single artist on the development plane with the
    field cloud, the observed path, and the dashed forecast. The right panel
    shows the raw yearly streams and placement adds, observed as solid lines and
    the recursive forecast as dashed continuations, so the reader sees both what
    this analysis adds (the normalized position and the forecast) and the raw
    totals a standard dashboard already reports.
    """
    name_by_id = artists.set_index("artist_id")["name"].to_dict()
    name = name_by_id.get(artist_id, artist_id)
    a = forecast["artists"][artist_id]
    obs, pred = a["obs"], a["pred"]

    fig, (axq, axr) = plt.subplots(1, 2, figsize=(12.6, 5.6))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.06, right=0.93, top=0.88, bottom=0.14, wspace=0.32)

    # Left: the plane.
    _draw_quadrant_base(axq, quadrant)
    q = {h["id"]: h for h in quadrant["highlight"]}[artist_id]
    _draw_artist_path(axq, q, pred, PALETTE[0], name)
    axq.set_title("Development plane position and forecast", fontsize=11)

    # Right: raw yearly totals with the forecast, streams on the left axis and
    # placement adds on a second axis.
    oy, py = obs["years"], pred["years"]
    axr.plot(oy, obs["streams"], color=PALETTE[0], linewidth=2,
             marker="o", label="Streams, observed")
    axr.plot([oy[-1]] + py, [obs["streams"][-1]] + pred["streams"],
             color=PALETTE[0], linewidth=1.8, linestyle="--",
             label="Streams, forecast")
    axr.set_ylabel("Total streams per year", color=PALETTE[0], fontsize=10)
    axr.tick_params(axis="y", labelcolor=PALETTE[0])
    axr.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
    axr.set_xlabel("Year", fontsize=10)

    ax2 = axr.twinx()
    ax2.plot(oy, obs["placements"], color=PALETTE[1], linewidth=2,
             marker="s", label="Placement adds, observed")
    ax2.plot([oy[-1]] + py, [obs["placements"][-1]] + pred["placements"],
             color=PALETTE[1], linewidth=1.8, linestyle="--",
             label="Placement adds, forecast")
    ax2.set_ylabel("Total placement adds per year", color=PALETTE[1], fontsize=10)
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(_millions))
    ax2.grid(False)
    axr.set_title("Yearly totals: observed and forecast", fontsize=11)
    h1, l1 = axr.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axr.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=8)

    fig.suptitle(headline, fontsize=12.5, y=0.97)
    fig.savefig(out_path)
    plt.close(fig)


def figure_investment_board(scored: pd.DataFrame, tables: dict, out_path: str) -> None:
    """Ranked bar chart of investment scores."""
    artists = tables["artists"].set_index("artist_id")
    df = scored.sort_values("investment_score")
    names = [artists.loc[a, "name"] for a in df["artist_id"]]
    scores = df["investment_score"].to_numpy()

    colors = ["#009E73" if r <= 5 else "#B0B7BF" for r in df["rank"]]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(names, scores, color=colors)
    for y, s in enumerate(scores):
        ax.text(s + 0.6, y, f"{s:.1f}", va="center", fontsize=9)
    ax.set_title("Investment scoreboard (top 5 highlighted)")
    ax.set_xlabel("Investment score (0 to 100)")
    ax.set_xlim(0, max(scores) * 1.12)
    ax.grid(axis="y", alpha=0)
    fig.savefig(out_path)
    plt.close(fig)


def figure_prediction(pool_yearly: pd.DataFrame, transform_name: str,
                      prediction: dict, out_path: str) -> None:
    """Breakthrough probability field over early position, with outcomes.

    The background shades the model's predicted breakthrough probability across
    early placements and streams, holding the movement features at their mean.
    Points are eligible artists at their early position, colored by whether they
    actually broke through. A short metrics caption sits under the plot.
    """
    samples = predict.eligible_samples(pool_yearly, transform_name)
    m = prediction["model"]
    b0 = m["intercept"]
    b = m["coef"]
    mdx, mdy = m["mean_dx"], m["mean_dy"]

    grid = np.linspace(-1, 1, 120)
    gx, gy = np.meshgrid(grid, grid)
    z = 1.0 / (1.0 + np.exp(-(b0 + b[0] * gx + b[1] * gy + b[2] * mdx + b[3] * mdy)))

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.17)

    cf = ax.contourf(gx, gy, z, levels=np.linspace(0, 1, 11), cmap="BuPu", alpha=0.75)
    ax.contour(gx, gy, z, levels=[0.5], colors="#333333", linewidths=1.2,
               linestyles="--")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Predicted breakthrough probability", fontsize=9)

    broke = samples[samples["label"] == 1]
    stayed = samples[samples["label"] == 0]
    ax.scatter(stayed["x0"], stayed["y0"], s=16, color="#8a8f99", alpha=0.6,
               edgecolor="none", label="Did not break through")
    ax.scatter(broke["x0"], broke["y0"], s=22, color="#009E73",
               edgecolor="white", linewidth=0.4, label="Broke through")

    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel("Early placements (normalized)", fontsize=10)
    ax.set_ylabel("Early streams (normalized)", fontsize=10)
    ax.set_title("Breakthrough model: probability field and actual outcomes")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8.5)

    mt = prediction["metrics"]
    bl = prediction["baselines"]
    cap = ("Held out test: accuracy " + f"{mt['accuracy']:.2f}"
           + ", ROC AUC " + f"{mt['roc_auc']:.2f}"
           + ", precision " + f"{mt['precision']:.2f}"
           + ", recall " + f"{mt['recall']:.2f}"
           + ", F1 " + f"{mt['f1']:.2f}" + ".\nBaselines: coin toss accuracy 0.50, "
           "majority class accuracy " + f"{bl['majority_class']['accuracy']:.2f}"
           + ". Base rate " + f"{prediction['base_rate']:.2f}"
           + " over " + str(prediction["n_samples"]) + " eligible artists.")
    fig.text(0.5, 0.02, cap, ha="center", va="bottom", fontsize=7.8, color="#555555")
    fig.savefig(out_path)
    plt.close(fig)


# Scenario artists for the README headline questions, as (artist_id, headline).
SCENARIO_FIGURES = [
    ("A04", "scenario_fund.png",
     "Who has the strongest case? Luma Vale, rising and forecast to keep climbing"),
    ("A01", "scenario_doubledown.png",
     "Do we double down on Nova Reyes? Already high, forecast flat, not fresh growth"),
    ("A10", "scenario_earlyflag.png",
     "Can we trust an early flag? Zephyr Kane, an early climber into the field"),
]


def generate_all(tables: dict, scored: pd.DataFrame, images_dir: str,
                 journey_artist: str, comparison_ids: list[str],
                 quadrant: dict, forecast: dict, prediction: dict,
                 quadrant_ids: list[str] | None = None) -> list[str]:
    """Produce every figure and return the list of written paths."""
    os.makedirs(images_dir, exist_ok=True)
    if quadrant_ids is None:
        quadrant_ids = comparison_ids
    paths = []

    p = os.path.join(images_dir, "journey_single.png")
    figure_journey(tables, journey_artist, p); paths.append(p)

    p = os.path.join(images_dir, "comparison_calendar.png")
    figure_comparison_calendar(tables, comparison_ids, p); paths.append(p)

    p = os.path.join(images_dir, "comparison_career.png")
    figure_comparison_career(tables, comparison_ids, p); paths.append(p)

    p = os.path.join(images_dir, "projection.png")
    figure_projection(forecast, tables["artists"], journey_artist, p)
    paths.append(p)

    p = os.path.join(images_dir, "development_quadrant.png")
    figure_development_quadrant(quadrant, forecast, tables["artists"],
                                quadrant_ids, p)
    paths.append(p)

    # Headline figure for the top of the README, the scenario artists on the
    # plane with their forecasts.
    p = os.path.join(images_dir, "development_quadrant_answers.png")
    figure_development_quadrant(
        quadrant, forecast, tables["artists"],
        [aid for aid, _, _ in SCENARIO_FIGURES], p,
        title="Development plane: the headline artists and their forecasts")
    paths.append(p)

    # One two panel figure per headline scenario.
    for aid, fname, headline in SCENARIO_FIGURES:
        if aid not in forecast["artists"]:
            continue
        p = os.path.join(images_dir, fname)
        figure_scenario(quadrant, forecast, tables["artists"], aid, p, headline)
        paths.append(p)

    p = os.path.join(images_dir, "investment_board.png")
    figure_investment_board(scored, tables, p); paths.append(p)

    p = os.path.join(images_dir, "breakthrough_model.png")
    figure_prediction(tables["pool_yearly"], quadrant["transform"], prediction, p)
    paths.append(p)

    return paths
