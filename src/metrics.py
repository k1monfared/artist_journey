"""Metrics and a simple projection for the investment lens.

We keep the math deliberately transparent so a stakeholder can follow it:

  momentum   compound weekly growth of streams over the recent short window.
  growth     compound weekly growth over a longer window.
  accel      momentum minus the growth rate, a proxy for acceleration.
  reach      current streaming scale on a log axis, normalized across roster.
  traction   playlist adds relative to streams, a demand side signal.
  steadiness inverse of recent volatility, rewards consistent artists.

The projection fits a log linear trend on the recent window and extends it,
which gives an exponential path with a plain confidence band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _compound_weekly_growth(series: np.ndarray, window: int) -> float:
    """Compound weekly growth rate over the last `window` steps.

    Uses a log linear fit rather than endpoint over endpoint so a single
    noisy week does not dominate the estimate.
    """
    y = series[-window:]
    if len(y) < 3 or np.all(y <= 0):
        return 0.0
    y = np.clip(y, 1.0, None)
    x = np.arange(len(y))
    slope = np.polyfit(x, np.log(y), 1)[0]
    return float(np.exp(slope) - 1.0)


def _recent_volatility(series: np.ndarray, window: int) -> float:
    """Standard deviation of weekly log returns over the recent window."""
    y = np.clip(series[-window:], 1.0, None)
    if len(y) < 3:
        return 0.0
    returns = np.diff(np.log(y))
    return float(np.std(returns))


def project(series: np.ndarray, weeks_ahead: int = config.PROJECTION_WEEKS,
            fit_window: int = config.GROWTH_WINDOW) -> dict:
    """Project streams forward with a log linear trend and a simple band.

    Returns central, lower, and upper paths. The band widens with the
    horizon using the residual scale of the fit.
    """
    y = np.clip(series[-fit_window:], 1.0, None)
    x = np.arange(len(y))
    coeffs = np.polyfit(x, np.log(y), 1)
    slope, intercept = coeffs[0], coeffs[1]
    resid = np.log(y) - (slope * x + intercept)
    sigma = float(np.std(resid)) if len(resid) > 2 else 0.1

    fx = np.arange(len(y), len(y) + weeks_ahead)
    center = np.exp(slope * fx + intercept)
    # Band widens as sqrt of horizon steps.
    horizon = np.arange(1, weeks_ahead + 1)
    widen = sigma * np.sqrt(horizon)
    lower = np.exp(slope * fx + intercept - 1.28 * widen)
    upper = np.exp(slope * fx + intercept + 1.28 * widen)

    return {
        "weekly_growth": float(np.exp(slope) - 1.0),
        "center": np.round(center).astype(int).tolist(),
        "lower": np.round(lower).astype(int).tolist(),
        "upper": np.round(upper).astype(int).tolist(),
    }


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1, robust to a flat vector."""
    lo, hi = np.min(values), np.max(values)
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def compute_artist_metrics(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Per artist raw signals, before cross roster normalization."""
    rows = []
    for artist_id, g in timeseries.groupby("artist_id"):
        g = g.sort_values("week_index")
        streams = g["streams"].to_numpy()
        playlist = g["playlist_adds"].to_numpy()
        followers = g["followers"].to_numpy()

        momentum = _compound_weekly_growth(streams, config.MOMENTUM_WINDOW)
        growth = _compound_weekly_growth(streams, config.GROWTH_WINDOW)
        accel = momentum - growth
        vol = _recent_volatility(streams, config.GROWTH_WINDOW)
        recent_streams = float(np.mean(streams[-4:]))
        traction = float(np.mean(playlist[-config.GROWTH_WINDOW:]) /
                         max(np.mean(streams[-config.GROWTH_WINDOW:]), 1.0))

        rows.append({
            "artist_id": artist_id,
            "current_streams": int(round(recent_streams)),
            "current_followers": int(followers[-1]),
            "momentum": round(momentum, 5),
            "growth": round(growth, 5),
            "acceleration": round(accel, 5),
            "volatility": round(vol, 5),
            "traction": round(traction, 6),
        })
    return pd.DataFrame(rows)


def score_investment(metrics: pd.DataFrame) -> pd.DataFrame:
    """Blend the raw signals into a 0..100 investment score.

    Weights are chosen to favor artists that are growing fast, accelerating,
    already have meaningful reach, convert to playlists, and are not wildly
    volatile. The weighting is a transparent business choice, not a fit.
    """
    m = metrics.copy()

    reach = _minmax(np.log1p(m["current_streams"].to_numpy()))
    momentum = _minmax(m["momentum"].to_numpy())
    accel = _minmax(m["acceleration"].to_numpy())
    traction = _minmax(m["traction"].to_numpy())
    steadiness = 1.0 - _minmax(m["volatility"].to_numpy())

    weights = {
        "momentum": 0.34,
        "acceleration": 0.22,
        "reach": 0.18,
        "traction": 0.14,
        "steadiness": 0.12,
    }
    score = (
        weights["momentum"] * momentum
        + weights["acceleration"] * accel
        + weights["reach"] * reach
        + weights["traction"] * traction
        + weights["steadiness"] * steadiness
    )

    m["reach_n"] = np.round(reach, 4)
    m["momentum_n"] = np.round(momentum, 4)
    m["acceleration_n"] = np.round(accel, 4)
    m["traction_n"] = np.round(traction, 4)
    m["steadiness_n"] = np.round(steadiness, 4)
    m["investment_score"] = np.round(100.0 * score, 2)
    m = m.sort_values("investment_score", ascending=False).reset_index(drop=True)
    m["rank"] = np.arange(1, len(m) + 1)
    return m


# ----------------------------------------------------------------------------
# Development quadrant
#
# The quadrant plots per artist per year totals: streams on the vertical axis
# and placements (playlist plus editorial, the reach dimension) on the
# horizontal axis. For each year the axis is transformed, then min max scaled
# across the pool to the fixed range -1 to 1, and the dividing cross hair is
# drawn at the median in that normalized space.
# ----------------------------------------------------------------------------


def skewness(v: np.ndarray) -> float:
    """Fisher Pearson skewness of a 1D array. Zero means symmetric."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n < 3:
        return 0.0
    m = v.mean()
    s = v.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((v - m) / s) ** 3))


def _transform(name: str, v: np.ndarray) -> np.ndarray:
    """Apply a named axis transform to a positive valued array."""
    v = np.asarray(v, dtype=float)
    if name == "identity":
        return v
    if name == "sqrt":
        return np.sqrt(v)
    if name == "log1p":
        return np.log1p(v)
    if name == "rank":
        # Quantile transform to a uniform distribution, for reference only.
        order = v.argsort().argsort()
        return (order + 0.5) / len(v)
    raise ValueError(f"unknown transform: {name}")


# Smooth transforms that preserve magnitude order. The rank transform is scored
# for reference but excluded from selection because it forces the median to the
# plane center and discards the magnitude information the quadrant is meant to show.
SMOOTH_TRANSFORMS = ["identity", "sqrt", "log1p"]
ALL_TRANSFORMS = SMOOTH_TRANSFORMS + ["rank"]


def analyze_transforms(pool_yearly: pd.DataFrame) -> dict:
    """Score candidate transforms and choose one from the data.

    Criterion: minimize the mean absolute skewness of the two axes over the
    pooled artist year totals, so that after normalization the points spread
    across the plane and the median line sits informatively rather than being
    crushed against an extreme. The chosen transform must be smooth and
    magnitude preserving, which is why the rank transform is reported as a
    reference lower bound but not selected.
    """
    axes = {
        "streams": pool_yearly["streams_total"].to_numpy(),
        "placements": pool_yearly["placements_total"].to_numpy(),
    }
    per_axis = {}
    mean_abs = {}
    for name in ALL_TRANSFORMS:
        axis_scores = {}
        for axis_name, values in axes.items():
            axis_scores[axis_name] = round(skewness(_transform(name, values)), 4)
        per_axis[name] = axis_scores
        mean_abs[name] = round(
            float(np.mean([abs(axis_scores[a]) for a in axes])), 4)

    chosen = min(SMOOTH_TRANSFORMS, key=lambda t: mean_abs[t])
    return {
        "criterion": ("minimize mean absolute skewness of the pooled artist "
                      "year totals across the two axes, restricted to smooth "
                      "magnitude preserving transforms"),
        "candidates": ALL_TRANSFORMS,
        "skewness_per_axis": per_axis,
        "mean_abs_skewness": mean_abs,
        "chosen": chosen,
        "chosen_mean_abs_skewness": mean_abs[chosen],
        "rank_reference_note": ("the rank transform reaches near zero skewness "
                                "by construction but is rejected because it "
                                "centers the median and discards magnitude"),
    }


def _center_params(v: np.ndarray) -> tuple[float, float]:
    """Return the median and the maximum absolute deviation from it.

    These are the two numbers that define the centering: subtract the median so
    it maps to 0, then divide by the maximum absolute deviation so every value
    lands within -1 to 1. A flat input returns a scale of 1 so division is safe.
    """
    med = float(np.median(v))
    s = float(np.max(np.abs(v - med)))
    if s < 1e-12:
        s = 1.0
    return med, s


def _center_to_unit(v: np.ndarray) -> np.ndarray:
    """Center on the median then scale into -1 to 1.

    Subtract the median so it maps to 0, then divide by the maximum absolute
    deviation from that median so every value lands within -1 to 1. Magnitude
    order is preserved and the median always sits at the origin. Flat input
    maps to zeros.
    """
    med, s = _center_params(v)
    return (v - med) / s


def year_norm_params(pool_yearly: pd.DataFrame, transform_name: str) -> dict:
    """Per year the transform plus centering parameters for each axis.

    Returns a dict keyed by year, each value holding the placements (x) and
    streams (y) centering parameters in transformed space, as
    {"x": (median, scale), "y": (median, scale)}. This is exactly the same
    cloud normalization used for the observed positions, exposed so that a
    forecast in raw units can be mapped onto the same plane.
    """
    years = sorted(int(y) for y in pool_yearly["year"].unique())
    params = {}
    for year in years:
        sub = pool_yearly[pool_yearly["year"] == year]
        if len(sub) < 3:
            continue
        xt = _transform(transform_name, sub["placements_total"].to_numpy())
        yt = _transform(transform_name, sub["streams_total"].to_numpy())
        params[year] = {"x": _center_params(xt), "y": _center_params(yt)}
    return params


def field_params(placements: np.ndarray, streams: np.ndarray,
                 transform_name: str) -> dict:
    """Centering parameters for one field of placements and streams.

    Transforms each axis and returns {"x": (median, scale), "y": (median, scale)}
    in the same form as one year_norm_params entry. Used to normalize a predicted
    field of artists against itself, so a forecast year is centered on the median
    of the whole field predicted for that same year.
    """
    xt = _transform(transform_name, np.asarray(placements, dtype=float))
    yt = _transform(transform_name, np.asarray(streams, dtype=float))
    return {"x": _center_params(xt), "y": _center_params(yt)}


def normalize_xy(streams: np.ndarray, placements: np.ndarray,
                 transform_name: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map raw streams and placements onto the plane using stored parameters.

    Applies the axis transform, then the median centering and scaling captured
    in params, and clips into the plane so forecast points sit in the same
    -1 to 1 space as the observed cloud. params is one year's entry from
    year_norm_params.
    """
    streams = np.asarray(streams, dtype=float)
    placements = np.asarray(placements, dtype=float)
    mx, sx = params["x"]
    my, sy = params["y"]
    x = (_transform(transform_name, placements) - mx) / sx
    y = (_transform(transform_name, streams) - my) / sy
    return np.clip(x, -1.0, 1.0), np.clip(y, -1.0, 1.0)


def normalized_positions(pool_yearly: pd.DataFrame, transform_name: str):
    """Per year median centered positions plus the streams top quartile.

    For each year the two axes are transformed, then centered on that year's
    median and scaled into -1 to 1, so the median maps to 0 on both axes.
    Returns a positions frame with columns artist_id, year, x, y, and a dict
    keyed by year with the 75th percentile of the centered streams axis. Both
    the quadrant and the breakthrough model read from here so they stay
    consistent, and the cross hair is always at the origin.
    """
    years = sorted(int(y) for y in pool_yearly["year"].unique())
    rows = []
    q75_streams = {}
    for year in years:
        sub = pool_yearly[pool_yearly["year"] == year]
        if len(sub) < 3:
            continue
        xt = _transform(transform_name, sub["placements_total"].to_numpy())
        yt = _transform(transform_name, sub["streams_total"].to_numpy())
        xn = _center_to_unit(xt)
        yn = _center_to_unit(yt)
        q75_streams[year] = float(np.percentile(yn, 75))
        for aid, vx, vy in zip(sub["artist_id"].to_numpy(), xn, yn):
            rows.append((aid, year, float(vx), float(vy)))
    positions = pd.DataFrame(rows, columns=["artist_id", "year", "x", "y"])
    return positions, q75_streams


def build_quadrant(pool_yearly: pd.DataFrame, transform_name: str,
                   highlight_ids: list[str]) -> dict:
    """Build the fixed zero centered quadrant payload.

    Because every year is centered on its median, the field looks the same each
    year, so one stable pooled cloud of all artist years is returned rather than
    a per year toggle. Highlighted artists keep their multi year trajectories.
    """
    positions, _ = normalized_positions(pool_yearly, transform_name)
    years = sorted(int(y) for y in positions["year"].unique())

    cloud = {
        "x": [round(v, 3) for v in positions["x"].tolist()],
        "y": [round(v, 3) for v in positions["y"].tolist()],
    }

    raw_by_key = pool_yearly.set_index(["artist_id", "year"])
    highlight = []
    for aid in highlight_ids:
        sub = positions[positions["artist_id"] == aid].sort_values("year")
        if len(sub):
            yrs = [int(y) for y in sub["year"].tolist()]
            streams = [int(raw_by_key.loc[(aid, yr), "streams_total"]) for yr in yrs]
            placements = [int(raw_by_key.loc[(aid, yr), "placements_total"]) for yr in yrs]
            highlight.append({
                "id": aid,
                "years": yrs,
                "x": [round(v, 3) for v in sub["x"].tolist()],
                "y": [round(v, 3) for v in sub["y"].tolist()],
                "streams": streams,
                "placements": placements,
            })

    return {
        "transform": transform_name,
        "years": years,
        "cloud": cloud,
        "highlight": highlight,
    }
