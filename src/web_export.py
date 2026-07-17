"""Export a single self contained JSON that the interactive page consumes.

The layout uses parallel arrays per artist rather than a list of point
objects, which keeps the file small and fast for the browser to parse. The
web page reads dates for the calendar view and career_month for the aligned
view, so both axes ship in the same payload.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, metrics


def _future_axis(last_date: pd.Timestamp, last_month: float, n: int):
    """Return future weekly dates and career months for the projection."""
    dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(n)]
    months = [round(last_month + (i + 1) * 7.0 / config.DAYS_PER_MONTH, 3)
              for i in range(n)]
    return [d.strftime("%Y-%m-%d") for d in dates], months


def build_payload(tables: dict, scored: pd.DataFrame, rankings: dict,
                  quadrant: dict, forecast: dict) -> dict:
    """Assemble the full JSON payload as a plain dict."""
    timeseries = tables["timeseries"]
    releases = tables["releases"]
    artists = tables["artists"].set_index("artist_id")
    scored_by_id = scored.set_index("artist_id")

    artist_payloads = []
    for artist_id, g in timeseries.groupby("artist_id"):
        g = g.sort_values("week_index")
        meta = artists.loc[artist_id]
        sc = scored_by_id.loc[artist_id]

        rel = releases[releases["artist_id"] == artist_id].sort_values("week_index")
        rel_list = [{
            "date": pd.Timestamp(r["release_date"]).strftime("%Y-%m-%d"),
            "career_month": float(r["career_month"]),
            "type": r["release_type"],
            "title": r["title"],
        } for _, r in rel.iterrows()]

        streams = g["streams"].to_numpy()
        proj = metrics.project(streams)
        last_date = pd.Timestamp(g["date"].iloc[-1])
        last_month = float(g["career_month"].iloc[-1])
        fut_dates, fut_months = _future_axis(last_date, last_month,
                                             config.PROJECTION_WEEKS)

        rank_row = next(item for item in rankings["full_board"]
                        if item["artist_id"] == artist_id)

        yearly = forecast["artists"].get(artist_id)

        artist_payloads.append({
            "id": artist_id,
            "name": meta["name"],
            "youtube_url": config.YOUTUBE_URLS.get(artist_id),
            "genre": meta["genre"],
            "region": meta["home_region"],
            "debut": meta["debut"],
            "archetype": meta["archetype"],
            "career_weeks": int(meta["career_weeks"]),
            "dates": [d.strftime("%Y-%m-%d") for d in g["date"]],
            "career_month": [float(x) for x in g["career_month"]],
            "streams": [int(x) for x in g["streams"]],
            "followers": [int(x) for x in g["followers"]],
            "playlist_adds": [int(x) for x in g["playlist_adds"]],
            "releases": rel_list,
            "metrics": {
                "investment_score": float(sc["investment_score"]),
                "rank": int(sc["rank"]),
                "momentum_weekly": float(sc["momentum"]),
                "growth_weekly": float(sc["growth"]),
                "acceleration": float(sc["acceleration"]),
                "current_streams": int(sc["current_streams"]),
                "current_followers": int(sc["current_followers"]),
                "rationale": rank_row["rationale"],
            },
            "projection": {
                "dates": fut_dates,
                "career_month": fut_months,
                "center": proj["center"],
                "lower": proj["lower"],
                "upper": proj["upper"],
            },
            "yearly": yearly,
        })

    # Order artists by investment rank so the page defaults to the top names.
    artist_payloads.sort(key=lambda a: a["metrics"]["rank"])

    payload = {
        "meta": {
            "seed": config.SEED,
            "as_of": config.AS_OF,
            "metric_options": ["streams", "followers", "playlist_adds"],
            "momentum_window_weeks": config.MOMENTUM_WINDOW,
            "growth_window_weeks": config.GROWTH_WINDOW,
            "projection_weeks": config.PROJECTION_WEEKS,
            "quadrant_axes": {
                "x": "total placements per year (playlist plus editorial adds), reach dimension",
                "y": "total streams per year",
                "normalization": ("per year, each axis is " + quadrant["transform"] +
                                  " transformed then centered on the year's median and "
                                  "scaled into -1 to 1, so the median maps to 0"),
                "cross_hair": "always at the origin (0, 0)",
                "pool_size": config.POOL_TOTAL,
            },
            "placements_unit": "adds per year (playlist plus editorial adds)",
            "generated_by": "scripts/run_demo.py",
        },
        "artists": artist_payloads,
        "rankings": rankings,
        "quadrant": quadrant,
        "forecast": forecast,
    }
    return payload


def write_json(payload: dict, path: str) -> None:
    """Write the payload as compact JSON."""
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
