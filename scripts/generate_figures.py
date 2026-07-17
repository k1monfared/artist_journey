"""Generate the static PNG figures used by the README.

Reads the committed CSV tables, recomputes scores and the quadrant, and writes
figures to docs/images. Run directly with:  python scripts/generate_figures.py
"""

import os

import pandas as pd

import _bootstrap
from src import config, figures, metrics, predict, forecast

# Artists featured in the README figures.
JOURNEY_ARTIST = "A01"
COMPARISON_IDS = ["A01", "A05", "A10", "A03", "A07"]


def load_tables(data_dir: str) -> dict:
    return {
        "artists": pd.read_csv(os.path.join(data_dir, "artists.csv")),
        "timeseries": pd.read_csv(os.path.join(data_dir, "timeseries.csv"),
                                  parse_dates=["date"]),
        "releases": pd.read_csv(os.path.join(data_dir, "releases.csv"),
                                parse_dates=["release_date"]),
        "pool_yearly": pd.read_csv(os.path.join(data_dir, "pool_yearly.csv")),
    }


def main() -> None:
    p = _bootstrap.paths()
    tables = load_tables(p["data"])
    raw = metrics.compute_artist_metrics(tables["timeseries"])
    scored = metrics.score_investment(raw)

    highlight_ids = tables["artists"].loc[
        tables["artists"]["highlight"], "artist_id"].tolist()
    transform_report = metrics.analyze_transforms(tables["pool_yearly"])
    quadrant = metrics.build_quadrant(
        tables["pool_yearly"], transform_report["chosen"], highlight_ids)
    prediction = predict.build_prediction(
        tables["pool_yearly"], transform_report["chosen"],
        tables["artists"], highlight_ids, config.SEED)
    position_forecast = forecast.build_forecast(
        tables["pool_yearly"], transform_report["chosen"],
        tables["artists"], highlight_ids, config.SEED)

    written = figures.generate_all(
        tables, scored, p["images"],
        journey_artist=JOURNEY_ARTIST, comparison_ids=COMPARISON_IDS,
        quadrant=quadrant, forecast=position_forecast, prediction=prediction,
        quadrant_ids=config.QUADRANT_FIGURE_IDS,
    )
    print("Wrote figures:")
    for path in written:
        print(f"  {os.path.relpath(path, p['root'])}")


if __name__ == "__main__":
    main()
