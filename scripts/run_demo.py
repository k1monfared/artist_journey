"""Single entry point that reproduces the whole demonstration.

Steps, in order:
  1. Generate the synthetic roster and the large pool, write CSV tables.
  2. Compute per artist metrics and the investment score.
  3. Choose the quadrant axis transform from the data and build the quadrant.
  4. Write outputs/ (rankings, metrics, transform analysis).
  5. Write the interactive page payload to docs/data.json.
  6. Render the static PNG figures to docs/images/.

Run with:  python scripts/run_demo.py
Everything is seeded, so repeated runs produce identical files.
"""

import json
import os

import _bootstrap
from src import config, data_gen, metrics, ranking, web_export, figures, predict, forecast

JOURNEY_ARTIST = "A01"
COMPARISON_IDS = ["A01", "A05", "A10", "A03", "A07"]


def main() -> None:
    p = _bootstrap.paths()
    print(f"Seed {config.SEED}, snapshot as of {config.AS_OF}")

    # 1. Data. Weekly series and releases for the highlighted roster, plus the
    # per year totals of the whole pool for the quadrant field.
    tables = data_gen.generate()
    highlight_ids = tables["artists"].loc[
        tables["artists"]["highlight"], "artist_id"].tolist()
    tables["artists"].to_csv(os.path.join(p["data"], "artists.csv"), index=False)
    tables["timeseries"].to_csv(os.path.join(p["data"], "timeseries.csv"), index=False)
    tables["releases"].to_csv(os.path.join(p["data"], "releases.csv"), index=False)
    tables["pool_yearly"].to_csv(os.path.join(p["data"], "pool_yearly.csv"), index=False)
    print(f"[1/6] Data: {len(tables['artists'])} artists in the pool "
          f"({len(highlight_ids)} highlighted), "
          f"{len(tables['pool_yearly'])} artist year rows")

    # 2. Metrics and score, on the highlighted roster.
    raw = metrics.compute_artist_metrics(tables["timeseries"])
    scored = metrics.score_investment(raw)
    print(f"[2/6] Metrics: scored {len(scored)} artists, "
          f"top pick {scored.iloc[0]['artist_id']}")

    # 3. Transform choice, quadrant construction, breakthrough model.
    transform_report = metrics.analyze_transforms(tables["pool_yearly"])
    quadrant = metrics.build_quadrant(
        tables["pool_yearly"], transform_report["chosen"], highlight_ids)
    prediction = predict.build_prediction(
        tables["pool_yearly"], transform_report["chosen"],
        tables["artists"], highlight_ids, config.SEED)
    position_forecast = forecast.build_forecast(
        tables["pool_yearly"], transform_report["chosen"],
        tables["artists"], highlight_ids, config.SEED)
    print(f"[3/6] Quadrant: transform '{transform_report['chosen']}', "
          f"{len(quadrant['years'])} years. Breakthrough model: "
          f"base rate {prediction['base_rate']}, "
          f"test accuracy {prediction['metrics']['accuracy']}, "
          f"AUC {prediction['metrics']['roc_auc']}")

    # 4. Outputs
    rankings = ranking.build_rankings(scored, tables["artists"])
    ranking.write_report(rankings, os.path.join(p["outputs"], "investment_report.md"))
    with open(os.path.join(p["outputs"], "rankings.json"), "w") as f:
        json.dump(rankings, f, indent=2)
    summary_cols = ["artist_id", "rank", "investment_score", "momentum",
                    "growth", "acceleration", "volatility", "traction",
                    "current_streams", "current_followers"]
    scored[summary_cols].to_json(
        os.path.join(p["outputs"], "metrics_summary.json"),
        orient="records", indent=2,
    )
    with open(os.path.join(p["outputs"], "transform_analysis.json"), "w") as f:
        json.dump(transform_report, f, indent=2)
    _write_transform_md(transform_report,
                        os.path.join(p["outputs"], "transform_analysis.md"))
    with open(os.path.join(p["outputs"], "prediction.json"), "w") as f:
        json.dump(prediction, f, indent=2)
    _write_prediction_md(prediction,
                         os.path.join(p["outputs"], "prediction.md"))
    with open(os.path.join(p["outputs"], "forecast.json"), "w") as f:
        json.dump(position_forecast, f, indent=2)
    _write_forecast_md(position_forecast,
                       os.path.join(p["outputs"], "forecast.md"))
    print("[4/6] Outputs: investment_report.md, rankings.json, "
          "metrics_summary.json, transform_analysis.*, prediction.*, forecast.*")

    # 5. Web payload
    payload = web_export.build_payload(tables, scored, rankings, quadrant,
                                       position_forecast)
    web_export.write_json(payload, os.path.join(p["docs"], "data.json"))
    size_kb = os.path.getsize(os.path.join(p["docs"], "data.json")) / 1024
    print(f"[5/6] Web: docs/data.json written ({size_kb:.0f} KB)")

    # 6. Figures
    written = figures.generate_all(
        tables, scored, p["images"],
        journey_artist=JOURNEY_ARTIST, comparison_ids=COMPARISON_IDS,
        quadrant=quadrant, forecast=position_forecast, prediction=prediction,
        quadrant_ids=config.QUADRANT_FIGURE_IDS,
    )
    print(f"[6/6] Figures: {len(written)} PNG files in docs/images/")

    print("\nTransform comparison (mean absolute skewness, lower is better):")
    for name in transform_report["candidates"]:
        tag = "  <- chosen" if name == transform_report["chosen"] else ""
        ref = "  (reference, not selected)" if name == "rank" else ""
        print(f"  {name:<9} {transform_report['mean_abs_skewness'][name]:.3f}{tag}{ref}")

    mt = prediction["metrics"]
    bl = prediction["baselines"]
    print("\nBreakthrough model vs baselines (held out test):")
    print(f"  model         accuracy {mt['accuracy']:.3f}  AUC {mt['roc_auc']:.3f}  "
          f"precision {mt['precision']:.3f}  recall {mt['recall']:.3f}  F1 {mt['f1']:.3f}")
    print(f"  coin toss     accuracy 0.500  AUC 0.500")
    print(f"  majority      accuracy {bl['majority_class']['accuracy']:.3f}  "
          f"(predicts class {bl['majority_class']['predicts']}, "
          f"base rate {prediction['base_rate']:.3f})")

    ev = position_forecast["eval"]
    print("\nRecursive forecast, one step held out error (median abs pct error):")
    print(f"  streams     model {ev['streams_mdape_pct']:.1f}%  "
          f"persistence {ev['streams_persistence_mdape_pct']:.1f}%")
    print(f"  placements  model {ev['placements_mdape_pct']:.1f}%  "
          f"persistence {ev['placements_persistence_mdape_pct']:.1f}%")

    print("\nDone. Top 5 shortlist:")
    for item in rankings["shortlist"]:
        print(f"  {item['rank']}. {item['name']:<16} "
              f"score {item['investment_score']:.1f}  ({item['genre']})")


def _write_forecast_md(fc: dict, path: str) -> None:
    """Write a readable summary of the recursive forecast and its error."""
    ev = fc["eval"]
    cs = fc["coef"]["streams_next"]
    cp = fc["coef"]["placements_next"]
    lines = []
    lines.append("# Recursive streams and placements forecast")
    lines.append("")
    lines.append("Model: " + fc["model"] + ".")
    lines.append("")
    lines.append(fc["note"])
    lines.append("")
    lines.append("## One step transition, log space coefficients")
    lines.append("")
    lines.append("| Next year | log streams | log placements | intercept |")
    lines.append("|-----------|------------:|---------------:|----------:|")
    lines.append(f"| log streams | {cs['log_streams']} | {cs['log_placements']} | "
                 f"{cs['intercept']} |")
    lines.append(f"| log placements | {cp['log_streams']} | {cp['log_placements']} | "
                 f"{cp['intercept']} |")
    lines.append("")
    lines.append("Each next year total loads on both current totals, which is how "
                 "the two variables carry each other forward.")
    lines.append("")
    lines.append("## One step held out error")
    lines.append("")
    lines.append(f"Fit on {ev['n_train_transitions']} pooled year over year "
                 f"transitions, evaluated on {ev['n_test_transitions']} "
                 "transitions from held out artists. Error is the median absolute "
                 "percentage error in raw units, against a persistence baseline "
                 "that predicts next year equals this year.")
    lines.append("")
    lines.append("| Axis | Model | Persistence |")
    lines.append("|------|------:|------------:|")
    lines.append(f"| Streams | {ev['streams_mdape_pct']:.1f}% | "
                 f"{ev['streams_persistence_mdape_pct']:.1f}% |")
    lines.append(f"| Placements | {ev['placements_mdape_pct']:.1f}% | "
                 f"{ev['placements_persistence_mdape_pct']:.1f}% |")
    lines.append("")
    lines.append("## Example ten year paths")
    lines.append("")
    lines.append("Predicted yearly totals stepping forward from each artist's most "
                 "recent observed year.")
    lines.append("")
    for aid, a in list(fc["artists"].items())[:3]:
        obs, pred = a["obs"], a["pred"]
        start = (f"{a['name']}, last observed {obs['years'][-1]} at "
                 f"{obs['streams'][-1]:,} streams and {obs['placements'][-1]:,} "
                 "placement adds")
        steps = ", ".join(
            f"{y} ({s:,} / {p:,})"
            for y, s, p in zip(pred["years"], pred["streams"], pred["placements"]))
        lines.append(f"- {start}: {steps}")
    lines.append("")
    lines.append("Each future year is written as year (streams / placement adds).")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_prediction_md(pred: dict, path: str) -> None:
    """Write a readable summary of the breakthrough model and samples."""
    mt = pred["metrics"]
    cm = mt["confusion"]
    bl = pred["baselines"]
    lines = []
    lines.append("# Breakthrough prediction")
    lines.append("")
    lines.append("Definition: " + pred["definition"])
    lines.append("")
    lines.append(f"Observation: {pred['observation']}. Horizon: "
                 f"{pred['horizon_years']} years.")
    lines.append("")
    lines.append(f"Eligible artists: {pred['n_samples']} "
                 f"({pred['n_train']} train, {pred['n_test']} test). "
                 f"Base rate of breakthrough: {pred['base_rate']:.3f}.")
    lines.append("")
    lines.append("## Held out performance vs baselines")
    lines.append("")
    lines.append("| Model | Accuracy | ROC AUC | Precision | Recall | F1 |")
    lines.append("|-------|---------:|--------:|----------:|-------:|---:|")
    lines.append(f"| Logistic regression | {mt['accuracy']:.3f} | {mt['roc_auc']:.3f} | "
                 f"{mt['precision']:.3f} | {mt['recall']:.3f} | {mt['f1']:.3f} |")
    lines.append(f"| Coin toss | 0.500 | 0.500 | . | . | . |")
    lines.append(f"| Majority class | {bl['majority_class']['accuracy']:.3f} | 0.500 | "
                 f". | . | . |")
    lines.append("")
    lines.append(f"Confusion matrix on test: true negatives {cm['tn']}, "
                 f"false positives {cm['fp']}, false negatives {cm['fn']}, "
                 f"true positives {cm['tp']}.")
    lines.append("")
    lines.append("## Sample predictions")
    lines.append("")
    lines.append("| Case | Artist | Early placements | Early streams | "
                 "Predicted probability | Prediction | Actual |")
    lines.append("|------|--------|-----------------:|--------------:|"
                 "----------------------:|:----------:|:------:|")
    label = {1: "breakthrough", 0: "no"}
    for e in pred["examples"]:
        lines.append(f"| {e['kind']} | {e['name']} | {e['x0']:.2f} | {e['y0']:.2f} | "
                     f"{e['proba']:.2f} | {label[e['pred']]} | {label[e['actual']]} |")
    lines.append("")
    lines.append("Cases: TP true positive, TN true negative, FP false positive, "
                 "FN false negative.")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_transform_md(report: dict, path: str) -> None:
    """Write a short markdown summary of the transform comparison."""
    lines = []
    lines.append("# Quadrant axis transform analysis")
    lines.append("")
    lines.append("Criterion: " + report["criterion"] + ".")
    lines.append("")
    lines.append("Each axis carries per artist per year totals across a pool "
                 "of a few hundred artists. Streams and placements are heavily "
                 "right skewed, so raw min max scaling would crush most artists "
                 "near the lower bound and stretch a few outliers to the upper "
                 "bound. The table below reports Fisher Pearson skewness after "
                 "each candidate transform, and the mean of the absolute values "
                 "across the two axes.")
    lines.append("")
    lines.append("| Transform | Streams skew | Placements skew | Mean abs skew |")
    lines.append("|-----------|-------------:|----------------:|--------------:|")
    for name in report["candidates"]:
        pa = report["skewness_per_axis"][name]
        note = " (reference)" if name == "rank" else ""
        star = " chosen" if name == report["chosen"] else ""
        lines.append(f"| {name}{note}{star} | {pa['streams']:.3f} | "
                     f"{pa['placements']:.3f} | "
                     f"{report['mean_abs_skewness'][name]:.3f} |")
    lines.append("")
    lines.append("Chosen transform: **" + report["chosen"] + "**.")
    lines.append("")
    lines.append(report["rank_reference_note"].capitalize() + ".")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
