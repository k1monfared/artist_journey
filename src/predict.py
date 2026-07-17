"""Breakthrough prediction from early quadrant position.

The question: can an artist's early standing on the development quadrant predict
whether they break through a few years later. The features are only observable
quadrant coordinates, the normalized placements and streams at an early year and
their movement over the first year. The generative archetype is never used, that
would be leakage.

Breakthrough is defined from the data, see BREAKTHROUGH_DEF below. The model is a
plain logistic regression, trained and evaluated on a held out split of the pool
and compared against a coin toss and the majority class baseline. The intent is
an honest read, the signal is expected to be only modest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

from . import metrics

# Years between the observation year and the target year at which breakthrough
# is measured. The observation year is the artist's first full year on the plane.
HORIZON = 3

FEATURES = ["x0", "y0", "dx", "dy"]

BREAKTHROUGH_DEF = (
    "An artist breaks through if, in the target year (the observation year plus "
    + str(HORIZON) + " years), they are in the upper right quadrant, both centered "
    "coordinates greater than zero, or in the top quartile of centered streams, "
    "having started below that at the observation year. Artists already in that "
    "region at observation are excluded, since the axes are centered on the "
    "yearly median the origin is the median."
)


def _build_samples(positions: pd.DataFrame, q75: dict) -> pd.DataFrame:
    """Assemble one labeled training row per eligible artist.

    Coordinates are median centered, so the origin is the yearly median and the
    upper right quadrant is both coordinates greater than zero.
    """
    rows = []
    for aid, g in positions.groupby("artist_id"):
        g = g.sort_values("year")
        years = [int(y) for y in g["year"].tolist()]
        xs = g["x"].tolist()
        ys = g["y"].tolist()
        idx = {y: i for i, y in enumerate(years)}
        obs = years[0]
        target = obs + HORIZON
        # Need the observation year, the next year for movement, and the target.
        if (obs + 1) not in idx or target not in idx:
            continue
        x0, y0 = xs[idx[obs]], ys[idx[obs]]
        x1, y1 = xs[idx[obs + 1]], ys[idx[obs + 1]]
        dx, dy = x1 - x0, y1 - y0

        already = (x0 > 0 and y0 > 0) or (y0 >= q75[obs])
        if already:
            continue

        xt, yt = xs[idx[target]], ys[idx[target]]
        label = int((xt > 0 and yt > 0) or (yt >= q75[target]))

        rows.append({
            "artist_id": aid, "obs_year": obs, "target_year": target,
            "x0": round(x0, 4), "y0": round(y0, 4),
            "dx": round(dx, 4), "dy": round(dy, 4),
            "label": label,
        })
    return pd.DataFrame(rows)


def eligible_samples(pool_yearly: pd.DataFrame, transform_name: str) -> pd.DataFrame:
    """Public helper returning the labeled sample rows used by the model."""
    positions, q75 = metrics.normalized_positions(pool_yearly, transform_name)
    return _build_samples(positions, q75)


def _round_metrics(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def build_prediction(pool_yearly: pd.DataFrame, transform_name: str,
                     artists: pd.DataFrame, highlight_ids: list[str],
                     seed: int) -> dict:
    """Fit the model, evaluate it, and assemble the report payload."""
    samples = eligible_samples(pool_yearly, transform_name)

    X = samples[FEATURES].to_numpy()
    y = samples["label"].to_numpy()
    base_rate = float(y.mean())

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, samples.index.to_numpy(), test_size=0.3, random_state=seed, stratify=y)

    # Breakthrough is the minority class, so balance the classes. Without this
    # the model predicts all negative at the 0.5 threshold, which hides its
    # ranking ability and leaves no positive predictions to inspect.
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    model_metrics = _round_metrics({
        "accuracy": accuracy_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    })

    # Baselines. Majority class is learned from the training split.
    majority = int(round(y_tr.mean()))
    maj_acc = float(np.mean(y_te == majority))
    baselines = {
        "coin_toss": {"accuracy": 0.5, "roc_auc": 0.5},
        "majority_class": {"accuracy": round(maj_acc, 4),
                           "predicts": majority,
                           "base_rate": round(base_rate, 4)},
    }

    # Sample right and wrong predictions from the held out test set.
    name_by_id = artists.set_index("artist_id")["name"].to_dict()
    test = samples.loc[idx_te].copy()
    test["proba"] = np.round(proba, 4)
    test["pred"] = pred
    kinds = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}
    examples = []
    for kind, (actual, predicted) in kinds.items():
        hit = test[(test["label"] == actual) & (test["pred"] == predicted)]
        for _, r in hit.head(2).iterrows():
            examples.append({
                "kind": kind, "artist_id": r["artist_id"],
                "name": name_by_id.get(r["artist_id"], r["artist_id"]),
                "obs_year": int(r["obs_year"]), "target_year": int(r["target_year"]),
                "x0": float(r["x0"]), "y0": float(r["y0"]),
                "dx": float(r["dx"]), "dy": float(r["dy"]),
                "proba": float(r["proba"]), "pred": int(r["pred"]),
                "actual": int(r["label"]),
            })

    # Predictions for the highlighted artists, for the interactive tab.
    by_id = samples.set_index("artist_id")
    highlight = []
    for aid in highlight_ids:
        if aid not in by_id.index:
            continue
        r = by_id.loc[aid]
        feats = np.array([[r["x0"], r["y0"], r["dx"], r["dy"]]])
        p = float(model.predict_proba(feats)[0, 1])
        highlight.append({
            "id": aid, "name": name_by_id.get(aid, aid),
            "obs_year": int(r["obs_year"]), "target_year": int(r["target_year"]),
            "x0": float(r["x0"]), "y0": float(r["y0"]),
            "dx": float(r["dx"]), "dy": float(r["dy"]),
            "proba": round(p, 4), "pred": int(p >= 0.5), "actual": int(r["label"]),
        })

    coef = model.coef_[0]
    model_export = {
        "type": "logistic_regression",
        "features": FEATURES,
        "coef": [round(float(c), 4) for c in coef],
        "intercept": round(float(model.intercept_[0]), 4),
        "mean_dx": round(float(samples["dx"].mean()), 4),
        "mean_dy": round(float(samples["dy"].mean()), 4),
    }

    return {
        "definition": BREAKTHROUGH_DEF,
        "horizon_years": HORIZON,
        "observation": "first full year on the plane, movement over the next year",
        "n_samples": int(len(samples)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "base_rate": round(base_rate, 4),
        "metrics": model_metrics,
        "baselines": baselines,
        "model": model_export,
        "examples": examples,
        "highlight": highlight,
    }
