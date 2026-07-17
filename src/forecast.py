"""Recursive forecast of yearly streams and placements.

The two yearly totals move together, so we forecast them together. Streams next
year are predicted from this year's streams and this year's placements, and
placements next year are predicted from this year's placements and this year's
streams, so each variable carries the other forward. This is a first order
vector autoregression, one linear step fit on the year over year transitions
pooled across every artist in the pool. Fitting in log space keeps the heavy
right skew of both totals in check and makes the step multiplicative, which is
the natural scale for growth.

To reach further out the single step is applied recursively: the predicted year
becomes the input for the next year, stepping forward up to PREDICT_YEARS years.
Because each artist starts from their own most recent totals and the two
variables feed each other, the paths differ per artist and keep evolving rather
than collapsing to a shared constant.

The raw forecast is then mapped onto the development plane with the same cloud
normalization used for the observed positions, so a predicted year sits in the
same quadrant space as the history it continues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from . import config, metrics

# How many years to step the recursion forward.
PREDICT_YEARS = 10

# Ridge penalty. Small, just enough to keep the two by two fit stable.
ALPHA = 1.0


def _transitions(pool_yearly: pd.DataFrame) -> np.ndarray:
    """Year over year log transitions pooled across artists.

    Each row is (log streams t, log placements t, log streams t+1,
    log placements t+1) for a pair of consecutive calendar years belonging to
    the same artist.
    """
    df = pool_yearly.copy()
    df["ls"] = np.log(df["streams_total"].clip(lower=1.0))
    df["lp"] = np.log(df["placements_total"].clip(lower=1.0))
    rows = []
    for _, g in df.groupby("artist_id"):
        g = g.sort_values("year")
        ls = g["ls"].tolist()
        lp = g["lp"].tolist()
        yr = [int(y) for y in g["year"].tolist()]
        for i in range(len(g) - 1):
            if yr[i + 1] == yr[i] + 1:
                rows.append((ls[i], lp[i], ls[i + 1], lp[i + 1]))
    return np.array(rows, dtype=float)


def _fit(transitions: np.ndarray) -> Ridge:
    """Fit the one step vector autoregression on log transitions."""
    return Ridge(alpha=ALPHA).fit(transitions[:, :2], transitions[:, 2:])


def _recurse(model: Ridge, streams0: float, placements0: float,
             n_years: int) -> tuple[np.ndarray, np.ndarray]:
    """Step the model forward n_years from a starting pair of raw totals."""
    ls = float(np.log(max(streams0, 1.0)))
    lp = float(np.log(max(placements0, 1.0)))
    out_s, out_p = [], []
    for _ in range(n_years):
        nls, nlp = model.predict(np.array([[ls, lp]]))[0]
        ls, lp = float(nls), float(nlp)
        out_s.append(float(np.exp(ls)))
        out_p.append(float(np.exp(lp)))
    return np.array(out_s), np.array(out_p)


def _evaluate(pool_yearly: pd.DataFrame, seed: int) -> dict:
    """One step held out error, split by artist, against a persistence baseline.

    Persistence predicts next year equals this year. The error is the median
    absolute percentage error in raw units, which is readable and robust to the
    heavy skew.
    """
    ids = sorted(pool_yearly["artist_id"].unique())
    train_ids, test_ids = train_test_split(ids, test_size=0.3, random_state=seed)
    tr = _transitions(pool_yearly[pool_yearly["artist_id"].isin(set(train_ids))])
    te = _transitions(pool_yearly[pool_yearly["artist_id"].isin(set(test_ids))])
    model = _fit(tr)
    pred = model.predict(te[:, :2])

    def mdape(pred_log, true_log):
        p = np.exp(pred_log)
        t = np.exp(true_log)
        return round(float(np.median(np.abs(p - t) / t) * 100), 1)

    return {
        "n_train_transitions": int(len(tr)),
        "n_test_transitions": int(len(te)),
        "streams_mdape_pct": mdape(pred[:, 0], te[:, 2]),
        "placements_mdape_pct": mdape(pred[:, 1], te[:, 3]),
        "streams_persistence_mdape_pct": mdape(te[:, 0], te[:, 2]),
        "placements_persistence_mdape_pct": mdape(te[:, 1], te[:, 3]),
    }


def build_forecast(pool_yearly: pd.DataFrame, transform_name: str,
                   artists: pd.DataFrame, highlight_ids: list[str],
                   seed: int) -> dict:
    """Fit the recursive forecast and precompute paths for the highlighted roster.

    Returns a compact payload: the fitted coefficients, the held out evaluation,
    and per highlighted artist the observed and predicted yearly series in both
    raw units and normalized plane coordinates. Every pool artist is forecast
    forward, so each predicted year is normalized against the field predicted for
    that same year, exactly as an observed year is normalized against its observed
    field. A predicted position therefore reflects standing relative to peers, and
    can move differently from the artist's own raw trend when the field moves too.
    """
    deploy = _fit(_transitions(pool_yearly))
    positions, _ = metrics.normalized_positions(pool_yearly, transform_name)
    name_by_id = artists.set_index("artist_id")["name"].to_dict()
    raw_by_key = pool_yearly.set_index(["artist_id", "year"])

    # Forecast every pool artist forward from their last observed year, so the
    # predicted positions can be normalized against the predicted field of each
    # future year, exactly as the observed positions are normalized against the
    # observed field of their year.
    last = (pool_yearly.sort_values("year").groupby("artist_id").tail(1)
            .set_index("artist_id"))
    field = {}  # future calendar year -> {"s": [...], "p": [...]}
    all_pred = {}  # artist_id -> (years, streams, placements)
    for aid, row in last.iterrows():
        s_path, p_path = _recurse(deploy, float(row["streams_total"]),
                                  float(row["placements_total"]), PREDICT_YEARS)
        years = [int(row["year"]) + i + 1 for i in range(PREDICT_YEARS)]
        all_pred[aid] = (years, s_path, p_path)
        for i, y in enumerate(years):
            f = field.setdefault(y, {"s": [], "p": []})
            f["s"].append(s_path[i])
            f["p"].append(p_path[i])
    pred_params = {y: metrics.field_params(f["p"], f["s"], transform_name)
                   for y, f in field.items() if len(f["s"]) >= 3}

    out = {}
    for aid in highlight_ids:
        pos = positions[positions["artist_id"] == aid].sort_values("year")
        if len(pos) < 2 or aid not in all_pred:
            continue
        obs_years = [int(y) for y in pos["year"].tolist()]
        obs_streams = [int(raw_by_key.loc[(aid, y), "streams_total"]) for y in obs_years]
        obs_plac = [int(raw_by_key.loc[(aid, y), "placements_total"]) for y in obs_years]

        pred_years, s_path, p_path = all_pred[aid]
        # Normalize each predicted year against the predicted field of that year.
        px, py = [], []
        for i, y in enumerate(pred_years):
            xi, yi = metrics.normalize_xy([s_path[i]], [p_path[i]],
                                          transform_name, pred_params[y])
            px.append(float(xi[0]))
            py.append(float(yi[0]))
        px, py = np.array(px), np.array(py)

        out[aid] = {
            "name": name_by_id.get(aid, aid),
            "obs": {
                "years": obs_years,
                "streams": obs_streams,
                "placements": obs_plac,
                "x": [round(v, 3) for v in pos["x"].tolist()],
                "y": [round(v, 3) for v in pos["y"].tolist()],
            },
            "pred": {
                "years": pred_years,
                "streams": [int(round(v)) for v in s_path],
                "placements": [int(round(v)) for v in p_path],
                "x": [round(float(v), 3) for v in px],
                "y": [round(float(v), 3) for v in py],
            },
        }

    coef = deploy.coef_
    intercept = deploy.intercept_
    return {
        "model": ("first order vector autoregression on log yearly totals, fit "
                  "on the pooled year over year transitions and applied "
                  "recursively for " + str(PREDICT_YEARS) + " years"),
        "predict_years": PREDICT_YEARS,
        "coef": {
            "streams_next": {"log_streams": round(float(coef[0, 0]), 4),
                             "log_placements": round(float(coef[0, 1]), 4),
                             "intercept": round(float(intercept[0]), 4)},
            "placements_next": {"log_streams": round(float(coef[1, 0]), 4),
                                "log_placements": round(float(coef[1, 1]), 4),
                                "intercept": round(float(intercept[1]), 4)},
        },
        "eval": _evaluate(pool_yearly, seed),
        "artists": out,
        "note": ("streams and placements predict each other forward, so the two "
                 "paths co evolve. Every pool artist is forecast forward, and each "
                 "predicted year is normalized against the field predicted for that "
                 "year, so a predicted position reflects standing relative to peers "
                 "and can diverge from the artist's own raw trend when the field "
                 "moves too."),
    }
