"""Synthetic data generation for the artist roster.

The generator builds three tables:

  artists     one row per artist, static attributes.
  timeseries  weekly streams, followers, and playlist adds per artist.
  releases    release events (single, EP, album) used as journey markers.

Each artist has a career "archetype" that shapes the streaming curve. On top
of the shape we layer release bumps, annual seasonality, and multiplicative
noise so the series look plausible without being tied to any real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _career_weeks(debut: pd.Timestamp, as_of: pd.Timestamp) -> pd.DatetimeIndex:
    """Weekly Monday index from the first Monday on or after debut to as_of."""
    start = debut + pd.offsets.Week(weekday=0, n=0)
    return pd.date_range(start=start, end=as_of, freq=config.FREQ)


def _logistic(x: np.ndarray, midpoint: float, steepness: float) -> np.ndarray:
    """Standard logistic curve rising from 0 to 1."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def _shape(archetype: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a relative streaming shape in roughly the 0.05 to 1.2 range.

    The shape is expressed over the career week index 0..n-1 and captures the
    qualitative story of each archetype. Scale is applied later.
    """
    w = np.arange(n, dtype=float)

    if archetype == "breakout":
        # Quiet start, then a sharp climb to a high plateau.
        mid = n * rng.uniform(0.28, 0.42)
        base = 0.08 + 0.95 * _logistic(w, mid, 0.09)
        return base

    if archetype == "slow_burn":
        # Gradual, near linear compounding growth that is still climbing.
        base = 0.12 + 0.9 * (w / n) ** 1.3
        return base

    if archetype == "steady_star":
        # Already established, mild growth, high floor.
        base = 0.7 + 0.35 * _logistic(w, n * 0.2, 0.05)
        return base

    if archetype == "one_hit":
        # A single strong peak that decays toward a modest floor.
        peak = n * rng.uniform(0.35, 0.5)
        rise = _logistic(w, peak - n * 0.08, 0.16)
        decay = np.exp(-np.maximum(w - peak, 0) / (n * 0.22))
        base = 0.1 + 1.0 * rise * decay
        return np.maximum(base, 0.08)

    if archetype == "declining":
        # Started high, slid down, now soft with a low tail.
        base = 1.0 * np.exp(-w / (n * 0.7)) + 0.12
        return base

    if archetype == "late_bloomer":
        # Flat and quiet, then a recent turn upward near the end.
        mid = n * rng.uniform(0.62, 0.75)
        base = 0.1 + 0.9 * _logistic(w, mid, 0.11)
        return base

    raise ValueError(f"unknown archetype: {archetype}")


def _plan_releases(spec: config.ArtistSpec, weeks: pd.DatetimeIndex,
                   rng: np.random.Generator) -> list[dict]:
    """Place a sequence of releases across the career at plausible spacing."""
    n = len(weeks)
    releases: list[dict] = []
    # Debut single anchors the timeline.
    idx = 2 if n > 3 else 0
    seq = 0
    types_cycle = ["Single", "Single", "EP", "Single", "Album", "Single", "EP", "Album"]
    while idx < n - 1:
        rtype = types_cycle[seq % len(types_cycle)]
        label = f"{rtype}: {_release_title(rng)}"
        releases.append({
            "artist_id": spec.artist_id,
            "release_date": weeks[idx],
            "release_type": rtype,
            "title": label,
            "week_index": int(idx),
        })
        seq += 1
        # Albums are followed by longer gaps than singles.
        gap = rng.integers(20, 30) if rtype == "Album" else rng.integers(10, 20)
        idx += int(gap)
    return releases


_TITLE_WORDS_A = ["Golden", "Midnight", "Paper", "Velvet", "Neon", "Silver",
                  "Wild", "Hollow", "Bright", "Slow", "Electric", "Crimson"]
_TITLE_WORDS_B = ["Hours", "Rivers", "Signal", "Gardens", "Static", "Horizon",
                  "Echoes", "Anthem", "Mirage", "Fever", "Lantern", "Tide"]


def _release_title(rng: np.random.Generator) -> str:
    a = _TITLE_WORDS_A[int(rng.integers(len(_TITLE_WORDS_A)))]
    b = _TITLE_WORDS_B[int(rng.integers(len(_TITLE_WORDS_B)))]
    return f"{a} {b}"


def _build_artist(spec: config.ArtistSpec, as_of: pd.Timestamp,
                  master_rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    """Build the weekly series and releases for one artist."""
    # Derive a per artist seed so each artist is independent yet reproducible.
    art_seed = int(master_rng.integers(0, 2**31 - 1))
    rng = np.random.default_rng(art_seed)

    debut = pd.Timestamp(spec.debut)
    weeks = _career_weeks(debut, as_of)
    n = len(weeks)

    shape = _shape(spec.archetype, n, rng)

    # Release plan and the streaming bump each release produces.
    releases = _plan_releases(spec, weeks, rng)
    bump = np.zeros(n)
    for rel in releases:
        i = rel["week_index"]
        strength = {"Single": 0.18, "EP": 0.30, "Album": 0.55}[rel["release_type"]]
        # An exponentially decaying bump over the weeks after release.
        span = np.arange(n - i)
        bump[i:] += strength * np.exp(-span / 6.0)

    # Annual seasonality: a mild lift late in the calendar year.
    doy = weeks.dayofyear.to_numpy()
    seasonal = 1.0 + 0.08 * np.sin(2 * np.pi * (doy - 300) / 365.0)

    # Multiplicative noise for week to week wobble.
    noise = rng.normal(1.0, 0.06, size=n)

    streams = spec.base_streams * (shape * (1.0 + bump)) * seasonal * noise
    streams = np.clip(streams, 500, None)

    # Playlist adds track the growth of streams plus release spikes, and are
    # noisier and smaller in scale.
    growth = np.gradient(streams)
    playlist = (
        0.9 * np.maximum(growth, 0) / 40.0
        + streams * 0.0009
        + spec.base_streams * 0.02 * bump
    )
    playlist = np.clip(playlist * rng.normal(1.0, 0.12, size=n), 0, None)

    # Followers accumulate a fraction of streams and never fall much. We treat
    # them as a cumulative stock with slow churn.
    weekly_gain = streams * rng.uniform(0.010, 0.016) + playlist * 4.0
    churn = 0.004
    followers = np.zeros(n)
    running = spec.base_streams * 0.05
    for t in range(n):
        running = running * (1 - churn) + weekly_gain[t]
        followers[t] = running

    # Editorial placements are rarer than playlist adds and spike around
    # releases when editorial teams feature a new record. Drawn last so the
    # other series stay byte identical to earlier versions.
    editorial = spec.base_streams * 0.004 * bump + streams * 0.00025
    editorial = np.clip(editorial * rng.normal(1.0, 0.15, size=n), 0, None)

    df = pd.DataFrame({
        "artist_id": spec.artist_id,
        "date": weeks,
        "week_index": np.arange(n),
        "career_month": np.round(np.arange(n) * 7.0 / config.DAYS_PER_MONTH, 3),
        "streams": np.round(streams).astype(int),
        "followers": np.round(followers).astype(int),
        "playlist_adds": np.round(playlist).astype(int),
        "editorial_adds": np.round(editorial).astype(int),
    })
    return df, releases


# Vocabulary for the procedurally generated pool artists.
_POOL_GENRES = ["Pop", "Rock", "Electronic", "R&B", "Latin", "Indie",
                "Country", "Hip-Hop", "Afrobeats", "Jazz", "Folk", "Metal"]
_POOL_REGIONS = ["North America", "Europe", "Asia", "Latin America",
                 "Africa", "Oceania", "Middle East"]
_POOL_ARCHETYPES = ["breakout", "slow_burn", "steady_star", "one_hit",
                    "declining", "late_bloomer"]
_POOL_ARCHETYPE_WEIGHTS = [0.18, 0.20, 0.12, 0.15, 0.15, 0.20]


def _random_pool_specs(n: int, rng: np.random.Generator) -> list[config.ArtistSpec]:
    """Create n procedurally generated pool artists for the field statistics."""
    specs = []
    for i in range(n):
        year = int(rng.integers(2015, 2025))
        month = int(rng.integers(1, 13))
        debut = f"{year}-{month:02d}-01"
        archetype = _POOL_ARCHETYPES[int(
            rng.choice(len(_POOL_ARCHETYPES), p=_POOL_ARCHETYPE_WEIGHTS))]
        # Log uniform base streams so the pool spans small to large acts.
        base = float(np.exp(rng.uniform(np.log(3e4), np.log(2.5e6))))
        specs.append(config.ArtistSpec(
            artist_id=f"P{i + 1:04d}",
            name=f"Pool Artist {i + 1:04d}",
            genre=_POOL_GENRES[int(rng.integers(len(_POOL_GENRES)))],
            home_region=_POOL_REGIONS[int(rng.integers(len(_POOL_REGIONS)))],
            debut=debut,
            archetype=archetype,
            base_streams=int(base),
        ))
    return specs


def _aggregate_yearly(ts_all: pd.DataFrame) -> pd.DataFrame:
    """Per artist per year totals of streams and placements.

    Placements are the reach dimension: playlist adds plus editorial adds.
    Only years with enough weeks of data are kept, which removes partial debut
    years and the partial final year so the yearly totals are comparable.
    """
    df = ts_all.copy()
    df["year"] = df["date"].dt.year
    df["placements"] = df["playlist_adds"] + df["editorial_adds"]
    agg = df.groupby(["artist_id", "year"]).agg(
        streams_total=("streams", "sum"),
        placements_total=("placements", "sum"),
        weeks=("week_index", "count"),
    ).reset_index()
    agg = agg[agg["weeks"] >= config.MIN_WEEKS_PER_YEAR].reset_index(drop=True)
    return agg[["artist_id", "year", "streams_total", "placements_total"]]


def generate() -> dict[str, pd.DataFrame]:
    """Generate the roster and the pool.

    Returns:
      artists      metadata for every artist, with a highlight flag.
      timeseries   weekly series for the highlighted roster only.
      releases     release events for the highlighted roster only.
      pool_yearly  per year totals for every artist, the quadrant field.
    """
    as_of = pd.Timestamp(config.AS_OF)

    artist_rows = []
    highlight_frames = []
    all_frames = []
    release_rows = []

    # Highlighted roster first, using the primary seed so these series stay
    # identical to earlier versions and the other views do not shift.
    master_rng = np.random.default_rng(config.SEED)
    for spec in config.ROSTER:
        df, releases = _build_artist(spec, as_of, master_rng)
        highlight_frames.append(df)
        all_frames.append(df)
        release_rows.extend(releases)
        artist_rows.append({
            "artist_id": spec.artist_id, "name": spec.name, "genre": spec.genre,
            "home_region": spec.home_region, "debut": spec.debut,
            "archetype": spec.archetype, "career_weeks": int(len(df)),
            "highlight": True,
        })

    # Pool artists on an independent seed so the highlighted series are unaffected.
    pool_rng = np.random.default_rng(config.SEED + 1)
    pool_specs = _random_pool_specs(config.POOL_TOTAL - len(config.ROSTER), pool_rng)
    for spec in pool_specs:
        df, _ = _build_artist(spec, as_of, pool_rng)
        all_frames.append(df)
        artist_rows.append({
            "artist_id": spec.artist_id, "name": spec.name, "genre": spec.genre,
            "home_region": spec.home_region, "debut": spec.debut,
            "archetype": spec.archetype, "career_weeks": int(len(df)),
            "highlight": False,
        })

    artists = pd.DataFrame(artist_rows)
    timeseries = pd.concat(highlight_frames, ignore_index=True)
    pool_yearly = _aggregate_yearly(pd.concat(all_frames, ignore_index=True))

    releases = pd.DataFrame(release_rows)
    releases = releases[["artist_id", "release_date", "release_type", "title", "week_index"]]
    releases["career_month"] = np.round(
        releases["week_index"] * 7.0 / config.DAYS_PER_MONTH, 3
    )

    return {
        "artists": artists,
        "timeseries": timeseries,
        "releases": releases,
        "pool_yearly": pool_yearly,
    }
