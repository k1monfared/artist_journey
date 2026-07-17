"""Build the investment shortlist and human readable rationales.

The rationale is generated from the same normalized signals that feed the
score, so the explanation always matches the number. This mirrors how a
data team would hand a ranked list to A and R or finance with a short reason
attached to each name.
"""

from __future__ import annotations

import pandas as pd

from . import config


def _pct(x: float) -> str:
    """Format a weekly growth rate as a readable percent per week."""
    return f"{x * 100:.1f}% per week"


def _rationale(row: pd.Series, spec: config.ArtistSpec) -> str:
    """Compose a one line rationale from the strongest normalized signals."""
    signals = {
        "recent momentum": row["momentum_n"],
        "acceleration": row["acceleration_n"],
        "audience reach": row["reach_n"],
        "playlist traction": row["traction_n"],
        "consistency": row["steadiness_n"],
    }
    top = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)[:2]
    strengths = " and ".join(name for name, _ in top)

    direction = "accelerating" if row["acceleration"] > 0 else "cooling"
    return (
        f"{spec.name} ({spec.genre}, {spec.home_region}) shows strong "
        f"{strengths}. Streams are growing {_pct(row['momentum'])} recently "
        f"and the trend is {direction}. Current weekly streams near "
        f"{row['current_streams']:,} with {row['current_followers']:,} followers."
    )


def build_rankings(scored: pd.DataFrame, artists: pd.DataFrame,
                   shortlist_size: int = 5) -> dict:
    """Return a rankings dict with the full board and a shortlist."""
    by_id = config.roster_by_id()
    meta = artists.set_index("artist_id")

    board = []
    for _, row in scored.iterrows():
        spec = by_id[row["artist_id"]]
        board.append({
            "rank": int(row["rank"]),
            "artist_id": row["artist_id"],
            "name": spec.name,
            "genre": spec.genre,
            "home_region": spec.home_region,
            "archetype": meta.loc[row["artist_id"], "archetype"],
            "investment_score": float(row["investment_score"]),
            "momentum_weekly": float(row["momentum"]),
            "growth_weekly": float(row["growth"]),
            "acceleration": float(row["acceleration"]),
            "current_streams": int(row["current_streams"]),
            "current_followers": int(row["current_followers"]),
            "rationale": _rationale(row, spec),
        })

    shortlist = board[:shortlist_size]
    return {
        "generated_from_seed": config.SEED,
        "as_of": config.AS_OF,
        "shortlist_size": shortlist_size,
        "shortlist": shortlist,
        "full_board": board,
    }


def write_report(rankings: dict, path: str) -> None:
    """Write a communication forward markdown report of the shortlist."""
    lines = []
    lines.append("# Artist Investment Shortlist")
    lines.append("")
    lines.append(f"Snapshot as of {rankings['as_of']}. "
                 f"Generated from seed {rankings['generated_from_seed']}.")
    lines.append("")
    lines.append("This report ranks the roster by an investment score that "
                 "blends recent momentum, acceleration, audience reach, "
                 "playlist traction, and consistency. The top names are the "
                 "artists most worth a closer look for additional marketing "
                 "and deal investment.")
    lines.append("")
    lines.append("## Top picks")
    lines.append("")
    for item in rankings["shortlist"]:
        lines.append(f"### {item['rank']}. {item['name']}  "
                     f"(score {item['investment_score']:.1f})")
        lines.append("")
        lines.append(f"- Genre: {item['genre']}")
        lines.append(f"- Region: {item['home_region']}")
        lines.append(f"- Recent momentum: {item['momentum_weekly'] * 100:.1f}% per week")
        lines.append(f"- Acceleration (recent vs longer trend): "
                     f"{item['acceleration'] * 100:.2f} percentage points")
        lines.append(f"- Current weekly streams: {item['current_streams']:,}")
        lines.append(f"- Followers: {item['current_followers']:,}")
        lines.append(f"- Rationale: {item['rationale']}")
        lines.append("")

    lines.append("## Full board")
    lines.append("")
    lines.append("| Rank | Artist | Genre | Score | Momentum/wk | Accel | Weekly streams |")
    lines.append("|-----:|--------|-------|------:|------------:|------:|---------------:|")
    for item in rankings["full_board"]:
        lines.append(
            f"| {item['rank']} | {item['name']} | {item['genre']} | "
            f"{item['investment_score']:.1f} | "
            f"{item['momentum_weekly'] * 100:.1f}% | "
            f"{item['acceleration'] * 100:.2f} | "
            f"{item['current_streams']:,} |"
        )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
