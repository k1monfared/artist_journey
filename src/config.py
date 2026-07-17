"""Central configuration: random seed, roster, and shared constants.

Everything downstream reads from this module so that a single seed makes
the whole pipeline reproducible. The roster is a hand written synthetic
cast of artists with distinct career archetypes so the journey and
comparison views show visibly different shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Global seed. Change this to regenerate a different but reproducible world.
SEED = 20260702

# Weekly cadence. All series are sampled on Mondays.
FREQ = "W-MON"

# The analysis "as of" date. Every artist series ends here regardless of
# when the artist debuted, which mirrors a real reporting snapshot.
AS_OF = "2026-06-29"

# How many weeks of future trajectory to project in the investment lens.
PROJECTION_WEEKS = 26

# Windows used by the momentum and growth metrics, in weeks.
MOMENTUM_WINDOW = 12
GROWTH_WINDOW = 26

# Average number of days in a month, used to convert weeks to career months.
DAYS_PER_MONTH = 30.437

# Development quadrant settings.
# The pool is a large synthetic field so that per year percentiles are
# meaningful. The named ROSTER below are the highlighted artists that keep full
# weekly series and can be traced on the quadrant. The rest of the pool only
# contributes per year aggregate points.
POOL_TOTAL = 300
# A calendar year needs at least this many weeks of data to count, which drops
# partial debut years and the partial final year from the yearly totals.
MIN_WEEKS_PER_YEAR = 45
# Artists featured as trajectories in the static quadrant figure.
QUADRANT_FIGURE_IDS = ["A01", "A05", "A03", "A07"]


@dataclass(frozen=True)
class ArtistSpec:
    """Static description of one synthetic artist.

    archetype drives the shape of the streaming curve. base_streams sets the
    scale of weekly streams around the peak of the shape. debut fixes the
    start of the career age axis.
    """

    artist_id: str
    name: str
    genre: str
    home_region: str
    debut: str
    archetype: str
    base_streams: int


# The roster. Debut dates are spread across several years so that the
# calendar view and the career age view tell clearly different stories.
ROSTER: list[ArtistSpec] = [
    ArtistSpec("A01", "Nova Reyes", "Pop", "North America", "2021-03-01", "breakout", 900_000),
    ArtistSpec("A02", "The Meridian", "Rock", "Europe", "2018-09-03", "steady_star", 1_400_000),
    ArtistSpec("A03", "Kaito Sol", "Electronic", "Asia", "2020-01-06", "slow_burn", 350_000),
    ArtistSpec("A04", "Luma Vale", "R&B", "North America", "2022-06-06", "late_bloomer", 500_000),
    ArtistSpec("A05", "Rio Bravo", "Latin", "Latin America", "2019-05-06", "breakout", 1_100_000),
    ArtistSpec("A06", "Echo Park Twins", "Indie", "North America", "2020-10-05", "one_hit", 600_000),
    ArtistSpec("A07", "Selene Frost", "Pop", "Europe", "2019-02-04", "declining", 800_000),
    ArtistSpec("A08", "Basslake", "Electronic", "Europe", "2021-08-02", "slow_burn", 280_000),
    ArtistSpec("A09", "Marla Quinn", "Country", "North America", "2020-04-06", "steady_star", 700_000),
    ArtistSpec("A10", "Zephyr Kane", "Hip-Hop", "North America", "2022-01-03", "breakout", 750_000),
    ArtistSpec("A11", "Amara Diop", "Afrobeats", "Africa", "2021-11-01", "late_bloomer", 620_000),
    ArtistSpec("A12", "Glass Harbor", "Indie", "Oceania", "2018-04-02", "declining", 950_000),
]


# Confirmed real artists mapped to a public YouTube URL. Only artists listed
# here are treated as real and linked from the dashboard and README. To link
# another artist later, add one line: artist_id -> youtube_url. Artists left out
# of this mapping stay unlinked and render as plain text.
YOUTUBE_URLS: dict[str, str] = {
    "A01": "https://www.youtube.com/watch?v=wFRi0yDHf0Y",
    "A02": "https://www.youtube.com/watch?v=jGAj1XSbe1c",
    "A03": "https://www.youtube.com/watch?v=OaOZiXUKNAA",
    "A04": "https://www.youtube.com/watch?v=_EZtJmNXXnA",
    "A05": "https://www.youtube.com/watch?v=YWfAZPK01QE",
    "A06": "https://www.youtube.com/watch?v=JqIvi1OoR6Y",
    "A08": "https://www.youtube.com/watch?v=lOMtO9gQvN0",
    "A11": "https://www.youtube.com/watch?v=eeSsCbbfnao",
    "A12": "https://www.youtube.com/watch?v=dVQw2jts0oU",
}


def roster_by_id() -> dict[str, ArtistSpec]:
    """Return the roster keyed by artist id for quick lookups."""
    return {a.artist_id: a for a in ROSTER}
