"""Generate the synthetic roster and pool, then write the CSV tables.

Run directly with:  python scripts/generate_data.py
"""

import os

import _bootstrap
from src import data_gen


def main() -> dict:
    p = _bootstrap.paths()
    tables = data_gen.generate()

    tables["artists"].to_csv(os.path.join(p["data"], "artists.csv"), index=False)
    tables["timeseries"].to_csv(os.path.join(p["data"], "timeseries.csv"), index=False)
    tables["releases"].to_csv(os.path.join(p["data"], "releases.csv"), index=False)
    tables["pool_yearly"].to_csv(os.path.join(p["data"], "pool_yearly.csv"), index=False)

    print("Wrote data tables:")
    print(f"  artists     rows={len(tables['artists'])}")
    print(f"  timeseries  rows={len(tables['timeseries'])}")
    print(f"  releases    rows={len(tables['releases'])}")
    print(f"  pool_yearly rows={len(tables['pool_yearly'])}")
    return tables


if __name__ == "__main__":
    main()
