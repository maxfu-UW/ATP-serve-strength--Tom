"""Refresh the public copy of the dashboard (index.html).

index.html is a standalone version of the claude.ai dashboard that any
static web host can serve as-is (GitHub Pages, Netlify, Cloudflare Pages).
Its data is embedded in the page, so the one file is the whole site. This
script rewrites that data from the serve-tier and tournament-odds tables in
output/; after re-running serve_kmeans.py and round_markov.py, run it and
upload index.html again.

Standard library only. Usage:
    python3 build_site.py                      # dated by data/atp_serve_52week.csv
    python3 build_site.py --as-of 2026-10-26   # date the serve stats were fetched
"""
import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "output"
SERVE_CSV = ROOT / "data" / "atp_serve_52week.csv"
PAGE = ROOT / "index.html"

# name the page uses -> table in output/
TABLES = {
    "serve_tiers": "serve_clusters_by_surface_level_k3.csv",
    "round_win": "round_win_probs.csv",
    "round_reach": "round_reach_probs.csv",
    "round_check": "round_model_check.csv",
    "round_matrix": "round_transition_matrix.csv",
}

DATA_BLOCK = re.compile(r'(<script id="dash-data" type="application/json">).*?(</script>)', re.S)


def read_table(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        columns = next(reader)
        return {"columns": columns, "rows": list(reader)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=dt.date.fromisoformat,
                    help="date the serve stats were fetched, YYYY-MM-DD")
    args = ap.parse_args()
    as_of = args.as_of or dt.date.fromtimestamp(SERVE_CSV.stat().st_mtime)

    tables = {name: read_table(OUT / file) for name, file in TABLES.items()}
    # escape "</" so nothing in the data can close the <script> element
    data = json.dumps({"as_of": as_of.isoformat(), "tables": tables},
                      ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    page = PAGE.read_text(encoding="utf-8")
    page, n = DATA_BLOCK.subn(lambda m: m.group(1) + data + m.group(2), page)
    if n != 1:
        raise SystemExit(f"{PAGE}: expected one dash-data block, found {n}")
    PAGE.write_text(page, encoding="utf-8")
    counts = ", ".join(f"{name} {len(t['rows'])}" for name, t in tables.items())
    print(f"index.html: data as of {as_of.isoformat()} ({counts} rows)")


if __name__ == "__main__":
    main()
