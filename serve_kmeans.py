"""K-means clustering of ATP serve stats into High / Mid / Low serve tiers.

Data: ATP Tour serve leaderboard, last 52 weeks, one table per surface
(all, hard, clay, grass) -> data/atp_serve_52week.csv

Clustering runs separately on each surface, because a "good" serve number on
grass (where aces are easy) isn't the same as one on clay.

Usage:
    python3 serve_kmeans.py                 # level tiers, k=3 -> output/
    python3 serve_kmeans.py --mode style    # group by serve style instead
    python3 serve_kmeans.py --k 4
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "atp_serve_52week.csv"
OUT = ROOT / "output"

# All stats in the CSV. serve_rating is never clustered on: ATP builds it by
# summing these same stats, so including it would count them twice. It is still
# reported next to each cluster as a sanity check.
FEATURES = [
    "first_serve_pct",
    "first_serve_pts_won_pct",
    "second_serve_pts_won_pct",
    "service_games_won_pct",
    "aces_per_match",
    "df_per_match",
]
# +1 = higher is better, -1 = lower is better (double faults)
DIRECTION = dict(zip(FEATURES, [1, 1, 1, 1, 1, -1]))

# Clustering modes.
#   level: each player's stats are standardized and averaged into one serve-
#          strength score (double faults flipped), and k-means runs on that 1-D
#          score. Clusters on one axis are always ordered, so High > Mid > Low
#          is guaranteed.
#   style: k-means on all six standardized stats. Several of them are risk
#          trade-offs (big servers hit more aces but miss more first serves and
#          double-fault more), so this groups players by *how* they serve, and
#          the tiers are only roughly ordered by quality.
MODES = ["level", "style"]

# Clip each stat to these percentiles before standardizing, so a player with a
# tiny sample (e.g. 28.6% service games won on grass) can't take a cluster alone.
CLIP_PCT = (2.5, 97.5)

SURFACES = ["all", "hard", "clay", "grass"]
TIER_NAMES = {3: ["High", "Mid", "Low"]}


def load(path):
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["surface"]].append(r)
    return rows


# ---------------------------------------------------------------- k-means ----

def kmeans_pp_init(X, k, rng):
    """k-means++ seeding: spread the initial centroids apart."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centers)


def kmeans(X, k, n_init=50, max_iter=300, tol=1e-6, seed=42):
    """Lloyd's algorithm with k-means++ init; keeps the best of n_init runs."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = kmeans_pp_init(X, k, rng)
        for _ in range(max_iter):
            labels = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                             for j in range(k)])
            shift = np.abs(newC - C).max()
            C = newC
            if shift < tol:
                break
        inertia = ((X - C[labels]) ** 2).sum()
        if best is None or inertia < best[2]:
            best = (labels, C, inertia)
    return best


def silhouette(X, labels):
    """Mean silhouette score (-1..1); higher means better-separated clusters."""
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, labels == j].mean() for j in set(labels) if j != labels[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


# ------------------------------------------------------------------- main ----

def cluster_surface(rows, k, mode):
    direction = np.array([DIRECTION[f] for f in FEATURES])
    full = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    lo, hi = np.percentile(full, CLIP_PCT, axis=0)
    clipped = np.clip(full, lo, hi)
    # standardize so aces (~1-20) don't swamp percentages
    Z = (clipped - clipped.mean(0)) / clipped.std(0)

    # Serve strength: average z-score, double faults flipped so bigger = better
    player_strength = (Z * direction).mean(1)
    X = player_strength[:, None] if mode == "level" else Z

    labels, _, inertia = kmeans(X, k)

    # Rank clusters by their members' mean strength, strongest first
    strength = np.array([player_strength[labels == c].mean() for c in range(k)])
    order = np.argsort(-strength)
    names = TIER_NAMES.get(k, [f"Tier {i + 1}" for i in range(k)])
    tier_of = {cluster: names[rank] for rank, cluster in enumerate(order)}

    results = [
        {**r, "tier": tier_of[l], "strength_score": round(float(s), 3)}
        for r, l, s in zip(rows, labels, player_strength)
    ]
    # Cluster profiles: member means of the raw (unclipped) stats
    centers = [
        {
            "tier": tier_of[c],
            "n": int((labels == c).sum()),
            "serve_rating": float(np.mean([float(rows[i]["serve_rating"])
                                           for i in np.where(labels == c)[0]])),
            **dict(zip(FEATURES, full[labels == c].mean(0))),
        }
        for c in order
    ]
    return results, centers, silhouette(X, labels), inertia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", choices=MODES, default="level")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    data = load(DATA)
    all_results = []
    by_player = defaultdict(dict)

    print(f"mode={args.mode}  k={args.k}")
    short = ["1st%", "1stWon%", "2ndWon%", "SvGms%", "Aces", "DFs"]
    for surface in SURFACES:
        results, centers, sil, inertia = cluster_surface(data[surface], args.k,
                                                         args.mode)
        all_results += results
        for r in results:
            by_player[r["player"]][surface] = r["tier"]

        print(f"\n=== {surface.upper()}  ({len(results)} players, "
              f"silhouette={sil:.3f}, inertia={inertia:.1f}) ===")
        print(f"{'Tier':<6}{'n':>4}{'Rating':>8}" + "".join(f"{s:>9}" for s in short))
        for c in centers:
            print(f"{c['tier']:<6}{c['n']:>4}{c['serve_rating']:>8.1f}"
                  + "".join(f"{c[f]:>9.1f}" for f in FEATURES))
        for c in centers:
            members = sorted((r for r in results if r["tier"] == c["tier"]),
                             key=lambda r: -r["strength_score"])
            print(f"  {c['tier']}: " + ", ".join(r["player"] for r in members))

    fields = ["surface", "tier", "strength_score", "player_id", "player", "country",
              "serve_rating"] + FEATURES
    tag = f"{args.mode}_k{args.k}"
    clusters_csv = OUT / f"serve_clusters_by_surface_{tag}.csv"
    summary_csv = OUT / f"serve_tiers_player_summary_{tag}.csv"
    with open(clusters_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player"] + SURFACES)
        for p in sorted(by_player):
            w.writerow([p] + [by_player[p].get(s, "") for s in SURFACES])

    print(f"\nWrote {clusters_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
