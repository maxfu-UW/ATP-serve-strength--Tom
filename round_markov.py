"""Tournament-round Markov chain by serve tier.

Combines the k-means serve tiers (output/serve_clusters_by_surface_level_k3.csv)
with last-52-week ATP main-draw singles results (data/atp_matches_52week.csv,
scraped from the ATP Tour results archive).

States:  R128 -> R64 -> R32 -> R16 -> QF -> SF -> F -> W (title), plus Out.
A player in round r moves to the next round with p_tier(r) = P(win a match in
round r | serve tier), and to Out otherwise. W and Out are absorbing.

p_tier(r) is that tier's win rate in round r, shrunk toward 0.5 (the all-player
win rate of any round, since every match has one winner) with a Beta prior worth
PRIOR_STRENGTH matches, so rounds with few matches don't give extreme values.
Resampling whole tournaments (bootstrap) gives 90% intervals.

Tiers: the all-surface model uses each player's all-surface tier; a surface
model uses their tier on that surface, falling back to the all-surface tier.
Players on no serve leaderboard (too few matches to qualify) are "Unrated".

Model check: for players who entered a draw at R32 (or R128), the share who
actually reached each later round, next to the chain's prediction.

Standard library only. Usage:
    python3 round_markov.py                  # all, hard, clay and grass
    python3 round_markov.py --surface clay   # one model
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
MATCHES = ROOT / "data" / "atp_matches_52week.csv"
TIERS_CSV = ROOT / "output" / "serve_clusters_by_surface_level_k3.csv"
OUT = ROOT / "output"

ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
STATES = ROUNDS + ["W", "Out"]
TIERS = ["High", "Mid", "Low", "Unrated"]
SURFACES = ["all", "hard", "clay", "grass"]
STARTS = ["R32", "R128"]  # usual 250/500/1000 entry, and Grand Slam / 96-draw entry
ROUND_NAMES = {
    "Round of 128": "R128", "Round of 64": "R64", "Round of 32": "R32",
    "Round of 16": "R16", "Quarterfinals": "QF", "Quarterfinal": "QF", "Quarter-Finals": "QF",
    "Semifinals": "SF", "Semifinal": "SF", "Semi-Finals": "SF", "Final": "F", "Finals": "F",
}


def round_label(raw):
    # older archive pages print the round as "Semi-Finals -" when no court is named
    return raw.strip().removesuffix("-").strip()
PRIOR_STRENGTH = 10   # pseudo-matches pulling each tier's rate toward 0.5
N_BOOT = 500
SEED = 7


def load_tiers():
    tiers = defaultdict(dict)  # player_id -> {surface: tier}
    with open(TIERS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            tiers[r["player_id"]][r["surface"]] = r["tier"]
    return tiers


def tier_of(tiers, pid, surface):
    t = tiers.get(pid, {})
    return t.get(surface) or t.get("all") or "Unrated"


def load_matches(tiers, surface):
    """Main-draw knockout matches for one model, with each player's tier.

    Returns (played, all_rows, skipped): `played` excludes walkovers and feeds the
    transition probabilities; `all_rows` keeps walkovers, since a walkover still
    moves its winner on, and feeds the check of how far players got.
    """
    rows, skipped = [], defaultdict(int)
    with open(MATCHES, newline="") as f:
        raw = list(csv.DictReader(f))
    round_robin = {r["event_key"] for r in raw if "Round Robin" in r["round"]}
    for r in raw:
        if surface != "all" and r["surface"].lower() != surface:
            continue
        if r["event_key"] in round_robin:
            skipped["round-robin event (ATP Finals, Next Gen Finals)"] += 1
            continue
        rnd = ROUND_NAMES.get(round_label(r["round"]))
        if rnd is None:
            skipped["not a main-draw round: " + round_label(r["round"])] += 1
            continue
        if not r["winner_id"]:
            skipped["no result"] += 1
            continue
        rows.append({
            "event": r["event_key"], "round": rnd, "walkover": r["walkover"] == "1",
            "wid": r["winner_id"], "lid": r["loser_id"],
            "w": tier_of(tiers, r["winner_id"], surface),
            "l": tier_of(tiers, r["loser_id"], surface),
        })
    played = [m for m in rows if not m["walkover"]]
    skipped["walkover (kept only for how far players got)"] += len(rows) - len(played)
    return played, rows, skipped


def advance_probs(matches):
    """Per tier and round: shrunk P(win). Returns ({tier: {round: p}}, counts)."""
    wins, played = defaultdict(int), defaultdict(int)
    for m in matches:
        wins[m["w"], m["round"]] += 1
        played[m["w"], m["round"]] += 1
        played[m["l"], m["round"]] += 1
    probs = {t: {} for t in TIERS}
    counts = {t: {} for t in TIERS}
    for t in TIERS:
        for r in ROUNDS:
            n, w = played[t, r], wins[t, r]
            probs[t][r] = (w + 0.5 * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)
            counts[t][r] = (w, n)
    return probs, counts


def transition_matrix(p):
    """Absorbing Markov chain over STATES for one tier's advance probabilities."""
    k = len(STATES)
    P = [[0.0] * k for _ in range(k)]
    out = STATES.index("Out")
    for i, r in enumerate(ROUNDS):
        P[i][i + 1] = p[r]        # win -> next round (F -> W)
        P[i][out] = 1 - p[r]      # lose -> Out
    P[STATES.index("W")][STATES.index("W")] = 1.0
    P[out][out] = 1.0
    return P


def invert(A):
    """Gauss-Jordan inverse of a small square matrix."""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [x / d for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def reach_probs(P, start):
    """P(reach each later round | start round), via the fundamental matrix.

    With transient block Q, N = (I - Q)^-1 gives expected visits; a player visits
    each round at most once, so N[start][j] = P(reach round j). Absorption
    B = N R gives P(title) in its W column.
    """
    t = len(ROUNDS)
    I_minus_Q = [[(1.0 if i == j else 0.0) - P[i][j] for j in range(t)] for i in range(t)]
    N = invert(I_minus_Q)
    s = ROUNDS.index(start)
    reach = {r: N[s][j] for j, r in enumerate(ROUNDS) if j > s}
    w = STATES.index("W")
    reach["W"] = sum(N[s][j] * P[j][w] for j in range(t))
    return reach


def observed_reach(rows, tiers, surface):
    """For each tier and start round: entries, and how many reached each round."""
    seen = defaultdict(lambda: [99, -1, None])  # (event, player) -> [entry, furthest, id]
    for m in rows:
        i = ROUNDS.index(m["round"])
        for pid, won in ((m["wid"], True), (m["lid"], False)):
            s = seen[m["event"], pid]
            s[0] = min(s[0], i)
            s[1] = max(s[1], i + 1 if (won and m["round"] == "F") else i)  # 7 = title
            s[2] = pid
    entries = defaultdict(int)
    reached = defaultdict(int)
    for (_, pid), (entry, furthest, _) in seen.items():
        tier = tier_of(tiers, pid, surface)
        start = ROUNDS[entry]
        if start not in STARTS:
            continue
        entries[tier, start] += 1
        for j in range(entry + 1, len(ROUNDS) + 1):
            if furthest >= j:
                reached[tier, start, STATES[j]] += 1
    return entries, reached


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run_surface(tiers, surface, rng):
    played, rows, skipped = load_matches(tiers, surface)
    events = sorted({m["event"] for m in rows})
    probs, counts = advance_probs(played)
    point = {t: {s: reach_probs(transition_matrix(probs[t]), s) for s in STARTS} for t in TIERS}

    # bootstrap: resample whole tournaments
    by_event = defaultdict(list)
    for m in played:
        by_event[m["event"]].append(m)
    keys = sorted(by_event)
    boot_p, boot_reach = defaultdict(list), defaultdict(list)
    for _ in range(N_BOOT):
        sample = [m for k in rng.choices(keys, k=len(keys)) for m in by_event[k]]
        bp, _ = advance_probs(sample)
        for t in TIERS:
            for r in ROUNDS:
                boot_p[t, r].append(bp[t][r])
            for s in STARTS:
                for r, v in reach_probs(transition_matrix(bp[t]), s).items():
                    boot_reach[t, s, r].append(v)
    ci = lambda xs: (percentile(xs, 5), percentile(xs, 95))

    entries, reached = observed_reach(rows, tiers, surface)

    win_rows, reach_rows, check_rows, matrix_rows = [], [], [], []
    for t in TIERS:
        for r in ROUNDS:
            lo, hi = ci(boot_p[t, r])
            w, n = counts[t][r]
            win_rows.append({"surface": surface, "tier": t, "round": r, "p_win": probs[t][r],
                             "ci90_low": lo, "ci90_high": hi, "wins": w, "played": n})
        for s in STARTS:
            for r, v in point[t][s].items():
                lo, hi = ci(boot_reach[t, s, r])
                reach_rows.append({"surface": surface, "tier": t, "start": s,
                                   "reach": "Title" if r == "W" else r,
                                   "prob": v, "ci90_low": lo, "ci90_high": hi})
                e = entries[t, s]
                check_rows.append({"surface": surface, "tier": t, "start": s,
                                   "reach": "Title" if r == "W" else r, "entries": e,
                                   "observed": reached[t, s, r] / e if e else "",
                                   "model": v})
        P = transition_matrix(probs[t])
        for i, frm in enumerate(STATES):
            matrix_rows.append({"surface": surface, "tier": t, "from": frm,
                                **{to: P[i][j] for j, to in enumerate(STATES)}})

    print(f"\n=== {surface.upper()}: {len(played)} played main-draw matches from {len(events)} tournaments ===")
    for why, n in sorted(skipped.items()):
        if n:
            print(f"  skipped {n}: {why}")
    print(f"  P(win a match) by round   [wins/played]")
    print(f"  {'Tier':<9}" + "".join(f"{r:>15}" for r in ROUNDS))
    for t in TIERS:
        print(f"  {t:<9}" + "".join(f"{probs[t][r]:>6.2f} [{counts[t][r][0]:>3}/{counts[t][r][1]:<3}]" for r in ROUNDS))
    for s in STARTS:
        later = ROUNDS[ROUNDS.index(s) + 1:] + ["W"]
        print(f"  P(reach | entering at {s}): model (90% interval) / observed share [entries]")
        for t in TIERS:
            cells = []
            for r in later:
                lo, hi = ci(boot_reach[t, s, r])
                e = entries[t, s]
                obs = f"{reached[t, s, r] / e:.3f}" if e else "  -  "
                cells.append(f"{('Title' if r == 'W' else r)} {point[t][s][r]:.3f} ({lo:.3f}-{hi:.3f}) / {obs}")
            print(f"    {t:<8} [{entries[t, s]:>4}]  " + "   ".join(cells))
    return win_rows, reach_rows, check_rows, matrix_rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    global MATCHES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", choices=SURFACES)
    ap.add_argument("--matches", type=Path, default=MATCHES)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    MATCHES, OUT = args.matches, args.out_dir
    surfaces = [args.surface] if args.surface else SURFACES

    tiers = load_tiers()
    rng = random.Random(SEED)
    out = {"win": [], "reach": [], "check": [], "matrix": []}
    for s in surfaces:
        for key, rows in zip(out, run_surface(tiers, s, rng)):
            out[key] += rows

    tag = "" if not args.surface else "_" + args.surface
    files = {
        "win": OUT / f"round_win_probs{tag}.csv",
        "reach": OUT / f"round_reach_probs{tag}.csv",
        "check": OUT / f"round_model_check{tag}.csv",
        "matrix": OUT / f"round_transition_matrix{tag}.csv",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for key, path in files.items():
        write_csv(path, out[key])
    print("\nWrote " + ", ".join(str(p) for p in files.values()))


if __name__ == "__main__":
    main()
