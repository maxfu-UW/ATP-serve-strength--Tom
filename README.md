# ATP serve strength

**Live page:** https://maxfu-uw.github.io/ATP-serve-strength--Tom/

Men's singles players grouped into High, Mid and Low serve tiers on each surface (all, hard, clay, grass) by k-means clustering of their last 52 weeks of serve stats, with each tier's tournament odds from a Markov chain over draw rounds.

## Files

| Path | What it is |
| --- | --- |
| `index.html` | The page GitHub Pages serves. It is standalone: the data tables are embedded in it. |
| `serve_kmeans.py` | Serve tiers. Six stats per surface (1st serve in, 1st and 2nd serve points won, service games won, aces and double faults per match) are clipped at the 2.5th and 97.5th percentiles and standardized. Their average, with double faults flipped, is a serve-strength score that k-means (k = 3, k-means++ seeding, best of 50 runs) splits into High, Mid and Low. `--mode style` clusters on the six stats directly instead. Needs numpy. |
| `round_markov.py` | Tournament odds. An absorbing Markov chain over draw rounds (R128, R64, R32, R16, QF, SF, F, then the title or out) for each tier. Round win rates are shrunk toward 50%, the chance of reaching each round comes from the chain's fundamental matrix, and 90% intervals come from 500 resamples of whole tournaments. Standard library only. |
| `build_site.py` | Embeds the tables from `output/` in `index.html`. Standard library only. |
| `data/` | Inputs: the ATP Tour 52-week serve leaderboards, and singles results (main draw and qualifying) for events ending 27 Oct 2025 to 26 Oct 2026. |
| `output/` | The tier and odds tables the page shows. |
| `archive/2026-10-23/` | Inputs and outputs from the 23 Oct 2026 run. |

## Rebuild

```sh
python3 serve_kmeans.py                    # serve tiers -> output/serve_*_level_k3.csv
python3 round_markov.py                    # odds -> output/round_*.csv
python3 build_site.py --as-of 2026-10-26   # date the serve stats were fetched
```

## Caveats

- Tiers and results come from the same 52 weeks, so the odds show how the tiers line up with results. They are not forecasts.
- Tiers are relative to the other players on each surface's leaderboard. Hard-court tiers shift the most from week to week.
- The ATP results archive has no results for 's-Hertogenbosch 2026, so that event is left out, and on grass the round of 128 rests on a single event, Wimbledon.

## Source

Serve stats and results were collected from atptour.com (the stats leaderboard and the results archive). This is independent analysis, not affiliated with or endorsed by the ATP Tour.
