# DVGC Experiment State

- HEAD: `3421c4700cfbb4b6f2f9a0d9440f0604aa473c4c`; worktree clean at handoff.
- Current: Landing complete/audited; next automatic step is Flight candidates.
- Runtime gate: v3 PASS; config `bab689c...ea58`; XML `d7e9f43...ce794c`.
- Landing policy: `landing-20260714-190401`, params `fa3a518b...34bb7e`.
- Landing candidates: `9784649...099505`; 96/96 eligible, 0 duplicates,
  one-step failure 0, 25-step physical failure 2/96, timeout 0.
- Landing Tube: `landing-98ee7a10f7`, `6f6b2ac...61ce8`; labels
  safe/boundary/dead/unknown = 79/12/4/1.
- Landing audit: 96 states, 1536 branches; Final=Chain=87.17%, physical
  failure=12.83%, timeout=horizon=0; precision/recall/coverage=98.73/96.30/
  82.29%, Brier=0.00888, ECE=0.06843. PASS; one held-out false-safe retained.
- Confirmed fixes: `9caaf23` chunked audit, `3421c47` bounded Warp replay gate.
- Next: validate `artifacts/flight_candidates.pkl` using Landing Tube
  `6f6b2ac...61ce8`, then Flight seed-0 pilot in a non-overwriting run path.
