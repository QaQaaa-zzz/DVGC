# DVGC Experiment State

- Validated controller commit: `af5ed79`; current worktree is committed after each fix.
- Current: Landing complete/audited; Flight candidate construction is the active step.
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
- Flight issue: revision v1 candidate process accumulated Warp memory and OOMed
  on an 8 KiB allocation after about 725 proposals; no candidate artifact or
  PPO was accepted. First repair uses deterministic <=450-attempt fresh-process
  chunks with atomic partial banks and aggregate build history.
- Flight v2: fresh-process chunks eliminated OOM, but the original 0.06 feature
  dedup radius saturated at 132/160 (one complete seed chunk accepted 0/450).
  Second repair resumes the valid 132-state partial bank with a recorded 0.03
  radius; existing states already have minimum nearest distance 0.06034.
- Next: resume as pipeline revision v3, validate `flight_candidates.pkl` using
  Landing Tube `6f6b2ac...61ce8`, then run the Flight seed-0 pilot.
