# DVGC Experiment State

- Current HEAD lineage: controller `af5ed79`, candidate fixes `a27ac53` and
  `24351a4`; Flight is paused at its candidate quality gate before PPO.
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
- Flight v3 bank: `7b903cf...bdd37`; 160/160 finite/eligible, 2443 proposals,
  dedup rate 93.45%, proposal failure/timeout 0. Candidate audit FAIL:
  72 deep penetrations, 56 body-terrain contacts, minimum contact distance
  -0.1778 m; 25-step physical failure 91/160 (90 pitch, 1 roll), timeout 0,
  nonfinite 0. Root-z range includes -0.0553 m, showing that raw reference CoM
  placement is not geometry/terrain safe for Flight.
- Pause reason: two evidence-based repairs have not cleared the same Flight
  candidate gate, matching the user-defined pause condition. No PPO started.
- Required next decision: authorize a third repair that makes Flight placement
  XML/terrain-clearance-aware and rejects initial body contacts before bank
  insertion, while retaining reference-envelope velocities and phase semantics.
