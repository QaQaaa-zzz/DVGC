# DVGC Experiment State

- Current HEAD lineage: controller `af5ed79`; terrain-clearance Flight repair
  `6f32933`. Flight is paused at its third candidate gate before PPO.
- Runtime gate: PASS/current; config `9711096...dae7c`; XML
  `d7e9f43...ce794c`.
- Landing policy: `landing-20260714-190401`, params `fa3a518b...34bb7e`.
- Landing candidates: `9784649...099505`; 96/96 eligible, 0 duplicates,
  one-step failure 0, 25-step physical failure 2/96, timeout 0.
- Landing Tube: `landing-98ee7a10f7`, `6f6b2ac...61ce8`; labels
  safe/boundary/dead/unknown = 79/12/4/1.
- Landing audit: 96 states, 1536 branches; Final=Chain=87.17%, physical
  failure=12.83%, timeout=horizon=0; precision/recall/coverage=98.73/96.30/
  82.29%, Brier=0.00888, ECE=0.06843. PASS; one held-out false-safe retained.
- Confirmed fixes: `9caaf23` chunked audit, `3421c47` bounded Warp replay gate.
- Flight geometry-v1 partial bank: `c00504b...93cda`; 83/160 finite/eligible.
  It uses authoritative-XML geom distance/contact forward, minimal root-z-only
  correction, unchanged pose/velocity/joints, and a 25-step pre-insert rollout.
  Correction min/p50/p95/max = 0/0.01236/0.15800/0.19023 m.
- Third-round build evidence: 900 proposals, 83 unique accepted; seed 0/1
  proposals covered 197/201 reference indices, so even accepting every unseen
  index caps the bank at 87. Rejections: pitch 289, roll 168, correction above
  0.20 m 47. Accepted ascent/apex/descent = 22/19/42; proposal acceptance =
  6.65/17.27/9.15%, with the largest deficit in ascent.
- Full partial-bank audit: expected-count FAIL only. All 83 are finite,
  eligible and Flight-semantic; robot-terrain contact=0, deep penetration=0,
  25-step physical failure=timeout=nonfinite=0; subinterval coverage and all
  provenance flags pass. Decision report:
  `runs/flight/pipeline_seed0_v4/candidate_decision.json`.
- Pause reason: the authorized third geometry/candidate repair cannot construct
  160 states from the unmodified 201-row reference under the required envelope,
  collision and short-rollout constraints. No PPO started. A next step would
  require a research choice about target size or permitted proposal diversity.
