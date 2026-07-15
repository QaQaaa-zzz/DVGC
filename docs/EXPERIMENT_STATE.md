# DVGC Experiment State

- Current HEAD: handoff repair `a0ca3c2`, initial curriculum controller
  `76c98ef`; retention repair is validated and pending commit.
- Runtime gate: PASS/current; source `c6f1b0e...65b93`, config
  `307f41a...d28e9`, XML `d7e9f43...ce794c`; authoritative action mapping is
  unchanged.
- Landing policy: `landing-20260714-190401`, params `fa3a518b...34bb7e`.
- Landing candidates: `9784649...099505`; 96/96 eligible, 0 duplicates,
  one-step failure 0, 25-step physical failure 2/96, timeout 0.
- Landing Tube: `landing-98ee7a10f7`, `6f6b2ac...61ce8`; labels
  safe/boundary/dead/unknown = 79/12/4/1.
- Landing audit: 96 states, 1536 branches; Final=Chain=87.17%, physical
  failure=12.83%, timeout=horizon=0; precision/recall/coverage=98.73/96.30/
  82.29%, Brier=0.00888, ECE=0.06843. PASS; one held-out false-safe retained.
- Confirmed fixes: `9caaf23` chunked audit, `3421c47` bounded Warp replay gate.
- Flight augmented bank: `2d5d7de...f62934`; 83 fixed reference anchors plus
  77 constrained local candidates. Automatic ascent/apex/descent quotas and
  counts are 63/20/77. Build used 178/3000 proposals, 58 unique parents,
  maximum two children/parent; normalized NN min/p50/p95/max =
  0.1546/0.2807/1.3375/3.5079 at threshold 0.15. Rejections: joint range 74,
  normalized duplicate 26, pitch 1.
- Candidate gate: PASS. Overall and anchor/augmented grouped audits have
  contact=deep penetration=25-step physical failure=timeout=nonfinite=0;
  all 160 are finite/eligible/Flight-semantic and provenance-current.
  Anchor correction min/p50/p95/max=0/0.01236/0.15800/0.19023 m; augmented=
  0/0/0.00256/0.03318 m.
- Flight pilot: `runs/flight/pipeline_seed0_v5/pilot`, policy
  `flight-20260715-124908`, seed 0, 102400 effective steps. Healthy runtime,
  no NaN/OOM/compile restart; throughput 7.5k--11.3k SPS after compile.
  Fixed evaluation: Final=7.5%, Chain=0, physical failure=92.5%, timeout=0;
  pitch/roll/recovery=101/47/12. Anchor Final=6.02%, augmented=9.09%.
  Ascent/apex Final=0 in both groups; descent=15.58% overall.
- Frozen Landing-policy baseline on the same bank: Final=1.875%, Chain=0,
  physical failure=98.125%, timeout=0. Pilot improves Final fourfold and mean
  steps 20.82->25.30, but evaluation oscillates at 3.1--11.7% and misses the
  50% pilot gate. Value loss converged to 0.137, KL to 0.0131, policy std
  remained 0.0502; reset mix is 90% candidates/10% downstream rehearsal.
- Canonical Landing-entry v2: construction seed 4100000, certification seed
  4200000, audit seed 5200000; 90 proposals, 79 safe / 8 boundary / 0 dead /
  3 unknown.  Three proposals are true first-valid-contact snapshots; one is
  Final-safe.  Independent entry audit: precision/recall/coverage =
  98.73/92.86/87.78%, physical failure 7.29%, timeout=horizon=0.  Matcher
  radius 1.10679; independent matcher precision/recall=98.77/95.24%; bank
  `a98a246...2d964`.
- Flight handoff v3, fixed 160 candidates: frozen Landing policy has
  Chain/Final=1/3, false progress=0, Chain reward=8 and all Final trajectories
  have valid landing.  Existing Flight pilot has Chain/Final=0/12, so its
  learned trajectory does not reach `C_L`; this is no longer a missing event
  or reward signal.
- Curriculum v6 late-descent from Landing completed 102400 effective steps:
  Chain=0, Final=4.375%, physical failure=95.625%, timeout=0, Landing retention
  53.125%.  Runtime was healthy but early KL spikes (613.95 and 117.94) and the
  10% rehearsal mix caused destructive forgetting; the gate stopped before
  descent unlock.
- Start-policy comparison now includes retention: frozen Landing has
  Flight Chain/Final=1/3 and Landing retention=89.58%; old Flight pilot has
  0/12 and retention=87.50%.  The pilot wins combined fixed Chain+Final while
  satisfying retention, so v7 selected it by evidence.  Its repair round used
  learning rate 1e-5 and 30% canonical Landing-entry rehearsal.
- Curriculum v7 late-descent completed 102400 effective steps from the old
  Flight pilot with LR=1e-5 and 30% canonical-entry rehearsal.  Runtime was
  healthy (NaN/OOM/compile restart=0), KL stayed 0.00063--0.00314 and value
  loss fell 4.10->0.258.  Fixed Flight: Chain=0, Final=13.125%, physical
  failure=86.875%, timeout=0; anchor/augmented Final=12.05/14.29%; termination
  pitch/roll/recovery=99/40/21.  Landing retention=61.458% versus 89.583%
  reference, so the same late-descent unlock gate failed again.
- PAUSED after two evidence-based attempts at the same gate.  Root cause now
  points to conflating the narrow canonical entry bank used for Chain matching
  with the rehearsal bank: 30% rehearsal preserves only `C_L`, not the full
  certified Landing envelope.  The next bounded repair should add a separate
  full Landing Tube rehearsal input while keeping `C_L` fixed for Chain; do not
  add PPO steps or widen the entry matcher before that distinction is reviewed.
