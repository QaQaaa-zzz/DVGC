# Unified Descent 8-teacher bootstrap v2

Result: **TEACHER_MEMORIZATION_OR_SUPPORT_GAP**. The eight exact-replay teachers and marginal representation gate were valid, but candidate-level physical transfer failed after the single authorized support-gated relabel round.

- Authority: exactly 8/8 teachers satisfy deterministic exact-replay gain >=4. The two old-summary mismatches are `da0679b1` (14 -> 10 ticks; gain 5 -> 1, removed) and `173ee307` (18 -> 17; gain 5 -> 4, retained).
- Representability: close opposite-label conflict 17.1875% against the 20% limit, therefore marginal pass, not strong pass.
- Relabel: 40 student-visited states audited, 11 accepted under all support/phase/contact/delay/precursor/4-tick gates. Excluded folds and held-out were not used.
- Candidate CV: Fold A gains `[1,1,0]` but its selected diagnostic checkpoint violates the anchor max-drift gate; Fold B `[0,0,0]`; Fold C `[0,0]`. Combined: 0/8 gain >=2, median 0, one positive-median fold.
- Final bootstrap, held-out evaluation, Landing retention, and PPO were not run. The original 24-tick survivor was retained by the hard-eligible Fold B/C checkpoints. PPO authorization is false.
- A monolithic CV process hit an engineering OOM. Recovery used independent fold processes, reused complete checkpoints, and retried only the incomplete learning-rate shard in a non-overwriting path; all folds then completed.

The protocol is exhausted. More student relabels, hidden-block unfreezing, held-out selection, final bootstrap, or PPO would violate this bounded experiment.

