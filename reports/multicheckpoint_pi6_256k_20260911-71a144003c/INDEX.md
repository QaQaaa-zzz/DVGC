# 256k training-length pilot

Training completed: 256,000 PPO + 392 fixed TRAIN-panel interactions; 84.061 seconds.

All checkpoint/trajectory payloads remain under training/. Full plotting data: checkpoint_panels.csv, all_panel_records.csv, training_metrics.csv; checkpoint_training.png/pdf/svg. Declaration: reservation.json; source and checkpoint hashes preserved.

Late exploration comparison: pi6 and same-run128k/192k/256k,8 shared perturbation trajectories each, original pi0–pi6 evaluator bank, historical9296-cell novelty exclusion, serial first-success labeling. Maximum6,000,000 interactions; one attempt;600s per acquisition/evaluator worker; no automatic extension/promotion/TEST. plan: exploration/plan.json.

After this comparison, stop fine-grained length tuning and move to matched residual training data. Pending earlier condition trials have not run.

## Exploration outcome

{
  "charged_interactions": 24916,
  "final_test_used": false,
  "plan_sha256": "943cce3d93c45540dead680ba7803019cf48784b7ec69b2cabcebf98972ad40e",
  "status": "completed",
  "training_transitions": 0,
  "wall_seconds": 1232.0735195550005
}

Complete metrics and equal-budget rows are in exploration/figures/; preserve unknowns. This closes the declared length pilot; no extra PPO is scheduled. Residual data audit is at ../.. /engineering/residual_data_readiness_20260911 (absolute path in next_stage.json).
