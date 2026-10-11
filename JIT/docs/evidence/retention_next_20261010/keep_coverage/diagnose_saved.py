"""Describe saved development failures and actual gradients; never generate TRAIN."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--update", type=int, help="Diagnose a rejected trained node without changing candidate selection")
args = parser.parse_args()
root, output = args.run.resolve(), args.output.resolve()
read = lambda p: json.loads(Path(p).read_text())
plan = read(root / "plan.json")
finish = read(root / "B_completed.json")
selected = str(finish["selected_update"] if args.update is None else args.update)
row = read(root / "B_checkpoint_results.json")[selected]
repeat_file = root / f"B{int(selected):04d}_warning_repeat.json"
pair = read(repeat_file) if repeat_file.exists() else dict(baseline=read(root/"B_checkpoint_results.json")["0"]["full"], original=row["full"])
repetitions = [("first", pair["baseline"], pair["original"])]
if repeat_file.exists():
    repetitions.append(("repeat", pair["repeated_pi0"], pair["repeated"]))
cases = plan["batches"][0]["cases"]


def events(archive, lane):
    mask = archive["prefix_mask"][:, lane]
    n = int(mask.sum())
    assert n > 0 and mask[:n].all() and not mask[n:].any()
    result = dict(steps=n, success=bool((archive["success"][:, lane] & mask).any()),
                  end_code=int(archive["end_code"][n - 1, lane]),
                  recovery_ticks_end=int(archive["recovery_ticks"][n - 1, lane]), first={})
    contact = np.flatnonzero(archive["first_valid_contact"][:, lane] & mask)
    for name in ("first_valid_contact", "roll_limit", "prohibited_contact", "physical_failure"):
        indices = np.flatnonzero(archive[name][:, lane] & mask)
        tick = int(indices[0]) if len(indices) else None
        phase = None if tick is None else "post_contact" if len(contact) and tick > contact[0] else "at_or_before_first_valid_contact"
        result["first"][name] = dict(control_tick=tick, phase=phase,
                                     substep="UNKNOWN; control-step trace only")
    return result


losses = []
for group in "ABCD":
    lanes = [i for i, c in enumerate(cases) if c["group"] == group]
    for index, lane in enumerate(lanes):
        labels = [int(pair[k]["labels"][group][index]) for k in (("baseline", "original", "repeated_pi0", "repeated") if repeat_file.exists() else ("baseline", "original"))]
        if any(labels[i] and not labels[i+1] for i in range(0,len(labels),2)):
            losses.append(dict(case=cases[lane]["case"], lane=lane, labels=labels,
                               labels_order=["pi0_first", "student_first", "pi0_repeat", "student_repeat"][:len(labels)],
                               gradient_allowed=False, trajectories=[]))
for repetition, baseline, student in repetitions:
    ap, bp = Path(baseline["output"]) / "prefixes.npz", Path(student["output"]) / "prefixes.npz"
    with np.load(ap) as a, np.load(bp) as b:
        for case in losses:
            lane = case["lane"]
            ae, be = events(a, lane), events(b, lane)
            case["trajectories"].append(dict(repetition=repetition, pi0=ae, student=be,
                                              pi0_trace=str(ap), student_trace=str(bp)))
metrics = [json.loads(line) for line in (root / "B/metrics.jsonl").read_text().splitlines()]
assert [x["update"] for x in metrics] == list(range(1, finish["completed_updates"]+1))
gradients = {}
for key in ("actual/cosine_demo_keep", "actual/gradient_norm_demo", "actual/gradient_norm_keep",
            "actual/global_clip_scale", "actual/actor_adam_delta_norm"):
    values = np.asarray([x[key] for x in metrics])
    assert np.isfinite(values).all()
    gradients[key] = dict(mean=float(values.mean()), minimum=float(values.min()),
                          maximum=float(values.max()), negative_fraction=float((values < 0).mean()))
result = dict(selected_update=finish["selected_update"], diagnostic_update=int(selected), complete_DEV_old_losses=losses,
              actual_same_update_gradients=gradients, supervision_updates=len(metrics),
              scope="Saved-data development diagnosis; no causal or capacity proof; no DEV to TRAIN",
              next_experiment="B: TRAIN student-visited-state guidance diagnosis only",
              next_budget_proposal=dict(charged_max=544000, wall_hours=3, BC_updates=0,
                                        PPO_updates=0, E_G_updates=0, execution_authorized=False))
with (output / "diagnostics.json").open("x") as stream:
    json.dump(result, stream, indent=2)
print(output / "diagnostics.json")
