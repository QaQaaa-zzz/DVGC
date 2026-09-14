"""Reproducible qualification reports from recorded physical frames only.

This module neither runs a simulator nor recomputes the task label. Missing
values stay missing, and engineering unknowns remain in the case denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any


_COLORS = ("#245c91", "#bd6b25", "#787633", "#a55587")
_ACTION_NAMES = ("steer", "drive", "hip", "knee")


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _display(value: Any, precision: int = 6) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _load_rows(run_dir: Path, reference: Any) -> tuple[list[dict], list[str], str | None]:
    if not reference:
        return [], ["No trajectory was recorded for this case."], None
    path = (run_dir / str(reference)).resolve()
    if not path.is_relative_to(run_dir):
        return [], ["Trajectory reference lies outside this run; it was not read."], None
    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"Trajectory unavailable: {type(exc).__name__}: {exc}"], None
    rows, warnings = [], []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("frame is not an object")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            # A missing frame remains a gap rather than joining its neighbors.
            rows.append({})
            warnings.append(f"Invalid trajectory line {line_number}: {exc}")
    if not rows:
        warnings.append("Trajectory contains no recorded frames.")
    times = [_number(row.get("time")) for row in rows]
    valid_times = [time for time in times if math.isfinite(time)]
    if len(valid_times) != len(times):
        warnings.append("Some frame times are missing or nonfinite; those frames cannot be positioned on the time axis.")
    if any(right <= left for left, right in zip(valid_times, valid_times[1:])):
        warnings.append("Recorded times are not strictly increasing; record order is retained without sorting.")
    missing = [key for key in ("x", "z", "vx", "roll", "pitch", "yaw", "wx", "wy", "wz", "action", "ctrl", "signal")
               if any(row.get(key) is None for row in rows)]
    if missing:
        warnings.append("Missing values are shown as gaps, never zero-filled: " + ", ".join(missing))
    return rows, warnings, hashlib.sha256(raw).hexdigest()


def _plot_case(rows: list[dict], case: dict, config: dict, png: Path, pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    times = np.array([_number(row.get("time")) for row in rows])
    finite_times = times[np.isfinite(times)]
    title = f"JUMP | {case.get('case_id', 'unnamed')} | {case.get('policy', 'policy not recorded')}"
    subtitle = f"{case.get('status', 'unknown')}: {case.get('reason', 'reason not recorded')}"
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#dddddd", "grid.alpha": 0.6,
                         "axes.prop_cycle": matplotlib.cycler(color=_COLORS), "pdf.fonttype": 42}):
        if not len(finite_times):
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis("off")
            ax.text(0.03, 0.95, "\n".join(textwrap.wrap(title, 95)), va="top", fontsize=14)
            ax.text(0.03, 0.68, "\n".join(textwrap.wrap(subtitle, 105)), va="top")
            ax.text(0.03, 0.4, "No time-positioned trajectory is available. No curves were fabricated.\n"
                    "Approach qualification and complete-task success are distinct outcomes.", va="top")
            ax.text(0.03, 0.1, f"Approach qualified: {_display(case.get('qualified'))}", va="bottom")
        else:
            fig, grid = plt.subplots(6, 2, figsize=(14, 19), sharex=True)
            axes = grid.ravel()

            def draw(ax, key: str, label: str, *, column: int | None = None, scale: float = 1.0,
                     color: str = _COLORS[0], step: bool = False, linestyle: str = "-") -> None:
                values = []
                for row in rows:
                    value = row.get(key)
                    if column is not None:
                        value = value[column] if isinstance(value, (tuple, list)) and len(value) > column else None
                    values.append(_number(value) * scale)
                values = np.asarray(values)
                if np.any(np.isfinite(values) & np.isfinite(times)):
                    ax.plot(times, values, label=label, color=color, linestyle=linestyle, linewidth=1.3,
                            drawstyle="steps-post" if step else "default", marker="." if len(rows) < 10 else None)

            draw(axes[0], "x", "body x")
            axes[0].set(title="Forward position", ylabel="x (m)")
            scene = config.get("scene", {})
            front, length = _number(scene.get("front_x")), _number(scene.get("length"))
            if math.isfinite(front) and math.isfinite(length):
                axes[0].axhspan(front, front + length, color="#d6cfc2", alpha=0.5, label="obstacle x interval")
            draw(axes[1], "z", "body z")
            axes[1].set(title="Body height", ylabel="z (m)")
            draw(axes[2], "vx", "true world vx")
            limits = config.get("limits", {})
            for key, label, color, style in (("nominal_speed", "declared nominal speed", "#444444", "--"),
                                              ("min_forward_speed", "minimum forward speed", _COLORS[1], ":")):
                value = _number(limits.get(key))
                if math.isfinite(value):
                    axes[2].axhline(value, label=label, color=color, linestyle=style, linewidth=1.2)
            axes[2].set(title="Forward speed and declared requirements", ylabel="vx (m/s)")
            draw(axes[3], "roll", "roll", scale=180 / math.pi)
            draw(axes[3], "pitch", "pitch", scale=180 / math.pi, color=_COLORS[1], linestyle="--")
            draw(axes[3], "yaw", "yaw", scale=180 / math.pi, color=_COLORS[2], linestyle=":")
            axes[3].set(title="Recorded attitude", ylabel="angle (deg)")
            draw(axes[4], "wx", "wx")
            draw(axes[4], "wy", "wy", color=_COLORS[1], linestyle="--")
            draw(axes[4], "wz", "wz", color=_COLORS[2], linestyle=":")
            axes[4].set(title="Recorded angular velocity components", ylabel="angular velocity (rad/s)")
            for key, label, color, style in (("signal", "jump signal", _COLORS[0], "-"),
                                            ("front_contact", "front floor contact", _COLORS[1], "--"),
                                            ("rear_contact", "rear floor contact", _COLORS[2], ":")):
                draw(axes[5], key, label, color=color, step=True, linestyle=style)
            axes[5].set(title="Signal and actual floor contacts", ylabel="recorded value")
            for i, name in enumerate(_ACTION_NAMES):
                draw(axes[6], "action", name, column=i, color=_COLORS[i], step=True)
            axes[6].set(title="Applied normalized actions", ylabel="normalized action")
            for i, (name, unit) in enumerate(zip(_ACTION_NAMES, ("rad", "rad/s", "rad", "rad"))):
                draw(axes[7 + i], "ctrl", name, column=i, color=_COLORS[i], step=True)
                axes[7 + i].set(title=f"Actual {name} actuator command", ylabel=f"ctrl ({unit})")
            axes[11].axis("off")
            notes = [
                f"Approach qualified: {_display(case.get('qualified'))}",
                f"Trigger: {_display(case.get('trigger_time'))} s",
                f"Liftoff: {_display(case.get('liftoff_time'))} s",
                f"Recorded frames: {len(rows)}",
                f"Last recorded time: {finite_times[-1]:.6f} s",
                f"Charged control steps: {_display(case.get('charged_control_steps'))}",
                f"Physical steps: {_display(case.get('physics_steps'))}",
                "",
                "Full recorded interval; missing values remain gaps.",
                "Approach qualification is not complete-task success.",
                "Angular components use the runtime frame convention;",
                "wx/wy/wz are not Euler-angle derivatives.",
            ]
            axes[11].text(0.02, 0.98, "\n".join(notes), transform=axes[11].transAxes, va="top", fontsize=10)
            events = [("trigger", _number(case.get("trigger_time")), "#444444", "--"),
                      ("liftoff", _number(case.get("liftoff_time")), "#787633", ":")]
            if case.get("status") == "failure":
                failure_time = _number(case.get("failure_time", case.get("terminal_time")))
                if not math.isfinite(failure_time):
                    failure_time = float(finite_times[-1])
                events.append(("failure endpoint", failure_time, "#a33d4b", "-."))
            else:
                events.append(("last recorded frame", float(finite_times[-1]), "#777777", "-."))
            for ax in axes[:11]:
                if not ax.lines:
                    ax.text(0.5, 0.5, "No recorded values", transform=ax.transAxes, ha="center", color="#777777")
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc="best", fontsize=8)
                for _, at, color, style in events:
                    if math.isfinite(at):
                        ax.axvline(at, color=color, linestyle=style, linewidth=1)
                ax.set_xlabel("Time (s)")
                ax.tick_params(labelbottom=True)
            event_handles = [Line2D([0], [0], color=color, linestyle=style, label=f"{label}: {at:.4f} s")
                             for label, at, color, style in events if math.isfinite(at)]
            fig.suptitle("\n".join(textwrap.wrap(title, 115)) + "\n" + "\n".join(textwrap.wrap(subtitle, 115)),
                         fontsize=13, y=0.996)
            fig.legend(handles=event_handles, loc="upper center", bbox_to_anchor=(0.5, 0.954), ncol=3, fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=1.4)
        try:
            fig.savefig(png, dpi=130, bbox_inches="tight")
            fig.savefig(pdf, bbox_inches="tight")
        finally:
            plt.close(fig)


def build_report(run_dir: Path) -> dict:
    """Write analysis/INDEX.md and one PNG/PDF per case; return artifact metadata."""
    run_dir = Path(run_dir).resolve()
    results = json.loads((run_dir / "results.json").read_text())
    metadata = results.get("metadata", {})
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else metadata.get("config", {})
    config = config if isinstance(config, dict) else {}
    analysis = run_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    plot_dir = analysis / "cases"
    plot_dir.mkdir(exist_ok=True)
    counts = {"success": 0, "failure": 0, "unknown": 0}
    artifacts, table_rows, sections = [], [], []
    qualified_count = 0
    trace_groups: dict[str, list[str]] = {}
    untriggered_recorded_case_count = 0
    for number, case in enumerate(results.get("cases", []), start=1):
        case_id = str(case.get("case_id", f"case-{number}"))
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", case_id).strip(".") or "case"
        # Prefix retains uniqueness even when sanitized IDs collide.
        stem = f"{number:04d}-{name}"
        rows, warnings, raw_sha = _load_rows(run_dir, case.get("trajectory"))
        status = case.get("status", "unknown")
        counts[status if status in counts else "unknown"] += 1
        qualified_count += case.get("qualified") is True
        finite_times = [_number(row.get("time")) for row in rows if math.isfinite(_number(row.get("time")))]
        if raw_sha and finite_times:
            trace_groups.setdefault(raw_sha, []).append(case_id)
        signals = [_number(row.get("signal")) for row in rows]
        no_recorded_trigger = (bool(finite_times) and not math.isfinite(_number(case.get("trigger_time")))
                               and all(signal == 0 for signal in signals))
        untriggered_recorded_case_count += no_recorded_trigger
        if no_recorded_trigger:
            warnings.append("No trigger occurred in the recorded trajectory. Trigger-delay settings that never activated "
                            "cannot serve as independent trigger-timing comparisons.")
        png, pdf = plot_dir / f"{stem}.png", plot_dir / f"{stem}.pdf"
        _plot_case(rows, case, config, png, pdf)
        item = {"case_id": case_id, "status": status, "recorded_rows": len(rows),
                "last_recorded_time": finite_times[-1] if finite_times else None,
                "trajectory_sha256": raw_sha, "warnings": warnings,
                "no_recorded_trigger": no_recorded_trigger,
                "png": str(png.relative_to(run_dir)), "pdf": str(pdf.relative_to(run_dir))}
        artifacts.append(item)
        values = [case_id, case.get("policy"), case.get("case_name"), case.get("seed"), status,
                  case.get("reason"), case.get("qualified"), case.get("trigger_time"), case.get("liftoff_time"),
                  item["last_recorded_time"], case.get("charged_control_steps"), case.get("physics_steps"),
                  case.get("wall_seconds")]
        table_rows.append("| " + " | ".join(_display(value) for value in values) + f" | [PNG](cases/{stem}.png) / [PDF](cases/{stem}.pdf) |")
        sections += [f"## {_display(case_id)}", "", f"- Policy / case / seed: {_display(case.get('policy'))} / "
                     f"{_display(case.get('case_name'))} / {_display(case.get('seed'))}",
                     f"- Recorded outcome: **{_display(status)}**; reason: `{_display(case.get('reason'))}`.",
                     f"- Approach qualified: {_display(case.get('qualified'))}; trigger {_display(case.get('trigger_time'))} s; "
                     f"actual liftoff {_display(case.get('liftoff_time'))} s.",
                     f"- Recorded frames: {len(rows)}; last recorded time: {_display(item['last_recorded_time'])} s.",
                     f"- Cost: {_display(case.get('charged_control_steps'))} charged control steps; "
                     f"{_display(case.get('physics_steps'))} physics steps; {_display(case.get('wall_seconds'))} wall seconds.",
                     f"- [Full PNG](cases/{stem}.png) / [PDF](cases/{stem}.pdf)"]
        if raw_sha:
            source = (run_dir / str(case["trajectory"])).resolve().relative_to(run_dir)
            sections += [f"- [Original physical frames](../{source.as_posix()}), SHA-256 `{raw_sha}`."]
        sections += [f"- Evidence note: {warning}" for warning in warnings]
        sections += ["", f"![{_display(case_id)}](cases/{stem}.png)", ""]
    report = {"index": "analysis/INDEX.md", "counts": counts, "case_count": len(artifacts),
              "qualified_count": qualified_count, "cases": artifacts, "accounting": results.get("accounting", {}),
              "unique_physical_trace_count": len(trace_groups), "physical_trace_groups": trace_groups,
              "recorded_trace_case_count": sum(len(case_ids) for case_ids in trace_groups.values()),
              "untriggered_recorded_case_count": untriggered_recorded_case_count}
    lines = ["# JUMP qualification experiment", "",
             "Source: [results.json](../results.json); [frozen config](../config.json). "
             "Data role: ENGINEERING_DEVELOPMENT; host-MuJoCo qualification/transfer diagnosis.", "",
             f"Declared result records: **{len(artifacts)}**; success **{counts['success']}**, failure **{counts['failure']}**, "
             f"unknown **{counts['unknown']}**. Approach qualified: **{qualified_count}/{len(artifacts)}**.", "",
             f"Unique physical traces by trajectory SHA-256: **{len(trace_groups)}**, among "
             f"**{report['recorded_trace_case_count']}** cases with time-positioned recordings. "
             f"Cases with no trigger throughout their recorded interval: **{untriggered_recorded_case_count}**.", "",
             "Trace grouping uses exact original JSONL byte identity. Different hashes do not establish independent "
             "initial states or random samples. Repeated traces retain their actual run costs and case labels. "
             "Trigger-delay settings that never activated cannot serve as independent trigger-timing comparisons. "
             "Missing or nonfinite signal observations are not asserted to be untriggered.", "",
             "Approach qualification is distinct from complete narrow-obstacle crossing and recovery. "
             "This report preserves the runtime labels; it does not establish general jumping capability, "
             "held-out performance, or equivalence to legacy MJX rollouts. Unknown cases remain in the denominator.", "",
             "Every chart uses the full recorded physical interval. Trigger and actual liftoff are separate markers. "
             "Failure endpoint is the declared failure/terminal time when present, otherwise the last recorded frame "
             "of a failure case. Unknown endpoints are labeled only as the last recorded frame. Missing observations "
             "remain gaps; unavailable trajectories receive an explicit empty-evidence figure. Axes use each case's "
             "actual duration and independent vertical scales; compare the numeric units, not apparent curve heights.", "",
             "## Per-case results", "",
             "| Case ID | Policy | Case | Seed | Status | Reason | Approach qualified | Trigger (s) | Liftoff (s) | Last frame (s) | Charged controls | Physics steps | Wall (s) | Figures |",
             "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    trace_rows = [f"| `{digest}` | {len(case_ids)} | {', '.join(_display(case_id) for case_id in case_ids)} |"
                  for digest, case_ids in trace_groups.items()]
    lines += table_rows + ["", "## Recorded physical trace groups", "",
                           "| Original trajectory SHA-256 | Case count | Case IDs |", "|---|---:|---|"] + trace_rows
    lines += ["", "## Run accounting", "", "Frozen declared budgets and actual simulator cost are separate quantities.",
                           "", "```json", json.dumps(results.get("accounting", {}), indent=2, ensure_ascii=False), "```", ""] + sections
    (analysis / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    (analysis / "report_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
