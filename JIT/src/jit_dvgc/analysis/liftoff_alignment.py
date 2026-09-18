"""Offline, explicitly diagnostic liftoff alignment of saved pulse witnesses."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .pulse_tube_xz import read, round_directories


def liftoff_index(front, rear, threshold=.01, hold=3):
    """First sustained clearance AFTER observed contact; not the success oracle."""
    front, rear = np.asarray(front), np.asarray(rear)
    if (front.ndim != 1 or front.shape != rear.shape or
            not np.isfinite(front).all() or not np.isfinite(rear).all() or
            not np.isfinite(threshold) or threshold <= 0 or
            not isinstance(hold, int) or hold < 1):
        raise ValueError('invalid clearance arrays or diagnostic parameters')
    contact = np.flatnonzero((front <= 0) | (rear <= 0))
    if not len(contact):
        return None
    airborne = (front > threshold) & (rear > threshold)
    for i in range(int(contact[0]) + 1, len(front) - hold + 1):
        if airborne[i:i + hold].all():
            return i
    return None


def align_xz(points, liftoff_x):
    result = np.array(points, dtype=float, copy=True)
    result[:, 0] -= liftoff_x
    return result


def build(lineage, output, *, per_round=24, threshold=.01, hold=3):
    """One successful attempt per sampled candidate; zero requests to simulation."""
    from jit_dvgc.config import load_config
    from jit_dvgc.model import load_host_model

    if per_round < 0:
        raise ValueError('per_round must be nonnegative; zero means all')
    liftoff_index([], [], threshold, hold)
    root, output = Path(lineage).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def check(path, expected=None):
        path = str(Path(path).resolve())
        if path not in hashes:
            with open(path, 'rb') as f:
                hashes[path] = hashlib.file_digest(f, 'sha256').hexdigest()
        if expected is not None and hashes[path] != expected:
            raise ValueError('source hash mismatch: ' + path)

    spec = read(root / 'declaration.json')['spec']
    bank = read(spec['bank'])
    member = next(m for m in bank['members'] if m['name'] == spec['proposer'])
    formal = read(member['policy']['formal_config'])
    phase = load_config(Path(formal['inputs']['up_config_path']), runtime_only=True)
    q0 = load_host_model(phase).model_index.root_qpos_address
    for p in (root / 'declaration.json', spec['bank'], member['policy']['formal_config'],
              formal['inputs']['up_config_path'], __file__):
        check(p)
    directories = round_directories(root)  # Freeze completed-round membership once.
    selected, counts = [], []
    for d in directories:
        check(d / 'outcomes.json')
        rows = read(d / 'outcomes.json')
        successes = sorted((r for r in rows if r['label'] == 1), key=lambda r: r['index'])
        indices = (np.linspace(0, len(successes) - 1, min(per_round, len(successes)), dtype=int)
                   if per_round else range(len(successes)))
        chosen = [successes[i] for i in indices]
        selected.extend(chosen)
        counts.append(dict(round=int(d.name.split('_')[-1]), total=len(rows),
                           successful=len(successes), selected=len(chosen)))
    groups = defaultdict(list)
    for row in selected:
        groups[row['prefix_file']].append(row)
    metadata, segments, segment_meta = [], [], []

    def load(path):
        with np.load(path) as tape:
            names = ['qpos', 'front_wheel_clearance', 'rear_wheel_clearance',
                     'physical_failure', 'time']
            names += [k for k in ('prefix_mask', 'mask', 'success') if k in tape]
            return {k: tape[k] for k in names}

    def lane(tape, index, mask):
        ticks = np.flatnonzero(tape[mask][:, index])
        if not len(ticks):
            raise ValueError('empty active trace')
        if tape['physical_failure'][ticks, index].any():
            raise ValueError('successful witness has physical failure')
        return dict(xz=tape['qpos'][ticks, index][:, [q0, q0 + 2]],
                    front=tape['front_wheel_clearance'][ticks, index],
                    rear=tape['rear_wheel_clearance'][ticks, index],
                    time=tape['time'][ticks, index], ticks=ticks)

    for prefix_path, rows in groups.items():
        check(prefix_path, rows[0]['prefix_sha256'])
        prefix = load(prefix_path)
        by_trace = defaultdict(list)
        for row in rows:
            check(prefix_path, row['prefix_sha256'])
            attempt = None if row.get('prefix_terminal') else next(
                (a for a in row['attempts'] if a['label'] == 1), None)
            if not row.get('prefix_terminal') and attempt is None:
                raise ValueError('successful candidate lacks witness')
            by_trace[attempt['trace'] if attempt else None].append((row, attempt))
        for trace, items in by_trace.items():
            suffix = load(trace) if trace else None
            for row, attempt in items:
                pre = lane(prefix, row['index'], 'prefix_mask')
                parts = [('prefix', pre)]
                if attempt:
                    check(trace, attempt['trace_sha256'])
                    if attempt['snapshot_context_sha256'] != row['snapshot_context_sha256']:
                        raise ValueError('prefix/suffix context mismatch')
                    suf = lane(suffix, attempt['trace_lane'], 'mask')
                    success = suffix['success'][suf['ticks'], attempt['trace_lane']]
                    if len(suf['ticks']) != attempt['steps'] or not success[-1]:
                        raise ValueError('witness endpoint mismatch')
                    stop = int(np.flatnonzero(success)[0]) + 1
                    suf = {k: v[:stop] for k, v in suf.items()}
                    # Suffix clocks reset. Preserve both local clocks in exported data.
                    parts.append(('suffix', suf))
                all_xz = np.concatenate([p['xz'] for _, p in parts])
                if not np.isfinite(all_xz).all():
                    raise ValueError('nonfinite witness positions')
                # Search across saved consecutive transitions; no geometric join is drawn.
                k = liftoff_index(np.concatenate([p['front'] for _, p in parts]),
                                  np.concatenate([p['rear'] for _, p in parts]), threshold, hold)
                record = dict(id=len(metadata), round=row['round'], candidate=row['index'],
                              context=row['snapshot_context_sha256'],
                              source_actor_sha256=row.get('source_actor_sha256'),
                              witness=attempt['policy'] if attempt else row.get('witness'),
                              actor_sha256=attempt['actor_sha256'] if attempt else None,
                              prefix=prefix_path, suffix=trace,
                              prefix_lane=row['index'], suffix_lane=attempt['trace_lane'] if attempt else None,
                              pulse_start_step=row.get('pulse_start_step'),
                              status='aligned' if k is not None else 'no_diagnostic_liftoff')
                if k is not None:
                    record.update(liftoff_x=float(all_xz[k, 0]), liftoff_z=float(all_xz[k, 1]),
                                  liftoff_transition_index=int(k),
                                  liftoff_segment='prefix' if k < len(pre['xz']) else 'suffix')
                    for kind, part in parts:
                        segments.append(np.column_stack((part['xz'], align_xz(part['xz'], record['liftoff_x'])[:, 0],
                                                         part['time'], part['front'], part['rear'])))
                        segment_meta.append(dict(witness_id=record['id'], kind=kind, round=row['round']))
                metadata.append(record)
        print(f'Extracted {len(metadata)}/{len(selected)} selected witnesses', flush=True)
    if not segments:
        raise ValueError('no witnesses with a diagnostic liftoff')
    points = np.concatenate(segments)
    offsets = np.r_[0, np.cumsum([len(s) for s in segments])]
    np.savez_compressed(output / 'trajectory_points.npz', points=points, offsets=offsets,
                        columns=['world_x', 'root_z', 'aligned_x', 'segment_local_time', 'front_clearance', 'rear_clearance'])
    (output / 'witnesses.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'segments.json').write_text(json.dumps(segment_meta, indent=2) + '\n')
    with (output / 'liftoff_points.csv').open('w') as f:
        fields = ['id', 'round', 'candidate', 'status', 'liftoff_x', 'liftoff_z', 'witness', 'context']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(metadata)
    manifest = dict(lineage=str(root), completed_rounds=len(directories), per_round=per_round,
                    selection='evenly spaced successful candidate indices per round; first successful attempt',
                    counts=counts, selected=len(metadata), aligned=sum(r['status'] == 'aligned' for r in metadata),
                    diagnostic=dict(clearance_threshold_m=threshold, hold_frames=hold, prior_contact_threshold_m=0,
                                    definition='first sustained double-wheel clearance after observed wheel contact; control-frame diagnostic, not formal liftoff oracle'),
                    root_qpos_address=q0, success_criterion=spec.get('success_criterion'),
                    coordinate_transform='x_relative = x_world - diagnostic_liftoff_x; root z unchanged',
                    prefix_suffix='separate segments; replay join is never drawn; time column is segment-local',
                    new_simulation_interactions=0, input_sha256=hashes)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    render(segments, segment_meta, metadata, output, manifest)
    return manifest


def render(segments, segment_meta, metadata, output, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib import font_manager

    font = Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    rounds = np.array([m['round'] + 1 for m in segment_meta])
    norm = Normalize(1, max(2, manifest['completed_rounds']))
    cmap = plt.get_cmap('viridis')
    allpts = np.concatenate(segments)
    for full in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, layout='constrained')
        for ax, column, title in zip(axes, (0, 2), ('原始世界坐标', '离地点对齐：只平移 x，保留真实高度')):
            collection = LineCollection([s[:, [column, 1]] for s in segments],
                                        colors=cmap(norm(rounds)), linewidths=.65, alpha=.24,
                                        rasterized=True)
            ax.add_collection(collection)
            xmin, xmax = allpts[:, column].min(), allpts[:, column].max()
            ax.set_xlim(xmin - .06, xmax + .06 if full else min(xmax + .06, xmin + 3.3))
            ax.set_ylim(max(0, allpts[:, 1].min() - .03), allpts[:, 1].max() + .05)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('世界位置 x（m）' if column == 0 else '距诊断离地点 x − x_LO（m）')
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_ylabel('车体根部高度 z（m）')
        aligned = [r for r in metadata if r['status'] == 'aligned']
        for ax, xs in ((axes[0], [r['liftoff_x'] for r in aligned]), (axes[1], np.zeros(len(aligned)))):
            ax.scatter(xs, [r['liftoff_z'] for r in aligned], s=8, c='#c43c39', alpha=.4, label='诊断离地点', zorder=3)
            ax.legend(loc='upper right', frameon=False)
        axes[1].axvline(0, color='#c43c39', linestyle='--', linewidth=1)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label='探索轮次', shrink=.7)
        fig.suptitle(f'跳跃轨迹离地点对齐｜{manifest["completed_rounds"]} 轮 · {manifest["aligned"]} 条成功见证'
                     + ('｜完整恢复' if full else '｜跳跃局部视图'), fontsize=16)
        fig.supxlabel('按轮次抽样；前缀/接续分段。接地后双轮间隙连续 '
                       f'{manifest["diagnostic"]["hold_frames"]} 帧 > '
                       f'{manifest["diagnostic"]["clearance_threshold_m"] * 100:g} cm 为诊断离地；非正式事件判据。\n'
                       '对齐图用于比较形状，不用于世界坐标可达性计数；z 是根部高度，不是轮胎净空。', fontsize=9)
        name = 'liftoff_aligned_full' if full else 'liftoff_aligned'
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'{name}.{ext}', dpi=180)
        plt.close(fig)
    (output / 'INDEX.md').write_text(
        '# 离地点对齐图\n\n'
        f'来源：`{manifest["lineage"]}`。冻结读取 {manifest["completed_rounds"]} 个已完成轮次；'
        f'成功候选 {sum(r["successful"] for r in manifest["counts"])} 条，抽取 {manifest["selected"]} 条，'
        f'可对齐 {manifest["aligned"]} 条。每候选只保留首个成功接续见证。\n\n'
        '- [主图](liftoff_aligned.png) · [PDF](liftoff_aligned.pdf) · [SVG](liftoff_aligned.svg)\n'
        '- [完整恢复图](liftoff_aligned_full.png) · [PDF](liftoff_aligned_full.pdf)\n'
        '- [离地点表](liftoff_points.csv) · [完整坐标](trajectory_points.npz)\n'
        '- [来源与未对齐记录](witnesses.json) · [段索引](segments.json) · [参数和哈希](manifest.json)\n\n'
        '离地诊断先要求已观察到至少一轮接地，再取双轮超过指定净空并保持指定帧数的首帧。'
        '这是控制帧诊断，排除初始悬空间隙；没有插值估计物理子步离地时刻。'
        '前缀和接续不连线，接续的时间列保留局部时钟。未找到诊断离地的见证明确排除，不伪造对齐点。'
        '仅平移x，真实z不变，故红色离地点位于x=0但高度不必相同。'
        '本图是成功前缀/接续见证的抽样投影，不是单策略包线或填充可达域；没有新增仿真。\n')
