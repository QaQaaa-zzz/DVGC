"""Read-only G billing reconciliation along explicit series/campaign ancestry."""
import json
from pathlib import Path
from .contracts import file_sha


def reconcile_generator_billing(boundary):
    """Retain billed reservations, deduplicate reused stages, never infer age as cost.

    Only explicit continuation_parent, parent_campaign and recovery.previous edges
    are followed. Missing evidence is an error, not a fabricated lifetime total.
    A durable pre-proposal charge proves all preceding proposals completed; the
    last charged proposal may have been interrupted before or after completion.
    """
    locks={};series=set();campaigns=set();entries={};active=set();fresh_origins=set()
    def read(path):
        path=Path(path).resolve();sha=file_sha(path)
        if str(path) in locks and locks[str(path)]!=sha:raise ValueError('G cost evidence changed')
        locks[str(path)]=sha
        return json.loads(path.read_text())
    def campaign_parent(root):
        return read(root/'production.json').get('recovery',{}).get('previous')
    def origin(root,stage):
        seen=set()
        while True:
            if root in seen:raise ValueError('G recovery cost ancestry cycle')
            seen.add(root);path=root/(stage+'_config.json')
            if path.exists():return path
            previous=campaign_parent(root) if (root/'production.json').exists() else None
            if previous is None:raise ValueError('missing generator origin config: '+str(path))
            root=Path(previous).resolve()
    def collect(root):
        path=root/'costs.json'
        if not path.exists():raise ValueError('missing generator cost ledger: '+str(path))
        for cost in read(path):
            stage=cost.get('stage','')
            if not stage.startswith('generator_'):continue
            config_path=origin(root,stage);config=read(config_path)
            key=str(config_path.resolve());charged=cost.get('charged_updates')
            if type(charged) is not int or charged<0:raise ValueError('invalid generator billed count')
            if key in entries:
                if entries[key]['charged_updates']!=charged:raise ValueError('inherited G billed count drift')
                continue
            output=Path(config['output']).resolve()
            if stage.startswith('generator_incremental_'):
                output=output/f"attempt_{int(stage.rsplit('_',1)[1]):04d}"
            elif not stage.startswith('generator_pretrain_'):
                raise ValueError('unknown generator billing stage')
            progress_path=output/'cost_progress.json';receipt_path=output/'cost_receipt.json'
            selection_path=output/'generator_selection.json'
            proposed=read(progress_path)['charged_updates'] if progress_path.exists() else 0
            if type(proposed) is not int or not 0<=proposed<=charged:
                raise ValueError('generator actual proposal charges exceed billed reservation')
            if receipt_path.exists():
                receipt=read(receipt_path)
                if receipt['charged_updates']!=proposed:raise ValueError('G proposal receipt drift')
            if selection_path.exists():
                selection=read(selection_path);completed=selection['updates'];lower=upper=completed
                if (stage.startswith('generator_pretrain_') and selection.get('initial_updates')==0
                        and selection.get('charged_updates_before')==0):fresh_origins.add(key)
                if completed!=proposed:raise ValueError('G completed/proposal count drift')
            else:lower=max(0,proposed-1);upper=proposed
            entries[key]=dict(config=str(config_path.resolve()),ledger=str(path.resolve()),
                stage=stage,charged_updates=charged,proposed_update_charges=proposed,
                completed_updates_lower_bound=lower,completed_updates_upper_bound=upper,
                status='completed' if selection_path.exists() else 'interrupted_or_unstarted',
                progress=str(progress_path) if progress_path.exists() else None)
    def visit_campaign(root):
        root=Path(root).resolve()
        if root in active:raise ValueError('generator billing ancestry cycle')
        if root in campaigns:return
        active.add(root);campaigns.add(root);collect(root)
        previous=campaign_parent(root)
        if previous:visit_campaign(previous)
        active.remove(root)
    def visit_series(root):
        root=Path(root).resolve()
        if root in active:raise ValueError('generator billing ancestry cycle')
        if root in series:return
        active.add(root);series.add(root);plan=read(root/'plan.json')
        for path in sorted(root.glob('round_*/costs.json')):collect(path.parent)
        if plan.get('continuation_parent'):visit_series(plan['continuation_parent'])
        if plan.get('parent_campaign'):visit_campaign(plan['parent_campaign'])
        active.remove(root)
    visit_series(boundary['parent_series']);visit_series(boundary['bundle_series'])
    manifest=read(boundary['bundle']['generator']['checkpoint_manifest']);age=manifest['updates']
    total=sum(e['charged_updates'] for e in entries.values())
    if len(fresh_origins)!=1 or not entries or type(age) is not int or total<age:
        raise ValueError('cannot reconcile G lifetime billing from explicit ancestry')
    return dict(schema='jit_generator_billing_reconciliation_v1',total_charged_updates=total,
        charged_updates_scope='lifetime_conservative_billed',state_updates=age,
        proposed_update_charges=sum(e['proposed_update_charges'] for e in entries.values()),
        completed_updates_lower_bound=sum(e['completed_updates_lower_bound'] for e in entries.values()),
        completed_updates_upper_bound=sum(e['completed_updates_upper_bound'] for e in entries.values()),
        reservation_rule='interrupted unused reservations remain billed; no checkpoint rollback refunds',
        entries=list(entries.values()),locks=locks)
