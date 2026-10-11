#!/usr/bin/env python3
"""Export a named checkpoint and verify saved TRAIN/DEV observations; no physics."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'task-config', 'observations-manifest', 'output', 'repository',
                 'expected-payload-sha256', 'expected-actor-sha256', 'expected-normalizer-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-observations', type=int, default=1024)
    args = parser.parse_args()
    # This is inference on saved arrays only, and must not compete for a training GPU.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from jit_dvgc.sim2sim_export import export_checkpoint
    report = export_checkpoint(**vars(args))
    print(json.dumps(dict(status=report['status'], output=args.output, parity=report['parity'],
                         physics=0, target_adapter_ready=False), indent=2))


if __name__ == '__main__':
    main()
