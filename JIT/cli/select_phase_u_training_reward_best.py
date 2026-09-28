#!/usr/bin/env python3
"""Select training-reward-best Phase U checkpoint after a complete run."""

import argparse
import json
from pathlib import Path

from jit_dvgc.config import load_config
from jit_dvgc.phase_u_training_reward_best import select_training_reward_best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.sim2sim_randomization is None:
        parser.error("training reward selector requires a randomized Phase U config")
    result = select_training_reward_best(
        args.run_dir,
        target=config.ppo.requested_transitions,
        block=config.ppo.block_transitions,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
