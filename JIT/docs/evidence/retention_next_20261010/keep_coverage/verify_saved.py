"""CPU-only saved-state audit. No environment, simulation, or optimizer updates."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from jit_dvgc.handoff_bank import pytree_sha256

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
assert jax.default_backend() == "cpu", "Use JAX_PLATFORMS=cpu"
root = args.run.resolve()
plan = json.loads((root / "plan.json").read_text())
source = plan["models"]["pi0"]["policy"]
checks = []
for step in (0, 100, 500, 1000, 2000):
    metadata = json.loads((root / "B" / f"learner_update_{step:04d}.json").read_text())
    path = Path(metadata["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == metadata["sha256"]
    with path.open("rb") as stream:
        state = pickle.load(stream)
    with (root / "B" / f"update_{step:04d}.pkl").open("rb") as stream:
        inference = pickle.load(stream)
    assert state["phase"] == "BC"
    assert state["completed_supervised_updates"] == step
    assert state["seed"] == plan["seed"]
    assert state["sampling"] == "fold_in(rng,absolute_update)"
    assert state["optimizer"] == "Adam1e-5/clip1"
    assert np.array_equal(state["rng"], jax.random.PRNGKey(plan["seed"]))
    normal = pytree_sha256(state["normalizer"])
    actor = pytree_sha256(state["actor"])
    critic = pytree_sha256(state["critic"])
    assert normal == source["normalizer_sha256"] == metadata["normalizer_sha256"]
    assert critic == source["critic_sha256"] == metadata["critic_sha256"]
    assert actor == metadata["actor_sha256"]
    assert pytree_sha256(state["optimizer_state"]) == metadata["optimizer_sha256"]
    assert [pytree_sha256(x) for x in inference] == [normal, actor, critic]
    adam = state["optimizer_state"][1][0]
    assert int(adam.count) == step
    assert jax.tree.structure(adam.mu) == jax.tree.structure(state["actor"])
    assert jax.tree.structure(adam.nu) == jax.tree.structure(state["actor"])
    for tree in (state["actor"], state["optimizer_state"]):
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))
    if step == 0:
        assert actor == source["actor_sha256"]
        assert all(np.count_nonzero(np.asarray(x)) == 0 for x in jax.tree.leaves((adam.mu, adam.nu)))
    else:
        assert actor != source["actor_sha256"]
    checks.append(dict(clock=step, Adam=int(adam.count), RNG=np.asarray(state["rng"]).tolist(),
                       normalizer=normal, critic=critic, actor=actor, file_sha256=digest))
result = dict(phase="passed", checkpoints=checks, resume_implemented=False,
              simulation_steps=0, supervised_updates=0, backend="cpu",
              scope="Actual serialized BC state, inference pairing, fresh initialization and absolute clocks")
target = root / "BC_full_state_verification.json"
with target.open("x") as stream:
    json.dump(result, stream, indent=2)
print(target)
