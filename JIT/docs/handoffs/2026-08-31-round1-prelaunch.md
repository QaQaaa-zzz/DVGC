# DVGC/JIT Round-1 Unified Prelaunch — 2026-08-31

## Status

Round-1 is **metadata-sealed but not started**. The runtime hardening blocker is closed on real GPU evidence, and the only remaining action before the 10,009,600-transition launch is post-seal verification of the final config/branch state.

## Single-variable intervention

Round-1 changes exactly one method variable relative to Round-0 unified training:

- reset distribution: 10% existing Phase-U natural reset + 90% existing learned Soft-Tube RSI;
- Bernoulli selection per episode;
- seed remains `821101`;
- requested PPO training remains `10,009,600` transitions;
- Actor, Critic, optimizer remain wholly fresh;
- PPO, environment, reward, XML, learned Soft Tube, checkpoint schedule, TRAIN panel, validation/TEST exclusion remain unchanged.

The learned Soft Tube remains training guidance only. TRAIN panels are not independent final-policy evaluation and cannot support `pi_unified_star`, JCE/JEL, or safety/certification claims.

## Locked Round-0 causal evidence

Causal classification:

`PI_UP_APEX_UNIFIED_PRE_JUMP_FAIL`

Existing reports are bound by SHA-256 and were **not regenerated**:

- natural-start Round-0 diagnostic: `eb8ccb07e1d8229abd794283fc3c48c63943f9a0356cf313c89186458a939721`;
- frozen `pi_up` vs Round-0 unified comparison: `53c708fd33df66c53a1ab60098d033ebf1284df3fe0c16f999a3942630bbca8f`.

The raw reports remain existing local evidence; this handoff records their identities rather than fabricating or regenerating them.

## Runtime hardening evidence

Validated runtime hardening HEAD:

`f9be2c6e4b5e5ef7e654bb7704e711177ffa89d6`

Real GPU checks reported on the RTX 4090 D:

- pre-forward natural reset vs existing natural semantics: `1 passed`;
- Brax full-reset-cycle resampling without pytree drift: `1 passed`;
- full unified reset-mixture GPU suite: `4 passed`.

The mixed reset now selects pre-forward JAX reset data and executes exactly one MJX/Warp forward per reset. Round-0 `natural_reset_probability=0` retains the direct existing Tube reset path.

## Pre-metadata-seal gates

The following gates passed at hardening HEAD `f9be2c6...` before the provenance-only metadata seal:

- zero-interaction reset smoke: completed, 256 samples, 31 natural / 225 Soft-Tube, configured 0.1 / 0.9, `runtime_naccdmax=1024`, 0 environment interactions, 0 training transitions, no expert switching, no validation/TEST;
- locked handoff archive: `1149 files, 39937705 bytes, largest=1861326` verified;
- local preflight: `374 passed, 29 deselected` plus `14 passed` GPU.

Because the metadata seal changes the canonical config SHA without changing the method/runtime, the final config must still be loaded/tested and the zero-interaction smoke/archive/preflight rerun once after synchronization.

## Final config and predeclared run

Final Round-1 config:

`JIT/configs/pi_unified_round1_natural10.json`

Canonical config SHA-256:

`fba8ac1975727ea77ec271a6196cc998eb63b47960aba25935656927d69f2ae1`

Predeclared run ID:

`pi_unified_round1_natural10_10009600_seed821101_20260831`

Output directory:

`JIT/runs/pi_unified/pi_unified_round1_natural10_10009600_seed821101_20260831`

Current declaration status: `predeclared_not_started`.

## Required post-seal gate

Before launch, synchronize the branch and verify:

1. focused Round-1 config/formal CPU tests;
2. zero-interaction reset smoke reports final config SHA `fba8ac...`;
3. locked handoff archive verifier passes;
4. `JIT/scripts/local_preflight.sh` passes;
5. no predeclared Round-1 output directory already exists.

Only after those checks is the fresh 10,009,600-transition Round-1 PPO permitted to start. Independent frozen-final-policy natural-start / Final-Recovery evaluation remains mandatory after training.
