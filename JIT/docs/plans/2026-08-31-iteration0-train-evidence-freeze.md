# Iteration-0 TRAIN Evidence Freeze Plan

1. Add failing tests for exact readiness, TRAIN-only rows, unique physical
   states, finite observations, policy identity, parent-group accounting, and
   post-write tamper detection.
2. Implement one JIT-local freeze/loader module and CLI.
3. Add the exact completed-run config and zero-interaction prelaunch record.
4. Run focused CPU tests, static compilation, the full JIT preflight, and the
   real freeze once.
5. Independently audit the frozen artifact, then predeclare group-disjoint
   expansion validation before any `C^0` fit.
