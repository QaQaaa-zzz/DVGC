# Iteration-0 Group-Disjoint Validation Implementation Plan

1. Add RED tests for the absent protocol audit and exact 3-up/2-down real
   declaration.
2. Implement config/hash/identity/split/parent/state/near-duplicate checks.
3. Recompute the fixed attempt and interaction ceilings from the declared
   panels; reject hand-edited budgets.
4. Add an audit-only CLI and prelaunch declaration.
5. Run focused tests and full JIT preflight, commit the implementation, and
   stop without launching validation or fitting `C^0`.
