# Phase IV-D Summary (Theorem-Facing Synthesis)

## Scope and Exclusions
- This package delivers Phase IV-B / IV-D only.
- Axis A (cost sensitivity): treated as already completed.
- Axis C (projection-rule comparison): excluded unless baseline instability is detected.

## Experimental Baseline
- Baseline config: `N80_M240`
- Shared measurement cost: `c_meas=0.020000`

## Numerical Robustness Readout
- Max abs diff across refinements: `V1=1.249999e-02`, `V0=9.994788e-03`, `D1=1.249999e-02`, `D0=2.125514e-02`
- Worst decision disagreement rate: `0.003252`
- Worst alpha disagreement rate (measure-pair): `1.000000`

Numerically, the qualitative continuation regions remain stable under moderate grid and action refinements.
The observed pattern is consistent with finite-grid / finite-library approximation discussions.
We observe no evidence that the main D0 structure is a coarse-grid artifact.

## Representative-Belief Audit
- Representative points covered: `A, B, C, D, E`
- Per-point tables track value/action/projection diagnostics across all configs.

## Caveats
- This summary is theorem-adjacent numerical confirmation, not a proof.
- Refinement trends support stability claims but do not establish strict convergence theorems.
