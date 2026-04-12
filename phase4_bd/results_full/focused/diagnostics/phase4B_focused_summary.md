# Phase IV-B Focused Rerun Summary

## Configuration
- Baseline: `N80_M240`
- Coarse comparator: `N40_M240`
- Refined comparator: `N120_M240`
- Shared cost: `c_meas=0.020000`
- Focused points: `40` (requested top-k=`40`)
- Dense local alpha resolution: `M_alpha=360`

## Local Reevaluation (D0 vs dense exact)
- Baseline abs error mean/max: `2.685120e-03` / `5.389143e-03`
- Coarse abs error mean/max: `1.093089e-02` / `2.282079e-02`
- Refined abs error mean/max: `2.254013e-03` / `5.722404e-03`

## Sign Stability Against Dense Exact D0
- Baseline sign flips: `6/40`
- Coarse sign flips: `6/40`
- Refined sign flips: `0/40`

Interpretation note: this is a focused numerical reevaluation around suspicious boundary points, not a proof.
