# Phase III Summary

## Configuration
- Phase I source: `/Users/absolute_injury/Desktop/2026QSD_simulation/outputs/paper_final/data/one_step_maps.npz`
- Belief resolution N: `80`
- Number of grid points: `3321`
- Alpha samples M_alpha: `240`
- Baseline cost c_meas (run0): `0.0`
- Epsilon cost c_meas (run_eps): `0.02`

## Baseline (run0) quick reads
- Stage-1 continue fraction: `0.9991`
- Stage-0 continue fraction: `0.9991`
- D1 quantiles (50/90/99%): `0.166764, 0.255637, 0.307083`
- D0 quantiles (50/90/99%): `0.119512, 0.170932, 0.181387`
- Center D1 / D0: `0.333385` / `0.143847`
- Optional action-difference fraction (delta alpha idx > 0): `0.8374`

## Diagnostics
- Probability normalization pass: `True`
- Posterior normalization pass: `True`
- Nonnegativity pass (prob/post): `True` / `True`

## Outputs
- `phase3_values_run0.npz`: `/Users/absolute_injury/Desktop/2026QSD_simulation/outputs/phase3_sequential/phase3_values_run0.npz`
- `phase3_diag_run0.json`: `/Users/absolute_injury/Desktop/2026QSD_simulation/outputs/phase3_sequential/phase3_diag_run0.json`
- `phase3_values_run_eps.npz`: `/Users/absolute_injury/Desktop/2026QSD_simulation/outputs/phase3_sequential/phase3_values_run_eps.npz`

## Notes
- D1 and D0 are kept raw in arrays. Tiny negative numerical residues are clipped only in plotting.
- Stage-1 and Stage-0 best-action maps are stored as alpha indices and alpha values.
