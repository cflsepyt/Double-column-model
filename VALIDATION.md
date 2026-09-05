# SBM implementation and validation results

SBM fixes 1--4 and the module split are implemented. Numerical budget closure
and timestep agreement pass. **Radiative-convective equilibrium has not been
established**, and the changes do not yet demonstrate that the original long-run
overheating problem is fully resolved.

The existing `climlab` conda environment was used: climlab 0.9.2,
climlab-sbm-convection 0.2, climlab-rrtmg 0.4.1, and NumPy 2.5.1.
No environment installation or modification was needed.

## Regression checks

All eight tests passed (`python -m unittest -v test_sbm`), covering:

- Subsaturated initial conditions, shared radiation/humidity arrays, and CO2
  propagation to both radiation components, including quadrupling.
- Condensation energy/water conservation and scalar-column precipitation output.
- Daily means versus extrema and round-trip NetCDF output.
- Catchable failure snapshots.
- Synchronous and asynchronous radiation budget accounting.
- DCM storage accounting with the separate legacy WTG source.
- Original RCE and control/abrupt-4xCO2 entry points.

Python compilation and `git diff --check` also passed.

## Single-column integrations

Independent runs used 60 levels, a 1 m slab, identical initial conditions,
300 ppm CO2, and synchronous radiation/physics at 600 s and 300 s.
The existing insolation, albedo, and surface exchange coefficients were retained.
There was no WTG, temperature clipping, prescribed evolving RH, or empirical
convection energy correction.

Both one-year runs and both five-year (1825-day) runs completed without an
RRTMG failure. The five-year comparison uses the final 90 days:

| Quantity | 600 s | 300 s |
| --- | ---: | ---: |
| Mean surface temperature (K) | 307.299245 | 307.299239 |
| Surface temperature trend (K/year) | +0.806544 | +0.806549 |
| Water-path trend (kg/m2/year) | +7.409249 | +7.409279 |
| Mean ASR - OLR (W/m2) | +1.149291 | +1.149290 |
| Mean surface energy storage (W/m2) | +0.107058 | +0.107057 |
| Mean atmospheric thermal storage (W/m2) | +0.454523 | +0.454524 |
| Mean atmospheric latent storage (W/m2) | +0.587506 | +0.587506 |

The surface-temperature difference is 0.0000065 K. The maximum difference in
the final-window atmospheric temperature profiles is 0.000024 K.

Over the entire integrations, the largest absolute total energy residual was
below 0.001 W/m2 and the largest water residual below 3.8e-10 kg/m2/s.
Storage minus the sum of the applied process energy tendencies closed within
1.2e-9 W/m2. Maximum instantaneous surface temperature stayed below 307.40 K,
atmospheric temperature below 305.43 K, and RH below 1.016.

The positive TOA input is accounted for by ongoing thermal and latent energy
storage. The runs are therefore stable and timestep-converged over the tested
period, but still warming and moistening. The validator correctly reports
`budget_ok=true`, `timestep_converged=true`, `equilibrated=false`, and
`passed=false` (exit status 2). This is not a runtime failure.

Detailed outputs:

- [Five-year comparison](data/sbm_validation_long/comparison.json)
- [600 s daily output](data/sbm_validation_long/dt600_daily.nc)
- [300 s daily output](data/sbm_validation_long/dt300_daily.nc)
- [One-year comparison](data/sbm_validation/comparison.json)

## Remaining warming transition

In the 600 s run, the highest level with appreciable mean SBM temperature
tendency changes from about 175 hPa on day 1550 to about 158 hPa on day 1575.
Column water increases from approximately 93.18 to 94.04 kg/m2, and net TOA
input rises from approximately 0.282 to 1.685 W/m2. The 300 s run reproduces
the late warming transition.

An offline radiation calculation on those daily-mean states supports a humidity
contribution: holding atmospheric and surface temperatures at their day-1550
values while replacing humidity with the day-1575 profile changes net TOA input
from 0.283 to 2.270 W/m2, an increase of approximately 1.987 W/m2. Both ASR and
OLR were recomputed, with all other radiation inputs unchanged. This is a
sensitivity calculation, not a complete causal decomposition of the integration.

These observations motivate a subsequent vertical-resolution and radiative
forcing/feedback investigation before assuming that longer spin-up alone will
produce an acceptable RCE. They do not justify silently changing albedo or SBM
reference humidity. The full experiment entry point now refuses to use an RCE
candidate that has not passed the equilibrium criteria.

The known WTG transport defects remain outside fixes 1--4. They are isolated in
`dcm_wtg.py` and separately accounted for in DCM output; the standalone results
above do not validate that circulation.
