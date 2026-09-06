# SBM implementation and validation results

SBM fixes 1--4 and the module split are implemented. Numerical budget closure
and timestep agreement pass. **Radiative-convective equilibrium has not been
established**, and the changes do not yet demonstrate that the original long-run
overheating problem is fully resolved.

The existing `climlab` conda environment was used: climlab 0.9.2,
climlab-sbm-convection 0.2, climlab-rrtmg 0.4.1, and NumPy 2.5.1.
No environment installation or modification was needed.

## Regression checks

All eleven tests passed (`python -m unittest -v test_sbm`), covering:

- Subsaturated initial conditions, shared radiation/humidity arrays, and CO2
  propagation to both radiation components, including quadrupling.
- Condensation energy/water conservation and scalar-column precipitation output.
- Daily means versus extrema and round-trip NetCDF output.
- Catchable failure snapshots.
- Synchronous and asynchronous radiation budget accounting.
- DCM storage accounting with separate conservative WTG convergence.
- Original RCE and control/abrupt-4xCO2 entry points.
- Enforcement of the one-to-eight sweep-worker limit.

Python compilation and `git diff --check` also passed.

## Refactor verification

The coupled integration now reuses the snapshot already taken for WTG, keeps a
daily accumulator for each column, and stacks the column results once per day.
Diagnostic mappings are read once per process and layer mass is reused within a
diagnostic pass. The donor-cell face loop was replaced by its equivalent NumPy
array expression. Exact dataset comparisons used during the performance audit
showed identical values, coordinates, and attributes across all 60 output
variables. The obsolete land-column wrapper and temporary thread benchmark were
removed; callers use `create_column(..., lh_resistance=...)` directly.

## Parallel parameter sweep

The nine independent mixed-layer-depth/evaporation-resistance cases now use a
spawned process pool with at most eight workers. Native OpenMP/BLAS thread
counts are fixed at one per worker to prevent oversubscription. Each worker
runs one control integration followed by its dependent abrupt-4xCO2
integration; the coupled timestep and WTG calculations within a case are not
reordered or parallelized.

A short 20-level comparison initialized serial and process-pool runs from the
same arrays, integrated two control days and one abrupt-4xCO2 day, and compared
the complete saved control and forcing datasets with
`xarray.testing.assert_identical`. Both datasets were identical. This verifies
the execution-path change; it is not a climate-equilibration test.

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

## WTG operator update

The legacy temperature-difference relaxation has been replaced by the
Shaevitz--Sobel (2004) mean-heating method with `C=1`. Free-tropospheric
temperature is shared, the diabatic-heating anomaly diagnoses opposite omega,
and omega tapers from 850 hPa to zero at the surface. Moisture transport now
uses closed, upwind vertical fluxes and exactly cancelling horizontal fluxes
derived from continuity. The humidity floor, moving cold-point stratospheric
relaxation, and convergence-gated vertical advection were removed.

Unit tests verify temperature projection, opposite omega, thermal-energy
redistribution, uniform-tracer invariance, moisture conservation, positivity,
and CFL subcycling. The standalone results above contain no WTG and therefore
do not validate the coupled DCM climate or its equilibration.

A 60-level, 20-day coupled smoke integration completed without invalid states.
The largest daily absolute equal-area pair-mean WTG energy convergence was
`6.95e-12 W/m2`; the corresponding water convergence was below
`5.3e-20 kg/m2/s`. The maximum pair-mean storage-minus-TOA residual was
`1.73e-4 W/m2`, and the maximum column process-budget residual was
`2.26e-9 W/m2`.

Independent 60-level five-day integrations at 600 s and 300 s produced maximum
final-profile differences of `0.00181 K` in atmospheric temperature and
`2.82e-7 kg/kg` in humidity. Final land and ocean surface-temperature
differences were `+5.44e-4 K` and `-4.00e-5 K`, respectively. These are short
numerical checks, not evidence that the long coupled climate has equilibrated.
