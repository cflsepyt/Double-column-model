# SBM double-column model

See [VALIDATION.md](VALIDATION.md) for the completed tests and integrations.
Budget closure and timestep agreement pass, but the tested five-year runs are
still warming; they have not established RCE.

Activate the existing environment before running:

```bash
conda activate climlab
python -m unittest -v test_sbm
python validate_sbm.py --days 1825 --mean-days 30
```

`validate_sbm.py` runs independent 600 s and 300 s single columns with identical
initial states and no WTG. It writes daily NetCDF files, equilibrium summaries,
and `comparison.json` under `data/sbm_validation/`. Exit status 0 means the stated
budget, equilibration, and timestep-comparison criteria passed; status 2 means
the model is not yet validated. A short `--days 10 --mean-days 5` run is useful
as a smoke check but should not be expected to reach equilibrium. Each call
starts fresh; it does not resume an earlier validation run.

## Modules

| File | Responsibility |
| --- | --- |
| `dcm_physics.py` | SBM, radiation, surface fluxes, condensation, initialization, CO2 |
| `dcm_diagnostics.py` | State checks, storage/flux budgets, extrema, equilibrium criteria |
| `dcm_io.py` | Dataset construction, NetCDF/JSON writes, failure snapshots, plotting |
| `dcm_wtg.py` | Shaevitz--Sobel mean-heating WTG and conservative moisture transport |
| `dcm_rundiag_0817.py` | Integration and original control/abrupt-4xCO2 workflow |
| `validate_sbm.py` | Standalone timestep-comparison command |

The original `rcm()` return tuple and constructor names remain accessible through
`dcm_rundiag_0817`. `rcm()` now saves both the requested mean file and a sibling
`*_daily.nc` containing time-resolved diagnostics. The mean is an RCE candidate,
with an explicit `equilibrated` attribute, rather than assumed equilibrium.
The full experiment entry point requires the RCE checks to pass before continuing:

```bash
python dcm_rundiag_0817.py
```

That command retains the original large parameter sweep and interactive profile
plot. It is **not** the short verification command. Radiation now runs every
physics step by default, so the full sweep is more expensive than before.

## Column changes

- The initial profile decreases from 295 K at the lowest air level to 213 K at
  200 hPa and is isothermal above. Initial surface temperature is 297 K.
- Tropospheric RH starts at 0.65; stratospheric specific humidity is capped at
  5e-6 kg/kg and remains subsaturated. This initializes humidity only once.
- CO2 is set through `absorber_vmr` and checked in both LW/SW components. The
  control is 300 ppm and quadrupling produces 1200 ppm.
- Radiation receives temperature state variables and the shared prognostic
  humidity array. Surface exchange uses `Cd`, not an unused `Ck` keyword.
- SBM uses `tau_bm=7200`, `rhbm=0.8`, and the documented default branch options.
- Large-scale condensation uses `RH_ref=1.0` and a 14400 s relaxation time.
  Finite-time relaxation can leave transient supersaturation. It does not
  prescribe RH in subsaturated layers or clip humidity.
- `ColumnCondensation` preserves climlab 0.9.2's moisture/heating equations but
  fixes its `precipitation[:, 0]` assignment for scalar-column surface fields.

`ColumnConfig` exposes timesteps and relaxation/initialization parameters. For
example, `create_column(60, 1., config=ColumnConfig(physics_dt=300., radiation_dt=300.))`.
The physics interval must divide both a day and the radiation interval.
Existing insolation, albedo, exchange coefficients, and slab-depth choices are
retained for controlled comparison. No empirical SBM energy correction is added.

## Budgets and output interpretation

Layer masses are `100 * diff(pressure_bounds_hPa) / g`. Energy is diagnosed as
slab heat content plus the mass integral of `cp * Tatm + Lhvap * q`.
Every actual `step_forward()` call is checked before and after, and its existing
subprocess tendencies are read without a second physics evaluation.

- `surface_storage`, `thermal_storage`, `latent_storage`: actual component
  energy changes per second, in W/m2.
- `energy_<process>`: tendency-integrated energy input for each process.
- `energy_residual`: total storage minus the applied TOA input.
- `budget_residual`: storage minus all recorded process inputs.
- `radiation_residual`: integrated radiative heating minus ASR - OLR.
- `water_residual`: water storage minus evaporation plus total precipitation;
  in DCM output the separately recorded `water_WTG` input is also subtracted.
- `thermal_WTG`, `latent_WTG`, and `energy_WTG`: transport convergence in each
  column. Their equal-area pair means vanish to numerical precision.
- `precipitation_convective` and `precipitation_large_scale`: separate rates;
  `precipitation` is their sum, all in kg/m2/s (multiply by 86400 for mm/day).
- `SBM_Tatm_tendency`, `SBM_q_tendency`, `cape`, `cin`, and `RH`: daily means.
- `*_min`, `*_max`, and `*_absmax`: extrema over physics timesteps, not averages
  of signed errors. State means are averages of post-step states; fluxes are
  those applied during each corresponding step.

The source-forcing bookkeeping residual should be near numerical roundoff.
SBM's compiled calculations can leave somewhat larger energy/water residuals;
these are measured and not suppressed. Equilibrium requires mean TOA input and
each storage term below 0.1 W/m2, a surface trend below 0.05 K/year, and a water
path trend below 0.1 kg/m2/year, plus numerical budget checks. Timestep-comparison
tolerances are reported explicitly in `comparison.json`.

Catchable exceptions and invalid states write a failure NetCDF with the last
input, failed output, available tendencies, stage, step number, and versions.
Negative humidity, nonfinite diagnostics/states, and nonpositive absolute
temperatures are rejected, not silently repaired by the column integrator.
A native Fortran abort/segmentation fault cannot be caught this way. The code
does not guarantee radiative accuracy outside RRTMG's tabulated regime merely
because temperatures are finite. Failure files are diagnostic snapshots, not
complete restart files.

## WTG coupling

The double-column operator follows Shaevitz and Sobel's mean-heating method with
the artificial heat-capacity factor fixed at `C=1`. Between the configured
100 hPa WTG top and the 850 hPa PBL top, the columns share one temperature:
their mean diabatic heating changes that state, and the heating anomaly diagnoses
equal-and-opposite omega. Omega tapers linearly from its 850 hPa value to zero
at the surface.

Moisture uses closed-boundary, donor-cell vertical fluxes and one exactly
cancelling horizontal intercolumn flux derived from continuity. CFL subcycling
preserves positivity without a humidity floor. The former moving cold-point
relaxation and stratospheric moisture source have been removed. PBL temperature
remains controlled by local column physics because this reduced model does not
contain the mechanical-energy closure needed for an additional PBL pressure-work
term.

Passing the standalone SBM validation still does **not** by itself establish the
scientific validity or equilibration of the coupled DCM.

## References

- [climlab SBM API](https://climlab.readthedocs.io/en/stable/api/climlab.convection.SimplifiedBettsMiller.html)
- [SBM implementation](https://github.com/climlab/climlab-sbm-convection/blob/main/climlab_sbm_convection/climlab_betts_miller.f90)
- [climlab condensation implementation](https://climlab.readthedocs.io/en/latest/_modules/climlab/dynamics/large_scale_condensation.html)
- [climlab radiation configuration](https://climlab.readthedocs.io/en/latest/_modules/climlab/radiation/radiation.html)
- [Shaevitz and Sobel (2004), full-vertical-structure WTG](https://doi.org/10.1175/1520-0493(2004)132%3C0662:ITWTGA%3E2.0.CO;2)
