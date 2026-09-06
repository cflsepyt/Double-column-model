DCM efficiency audit — 2026-09-06

Reviewed the current working tree of `dcm_physics.py`, `dcm_wtg.py`,
`dcm_diagnostics.py`, `dcm_io.py`, `dcm_rundiag_0817.py`, and the validation
scripts. Existing edits were preserved. The low-risk optimizations measured by
this audit are now implemented in the production source.

The best measured immediate candidates are cheaper finite-value checks and
separate column accumulators. Vectorized WTG donor fluxes preserve tested
outputs but provide little benefit at this resolution. Larger gains
would require work inside climlab's radiation preparation or process dispatch.

A three-day, 60-level profile used 1 m land, 20 m ocean, evaporation resistance
0.6, and unchanged 600 s physics/radiation intervals. It covered 432 coupled
steps and daily dataset construction, excluding initialization and file writes.
Total profiled time was 3.468 s. The profile ran alongside the baseline unit
suite, so percentages are approximate; the separate timing experiment below
ran without that test workload. Cumulative entries overlap and must not be added.

| Path | Cumulative time | Share of total |
| --- | ---: | ---: |
| climlab `step_forward` | 2.543 s | 73.3% |
| RRTMG SW + LW heating methods, including preparation | 1.670 s | 48.2% |
| RRTMG common argument preparation | 0.577 s | 16.6% |
| Local `step_diagnostics` | 0.234 s | 6.7% |
| WTG `advance_dcm_one_step` | 0.182 s | 5.2% |
| Per-field diagnostic finite checks (generator) | 0.166 s | 4.8% |
| Post-WTG diagnostic update | 0.095 s | 2.7% |

Measured prototypes use five repetitions per variant, each integrating five
days from identical fresh profiles. Repetitions alternate 300 and 1200 ppm
CO2 and reverse variant order on alternating repetitions. These are startup
trajectories, not equilibrated control/forcing experiments. Times exclude
initialization, include daily dataset assembly, and do not use cProfile.

| Variant | Median seconds | Elapsed-time reduction |
| --- | ---: | ---: |
| Unchanged baseline | 4.1032 | 0.00% |
| Vectorized donor flux | 4.0953 | 0.19% |
| Scalar-aware finite checks | 4.0462 | 1.39% |
| Separate column accumulators | 4.0632 | 0.98% |
| All three combined | 3.9923 | 2.70% |

The combined change reduced median elapsed time by 2.70% (1.028× speedup).
The vector-only result is too small relative to run-to-run spread to establish
a meaningful full-run speedup. These numbers are workload-specific, not a
forecast for 15,000-day runs or eight concurrent workers.

All prototype daily datasets matched the reference exactly using
`xarray.testing.assert_identical`, including all 60 variables, coordinates,
and attributes. The vectorized flux also matched the original exactly on
400 random inputs covering 3, 20, 60, and 120 levels, nonuniform layer widths,
and both velocity signs. This establishes exact agreement on the tested paths;
it does not prove every long climate trajectory or configuration is unchanged.

Implemented changes and implementation details:

1. **Reuse diagnostic mappings and cheaper finite-value reductions.**
   [dcm_rundiag_0817.py:61](/home/garywu/dcm/dcm_rundiag_0817.py:61) reduces every scalar through
   `np.all`, producing 48,384 generator calls over 864 physics steps in the
   profile. The refactor fetches each diagnostic mapping once and uses the array
   method for finite checks. It keeps every existing check, exception, and
   failure snapshot. This
   changes no tendency, state, accumulation order, or validation threshold.

2. **Vectorize the vertical donor-flux loop.**
   [dcm_wtg.py:111](/home/garywu/dcm/dcm_wtg.py:111) loops over every interior face twice per
   transport substep. The equivalent interior assignment is
   `flux[1:-1] = omega_edge[1:-1] * np.where(omega_edge[1:-1] >= 0., q[:-1], q[1:])`.
   Keep the two zero boundary fluxes, existing flux difference and division,
   horizontal flux, CFL selection, and sequential substeps. This avoids changing
   the transport scheme. The original loop consumed only 0.013 s in this profile;
   benefits may grow with resolution or stronger circulation requiring subcycling.

3. **Avoid stacking 60 fields at every coupled timestep.**
   The coupled loop now maintains one
   `DailyAccumulator` per column and stack their means once per day. This cuts
   timestep field stacks by a factor of 144 at 600 s. Each column retains the
   same chronological addition order, extrema rules, and final output shape.
   Two accumulator calls add Python overhead, so fewer allocations do not
   necessarily translate to a large elapsed-time gain on this small grid.

4. **Cache immutable geometry.**
   [dcm_wtg.py:47](/home/garywu/dcm/dcm_wtg.py:47) reconstructs pressure interfaces three times
   per coupled step; [dcm_diagnostics.py:26](/home/garywu/dcm/dcm_diagnostics.py:26) computes
   layer mass eight times. Together these cost 0.047 s, about 1.35% of the
   profile. Layer mass is now reused within each diagnostic pass. Pressure-grid
   caching remains a possible follow-up; it must not cache evolving stability,
   temperature, humidity, or omega.

5. **Reduce peak output memory in long sweeps.**
   [dcm_io.py:59](/home/garywu/dcm/dcm_io.py:59) retains a dictionary of small arrays for every
   day and then stacks a second copy into the dataset. The profiled schema
   contains 6,624 bytes of numerical data per day: a 15,000-day control holds
   99.36 MB before dictionary/array-object overhead and construction copies.
   This multiplies across workers. Preallocated per-variable daily arrays would
   preserve daily arithmetic and avoid the list-of-dictionaries overhead.
   [dcm_rundiag_0817.py:234](/home/garywu/dcm/dcm_rundiag_0817.py:234) also keeps the complete
   control dataset alive during forcing; release it after saving and extracting
   its mean state. Memory improvement is inferred from data sizes and lifetimes,
   not measured peak RSS. Blocked output is a further option if needed.

6. **Investigate radiation preparation for larger gains.**
   [dcm_physics.py:97](/home/garywu/dcm/dcm_physics.py:97) creates climlab RRTMG objects whose
   installed wrappers rebuild pressure, gas/cloud arrays and interpolation
   objects every call. Common preparation alone accounts for 16.6% of the
   profile, although only part of it is invariant. An explicit cached adapter
   could reuse pressure arrays and fixed interpolation geometry while refreshing
   every evolving input. CO2 caches must refresh on `set_co2`/quadrupling;
   temperature, humidity, surface inputs, and configurable gases/clouds must
   never go stale. This is a higher-complexity follow-up, with no measured
   speedup or demonstrated equivalence yet. Avoid silently modifying the shared
   installed climlab environment.

The coupled loop now passes its pre-physics snapshot into
`checked_physics_step`, removing the duplicate snapshot while preserving the
same failure-file state. All snapshot calls combined cost just 0.0084 s (0.24%)
in the original profile, so this is primarily a cleanup.

Keep physics and radiation cadence, vertical resolution, precision, process
ordering, WTG/CFL rules, conservation checks, and equilibrium criteria unchanged.
The temporary thread benchmark was removed after the audit. Its larger-timestep
and less-frequent-radiation experiments did not meet the requirement of
unchanged results. Independent process-level sweep parallelism already exists.
Land/ocean threading requires native-library thread-safety evidence and separate
speed and identity tests; it is not an established improvement from this audit.

Validation: all 11 existing tests passed on the baseline, and all 11 passed
again with the three prototypes enabled, including WTG positivity/subcycling,
conservation, failure snapshots, asynchronous radiation accounting, and output
checks. `git diff --check` passed. Before production adoption, extend exact
comparisons to longer runs, the full depth/resistance sweep, both physics
timesteps, asynchronous radiation, and control-to-forcing transitions.

Reproduction artifacts are in `/tmp/dcm_efficiency_audit/`: `profile_run.py`,
`profile.txt`, `profile.pstats`, `compare.py`, `compare.txt`, `prototype_tests.py`,
and test logs. They use `/home/garywu/.conda/envs/climlab/bin/python` and import
the current working tree. Temporary artifacts may be removed by system cleanup.
