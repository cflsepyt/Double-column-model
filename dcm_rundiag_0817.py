"""Run SBM RCE and conservative double-column WTG experiments.

Physics, WTG transport, diagnostics, and persistence live in separate modules.

Original framework: Jonathan Lin (jonathanlin@cornell.edu).
Original flux diagnostics: Pei-Tzu Wu.
"""
from pathlib import Path
import warnings

import numpy as np

from dcm_physics import (
    ColumnConfig, create_column, create_land_column, create_sbm_convection,
    create_dcm_pair, apply_abrupt_4xco2,
)
from dcm_diagnostics import (
    DailyAccumulator, check_state, snapshot, step_diagnostics, timestep_seconds,
    equilibrium_summary, diagnose_control_equilibrium, get_mean_control_state,
    diag_scalar, include_wtg_diagnostics,
)
from dcm_io import (
    build_daily_dataset, save_dataset, save_failure, mean_rce_dataset, plot_rce_temp_q,
    add_control_reference,
)
from dcm_wtg import WTGConfig, advance_dcm_one_step


def checked_physics_step(scm, failure_path):
    before = snapshot(scm)
    stage = "before physics"
    try:
        check_state(scm)
        stage = "physics integration"
        scm.step_forward()
        stage = "after physics"
        check_state(scm)
        record = step_diagnostics(scm, before)
        if not all(np.all(np.isfinite(value)) for value in record.values()):
            raise FloatingPointError("Nonfinite physics diagnostic")
        return record
    except Exception as error:
        save_failure(scm, before, failure_path, stage, error)
        raise


def integrate_rce_daily(scm, n_days, failure_path="data/rce_failure.nc", progress=True):
    if not isinstance(n_days, (int, np.integer)) or n_days < 1:
        raise ValueError("n_days must be a positive integer")
    dt = timestep_seconds(scm)
    steps = int(round(86400. / dt))
    if not np.isclose(steps * dt, 86400.):
        raise ValueError("Timestep must divide a day")
    daily = []
    for day in range(n_days):
        acc = DailyAccumulator()
        for _ in range(steps):
            acc.add(checked_physics_step(scm, failure_path))
        daily.append(acc.mean())
        if progress and (day == 0 or (day + 1) % 100 == 0):
            print(f"RCE day {day + 1}: Ts={daily[-1]['Ts']:.3f} K, "
                  f"TOA={daily[-1]['TOA_imbalance']:+.4f} W/m2", flush=True)
    attrs = scm.column_config.attributes()
    attrs.update(water_depth=float(np.asarray(scm.Ts.domain.depth.bounds)[-1]),
                 co2_ppm=float(scm.subprocess["Radiation"].absorber_vmr["CO2"]) * 1e6,
                 description="Daily SBM single-column RCE diagnostics; no WTG")
    return build_daily_dataset(daily, scm.lev, attrs)


def rcm(num_lev=60, n_days=1825, out_nc="data/rce_mean.nc", *, config=None,
        mean_days=30, require_equilibrium=False):
    if n_days < mean_days or mean_days < 2:
        raise ValueError("RCE run must include an averaging window of at least two days")
    scm = create_column(num_lev, water_depth=1., config=config)
    out_nc = Path(out_nc)
    daily = integrate_rce_daily(scm, n_days, out_nc.with_name(out_nc.stem + "_failure.nc"))
    summary = equilibrium_summary(daily, mean_days)
    mean = mean_rce_dataset(daily, mean_days, summary)
    save_dataset(daily, out_nc.with_name(out_nc.stem + "_daily.nc"))
    save_dataset(mean, out_nc)
    print(f"RCE assessment: {summary}")
    if not summary["equilibrated"]:
        message = "Final RCE window has not met the equilibrium criteria; inspect daily diagnostics."
        if require_equilibrium:
            raise RuntimeError(message)
        warnings.warn(message)
    return mean.Tatm.values, mean.q.values, float(mean.Ts), scm.lev.copy(), mean


def integrate_dcm_daily(scm_land, scm_ocean, lev, ndays, land_MLD, ocean_MLD,
                        lh_r, description, print_label="DCM", *, wtg_config=None):
    """Run mean-heating WTG with checked physics and transport accounting."""
    dt = timestep_seconds(scm_land)
    if not np.isclose(dt, timestep_seconds(scm_ocean)):
        raise ValueError("Column timesteps differ")
    steps = int(round(86400. / dt))
    if ndays < 1 or not np.isclose(steps * dt, 86400.):
        raise ValueError("Invalid duration or timestep")
    p = lev * 100.
    wtg_config = wtg_config or WTGConfig()
    models = (scm_land, scm_ocean)
    records = []
    for day in range(ndays):
        acc = DailyAccumulator()
        for _ in range(steps):
            before_physics = [snapshot(scm) for scm in models]
            current = [checked_physics_step(scm, f"data/{print_label}_{label}_failure.nc")
                       for label, scm in zip(("land", "ocean"), models)]
            before_wtg = [snapshot(scm) for scm in models]
            try:
                omegas = advance_dcm_one_step(
                    scm_land, scm_ocean, p, dt,
                    Tatm_before=tuple(state["Tatm"] for state in before_physics),
                    config=wtg_config, advance_physics=False,
                )
                for scm in models:
                    check_state(scm)
            except Exception as error:
                for label, scm, before in zip(("land", "ocean"), models, before_wtg):
                    save_failure(scm, before, f"data/{print_label}_{label}_failure.nc", "WTG", error)
                raise
            for scm, record, before, omega in zip(models, current, before_wtg, omegas):
                include_wtg_diagnostics(scm, record, before, omega)
            pair_wtg_energy = sum(record["energy_WTG"] for record in current)
            pair_wtg_water = sum(record["water_WTG"] for record in current)
            if abs(pair_wtg_energy) > 1.0e-7 or abs(pair_wtg_water) > 1.0e-13:
                raise RuntimeError(
                    "WTG pair conservation failed: "
                    f"energy={pair_wtg_energy:.3e} W/m2, "
                    f"water={pair_wtg_water:.3e} kg/m2/s"
                )
            acc.add({key: np.stack([r[key] for r in current]) for key in current[0]})
        records.append(acc.mean())
        if day == 0 or (day + 1) % 100 == 0:
            print(f"{print_label} day {day + 1}: Ts={records[-1]['Ts']}, "
                  f"TOA={records[-1]['TOA_imbalance']}", flush=True)
    attrs = scm_land.column_config.attributes()
    attrs.update(description=description, land_MLD=float(land_MLD), ocean_MLD=float(ocean_MLD),
                 lh_resistance=float(lh_r),
                 wtg_status="Conservative mean-heating and flux-form moisture transport",
                 co2_ppm=float(scm_land.subprocess["Radiation"].absorber_vmr["CO2"]) * 1e6)
    attrs.update(wtg_config.attributes())
    ds = build_daily_dataset(records, lev, attrs, columns=["land", "ocean"])
    return ds.assign_coords(water_depth=("column", [land_MLD, ocean_MLD], {"units": "m"}))


def run_dcm_spinup(
    num_lev, Tatm_rce, qatm_rce, Ts_rce, lev, spinup_days=15000,
    restart_mean_days=365, land_MLD=1, ocean_MLD=100, lh_r=1,
):
    """Run the 300 ppm DCM control and return its final-window mean state."""
    land, ocean = create_dcm_pair(
        num_lev, np.stack([Tatm_rce, Tatm_rce]), np.stack([qatm_rce, qatm_rce]),
        np.array([Ts_rce, Ts_rce]), land_MLD, ocean_MLD, lh_r,
    )
    ds = integrate_dcm_daily(
        land, ocean, lev, spinup_days, land_MLD, ocean_MLD, lh_r,
        description="Daily-mean coupled double-column control spin-up at 300 ppm CO2.",
        print_label="CONTROL",
    )
    ds.attrs.update(
        experiment="DCM control spin-up", CO2="300 ppm", spinup_days=int(spinup_days),
        equilibrium_mean_days=int(restart_mean_days),
    )
    diagnose_control_equilibrium(ds, mean_days=restart_mean_days)
    mean_state, ds_mean = get_mean_control_state(ds, mean_days=restart_mean_days)
    return ds, mean_state, ds_mean


def run_abrupt4xco2(
    num_lev, mean_state, lev, forced_days=15000, land_MLD=1,
    ocean_MLD=100, lh_r=1, restart_mean_days=365,
):
    """Initialize from the mean control state and change CO2 to 1200 ppm."""
    land, ocean = create_dcm_pair(
        num_lev, mean_state["Tatm"], mean_state["q"], mean_state["Ts"],
        land_MLD, ocean_MLD, lh_r,
    )
    print("Applying abrupt 4xCO2 to both columns: 300 ppm -> 1200 ppm")
    apply_abrupt_4xco2(land)
    apply_abrupt_4xco2(ocean)
    ds = integrate_dcm_daily(
        land, ocean, lev, forced_days, land_MLD, ocean_MLD, lh_r,
        description="Daily-mean abrupt-4xCO2 DCM initialized from mean control state.",
        print_label="4xCO2",
    )
    add_control_reference(ds, mean_state)
    ds.attrs.update(
        experiment="abrupt 4xCO2", control_CO2="300 ppm", forced_CO2="1200 ppm",
        forcing="instantaneous quadrupling of CO2", forcing_applied_to="land and ocean columns",
        forced_days=int(forced_days),
        initial_state=f"Mean of final {restart_mean_days} days of 300-ppm control DCM",
    )
    return ds


def main():
    num_lev, land_MLD = 60, 1
    ocean_MLD_list = [100, 60, 20]
    lh_resistance_list = [0.6, 0.8, 1]
    spinup_days, restart_mean_days, forced_days = 15000, 365, 5000

    Tatm_rce, qatm_rce, Ts_rce, lev, ds_rce = rcm(
        num_lev=num_lev, out_nc="data/rce_mean.nc", require_equilibrium=True,
    )
    print(ds_rce)
    plot_rce_temp_q(Tatm_rce, qatm_rce, lev)
    for ocean_MLD in ocean_MLD_list:
        for resistance in lh_resistance_list:
            label = f"r{resistance}_land{land_MLD}_ocean{ocean_MLD}"
            ds_control, mean_state, _ = run_dcm_spinup(
                num_lev, Tatm_rce, qatm_rce, Ts_rce, lev,
                spinup_days=spinup_days, restart_mean_days=restart_mean_days,
                land_MLD=land_MLD, ocean_MLD=ocean_MLD, lh_r=resistance,
            )
            control_path = save_dataset(ds_control, f"0817data/dcm_control_spinup_{label}.nc")
            print(f"Saved control: {control_path}")
            print(f"Mean control Ts: land={mean_state['Ts'][0]:.3f}, "
                  f"ocean={mean_state['Ts'][1]:.3f} K")
            ds_forcing = run_abrupt4xco2(
                num_lev, mean_state, lev, forced_days=forced_days,
                land_MLD=land_MLD, ocean_MLD=ocean_MLD, lh_r=resistance,
                restart_mean_days=restart_mean_days,
            )
            forcing_path = save_dataset(ds_forcing, f"0817data/dcm_abrupt4xco2_{label}.nc")
            print(f"Saved forcing: {forcing_path}")


if __name__ == "__main__":
    main()
