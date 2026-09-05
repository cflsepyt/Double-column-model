"""Standalone, identical-initial-state SBM timestep comparison (no WTG)."""

import argparse
from pathlib import Path
import numpy as np

from dcm_physics import ColumnConfig, create_column
from dcm_rundiag_0817 import integrate_rce_daily
from dcm_diagnostics import equilibrium_summary
from dcm_io import save_dataset, save_json


def run_single(dt, days, mean_days, num_lev, output_dir):
    """Run one member independently; useful for concurrent long validations."""
    if not 2 <= mean_days <= days:
        raise ValueError("Require 2 <= mean_days <= days")
    output_dir = Path(output_dir)
    config = ColumnConfig(physics_dt=dt, radiation_dt=dt)
    model = create_column(num_lev, water_depth=1., config=config)
    label = f"dt{int(dt)}"
    print(f"Validating {label}, {days} days, {num_lev} levels", flush=True)
    ds = integrate_rce_daily(model, days, output_dir / f"{label}_failure.nc")
    summary = equilibrium_summary(ds, mean_days)
    save_dataset(ds, output_dir / f"{label}_daily.nc")
    save_json(summary, output_dir / f"{label}_summary.json")
    print(summary, flush=True)
    return ds, summary


def validate(days=1825, mean_days=30, num_lev=60, output_dir="data/sbm_validation"):
    if not 2 <= mean_days <= days:
        raise ValueError("Require 2 <= mean_days <= days")
    output_dir = Path(output_dir)
    runs, summaries = [], {}
    initial = None
    for dt in (600., 300.):
        config = ColumnConfig(physics_dt=dt, radiation_dt=dt)
        model = create_column(num_lev, water_depth=1., config=config)
        state = {k: np.asarray(v).copy() for k, v in model.state.items()}
        if initial is None:
            initial = state
        else:
            for key in initial:
                np.testing.assert_array_equal(state[key], initial[key])
        ds, summary = run_single(dt, days, mean_days, num_lev, output_dir)
        summaries[f"dt{int(dt)}"] = summary
        runs.append(ds)
    return compare_runs(runs, summaries, mean_days, output_dir)


def compare_runs(runs, summaries, mean_days, output_dir):
    output_dir = Path(output_dir)
    means = [ds.isel(time=slice(-mean_days, None)).mean("time") for ds in runs]
    differences = {
        "Ts_K": abs(float(means[0].Ts - means[1].Ts)),
        "Tatm_max_K": float(abs(means[0].Tatm - means[1].Tatm).max()),
        "q_max_kg_kg": float(abs(means[0].q - means[1].q).max()),
        "water_path_kg_m2": abs(float(means[0].water_path - means[1].water_path)),
        "TOA_W_m2": abs(float(means[0].TOA_imbalance - means[1].TOA_imbalance)),
    }
    # Scientific tolerances, distinct from per-step numerical closure limits.
    tolerances = dict(Ts_K=0.2, Tatm_max_K=0.2, q_max_kg_kg=1e-4,
                      water_path_kg_m2=0.2, TOA_W_m2=0.1)
    converged = all(differences[k] < v for k, v in tolerances.items())
    budget_ok = all(float(ds.energy_residual_absmax.max()) < 0.1
                    and float(ds.water_residual_absmax.max()) < 1e-8 for ds in runs)
    equilibrated = all(s["equilibrated"] for s in summaries.values())
    passed = budget_ok and equilibrated and converged
    report = dict(runs=summaries, differences=differences, tolerances=tolerances,
                  budget_ok=budget_ok, timestep_converged=converged,
                  equilibrated=equilibrated, passed=passed,
                  status="passed" if passed else "not validated; inspect budgets, convergence, and equilibration")
    save_json(report, output_dir / "comparison.json")
    print(report, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--mean-days", type=int, default=30)
    parser.add_argument("--num-lev", type=int, default=60)
    parser.add_argument("--output-dir", default="data/sbm_validation")
    parser.add_argument("--timestep", type=float, choices=(300., 600.),
                        help="Run just one comparison member")
    args = parser.parse_args()
    if args.timestep is not None:
        _, summary = run_single(args.timestep, args.days, args.mean_days, args.num_lev, args.output_dir)
        raise SystemExit(0 if summary["equilibrated"] else 2)
    report = validate(args.days, args.mean_days, args.num_lev, args.output_dir)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
