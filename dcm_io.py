"""Dataset construction, NetCDF/JSON persistence, and plotting."""

from importlib import metadata
import json
from pathlib import Path

import numpy as np
import xarray as xr


def package_versions():
    versions = {}
    for name in ("climlab", "climlab-sbm-convection", "climlab-rrtmg", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def save_dataset(ds, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def variable_units(name):
    base = name.removesuffix("_absmax")
    if base in ("Tatm", "Ts", "Tatm_min", "Tatm_max", "Ts_min", "Ts_max"):
        return "K"
    if base in ("q", "q_min", "q_max"):
        return "kg kg-1"
    if base.startswith("RH"):
        return "1"
    if base in ("cape", "cin"):
        return "J kg-1"
    if base == "water_path":
        return "kg m-2"
    if base.startswith("precipitation") or base in ("evaporation", "water_storage", "water_residual", "water_WTG"):
        return "kg m-2 s-1"
    if base in ("surface_energy", "thermal_energy", "latent_energy"):
        return "J m-2"
    if base == "SBM_Tatm_tendency":
        return "K s-1"
    if base == "SBM_q_tendency":
        return "kg kg-1 s-1"
    if base == "omega":
        return "Pa s-1"
    return "W m-2"


def build_daily_dataset(records, lev, attrs=None, columns=None):
    """Records are daily scalars/profiles, optionally with a leading column axis."""
    if not records:
        raise ValueError("No daily records")
    variables = {}
    for name in records[0]:
        values = np.stack([record[name] for record in records])
        dims = ["time"]
        if columns is not None:
            values = np.swapaxes(values, 0, 1)
            dims = ["column", "time"]
        if values.ndim == len(dims) + 1:
            dims.append("lev")
        method = "minimum" if name.endswith("_min") else (
            "maximum" if name.endswith(("_max", "_absmax")) else "mean")
        variables[name] = (dims, values, {"units": variable_units(name), "cell_methods": f"time: {method}"})
    coords = {"time": np.arange(1, len(records) + 1), "lev": lev}
    if columns is not None:
        coords["column"] = columns
    ds = xr.Dataset(variables, coords=coords, attrs=attrs or {})
    ds.attrs["package_versions"] = json.dumps(package_versions(), sort_keys=True)
    ds.time.attrs = {"long_name": "Elapsed model time at end of averaging interval", "units": "days"}
    ds.lev.attrs = {"units": "hPa", "positive": "down"}
    return ds


def mean_rce_dataset(daily, mean_days, summary):
    mean = daily.isel(time=slice(-mean_days, None)).mean("time", keep_attrs=True)
    for name in daily:
        if name.endswith("_min"):
            mean[name] = daily[name].isel(time=slice(-mean_days, None)).min("time", keep_attrs=True)
        elif name.endswith(("_max", "_absmax")):
            mean[name] = daily[name].isel(time=slice(-mean_days, None)).max("time", keep_attrs=True)
    mean.attrs.update(description="Final-window RCE candidate; equilibrium is diagnosed, not assumed.",
                      mean_period_days=mean_days, equilibrium_diagnostics=json.dumps(summary),
                      equilibrated=int(summary["equilibrated"]))
    return mean


def add_control_reference(ds, mean_state):
    for name, units in (("Tatm", "K"), ("q", "kg kg-1"), ("Ts", "K")):
        dims = ("column",) if name == "Ts" else ("column", "lev")
        ds[f"control_{name}"] = (dims, mean_state[name], {
            "long_name": f"Mean control {name} used to initialize forcing run", "units": units,
        })
    return ds


def save_failure(scm, before, path, stage, error):
    """Save the last input and failed output for catchable Python/state errors."""
    variables = {}
    for prefix, state in (("before", before), ("failed", scm.state)):
        for name in ("Tatm", "q", "Ts"):
            value = np.asarray(state[name]).squeeze()
            variables[f"{prefix}_{name}"] = (("lev",) if value.ndim else (), value)
    for procname, proc in scm.subprocess.items():
        for name, value in proc.tendencies.items():
            value = np.asarray(value).squeeze()
            variables[f"tendency_{procname}_{name}"] = (("lev",) if value.ndim else (), value)
    ds = xr.Dataset(variables, coords={"lev": scm.lev}, attrs={
        "stage": stage, "error": str(error), "step": int(scm.time["steps"]),
        "package_versions": json.dumps(package_versions()),
    })
    return save_dataset(ds, path)


def plot_rce_temp_q(Tatm_rce, qatm_rce, lev):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, values, label in zip(axes, (Tatm_rce, qatm_rce * 1e3),
                                 ("Temperature (K)", "Humidity (g kg-1)")):
        ax.plot(values, lev)
        ax.set(xlabel=label, ylabel="Pressure (hPa)")
        ax.invert_yaxis()
        ax.grid()
    fig.tight_layout()
    plt.show()
