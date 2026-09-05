"""State checks and energy/water diagnostics; no filesystem writes."""

import numpy as np
from climlab import constants as const
from climlab.utils.thermo import qsat

FLUX_NAMES = (
    "ASR", "ASRclr", "ASRcld", "OLR", "OLRclr", "OLRcld",
    "SW_sfc", "SW_sfc_clr", "LW_sfc", "LW_sfc_clr", "LHF", "SHF",
)


def timestep_seconds(scm):
    dt = scm.timestep
    return float(dt / np.timedelta64(1, "s")) if isinstance(dt, np.timedelta64) else float(dt)


def diag_scalar(scm, name):
    return float(np.asarray(scm.diagnostics[name]).squeeze())


def snapshot(scm):
    return {name: np.asarray(value).copy() for name, value in scm.state.items()}


def layer_mass(scm):
    return np.diff(scm.Tatm.domain.lev.bounds) * 100. / const.g


def check_state(scm):
    """Reject invalid states before the next radiation call; never clip."""
    for name in ("Tatm", "Ts", "q"):
        value = np.asarray(scm.state[name])
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Nonfinite {name} at model step {scm.time['steps']}")
    if np.any(scm.q < 0.) or np.any(scm.q >= 1.):
        raise ValueError("Specific humidity must satisfy 0 <= q < 1")
    if np.any(scm.Tatm <= 0.) or np.any(scm.Ts <= 0.):
        raise ValueError("Absolute temperature must be positive")


def state_diagnostics(scm):
    rh = np.asarray(scm.q / qsat(scm.Tatm, scm.lev))
    mass = layer_mass(scm)
    return {
        "Tatm": np.asarray(scm.Tatm).copy(), "q": np.asarray(scm.q).copy(),
        "Ts": float(np.asarray(scm.Ts).squeeze()), "RH": rh.copy(),
        "Ts_min": float(np.min(scm.Ts)), "Ts_max": float(np.max(scm.Ts)),
        "Tatm_min": float(np.min(scm.Tatm)), "Tatm_max": float(np.max(scm.Tatm)),
        "q_min": float(np.min(scm.q)), "q_max": float(np.max(scm.q)),
        "RH_min": float(np.min(rh)), "RH_max": float(np.max(rh)),
        "water_path": float(np.sum(mass * scm.q)),
        "surface_energy": float(np.sum(scm.Ts.domain.heat_capacity * scm.Ts)),
        "thermal_energy": float(np.sum(mass * const.cp * scm.Tatm)),
        "latent_energy": float(np.sum(mass * const.Lhvap * scm.q)),
    }


def step_diagnostics(scm, before):
    """Read tendencies already used by step_forward(), without recomputing physics."""
    dt = timestep_seconds(scm)
    mass = layer_mass(scm)
    cs = scm.Ts.domain.heat_capacity
    result = state_diagnostics(scm)
    result.update({key: diag_scalar(scm, key) for key in FLUX_NAMES})
    rad = scm.subprocess["Radiation"]
    result["TOA_imbalance"] = diag_scalar(rad, "ASR") - diag_scalar(rad, "OLR")
    result["surface_storage"] = float(np.sum(cs * (scm.Ts - before["Ts"])) / dt)
    result["thermal_storage"] = float(np.sum(mass * const.cp * (scm.Tatm - before["Tatm"])) / dt)
    result["latent_storage"] = float(np.sum(mass * const.Lhvap * (scm.q - before["q"])) / dt)
    result["energy_storage"] = sum(result[k] for k in ("surface_storage", "thermal_storage", "latent_storage"))
    result["water_storage"] = float(np.sum(mass * (scm.q - before["q"])) / dt)
    for name, proc in scm.subprocess.items():
        tend = proc.tendencies
        energy = np.sum(mass * (const.cp * tend.get("Tatm", 0.) + const.Lhvap * tend.get("q", 0.)))
        energy += np.sum(cs * tend.get("Ts", 0.))
        result[f"energy_{name}"] = float(energy)
    conv = scm.subprocess["Convection"]
    result["cape"] = diag_scalar(conv, "cape")
    result["cin"] = diag_scalar(conv, "cin")
    for name in ("Tatm", "q"):
        result[f"SBM_{name}_tendency"] = np.asarray(conv.tendencies[name]).copy()
    result["precipitation_convective"] = diag_scalar(conv, "precipitation")
    result["precipitation_large_scale"] = diag_scalar(scm.subprocess["Condensation"], "precipitation")
    result["precipitation"] = result["precipitation_convective"] + result["precipitation_large_scale"]
    result["evaporation"] = diag_scalar(scm.subprocess["LHF"], "evaporation")
    result["energy_residual"] = result["energy_storage"] - result["TOA_imbalance"]
    result["budget_residual"] = result["energy_storage"] - sum(result[f"energy_{n}"] for n in scm.subprocess)
    result["water_residual"] = result["water_storage"] - result["evaporation"] + result["precipitation"]
    result["radiation_residual"] = result["energy_Radiation"] - result["TOA_imbalance"]
    # Preserve within-day peaks, including cancellation in signed budget errors.
    for name in ("energy_residual", "budget_residual", "water_residual", "energy_Convection"):
        result[f"{name}_absmax"] = abs(result[name])
    return result


def include_wtg_diagnostics(scm, record, before, omega):
    """Account for legacy transport separately from column-physics closure."""
    dt = timestep_seconds(scm)
    mass = layer_mass(scm)
    dthermal = float(np.sum(mass * const.cp * (scm.Tatm - before["Tatm"])) / dt)
    dwater = float(np.sum(mass * (scm.q - before["q"])) / dt)
    transport = dthermal + const.Lhvap * dwater
    updated = state_diagnostics(scm)
    for name, value in updated.items():
        if name.endswith("_min"):
            record[name] = min(record[name], value)
        elif name.endswith("_max"):
            record[name] = max(record[name], value)
        else:
            record[name] = value
    record["omega"] = omega
    record["energy_WTG"] = transport
    record["water_WTG"] = dwater
    record["thermal_storage"] += dthermal
    record["latent_storage"] += const.Lhvap * dwater
    record["energy_storage"] += transport
    record["water_storage"] += dwater
    # budget_residual and water_residual exclude the separately recorded WTG
    # source. energy_residual is total storage minus TOA, including transport.
    record["energy_residual"] += transport
    record["energy_residual_absmax"] = abs(record["energy_residual"])


class DailyAccumulator:
    """Mean rates/profiles, but retain extrema over every physics timestep."""

    def __init__(self):
        self.values = {}
        self.count = 0

    def add(self, record):
        for name, value in record.items():
            value = np.asarray(value)
            if name not in self.values:
                self.values[name] = value.copy()
            elif name.endswith("_min"):
                self.values[name] = np.minimum(self.values[name], value)
            elif name.endswith(("_max", "_absmax")):
                self.values[name] = np.maximum(self.values[name], value)
            else:
                self.values[name] += value
        self.count += 1

    def mean(self):
        if not self.count:
            raise ValueError("Cannot average an empty day")
        return {k: v.copy() if k.endswith(("_min", "_max", "_absmax")) else v / self.count
                for k, v in self.values.items()}


def equilibrium_summary(ds, mean_days=30, flux_tolerance=0.1, trend_tolerance=0.05):
    """Equilibrium requires negligible storage and trends, not just elapsed time."""
    if mean_days < 2 or ds.sizes["time"] < mean_days:
        raise ValueError("Equilibrium assessment needs the full window of at least two days")
    eq = ds.isel(time=slice(-mean_days, None))
    x = np.asarray(eq.time)
    def trend(name):
        return float(np.polyfit(x, np.asarray(eq[name]), 1)[0] * 365.)
    result = {
        "mean_days": mean_days,
        "Ts_mean_K": float(eq.Ts.mean()),
        "Ts_trend_K_per_year": trend("Ts"),
        "water_trend_kg_m2_per_year": trend("water_path"),
        "TOA_W_m2": float(eq.TOA_imbalance.mean()),
        "max_daily_budget_error_W_m2": float(eq.budget_residual_absmax.max()),
        "max_daily_energy_error_W_m2": float(eq.energy_residual_absmax.max()),
        "max_daily_water_error_kg_m2_s": float(eq.water_residual_absmax.max()),
    }
    for name in ("surface_storage", "thermal_storage", "latent_storage"):
        result[name + "_W_m2"] = float(eq[name].mean())
    result["equilibrated"] = bool(
        abs(result["TOA_W_m2"]) < flux_tolerance
        and all(abs(result[n + "_W_m2"]) < flux_tolerance
                for n in ("surface_storage", "thermal_storage", "latent_storage"))
        and abs(result["Ts_trend_K_per_year"]) < trend_tolerance
        and abs(result["water_trend_kg_m2_per_year"]) < 0.1
        and result["max_daily_energy_error_W_m2"] < flux_tolerance
        and result["max_daily_water_error_kg_m2_s"] < 1e-8
    )
    return result


def diagnose_control_equilibrium(ds, mean_days=365):
    if mean_days < 2 or mean_days > ds.sizes["time"]:
        raise ValueError("Invalid equilibrium averaging window")
    eq = ds.isel(time=slice(-mean_days, None))
    for column in eq.column.values:
        sub = eq.sel(column=column)
        trend = np.polyfit(sub.time, sub.Ts, 1)[0] * 365.
        print(f"{column}: Ts={float(sub.Ts.mean()):.3f} K, trend={trend:+.5f} K/yr, "
              f"TOA={float(sub.TOA_imbalance.mean()):+.4f} W/m2")
    print(f"Domain TOA={float(eq.TOA_imbalance.mean()):+.4f} W/m2; "
          "legacy WTG transport is not energy-conservative.")


def get_mean_control_state(ds_control, mean_days=365):
    if not 1 <= mean_days <= ds_control.sizes["time"]:
        raise ValueError("Invalid control averaging window")
    ds_mean = ds_control.isel(time=slice(-mean_days, None)).mean("time", keep_attrs=True)
    return {k: ds_mean[k].values.copy() for k in ("Tatm", "q", "Ts")}, ds_mean
