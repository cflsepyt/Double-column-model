"""Legacy WTG physics, isolated without repairing its known transport defects.

The moisture sign error, energy nonconservation, relaxation, and floor remain
pending separate circulation-model work (outside SBM fixes 1--4).
"""
import numpy as np
from climlab import constants as const


def find_tropopause(T, p):
    """
    Tropopause = cold point (temperature minimum), searching from surface upward.
    Returns index into lev array (lev[0]=TOA, lev[-1]=surface).
    """
    mask = (p >= 5000.0) & (p <= 25000.0)
    idx_range = np.where(mask)[0]
    idx_min = idx_range[np.argmin(T[idx_range])]
    return idx_min


def wtg_moisture_advection(omega, q_column, q_other, T_column, p, dt, tau_strat=10.0 * 86400.0):
    """
    Moisture tendency associated with the WTG circulation.
    Positive omega denotes subsidence.
    """
    nlev = len(p)
    dq = np.zeros(nlev)
    idx_tp = find_tropopause(T_column, p)
    p_tp = p[idx_tp]
    q_tp = q_column[idx_tp]
    above_tp = p < p_tp
    for k in range(1, nlev - 1):
        fradv = 0.0
        dwdp = max(omega[k - 1] - omega[k + 1], 0.0) / (p[k - 1] - p[k + 1])
        if dwdp > 0.0:
            vvp = max(omega[k + 1], 0.0) * dt
            vvm = min(omega[k], 0.0) * dt
            fradv = -vvp * (q_column[k] - q_column[k + 1]) / (p[k] - p[k + 1]) - vvm * (q_column[k - 1] - q_column[k]) / (p[k - 1] - p[k])
            fradv += dwdp * (q_other[k] - q_column[k]) * dt
        if above_tp[k]:
            fradv = (q_tp - q_column[k]) * dt / tau_strat
        dq[k] = fradv
    return (dq, idx_tp, q_tp)


def advance_dcm_one_step(scm_land, scm_ocean, p, above_850, tau_wtg, dt, *, advance_physics=True):
    """
    Advance the mass-conserving two-column model by one timestep.

    Equal-area mass conservation:

        omega_land + omega_ocean = 0

    WTG relaxation:

        d(T_land - T_ocean)/dt
            = -(T_land - T_ocean)/tau_wtg
    """
    if advance_physics:
        scm_land.step_forward()
        scm_ocean.step_forward()
    T_land = np.asarray(scm_land.state['Tatm']).squeeze().copy()
    T_ocean = np.asarray(scm_ocean.state['Tatm']).squeeze().copy()
    q_land = np.asarray(scm_land.state['q']).squeeze().copy()
    q_ocean = np.asarray(scm_ocean.state['q']).squeeze().copy()
    Rd = const.Rd
    cp = const.cp
    dTdp_land = np.gradient(T_land, p)
    dTdp_ocean = np.gradient(T_ocean, p)
    sigma_land = -dTdp_land + Rd * T_land / (cp * p)
    sigma_ocean = -dTdp_ocean + Rd * T_ocean / (cp * p)
    sigma_floor = 1e-07
    sigma_land = np.maximum(sigma_land, sigma_floor)
    sigma_ocean = np.maximum(sigma_ocean, sigma_floor)
    dT = T_land - T_ocean
    omega_ocean = dT / (tau_wtg * (sigma_land + sigma_ocean))
    omega_land = -omega_ocean
    idx_tp_land = find_tropopause(T_land, p)
    idx_tp_ocean = find_tropopause(T_ocean, p)
    p_tp_land = p[idx_tp_land]
    p_tp_ocean = p[idx_tp_ocean]
    p_tp_shared = max(p_tp_land, p_tp_ocean)
    wtg_mask = (p < 85000.0) & (p >= p_tp_shared)
    omega_land[~wtg_mask] = 0.0
    omega_ocean[~wtg_mask] = 0.0
    mass_residual = omega_land + omega_ocean
    if not np.allclose(mass_residual, 0.0, atol=1e-14, rtol=0.0):
        raise RuntimeError(f'Two-column WTG mass conservation failed: max residual = {np.max(np.abs(mass_residual)):.3e} Pa s-1')
    dTdt_land_wtg = sigma_land * omega_land
    dTdt_ocean_wtg = sigma_ocean * omega_ocean
    scm_land.state['Tatm'][wtg_mask] += dTdt_land_wtg[wtg_mask] * dt
    scm_ocean.state['Tatm'][wtg_mask] += dTdt_ocean_wtg[wtg_mask] * dt
    (dq_land, _, _) = wtg_moisture_advection(omega=omega_land, q_column=q_land, q_other=q_ocean, T_column=T_land, p=p, dt=dt)
    (dq_ocean, _, _) = wtg_moisture_advection(omega=omega_ocean, q_column=q_ocean, q_other=q_land, T_column=T_ocean, p=p, dt=dt)
    scm_land.state['q'][above_850] += dq_land[above_850]
    scm_ocean.state['q'][above_850] += dq_ocean[above_850]
    scm_land.state['q'][:] = np.maximum(scm_land.state['q'], 1e-08)
    scm_ocean.state['q'][:] = np.maximum(scm_ocean.state['q'], 1e-08)
    return (omega_land, omega_ocean)

