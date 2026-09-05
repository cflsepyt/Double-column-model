# Double Column Model Framework
# 2026-06-25 Add TOA flux and surface fluxes diagnosis (Pei-Tzu Wu)
# Modified:
#   1. Mass-conserving two-column WTG (omega_ocean + omega_land = 0)
#   2. Separate 300 ppm DCM spin-up and abrupt-4xCO2 experiment
#   3. Initialize abrupt-4xCO2 experiment from mean equilibrium state
#   4. Store daily means directly to reduce memory usage
#   5. Use Simplified Betts-Miller convection with explicit relaxation settings
#
# Author: Jonathan Lin (jonathanlin@cornell.edu)

import climlab
import matplotlib.pyplot as plt
import numpy as np
from climlab import constants as const
from climlab.solar.insolation import daily_insolation
import xarray as xr

from pathlib import Path

# ======================================================================
# Basic model construction
# ======================================================================

# Synchronous 10-minute baseline for diagnosing SBM/radiation stability.
# Increase the radiation interval only after timestep-convergence checks.
PHYSICS_TIMESTEP = 10 * const.seconds_per_minute
RADIATION_TIMESTEP = PHYSICS_TIMESTEP
SBM_RELAXATION_TIME = 2 * const.seconds_per_hour
SBM_REFERENCE_RH = 0.8  # Convective reference; actual humidity remains prognostic.


def create_sbm_convection(state):
    """Use the same SBM settings in the RCE, land, and ocean columns."""
    return climlab.convection.SimplifiedBettsMiller(
        name="Convection",
        state=state,
        timestep=PHYSICS_TIMESTEP,
        tau_bm=SBM_RELAXATION_TIME,
        rhbm=SBM_REFERENCE_RH,
        do_simp=False,
        do_shallower=True,
        do_changeqref=True,
        do_envsat=True,
        do_taucape=False,
    )


def create_column(
    num_lev,
    water_depth,
    co2_forcing=False,
):
    """
    Build a single-column radiative-convective model.

    Physics:
      - RRTMG clear-sky radiation
      - Simplified Betts-Miller convection with prognostic humidity
      - Interactive sensible heat flux
      - Interactive latent heat flux

    Parameters
    ----------
    num_lev : int
        Number of atmospheric pressure levels.

    water_depth : float
        Slab-ocean mixed-layer depth (m).

    co2_forcing : bool
        If True, immediately quadruple CO2 from 300 to 1200 ppm.

    Returns
    -------
    rcm : climlab Process
        Single-column radiative-convective model.
    """

    # --------------------------------------------------------------
    # State
    # --------------------------------------------------------------
    state = climlab.column_state(
        num_lev=num_lev,
        water_depth=water_depth,
    )

    lev = state["Tatm"].domain.axes["lev"].points

    # Initial humidity profile
    q0 = 10e-3 * (lev / lev[-1])**3

    state["q"] = climlab.Field(
        q0,
        domain=state.Tatm.domain,
    )

    # Initial atmospheric temperature
    state["Tatm"] = climlab.Field(
        280.,
        domain=state.Tatm.domain,
    )

    # --------------------------------------------------------------
    # Annual-mean insolation
    # --------------------------------------------------------------
    S0 = 1360.
    lat = 25.

    Q_annual = np.mean([daily_insolation(lat=lat,day=d,S0=S0,) for d in range(365)])

    # --------------------------------------------------------------
    # Radiation
    # --------------------------------------------------------------
    rad = climlab.radiation.RRTMG(
        name="Radiation",
        state=state,
        specific_humidity=state["q"],
        CO2=300.,
        albedo=0.33,
        insolation=Q_annual,
        coszen=0.5,
        icld=0,
        timestep=RADIATION_TIMESTEP,
    )

    if co2_forcing:
        rad.absorber_vmr["CO2"] *= 4.

    # --------------------------------------------------------------
    # Convection
    # --------------------------------------------------------------
    conv = create_sbm_convection(state)

    # --------------------------------------------------------------
    # Surface sensible heat flux
    # --------------------------------------------------------------
    shf = climlab.surface.SensibleHeatFlux(
        name="SHF",
        state={
            "Tatm": state.Tatm,
            "Ts": state.Ts,
        },
        Cd=1.e-3,
        timestep=PHYSICS_TIMESTEP,
    )

    # --------------------------------------------------------------
    # Surface latent heat flux
    # --------------------------------------------------------------
    lhf = climlab.surface.LatentHeatFlux(
        name="LHF",
        state=state,
        Cd=1.e-3,
        Ck=1.e-3,
        timestep=PHYSICS_TIMESTEP,
    )

    rcm = climlab.couple([rad, conv, lhf, shf], name="Tropical RCE",)
    return rcm


def create_land_column(
    num_lev,
    Tatm_init,
    qatm_init,
    water_depth,
    lh_resistance,
    co2_forcing=False,
):
    """
    Build the land column with Simplified Betts-Miller convection.

    The principal difference from create_column() is the latent heat
    flux resistance.

    CO2 is 300 ppm initially unless co2_forcing=True.
    """

    state = climlab.column_state(num_lev=num_lev, water_depth=water_depth)
    state["q"] = climlab.Field(qatm_init, domain=state.Tatm.domain)
    state["Tatm"] = climlab.Field(Tatm_init, domain=state.Tatm.domain)

    # Annual-mean insolation
    S0 = 1360.
    lat = 25.
    Q_annual = np.mean([daily_insolation(lat=lat,day=d,S0=S0,) for d in range(365)])
    
    # Radiation
    rad = climlab.radiation.RRTMG(
        name="Radiation",
        state=state,
        specific_humidity=state["q"],
        CO2=300.,
        albedo=0.33,
        insolation=Q_annual,
        coszen=0.5,
        icld=0,
        timestep=RADIATION_TIMESTEP,
    )

    if co2_forcing:
        rad.absorber_vmr["CO2"] *= 4.

    
    # Convection
    conv = create_sbm_convection(state)

    # Surface sensible heat flux
    shf = climlab.surface.SensibleHeatFlux(
        name="SHF",
        state={
            "Tatm": state.Tatm,
            "Ts": state.Ts,
        },
        Cd=1.e-3,
        timestep=PHYSICS_TIMESTEP,
    )

    # Surface latent heat flux with land resistance
    lhf = climlab.surface.LatentHeatFlux(
        name="LHF",
        state=state,
        Cd=1.e-3,
        Ck=1.e-3,
        resistance=lh_resistance,
        timestep=PHYSICS_TIMESTEP,
    )

    rcm = climlab.couple(
        [rad, conv, lhf, shf],
        name="Tropical RCE",
    )

    return rcm


# ======================================================================
# Diagnostics
# ======================================================================

def diag_scalar(scm, var_name):
    """Read a scalar climlab diagnostic."""
    return float(np.asarray(scm.diagnostics[var_name]).squeeze())


def apply_abrupt_4xco2(scm):
    """
    Instantaneously quadruple CO2 in an existing column.

    Should only be called once per model object.
    """

    rad = scm.subprocess["Radiation"]
    rad.absorber_vmr["CO2"] *= 4.


# ======================================================================
# Single-column RCE
# ======================================================================

def rcm(
    num_lev=60,
    n_days=1825,
    out_nc="data/rce_mean.nc",
):
    scm = create_column(num_lev, water_depth=1.)


    # Output fields to store the time evolution of temperature, 
    # humidity, surface temperature, and energy imbalance
    Tatm_time      = np.empty((n_days, num_lev))
    q_time         = np.empty((n_days, num_lev))
    Ts_time        = np.empty((n_days))
    imbalance_time = np.empty((n_days))

    # TOA diagnostics
    ASR_time = np.full(n_days, np.nan)
    ASRclr_time = np.full(n_days, np.nan)
    ASRcld_time = np.full(n_days, np.nan)
    OLR_time = np.full(n_days, np.nan)
    OLRclr_time = np.full(n_days, np.nan)
    OLRcld_time = np.full(n_days, np.nan)

    # Surface diagnostics
    SW_sfc_time = np.full(n_days, np.nan)
    SW_sfc_clr_time = np.full(n_days, np.nan)
    LW_sfc_time = np.full(n_days, np.nan)
    LW_sfc_clr_time = np.full(n_days, np.nan)
    LHF_time = np.full(n_days, np.nan)
    SHF_time = np.full(n_days, np.nan)

    
    for i in range(n_days):
        scm.integrate_days(1, verbose=False)

        Tatm_time[i, :] = np.asarray(scm.Tatm).squeeze()
        q_time[i, :]    = np.asarray(scm.q).squeeze()
        Ts_time[i]      = np.asarray(scm.Ts).squeeze()

        ASR_time[i]    = diag_scalar(scm, "ASR")
        ASRclr_time[i] = diag_scalar(scm, "ASRclr")
        ASRcld_time[i] = diag_scalar(scm, "ASRcld")
        OLR_time[i]    = diag_scalar(scm, "OLR")
        OLRclr_time[i] = diag_scalar(scm, "OLRclr")
        OLRcld_time[i] = diag_scalar(scm, "OLRcld")

        SW_sfc_time[i]     = diag_scalar(scm, "SW_sfc")
        SW_sfc_clr_time[i] = diag_scalar(scm, "SW_sfc_clr")
        LW_sfc_time[i]     = diag_scalar(scm, "LW_sfc")
        LW_sfc_clr_time[i] = diag_scalar(scm, "LW_sfc_clr")
        LHF_time[i]        = diag_scalar(scm, "LHF")
        SHF_time[i]        = diag_scalar(scm, "SHF")

        imbalance_time[i] = ASR_time[i] - OLR_time[i]

        if (i % 100) == 0:
            print("RCE day %d, Ts: %.2f K" % (i + 1, Ts_time[i]))


    lev = scm.state["Tatm"].domain.axes["lev"].points

    # Last 30 days mean = RCE state
    sl = slice(-30, None)

    Tatm_rce      = np.nanmean(Tatm_time[sl, :], axis=0)
    qatm_rce      = np.nanmean(q_time[sl, :], axis=0)
    Ts_rce        = np.nanmean(Ts_time[sl])
    imbalance_rce = np.nanmean(imbalance_time[sl])

    print("TOA imbalance in RCE state: %.2f W/m²" % imbalance_rce)

    # Output mean RCE state and diagnostics to xarray Dataset
    ds_rce = xr.Dataset(
        data_vars={
            "Tatm": (
                ("lev",),
                Tatm_rce,
                {"long_name": "RCE mean atmospheric temperature", "units": "K"},
            ),
            "q": (
                ("lev",),
                qatm_rce,
                {"long_name": "RCE mean specific humidity", "units": "kg kg-1"},
            ),
            "Ts": (
                (),
                Ts_rce,
                {"long_name": "RCE mean surface temperature", "units": "K"},
            ),
            "TOA_imbalance": (
                (),
                imbalance_rce,
                {"long_name": "RCE mean TOA energy imbalance, ASR - OLR", "units": "W m-2"},
            ),

            "ASR": (
                (),
                np.nanmean(ASR_time[sl]),
                {"long_name": "RCE mean absorbed shortwave radiation at TOA", "units": "W m-2"},
            ),
            "ASRclr": (
                (),
                np.nanmean(ASRclr_time[sl]),
                {"long_name": "RCE mean clear-sky absorbed shortwave radiation at TOA", "units": "W m-2"},
            ),
            "ASRcld": (
                (),
                np.nanmean(ASRcld_time[sl]),
                {"long_name": "RCE mean cloud shortwave radiative effect at TOA", "units": "W m-2"},
            ),
            "OLR": (
                (),
                np.nanmean(OLR_time[sl]),
                {"long_name": "RCE mean outgoing longwave radiation at TOA", "units": "W m-2"},
            ),
            "OLRclr": (
                (),
                np.nanmean(OLRclr_time[sl]),
                {"long_name": "RCE mean clear-sky outgoing longwave radiation at TOA", "units": "W m-2"},
            ),
            "OLRcld": (
                (),
                np.nanmean(OLRcld_time[sl]),
                {"long_name": "RCE mean cloud longwave radiative effect at TOA", "units": "W m-2"},
            ),

            "SW_sfc": (
                (),
                np.nanmean(SW_sfc_time[sl]),
                {"long_name": "RCE mean net shortwave radiation at surface", "units": "W m-2"},
            ),
            "SW_sfc_clr": (
                (),
                np.nanmean(SW_sfc_clr_time[sl]),
                {"long_name": "RCE mean clear-sky net shortwave radiation at surface", "units": "W m-2"},
            ),
            "LW_sfc": (
                (),
                np.nanmean(LW_sfc_time[sl]),
                {"long_name": "RCE mean net longwave radiation at surface", "units": "W m-2"},
            ),
            "LW_sfc_clr": (
                (),
                np.nanmean(LW_sfc_clr_time[sl]),
                {"long_name": "RCE mean clear-sky net longwave radiation at surface", "units": "W m-2"},
            ),
            "LHF": (
                (),
                np.nanmean(LHF_time[sl]),
                {"long_name": "RCE mean surface latent heat flux", "units": "W m-2"},
            ),
            "SHF": (
                (),
                np.nanmean(SHF_time[sl]),
                {"long_name": "RCE mean surface sensible heat flux", "units": "W m-2"},
            ),
        },
        coords={
            "lev": lev,
        },
        attrs={
            "description": "Last 30-day mean RCE state and diagnostics from single-column model.",
            "mean_period_days": "last 30 days",
            "water_depth": 1.0,
            "CO2": "300 ppm",
        },
    )

    Path(out_nc).parent.mkdir(parents=True, exist_ok=True)
    ds_rce.to_netcdf(out_nc)
    print(f"Saved RCE mean diagnostics to: {out_nc}")

    return Tatm_rce, qatm_rce, Ts_rce, lev, ds_rce


def plot_rce_temp_q(Tatm_rce, qatm_rce, lev):
    # Plot the RCE temperature and humidity profiles
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].plot(Tatm_rce, lev, 'b-')
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('Pressure (hPa)')
    axes[0].grid()
    axes[0].set_title('Temperature profile of RCE')
    axes[0].invert_yaxis()

    axes[1].plot(qatm_rce * 1e3, lev, 'g-')
    axes[1].set_xlabel('Specific humidity (g kg⁻¹)')
    axes[1].set_ylabel('Pressure (hPa)')
    axes[1].set_title('Humidity profile of RCE')
    axes[1].grid()
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()
    return


# WTG physics
def find_tropopause(T, p):
    """
    Tropopause = cold point (temperature minimum), searching from surface upward.
    Returns index into lev array (lev[0]=TOA, lev[-1]=surface).
    """
    # Search only in plausible tropopause range (50-250 hPa)
    mask = (p >= 5000.) & (p <= 25000.)   # Pa
    idx_range = np.where(mask)[0]
    idx_min   = idx_range[np.argmin(T[idx_range])]
    return idx_min


def wtg_moisture_advection(omega, q_column, q_other, T_column, p, dt, tau_strat=10. * 86400.):
    """
    Moisture tendency associated with the WTG circulation.
    Positive omega denotes subsidence.
    """

    nlev = len(p)
    dq = np.zeros(nlev)

    # cold-point tropopause
    idx_tp = find_tropopause(T_column,p,)
    p_tp = p[idx_tp]
    q_tp = q_column[idx_tp] # q at cold point — stratospheric target
    above_tp = p < p_tp # True for levels above tropopause

    for k in range(1, nlev - 1,):
        fradv = 0.
        dwdp = (max(omega[k - 1] - omega[k + 1], 0.) / (p[k - 1] - p[k + 1]))
        if dwdp > 0.: # only advect when there is convergence
            vvp = (max(omega[k + 1], 0.) * dt)
            vvm = min(omega[k],   0.0) * dt

            fradv = (
                -vvp * (q_column[k] - q_column[k + 1]) / (p[k] - p[k + 1])
                -vvm * (q_column[k - 1] - q_column[k]) / (p[k - 1] - p[k]))

            fradv += (dwdp * (q_other[k] - q_column[k]) * dt)

        # Stratospheric humidity relaxation
        if above_tp[k]:
            fradv = (q_tp - q_column[k]) * dt / tau_strat

        dq[k] = fradv

    return dq, idx_tp, q_tp,
    


def advance_dcm_one_step(
    scm_land,
    scm_ocean,
    p,
    above_850,
    tau_wtg,
    dt,
):
    """
    Advance the mass-conserving two-column model by one timestep.

    Equal-area mass conservation:

        omega_land + omega_ocean = 0

    WTG relaxation:

        d(T_land - T_ocean)/dt
            = -(T_land - T_ocean)/tau_wtg
    """

    # --------------------------------------------------------------
    # First advance column physics
    # --------------------------------------------------------------
    scm_land.step_forward()
    scm_ocean.step_forward()

    # States after radiation/convection/surface fluxes
    T_land = np.asarray(scm_land.state["Tatm"]).squeeze().copy()
    T_ocean = np.asarray(scm_ocean.state["Tatm"]).squeeze().copy()

    q_land = np.asarray(scm_land.state["q"]).squeeze().copy()
    q_ocean = np.asarray(scm_ocean.state["q"]).squeeze().copy()

    # Static stability sigma = -dT/dp + kappa T/p
    # dT/dt |_WTG = sigma * omega
    Rd = const.Rd
    cp = const.cp

    dTdp_land = np.gradient(T_land,p)
    dTdp_ocean = np.gradient(T_ocean,p)

    sigma_land = (-dTdp_land + Rd * T_land / (cp * p))
    sigma_ocean = (-dTdp_ocean + Rd * T_ocean / (cp * p))

    # Avoid singular omega in nearly neutral layers
    sigma_floor = 1.e-7
    sigma_land = np.maximum(sigma_land,sigma_floor)
    sigma_ocean = np.maximum(sigma_ocean,sigma_floor,)

    # WTG vertical velocity omega
    # WTG balance: sigma*omega = delta T / tau_relax
    # tau_wtg = 2 days is the standard choice (Sobel & Bretherton 2000)
    dT = T_land - T_ocean
    omega_ocean = (dT / (tau_wtg * (sigma_land + sigma_ocean)))
    omega_land = -omega_ocean # omega_ocean + omega_land = 0

    # Shared WTG domain
    idx_tp_land = find_tropopause(T_land, p,)

    idx_tp_ocean = find_tropopause(T_ocean, p,)

    p_tp_land = p[idx_tp_land]
    p_tp_ocean = p[idx_tp_ocean]

    # Do not apply WTG where either column considers
    # the level stratospheric.
    p_tp_shared = max(p_tp_land,p_tp_ocean,)
    wtg_mask = ((p < 85000.)& (p >= p_tp_shared))

    omega_land[~wtg_mask] = 0.
    omega_ocean[~wtg_mask] = 0.

    # Verify mass conservation
    mass_residual = (omega_land + omega_ocean)

    if not np.allclose(
        mass_residual,
        0.,
        atol=1.e-14,
        rtol=0.,
    ):
        raise RuntimeError(
            "Two-column WTG mass conservation failed: "
            f"max residual = "
            f"{np.max(np.abs(mass_residual)):.3e} Pa s-1"
        )

    # WTG temperature tendencies
    dTdt_land_wtg = sigma_land * omega_land
    dTdt_ocean_wtg = sigma_ocean * omega_ocean
    scm_land.state["Tatm"][wtg_mask] += dTdt_land_wtg[wtg_mask] * dt
    scm_ocean.state["Tatm"][wtg_mask] += dTdt_ocean_wtg[wtg_mask] * dt
    
    # Moisture tendencies
    dq_land, _, _ = (
        wtg_moisture_advection(
            omega=omega_land,
            q_column=q_land,
            q_other=q_ocean,
            T_column=T_land,
            p=p,
            dt=dt,
        )
    )

    dq_ocean, _, _ = (
        wtg_moisture_advection(
            omega=omega_ocean,
            q_column=q_ocean,
            q_other=q_land,
            T_column=T_ocean,
            p=p,
            dt=dt,
        )
    )

    scm_land.state["q"][above_850] += (
        dq_land[above_850]
    )

    scm_ocean.state["q"][above_850] += (
        dq_ocean[above_850]
    )

    # Prevent negative humidity
    scm_land.state["q"][:] = np.maximum(
        scm_land.state["q"],
        1.e-8,
    )

    scm_ocean.state["q"][:] = np.maximum(
        scm_ocean.state["q"],
        1.e-8,
    )

    return (
        omega_land,
        omega_ocean,
    )


# ======================================================================
# DCM initialization
# ======================================================================

def create_dcm_pair(
    num_lev,
    Tatm_init,
    q_init,
    Ts_init,
    land_MLD,
    ocean_MLD,
    lh_r,
):
    """
    Construct a fresh land-ocean DCM pair at 300 ppm CO2.

    Parameters
    ----------
    Tatm_init : ndarray
        Shape (2, nlev).
        [land, ocean]

    q_init : ndarray
        Shape (2, nlev).

    Ts_init : ndarray
        Shape (2,).
    """

    # --------------------------------------------------------------
    # Land
    # --------------------------------------------------------------
    scm_land = create_land_column(
        num_lev=num_lev,
        Tatm_init=Tatm_init[0],
        qatm_init=q_init[0],
        water_depth=land_MLD,
        lh_resistance=lh_r,
        co2_forcing=False,
    )

    # --------------------------------------------------------------
    # Ocean
    # --------------------------------------------------------------
    scm_ocean = create_column(
        num_lev=num_lev,
        water_depth=ocean_MLD,
        co2_forcing=False,
    )

    # --------------------------------------------------------------
    # Assign initial states
    # --------------------------------------------------------------
    scm_land.state["Tatm"][:] = (
        Tatm_init[0]
    )

    scm_land.state["q"][:] = (
        q_init[0]
    )

    scm_land.state["Ts"][:] = (
        Ts_init[0]
    )

    scm_ocean.state["Tatm"][:] = (
        Tatm_init[1]
    )

    scm_ocean.state["q"][:] = (
        q_init[1]
    )

    scm_ocean.state["Ts"][:] = (
        Ts_init[1]
    )

    scm_land.time["steps"] = 0
    scm_ocean.time["steps"] = 0

    return (
        scm_land,
        scm_ocean,
    )


# ======================================================================
# Daily-output infrastructure
# ======================================================================

def allocate_daily_output(
    ndays,
    nlev,
):
    """
    Allocate daily-mean output arrays only.

    This is much smaller than storing every physics timestep.
    """

    shape_profile = (
        2,
        ndays,
        nlev,
    )

    shape_scalar = (
        2,
        ndays,
    )

    output = {
        "Tatm": np.full(
            shape_profile,
            np.nan,
        ),

        "q": np.full(
            shape_profile,
            np.nan,
        ),

        "omega": np.full(
            shape_profile,
            np.nan,
        ),

        "Ts": np.full(
            shape_scalar,
            np.nan,
        ),
    }

    for name in [
        "ASR",
        "ASRclr",
        "ASRcld",
        "OLR",
        "OLRclr",
        "OLRcld",
        "SW_sfc",
        "SW_sfc_clr",
        "LW_sfc",
        "LW_sfc_clr",
        "LHF",
        "SHF",
    ]:

        output[name] = np.full(
            shape_scalar,
            np.nan,
        )

    return output


def zero_daily_accumulator(
    nlev,
):
    """
    Create accumulators for one model day.
    """

    acc = {
        "Tatm": np.zeros(
            (2, nlev)
        ),

        "q": np.zeros(
            (2, nlev)
        ),

        "omega": np.zeros(
            (2, nlev)
        ),

        "Ts": np.zeros(2),
    }

    for name in [
        "ASR",
        "ASRclr",
        "ASRcld",
        "OLR",
        "OLRclr",
        "OLRcld",
        "SW_sfc",
        "SW_sfc_clr",
        "LW_sfc",
        "LW_sfc_clr",
        "LHF",
        "SHF",
    ]:

        acc[name] = np.zeros(2)

    return acc


def accumulate_timestep(
    acc,
    scm_land,
    scm_ocean,
    omega_land,
    omega_ocean,
):
    """
    Add one timestep to daily accumulators.
    """

    models = [
        scm_land,
        scm_ocean,
    ]

    omegas = [
        omega_land,
        omega_ocean,
    ]

    for j, scm in enumerate(
        models
    ):

        acc["Tatm"][j] += np.asarray(
            scm.state["Tatm"]
        ).squeeze()

        acc["q"][j] += np.asarray(
            scm.state["q"]
        ).squeeze()

        acc["Ts"][j] += float(
            np.asarray(
                scm.state["Ts"]
            ).squeeze()
        )

        acc["omega"][j] += (
            omegas[j]
        )

        for name in [
            "ASR",
            "ASRclr",
            "ASRcld",
            "OLR",
            "OLRclr",
            "OLRcld",
            "SW_sfc",
            "SW_sfc_clr",
            "LW_sfc",
            "LW_sfc_clr",
            "LHF",
            "SHF",
        ]:

            acc[name][j] += (
                diag_scalar(
                    scm,
                    name,
                )
            )


def store_daily_mean(
    output,
    acc,
    day_index,
    steps_per_day,
):
    """
    Divide daily accumulator by number of timesteps and store.
    """

    for name in output:

        output[name][
            :,
            day_index,
            ...
        ] = (
            acc[name]
            / steps_per_day
        )


def build_daily_dataset(
    output,
    lev,
    land_MLD,
    ocean_MLD,
    lh_r,
    description,
):
    """
    Convert daily arrays into an xarray Dataset.
    """

    ndays = output["Ts"].shape[1]

    ds = xr.Dataset(
        data_vars={
            "Tatm": (
                (
                    "column",
                    "time",
                    "lev",
                ),
                output["Tatm"],
                {
                    "long_name":
                        "Atmospheric temperature",
                    "units": "K",
                },
            ),

            "q": (
                (
                    "column",
                    "time",
                    "lev",
                ),
                output["q"],
                {
                    "long_name":
                        "Specific humidity",
                    "units": "kg kg-1",
                },
            ),

            "omega": (
                (
                    "column",
                    "time",
                    "lev",
                ),
                output["omega"],
                {
                    "long_name":
                        "Mass-conserving two-column WTG pressure vertical velocity",
                    "units": "Pa s-1",
                    "positive": "down",
                },
            ),

            "Ts": (
                (
                    "column",
                    "time",
                ),
                output["Ts"],
                {
                    "long_name":
                        "Surface temperature",
                    "units": "K",
                },
            ),

            "ASR": (
                ("column", "time"),
                output["ASR"],
                {"units": "W m-2"},
            ),

            "ASRclr": (
                ("column", "time"),
                output["ASRclr"],
                {"units": "W m-2"},
            ),

            "ASRcld": (
                ("column", "time"),
                output["ASRcld"],
                {"units": "W m-2"},
            ),

            "OLR": (
                ("column", "time"),
                output["OLR"],
                {"units": "W m-2"},
            ),

            "OLRclr": (
                ("column", "time"),
                output["OLRclr"],
                {"units": "W m-2"},
            ),

            "OLRcld": (
                ("column", "time"),
                output["OLRcld"],
                {"units": "W m-2"},
            ),

            "SW_sfc": (
                ("column", "time"),
                output["SW_sfc"],
                {"units": "W m-2"},
            ),

            "SW_sfc_clr": (
                ("column", "time"),
                output["SW_sfc_clr"],
                {"units": "W m-2"},
            ),

            "LW_sfc": (
                ("column", "time"),
                output["LW_sfc"],
                {"units": "W m-2"},
            ),

            "LW_sfc_clr": (
                ("column", "time"),
                output["LW_sfc_clr"],
                {"units": "W m-2"},
            ),

            "LHF": (
                ("column", "time"),
                output["LHF"],
                {"units": "W m-2"},
            ),

            "SHF": (
                ("column", "time"),
                output["SHF"],
                {"units": "W m-2"},
            ),
        },

        coords={
            "column": [
                "land",
                "ocean",
            ],

            "time": (
                np.arange(ndays)
                + 1
            ),

            "lev": lev,
        },

        attrs={
            "description": description,
            "land_MLD": float(
                land_MLD
            ),
            "ocean_MLD": float(
                ocean_MLD
            ),
            "lh_resistance": float(
                lh_r
            ),
        },
    )

    ds["time"].attrs = {
        "long_name": "Elapsed time",
        "units": "days",
    }

    ds = ds.assign_coords(
        water_depth=(
            "column",
            [
                land_MLD,
                ocean_MLD,
            ],
        )
    )

    ds["water_depth"].attrs = {
        "long_name":
            "Surface mixed-layer depth",
        "units": "m",
    }

    return ds


# ======================================================================
# Generic DCM integration
# ======================================================================

def integrate_dcm_daily(
    scm_land,
    scm_ocean,
    lev,
    ndays,
    land_MLD,
    ocean_MLD,
    lh_r,
    description,
    print_label="DCM",
):
    """
    Run the coupled DCM and return daily-mean output.
    """

    p = (
        lev
        * 100.
    )

    above_850 = (
        lev
        < 850.
    )

    tau_wtg = (
        2.
        * const.seconds_per_day
    )

    dt_land = float(
        scm_land.timestep
    )

    dt_ocean = float(
        scm_ocean.timestep
    )

    if not np.isclose(
        dt_land,
        dt_ocean,
        rtol=0.,
        atol=1.e-12,
    ):
        raise ValueError(
            "Land and ocean model timesteps differ: "
            f"{dt_land} vs {dt_ocean} s."
        )

    dt = dt_land

    steps_per_day_float = (
        const.seconds_per_day
        / dt
    )

    steps_per_day = int(
        round(
            steps_per_day_float
        )
    )

    if not np.isclose(
        steps_per_day,
        steps_per_day_float,
        rtol=0.,
        atol=1.e-10,
    ):
        raise ValueError(
            "Model timestep does not divide evenly into one day. "
            f"dt = {dt} s"
        )

    print(
        f"{print_label}:"
    )

    print(
        f"  dt = {dt:.1f} s"
    )

    print(
        f"  steps/day = {steps_per_day}"
    )

    print(
        f"  integration = {ndays} days"
    )

    # --------------------------------------------------------------
    # Allocate DAILY output
    # --------------------------------------------------------------
    output = allocate_daily_output(
        ndays=ndays,
        nlev=len(lev),
    )

    # --------------------------------------------------------------
    # Day-by-day integration
    # --------------------------------------------------------------
    for day in range(ndays):

        acc = zero_daily_accumulator(
            nlev=len(lev),
        )

        for _ in range(
            steps_per_day
        ):

            omega_land, omega_ocean = (
                advance_dcm_one_step(
                    scm_land=scm_land,
                    scm_ocean=scm_ocean,
                    p=p,
                    above_850=above_850,
                    tau_wtg=tau_wtg,
                    dt=dt,
                )
            )

            accumulate_timestep(
                acc=acc,
                scm_land=scm_land,
                scm_ocean=scm_ocean,
                omega_land=omega_land,
                omega_ocean=omega_ocean,
            )

        # End of one model day
        store_daily_mean(
            output=output,
            acc=acc,
            day_index=day,
            steps_per_day=steps_per_day,
        )

        if (
            day == 0
            or (day + 1) % 100 == 0
        ):

            Ts_land = (
                output["Ts"][
                    0,
                    day,
                ]
            )

            Ts_ocean = (
                output["Ts"][
                    1,
                    day,
                ]
            )

            toa_land = (
                output["ASR"][
                    0,
                    day,
                ]
                - output["OLR"][
                    0,
                    day,
                ]
            )

            toa_ocean = (
                output["ASR"][
                    1,
                    day,
                ]
                - output["OLR"][
                    1,
                    day,
                ]
            )

            print(
                f"{print_label} day {day + 1:6d}: "
                f"Ts_land={Ts_land:.3f} K, "
                f"Ts_ocean={Ts_ocean:.3f} K, "
                f"N_land={toa_land:+.3f}, "
                f"N_ocean={toa_ocean:+.3f} W m-2"
            )

    ds = build_daily_dataset(
        output=output,
        lev=lev,
        land_MLD=land_MLD,
        ocean_MLD=ocean_MLD,
        lh_r=lh_r,
        description=description,
    )

    ds.attrs[
        "wtg_timescale_days"
    ] = float(
        tau_wtg
        / const.seconds_per_day
    )

    return ds


# ======================================================================
# Equilibrium diagnostics
# ======================================================================

def diagnose_control_equilibrium(
    ds,
    mean_days=365,
):
    """
    Diagnose the final portion of the control run.

    Reports:
      - mean land and ocean surface temperatures
      - linear Ts trend
      - mean TOA imbalance
    """

    if mean_days > ds.sizes["time"]:
        raise ValueError(
            "mean_days is longer than the available control integration."
        )

    eq = ds.isel(
        time=slice(
            -mean_days,
            None,
        )
    )

    print()
    print(
        "=" * 70
    )
    print(
        f"CONTROL EQUILIBRIUM DIAGNOSTICS "
        f"(last {mean_days} days)"
    )
    print(
        "=" * 70
    )

    x_days = np.arange(
        mean_days,
        dtype=float,
    )

    for column in [
        "land",
        "ocean",
    ]:

        Ts = (
            eq["Ts"]
            .sel(column=column)
            .values
        )

        # K / day
        slope = np.polyfit(
            x_days,
            Ts,
            1,
        )[0]

        # K / year
        slope_year = (
            slope
            * 365.
        )

        N = (
            eq["ASR"]
            .sel(column=column)
            - eq["OLR"]
            .sel(column=column)
        )

        print(
            f"{column.capitalize():5s}: "
            f"mean Ts = {Ts.mean():.3f} K, "
            f"Ts trend = {slope_year:+.5f} K yr-1, "
            f"mean TOA imbalance = "
            f"{float(N.mean()):+.4f} W m-2"
        )

    N_domain = (
        (
            eq["ASR"]
            - eq["OLR"]
        )
        .mean(
            dim="column"
        )
    )

    print(
        f"Domain-mean TOA imbalance = "
        f"{float(N_domain.mean()):+.4f} W m-2"
    )

    print(
        "=" * 70
    )
    print()


def get_mean_control_state(
    ds_control,
    mean_days=365,
):
    """
    Calculate the mean control state used to initialize the
    abrupt-4xCO2 experiment.
    """

    if mean_days > ds_control.sizes[
        "time"
    ]:
        raise ValueError(
            "mean_days exceeds length of control integration."
        )

    ds_mean = (
        ds_control
        .isel(
            time=slice(
                -mean_days,
                None,
            )
        )
        .mean(
            dim="time",
            keep_attrs=True,
        )
    )

    mean_state = {
        "Tatm": (
            ds_mean["Tatm"]
            .values
            .copy()
        ),

        "q": (
            ds_mean["q"]
            .values
            .copy()
        ),

        "Ts": (
            ds_mean["Ts"]
            .values
            .copy()
        ),
    }

    return (
        mean_state,
        ds_mean,
    )


# ======================================================================
# CONTROL SPIN-UP
# ======================================================================

def run_dcm_spinup(
    num_lev,
    Tatm_rce,
    qatm_rce,
    Ts_rce,
    lev,
    spinup_days=15000,
    restart_mean_days=365,
    land_MLD=1,
    ocean_MLD=100,
    lh_r=1,
):
    """
    Run the coupled land-ocean DCM at 300 ppm CO2.

    The final restart_mean_days are averaged to define the
    equilibrium state used for the forcing experiment.
    """

    # Both columns initially use the single-column RCE state
    Tatm_init = np.stack(
        [
            Tatm_rce,
            Tatm_rce,
        ]
    )

    q_init = np.stack(
        [
            qatm_rce,
            qatm_rce,
        ]
    )

    Ts_init = np.array(
        [
            Ts_rce,
            Ts_rce,
        ]
    )

    # --------------------------------------------------------------
    # Fresh 300-ppm DCM
    # --------------------------------------------------------------
    scm_land, scm_ocean = (
        create_dcm_pair(
            num_lev=num_lev,
            Tatm_init=Tatm_init,
            q_init=q_init,
            Ts_init=Ts_init,
            land_MLD=land_MLD,
            ocean_MLD=ocean_MLD,
            lh_r=lh_r,
        )
    )

    # --------------------------------------------------------------
    # Control integration
    # --------------------------------------------------------------
    ds_control = (
        integrate_dcm_daily(
            scm_land=scm_land,
            scm_ocean=scm_ocean,
            lev=lev,
            ndays=spinup_days,
            land_MLD=land_MLD,
            ocean_MLD=ocean_MLD,
            lh_r=lh_r,
            description=(
                "Daily-mean coupled double-column control "
                "spin-up at 300 ppm CO2."
            ),
            print_label="CONTROL",
        )
    )

    ds_control.attrs.update(
        {
            "experiment":
                "DCM control spin-up",

            "CO2":
                "300 ppm",

            "spinup_days":
                int(spinup_days),

            "equilibrium_mean_days":
                int(restart_mean_days),
        }
    )

    # --------------------------------------------------------------
    # Equilibrium diagnostics
    # --------------------------------------------------------------
    diagnose_control_equilibrium(
        ds_control,
        mean_days=restart_mean_days,
    )

    # --------------------------------------------------------------
    # Mean equilibrium state
    # --------------------------------------------------------------
    mean_state, ds_mean = (
        get_mean_control_state(
            ds_control=ds_control,
            mean_days=restart_mean_days,
        )
    )

    return (
        ds_control,
        mean_state,
        ds_mean,
    )


# ======================================================================
# ABRUPT 4xCO2 EXPERIMENT
# ======================================================================

def run_abrupt4xco2(
    num_lev,
    mean_state,
    lev,
    forced_days=15000,
    land_MLD=1,
    ocean_MLD=100,
    lh_r=1,
    restart_mean_days=365,
):
    """
    Initialize a fresh DCM from the mean equilibrium control state,
    then instantaneously change CO2 from 300 ppm to 1200 ppm.
    """

    # --------------------------------------------------------------
    # Fresh DCM at 300 ppm
    #
    # This is initialized from the AVERAGED equilibrium control
    # state rather than the instantaneous last control timestep.
    # --------------------------------------------------------------
    scm_land, scm_ocean = (
        create_dcm_pair(
            num_lev=num_lev,
            Tatm_init=mean_state["Tatm"],
            q_init=mean_state["q"],
            Ts_init=mean_state["Ts"],
            land_MLD=land_MLD,
            ocean_MLD=ocean_MLD,
            lh_r=lh_r,
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Applying abrupt 4xCO2 forcing:"
    )

    print(
        "    land  : 300 ppm -> 1200 ppm"
    )

    print(
        "    ocean : 300 ppm -> 1200 ppm"
    )

    print(
        "=" * 70
    )
    print()

    # --------------------------------------------------------------
    # Abrupt forcing
    # --------------------------------------------------------------
    apply_abrupt_4xco2(
        scm_land
    )

    apply_abrupt_4xco2(
        scm_ocean
    )

    # --------------------------------------------------------------
    # Forced integration
    # --------------------------------------------------------------
    ds_forcing = (
        integrate_dcm_daily(
            scm_land=scm_land,
            scm_ocean=scm_ocean,
            lev=lev,
            ndays=forced_days,
            land_MLD=land_MLD,
            ocean_MLD=ocean_MLD,
            lh_r=lh_r,
            description=(
                "Daily-mean abrupt-4xCO2 double-column "
                "experiment initialized from the mean "
                "equilibrium 300-ppm control state."
            ),
            print_label="4xCO2",
        )
    )

    # --------------------------------------------------------------
    # Save control reference state in forcing file
    # --------------------------------------------------------------
    ds_forcing[
        "control_Tatm"
    ] = (
        (
            "column",
            "lev",
        ),
        mean_state["Tatm"],
        {
            "long_name":
                "Mean control atmospheric temperature used to initialize forcing run",
            "units":
                "K",
        },
    )

    ds_forcing[
        "control_q"
    ] = (
        (
            "column",
            "lev",
        ),
        mean_state["q"],
        {
            "long_name":
                "Mean control specific humidity used to initialize forcing run",
            "units":
                "kg kg-1",
        },
    )

    ds_forcing[
        "control_Ts"
    ] = (
        ("column",),
        mean_state["Ts"],
        {
            "long_name":
                "Mean control surface temperature used to initialize forcing run",
            "units":
                "K",
        },
    )

    ds_forcing.attrs.update(
        {
            "experiment":
                "abrupt 4xCO2",

            "control_CO2":
                "300 ppm",

            "forced_CO2":
                "1200 ppm",

            "forcing":
                "instantaneous quadrupling of CO2",

            "forcing_applied_to":
                "land and ocean columns",

            "forced_days":
                int(forced_days),

            "initial_state":
                (
                    f"Mean of final "
                    f"{restart_mean_days} days "
                    f"of 300-ppm control DCM"
                ),
        }
    )

    return ds_forcing


# ======================================================================
# MAIN
# ======================================================================

def main():

    # ==============================================================
    # Experiment configuration
    # ==============================================================

    num_lev = 60

    land_MLD = 1

    ocean_MLD_list = [100,60,20]

    lh_resistance_list = [0.6,0.8,1]

    # --------------------------------------------------------------
    # DCM control
    #
    # 15000 days ~= 41 years
    # --------------------------------------------------------------
    spinup_days = 15000

    # Mean equilibrium state used to initialize 4xCO2 experiment
    restart_mean_days = 365

    # --------------------------------------------------------------
    # Abrupt 4xCO2 integration
    # --------------------------------------------------------------
    forced_days = 5000

    Path(
        "data"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # 1. Generate single-column RCE reference state
    # ==============================================================

    rce_nc = Path(
        "data/rce_mean.nc"
    )

    (
        Tatm_rce,
        qatm_rce,
        Ts_rce,
        lev,
        ds_rce,
    ) = rcm(
        num_lev=num_lev,
        out_nc=rce_nc,
    )

    print(
        ds_rce
    )

    plot_rce_temp_q(
        Tatm_rce,
        qatm_rce,
        lev,
    )

    # ==============================================================
    # 2. Loop over DCM configurations
    # ==============================================================

    for ocean_MLD in (
        ocean_MLD_list
    ):

        for lh_resistance in (
            lh_resistance_list
        ):

            label = (
                f"r{lh_resistance}_"
                f"land{land_MLD}_"
                f"ocean{ocean_MLD}"
            )

            # ------------------------------------------------------
            # Output file 1:
            # 300-ppm control DCM spin-up
            # ------------------------------------------------------
            control_nc = Path(
                "0817data/"
                f"dcm_control_spinup_{label}.nc"
            )

            # ------------------------------------------------------
            # Output file 2:
            # abrupt-4xCO2 experiment
            # ------------------------------------------------------
            forcing_nc = Path(
                "0817data/"
                f"dcm_abrupt4xco2_{label}.nc"
            )

            # ======================================================
            # CONTROL SPIN-UP
            # ======================================================

            (
                ds_control,
                mean_state,
                ds_control_mean,
            ) = run_dcm_spinup(
                num_lev=num_lev,
                Tatm_rce=Tatm_rce,
                qatm_rce=qatm_rce,
                Ts_rce=Ts_rce,
                lev=lev,
                spinup_days=spinup_days,
                restart_mean_days=restart_mean_days,
                land_MLD=land_MLD,
                ocean_MLD=ocean_MLD,
                lh_r=lh_resistance,
            )

            # ------------------------------------------------------
            # Save spin-up separately
            # ------------------------------------------------------
            ds_control.to_netcdf(
                control_nc
            )

            print()
            print(
                f"Saved CONTROL spin-up:"
            )

            print(
                f"    {control_nc}"
            )

            # ------------------------------------------------------
            # Print the equilibrium state used for forcing
            # ------------------------------------------------------
            print()
            print(
                f"Mean state over final "
                f"{restart_mean_days} control days:"
            )

            print(
                f"    Land Ts  = "
                f"{mean_state['Ts'][0]:.3f} K"
            )

            print(
                f"    Ocean Ts = "
                f"{mean_state['Ts'][1]:.3f} K"
            )

            print(
                f"    Land-ocean Ts contrast = "
                f"{mean_state['Ts'][0] - mean_state['Ts'][1]:+.3f} K"
            )

            # ======================================================
            # ABRUPT 4xCO2
            # ======================================================

            ds_forcing = run_abrupt4xco2(
                num_lev=num_lev,
                mean_state=mean_state,
                lev=lev,
                forced_days=forced_days,
                land_MLD=land_MLD,
                ocean_MLD=ocean_MLD,
                lh_r=lh_resistance,
                restart_mean_days=restart_mean_days,
            )

            # ------------------------------------------------------
            # Save forcing run separately
            # ------------------------------------------------------
            ds_forcing.to_netcdf(
                forcing_nc
            )

            print()
            print(
                f"Saved ABRUPT-4xCO2 run:"
            )

            print(
                f"    {forcing_nc}"
            )

            print()
            print(
                "=" * 70
            )

            print(
                "Experiment completed."
            )

            print(
                "=" * 70
            )


if __name__ == "__main__":
    main()
