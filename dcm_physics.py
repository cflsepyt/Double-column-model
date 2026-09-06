"""SBM column construction and common numerical/physical settings."""

from dataclasses import asdict, dataclass
import climlab
import numpy as np
from climlab import constants as const
from climlab.solar.insolation import daily_insolation
from climlab.utils.thermo import qsat


@dataclass(frozen=True)
class ColumnConfig:
    physics_dt: float = 600.
    radiation_dt: float = 600.
    tau_bm: float = 7200.
    rhbm: float = 0.7
    condensation_time: float = 14400.
    initial_rh: float = 0.65
    co2_ppm: float = 300.

    def __post_init__(self):
        for name in ("physics_dt", "radiation_dt", "tau_bm", "condensation_time", "co2_ppm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.rhbm <= 1 or not 0 < self.initial_rh < 1:
            raise ValueError("Invalid reference or initial relative humidity")
        for ratio in (86400. / self.physics_dt, self.radiation_dt / self.physics_dt):
            if ratio < 1 or not np.isclose(ratio, round(ratio)):
                raise ValueError("Physics timestep must divide a day and the radiation interval")
        if self.physics_dt > min(self.tau_bm, self.condensation_time):
            raise ValueError("Physics timestep exceeds a moisture relaxation timescale")

    def attributes(self):
        return {**asdict(self), "convection_scheme": "SimplifiedBettsMiller", "condensation_RH_ref": 1.0}


class ColumnCondensation(climlab.dynamics.LargeScaleCondensation):
    """Upstream condensation physics with scalar-column precipitation indexing.

    climlab 0.9.2 uses precipitation[:, 0], although column_state has a
    one-dimensional surface field. This changes only diagnostic assignment.
    """

    def _compute(self):
        dq = np.minimum(-(self.q - self.RH_ref * qsat(self.Tatm, self.lev)) /
                        self.condensation_time, 0.)
        dT = -const.Lhvap / const.cp * dq
        self.latent_heating[:] = dT * self.Tatm.domain.heat_capacity
        rain = np.sum(self.latent_heating, axis=-1) / const.Lhvap
        self.precipitation[...] = np.asarray(rain).reshape(self.precipitation.shape)
        return {"Tatm": dT, "q": dq}


def create_sbm_convection(state, config=None):
    config = config or ColumnConfig()
    return climlab.convection.SimplifiedBettsMiller(
        name="Convection", state=state, timestep=config.physics_dt,
        tau_bm=config.tau_bm, rhbm=config.rhbm,
        do_simp=False, do_shallower=True, do_changeqref=True,
        do_envsat=True, do_taucape=False,
    )


def set_co2(scm, ppm):
    """Set and verify container and LW/SW inputs, preserving shared dictionaries."""
    rad = scm.subprocess["Radiation"]
    processes = [rad, *rad.subprocess.values()]
    for process in processes:
        process.absorber_vmr["CO2"] = ppm * 1e-6
    if not all(np.allclose(p.absorber_vmr["CO2"], ppm * 1e-6, rtol=0, atol=1e-15)
               for p in processes):
        raise RuntimeError("CO2 was not propagated to both radiation components")


def apply_abrupt_4xco2(scm):
    ppm = float(scm.subprocess["Radiation"].absorber_vmr["CO2"]) * 1e6
    set_co2(scm, 4. * ppm)


def create_column(num_lev, water_depth, co2_forcing=False, *, config=None,
                  lh_resistance=1., Tatm_init=None, qatm_init=None, Ts_init=None):
    config = config or ColumnConfig()
    state = climlab.column_state(num_lev=num_lev, water_depth=water_depth)
    lev = state.Tatm.domain.lev.points
    # 295 K at the bottom, 213 K at/above 200 hPa; RH is initialized only once.
    fraction = np.clip(np.log(lev / 200.) / np.log(lev[-1] / 200.), 0., 1.)
    state.Tatm[:] = 213. + 82. * fraction if Tatm_init is None else Tatm_init
    state.Ts[:] = 297. if Ts_init is None else Ts_init
    q0 = config.initial_rh * qsat(state.Tatm, lev)
    q0[lev <= 200.] = np.minimum(q0[lev <= 200.], 5e-6)
    state["q"] = climlab.Field(q0 if qatm_init is None else qatm_init, domain=state.Tatm.domain)
    temperature_state = {"Tatm": state.Tatm, "Ts": state.Ts}
    insolation = float(np.mean(daily_insolation(lat=25., day=np.arange(365), S0=1360.)))
    rad = climlab.radiation.RRTMG(
        name="Radiation", state=temperature_state, specific_humidity=state.q,
        albedo=0.33, insolation=insolation, coszen=0.5, icld=0,
        timestep=config.radiation_dt,
    )
    conv = create_sbm_convection(state, config)
    condensation = ColumnCondensation(
        name="Condensation", state=state, RH_ref=1.,
        condensation_time=config.condensation_time, timestep=config.physics_dt,
    )
    shf = climlab.surface.SensibleHeatFlux(
        name="SHF", state=temperature_state, Cd=1e-3, timestep=config.physics_dt,
    )
    lhf = climlab.surface.LatentHeatFlux(
        name="LHF", state=state, Cd=1e-3, resistance=lh_resistance, timestep=config.physics_dt,
    )
    model = climlab.couple([rad, conv, condensation, lhf, shf], name="Tropical SBM RCE")
    model.column_config = config
    set_co2(model, config.co2_ppm * (4. if co2_forcing else 1.))
    return model


def create_dcm_pair(num_lev, Tatm_init, q_init, Ts_init, land_MLD, ocean_MLD, lh_r,
                    *, config=None):
    return tuple(create_column(num_lev, depth, config=config, lh_resistance=resistance,
                               Tatm_init=Tatm_init[i], qatm_init=q_init[i], Ts_init=Ts_init[i])
                 for i, (depth, resistance) in enumerate(((land_MLD, lh_r), (ocean_MLD, 1.))))
