"""Conservative two-column WTG coupling following Shaevitz and Sobel (2004).

Within the configured free-tropospheric region the columns share one
temperature. The area-mean diabatic heating changes that profile, while the
heating anomaly diagnoses equal-and-opposite pressure velocity. Moisture is
transported in flux form using the horizontal divergence implied by continuity.

The boundary-layer pressure velocity is tapered linearly to zero at the
surface, as in the paper. Boundary-layer temperature remains governed by the
local column physics: adding its pressure-work term would require a mechanical
energy closure that this diagnostic circulation model does not contain.
"""

from dataclasses import dataclass

import numpy as np
from climlab import constants as const


@dataclass(frozen=True)
class WTGConfig:
    """Numerical configuration for the two-column WTG operator."""

    top_pressure_pa: float = 10_000.0
    pbl_top_pressure_pa: float = 85_000.0
    transport_cfl: float = 0.45
    minimum_stability_K_Pa: float = 1.0e-8

    def __post_init__(self):
        if not 0.0 < self.top_pressure_pa < self.pbl_top_pressure_pa:
            raise ValueError("WTG top must lie above the PBL top")
        if not 0.0 < self.transport_cfl < 1.0:
            raise ValueError("transport_cfl must lie between zero and one")
        if self.minimum_stability_K_Pa <= 0.0:
            raise ValueError("minimum_stability_K_Pa must be positive")

    def attributes(self):
        return {
            "wtg_method": "Shaevitz-Sobel-2004 mean-heating WTG",
            "wtg_top_pressure_hPa": self.top_pressure_pa / 100.0,
            "wtg_pbl_top_pressure_hPa": self.pbl_top_pressure_pa / 100.0,
            "wtg_transport_cfl": self.transport_cfl,
            "wtg_heat_capacity_factor": 1.0,
        }


def pressure_interfaces(p):
    """Return pressure-cell interfaces from monotonically increasing centers."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size < 3 or not np.all(np.diff(p) > 0.0):
        raise ValueError("Pressure centers must be a 1-D increasing array")
    edges = np.empty(p.size + 1)
    edges[1:-1] = 0.5 * (p[:-1] + p[1:])
    edges[0] = p[0] - 0.5 * (p[1] - p[0])
    edges[-1] = p[-1] + 0.5 * (p[-1] - p[-2])
    if edges[0] < -1.0e-8 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("Invalid pressure-cell interfaces")
    edges[0] = max(edges[0], 0.0)
    return edges


def common_static_stability(T, p):
    """Return S=(T/theta)d(theta)/dp in K Pa-1 (negative when stable)."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)
    if T.shape != p.shape or np.any(T <= 0.0) or np.any(p <= 0.0):
        raise ValueError("Temperature and pressure must be positive 1-D profiles")
    kappa = const.Rd / const.cp
    theta = T * (100_000.0 / p) ** kappa
    return T / theta * np.gradient(theta, p, edge_order=2)


def _omega_on_interfaces(omega_free, p, config):
    """Interpolate free-tropospheric omega and close it through the PBL."""
    p = np.asarray(p, dtype=float)
    omega_free = np.asarray(omega_free, dtype=float)
    edges = pressure_interfaces(p)
    if omega_free.shape != p.shape:
        raise ValueError("omega and pressure shapes differ")
    if config.pbl_top_pressure_pa >= edges[-1]:
        raise ValueError("PBL top must lie above the surface pressure")

    free = ((p >= config.top_pressure_pa) &
            (p < config.pbl_top_pressure_pa))
    if np.count_nonzero(free) < 2:
        raise ValueError("WTG region contains fewer than two model levels")

    support_p = np.concatenate((
        [config.top_pressure_pa], p[free], [config.pbl_top_pressure_pa],
    ))
    support_omega = np.concatenate(([0.0], omega_free[free], [omega_free[free][-1]]))
    omega_edge = np.zeros_like(edges)
    free_edges = ((edges >= config.top_pressure_pa) &
                  (edges <= config.pbl_top_pressure_pa))
    omega_edge[free_edges] = np.interp(
        edges[free_edges], support_p, support_omega,
    )

    pbl_edges = ((edges > config.pbl_top_pressure_pa) &
                 (edges < edges[-1]))
    omega_850 = support_omega[-1]
    omega_edge[pbl_edges] = omega_850 * (
        (edges[-1] - edges[pbl_edges]) /
        (edges[-1] - config.pbl_top_pressure_pa)
    )
    omega_edge[0] = 0.0
    omega_edge[-1] = 0.0
    return omega_edge


def _vertical_tracer_tendency(q, omega_edge, dp):
    """Conservative donor-cell vertical tracer transport in pressure space."""
    flux = np.zeros(q.size + 1)
    interior_omega = omega_edge[1:-1]
    flux[1:-1] = interior_omega * np.where(
        interior_omega >= 0.0, q[:-1], q[1:]
    )
    return -(flux[1:] - flux[:-1]) / dp


def transport_moisture_pair(q_land, q_ocean, omega_land_edge, p, dt, *, cfl=0.45):
    """Transport moisture with exactly cancelling two-column horizontal fluxes.

    ``omega_land_edge`` is positive for downward motion. Ocean omega is its
    exact negative for equal-area columns. Both vertical boundary fluxes are
    required to be zero.
    """
    q_land = np.asarray(q_land, dtype=float).copy()
    q_ocean = np.asarray(q_ocean, dtype=float).copy()
    omega_land_edge = np.asarray(omega_land_edge, dtype=float)
    edges = pressure_interfaces(p)
    dp = np.diff(edges)
    if q_land.shape != dp.shape or q_ocean.shape != dp.shape:
        raise ValueError("Humidity and pressure shapes differ")
    if omega_land_edge.shape != edges.shape:
        raise ValueError("Interface omega has the wrong shape")
    if not np.isfinite(dt) or dt <= 0.0 or not 0.0 < cfl < 1.0:
        raise ValueError("Invalid transport timestep or CFL number")
    if not all(np.all(np.isfinite(value))
               for value in (q_land, q_ocean, omega_land_edge)):
        raise ValueError("Moisture transport inputs must be finite")
    if np.any(q_land < 0.0) or np.any(q_ocean < 0.0):
        raise ValueError("Cannot transport negative humidity")
    if not np.allclose(omega_land_edge[[0, -1]], 0.0, atol=1e-15, rtol=0.0):
        raise ValueError("Moisture transport requires closed vertical boundaries")

    omega_ocean_edge = -omega_land_edge
    # Horizontal divergence follows directly from du/dx = -domega/dp.
    divergence_land = -(omega_land_edge[1:] - omega_land_edge[:-1]) / dp
    divergence_ocean = -divergence_land

    def outflow_rate(omega_edge, divergence):
        vertical = (np.maximum(-omega_edge[:-1], 0.0) +
                    np.maximum(omega_edge[1:], 0.0)) / dp
        return vertical + np.maximum(divergence, 0.0)

    max_rate = float(max(np.max(outflow_rate(omega_land_edge, divergence_land)),
                         np.max(outflow_rate(omega_ocean_edge, divergence_ocean))))
    nsubsteps = max(1, int(np.ceil(dt * max_rate / cfl)))
    subdt = dt / nsubsteps
    initial_water = float(np.sum(dp * (q_land + q_ocean)))

    for _ in range(nsubsteps):
        vertical_land = _vertical_tracer_tendency(q_land, omega_land_edge, dp)
        vertical_ocean = _vertical_tracer_tendency(q_ocean, omega_ocean_edge, dp)

        # One donor-cell horizontal transfer, applied with opposite signs.
        horizontal_land = np.where(
            divergence_land >= 0.0,
            -divergence_land * q_land,
            -divergence_land * q_ocean,
        )
        horizontal_ocean = -horizontal_land
        q_land += subdt * (vertical_land + horizontal_land)
        q_ocean += subdt * (vertical_ocean + horizontal_ocean)
        if np.min(q_land) < -1.0e-14 or np.min(q_ocean) < -1.0e-14:
            raise FloatingPointError("WTG moisture transport produced negative humidity")

    final_water = float(np.sum(dp * (q_land + q_ocean)))
    tolerance = 5.0e-13 * max(1.0, abs(initial_water))
    if abs(final_water - initial_water) > tolerance:
        raise RuntimeError("Two-column WTG moisture transport is not conservative")
    return q_land, q_ocean, nsubsteps


def advance_dcm_one_step(scm_land, scm_ocean, p, dt, *, Tatm_before=None,
                         config=None, advance_physics=True):
    """Apply one Shaevitz--Sobel mean-heating WTG coupling step.

    Column physics supplies provisional diabatic temperature increments. The
    free-tropospheric mean increment changes the shared temperature, and its
    anomaly diagnoses omega through ``omega*S = Q_T - Q_TM``.
    """
    config = config or WTGConfig()
    p = np.asarray(p, dtype=float)
    free = ((p >= config.top_pressure_pa) &
            (p < config.pbl_top_pressure_pa))

    if advance_physics:
        Tatm_before = (
            np.asarray(scm_land.state["Tatm"]).squeeze().copy(),
            np.asarray(scm_ocean.state["Tatm"]).squeeze().copy(),
        )
        scm_land.step_forward()
        scm_ocean.step_forward()
    elif Tatm_before is None:
        raise ValueError("Tatm_before is required when physics was advanced externally")

    T_before_land = np.asarray(Tatm_before[0], dtype=float).squeeze()
    T_before_ocean = np.asarray(Tatm_before[1], dtype=float).squeeze()
    T_star_land = np.asarray(scm_land.state["Tatm"]).squeeze().copy()
    T_star_ocean = np.asarray(scm_ocean.state["Tatm"]).squeeze().copy()
    q_star_land = np.asarray(scm_land.state["q"]).squeeze().copy()
    q_star_ocean = np.asarray(scm_ocean.state["q"]).squeeze().copy()
    for profile in (T_before_land, T_before_ocean, T_star_land, T_star_ocean,
                    q_star_land, q_star_ocean):
        if profile.shape != p.shape or not np.all(np.isfinite(profile)):
            raise ValueError("Invalid WTG input profile")
    if not np.allclose(T_before_land[free], T_before_ocean[free],
                       atol=1.0e-10, rtol=0.0):
        raise ValueError("WTG requires a shared free-tropospheric temperature before physics")

    heating_land = (T_star_land - T_before_land) / dt
    heating_ocean = (T_star_ocean - T_before_ocean) / dt
    mean_heating = 0.5 * (heating_land + heating_ocean)
    heating_anomaly_land = heating_land - mean_heating

    # Equation (4) with C=1: only mean diabatic heating changes temperature.
    shared_temperature = 0.5 * (T_star_land + T_star_ocean)
    T_new_land = T_star_land.copy()
    T_new_ocean = T_star_ocean.copy()
    T_new_land[free] = shared_temperature[free]
    T_new_ocean[free] = shared_temperature[free]

    # Equation (5): omega_i S_M = Q_Ti - Q_TM.
    stability = common_static_stability(shared_temperature, p)
    bad_stability = free & (stability >= -config.minimum_stability_K_Pa)
    if np.any(bad_stability):
        levels = p[bad_stability] / 100.0
        raise FloatingPointError(
            f"WTG common state is insufficiently stable at {levels.tolist()} hPa"
        )
    omega_land_free = np.zeros_like(p)
    omega_land_free[free] = heating_anomaly_land[free] / stability[free]
    omega_land_edge = _omega_on_interfaces(omega_land_free, p, config)
    q_new_land, q_new_ocean, _ = transport_moisture_pair(
        q_star_land, q_star_ocean, omega_land_edge, p, dt,
        cfl=config.transport_cfl,
    )

    scm_land.state["Tatm"][:] = T_new_land
    scm_ocean.state["Tatm"][:] = T_new_ocean
    scm_land.state["q"][:] = q_new_land
    scm_ocean.state["q"][:] = q_new_ocean

    # Return center values for existing output files; transport uses interfaces.
    omega_land = np.zeros_like(p)
    omega_land[free] = omega_land_free[free]
    pbl = p >= config.pbl_top_pressure_pa
    edges = pressure_interfaces(p)
    surface_pressure = edges[-1]
    omega_850 = np.interp(config.pbl_top_pressure_pa, edges, omega_land_edge)
    omega_land[pbl] = omega_850 * (
        (surface_pressure - p[pbl]) /
        (surface_pressure - config.pbl_top_pressure_pa)
    )
    omega_ocean = -omega_land
    if not np.allclose(omega_land + omega_ocean, 0.0, atol=1e-15, rtol=0.0):
        raise RuntimeError("Two-column WTG mass conservation failed")
    return omega_land, omega_ocean
