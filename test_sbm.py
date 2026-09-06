"""Focused numerical regression checks; run with python -m unittest -v test_sbm."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from climlab import constants as const
from climlab.utils.thermo import qsat

from dcm_physics import ColumnConfig, apply_abrupt_4xco2, create_column
from dcm_diagnostics import (
    DailyAccumulator, check_state, equilibrium_summary, layer_mass, snapshot,
    step_diagnostics,
)
from dcm_rundiag_0817 import (
    checked_physics_step, integrate_rce_daily, integrate_dcm_daily,
    rcm, run_dcm_spinup, run_abrupt4xco2, run_parameter_sweep, MAX_WORKERS,
)
from dcm_io import save_dataset, mean_rce_dataset
from dcm_wtg import (
    WTGConfig, advance_dcm_one_step, common_static_stability, pressure_interfaces,
    transport_moisture_pair,
)


class SBMTests(unittest.TestCase):
    def test_initial_state_and_shared_radiation_inputs(self):
        model = create_column(60, 1.)
        check_state(model)
        self.assertLessEqual(float(np.max(model.q / qsat(model.Tatm, model.lev))), 0.65000001)
        self.assertLess(float(model.Tatm[0]), float(model.Tatm[-1]))
        rad = model.subprocess["Radiation"]
        for proc in (rad, *rad.subprocess.values()):
            self.assertTrue(np.shares_memory(proc.specific_humidity, model.q))
            self.assertNotIn("q", proc.state)
            self.assertAlmostEqual(proc.absorber_vmr["CO2"], 300e-6)
        apply_abrupt_4xco2(model)
        for proc in (rad, *rad.subprocess.values()):
            self.assertAlmostEqual(proc.absorber_vmr["CO2"], 1200e-6)
        land = create_column(
            60, 1., lh_resistance=0.6,
            Tatm_init=model.Tatm.copy(), qatm_init=model.q.copy(),
        )
        self.assertEqual(land.subprocess["LHF"].resistance, 0.6)
        np.testing.assert_array_equal(land.q, model.q)

    def test_condensation_conserves_energy_and_precipitation(self):
        model = create_column(60, 1.)
        process = model.subprocess["Condensation"]
        model.q[:] = 1.1 * qsat(model.Tatm, model.lev)
        before = snapshot(model)
        tendency = process.compute()
        mass = layer_mass(model)
        self.assertGreater(float(np.asarray(process.precipitation).squeeze()), 0.)
        self.assertAlmostEqual(float(np.sum(mass * (const.cp * tendency["Tatm"] +
                                                    const.Lhvap * tendency["q"]))), 0., places=10)
        self.assertAlmostEqual(float(np.sum(mass * tendency["q"])) +
                               float(np.asarray(process.precipitation).squeeze()), 0., places=12)
        np.testing.assert_allclose(model.q, before["q"], atol=1e-16)
        model.q[:] = 0.65 * qsat(model.Tatm, model.lev)
        np.testing.assert_array_equal(process.compute()["q"], np.zeros_like(model.q))

    def test_day_budgets_and_io(self):
        model = create_column(30, 1.)
        with tempfile.TemporaryDirectory() as tmp:
            ds = integrate_rce_daily(model, 2, Path(tmp) / "failure.nc", progress=False)
            self.assertLess(float(ds.energy_residual_absmax.max()), 0.1)
            self.assertLess(float(ds.water_residual_absmax.max()), 1e-8)
            self.assertGreater(float(ds.cape.max()), 0.)
            self.assertEqual(ds.SBM_Tatm_tendency.shape, (2, 30))
            mean = mean_rce_dataset(ds, 2, equilibrium_summary(ds, 2))
            self.assertEqual(mean.Tatm.shape, (30,))
            self.assertFalse(mean.attrs["equilibrated"])
            path = save_dataset(ds, Path(tmp) / "nested" / "daily.nc")
            with xr.open_dataset(path, decode_times=False) as restored:
                xr.testing.assert_allclose(restored, ds)

    def test_failure_snapshot(self):
        model = create_column(30, 1.)
        model.q[0] = -1e-6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.nc"
            with self.assertRaises(ValueError):
                checked_physics_step(model, path)
            with xr.open_dataset(path) as ds:
                self.assertEqual(ds.attrs["stage"], "before physics")
                self.assertLess(float(ds.failed_q[0]), 0.)
                self.assertIn("before_Tatm", ds)

    def test_daily_extrema_are_not_averaged(self):
        acc = DailyAccumulator()
        acc.add(dict(Ts=290., RH_max=1.2, q_min=1e-5, budget_residual_absmax=2.))
        acc.add(dict(Ts=300., RH_max=0.8, q_min=2e-5, budget_residual_absmax=0.))
        result = acc.mean()
        self.assertEqual(result["Ts"], 295.)
        self.assertEqual(result["RH_max"], 1.2)
        self.assertEqual(result["budget_residual_absmax"], 2.)

    def test_asynchronous_budget(self):
        model = create_column(30, 1., config=ColumnConfig(radiation_dt=1800.))
        for _ in range(5):
            before = snapshot(model)
            model.step_forward()
            record = step_diagnostics(model, before)
            self.assertLess(abs(record["budget_residual"]), 1e-6)
            self.assertLess(abs(record["radiation_residual"]), 1e-6)

    def test_dcm_transport_is_accounted_separately(self):
        land = create_column(30, 1., lh_resistance=0.6)
        ocean = create_column(30, 20.)
        ds = integrate_dcm_daily(land, ocean, land.lev, 1, 1., 20., 0.6, "test")
        self.assertEqual(ds.Tatm.shape, (2, 1, 30))
        self.assertIn("energy_WTG", ds)
        sources = sum(ds[f"energy_{name}"] for name in land.subprocess) + ds.energy_WTG
        np.testing.assert_allclose(ds.energy_storage - sources, ds.budget_residual, atol=1e-7)
        np.testing.assert_allclose(ds.water_storage - ds.evaporation + ds.precipitation - ds.water_WTG,
                                   ds.water_residual, atol=1e-12)
        self.assertEqual(ds.water_WTG.attrs["units"], "kg m-2 s-1")
        self.assertEqual(ds.attrs["wtg_heat_capacity_factor"], 1.0)
        self.assertIn("Shaevitz-Sobel-2004", ds.attrs["wtg_method"])

    def test_wtg_mean_heating_projection_and_mass_conservation(self):
        land = create_column(30, 1., lh_resistance=0.6)
        ocean = create_column(30, 20.)
        p = land.lev * 100.
        config = WTGConfig()
        free = ((p >= config.top_pressure_pa) &
                (p < config.pbl_top_pressure_pa))
        before = (land.Tatm.copy(), ocean.Tatm.copy())
        land.Tatm[free] += 0.03
        ocean.Tatm[free] -= 0.01
        expected = 0.5 * (land.Tatm.copy() + ocean.Tatm.copy())
        mass = layer_mass(land)
        thermal_before = float(np.sum(mass * const.cp * (land.Tatm + ocean.Tatm)))
        omega_land, omega_ocean = advance_dcm_one_step(
            land, ocean, p, 600., Tatm_before=before,
            config=config, advance_physics=False,
        )
        np.testing.assert_allclose(land.Tatm[free], expected[free], atol=1e-13)
        np.testing.assert_allclose(ocean.Tatm[free], expected[free], atol=1e-13)
        np.testing.assert_allclose(omega_land, -omega_ocean, atol=0., rtol=0.)
        stability = common_static_stability(expected, p)
        np.testing.assert_allclose(
            omega_land[free], (0.02 / 600.) / stability[free], rtol=1e-12,
        )
        self.assertGreater(float(np.max(np.abs(omega_land[free]))), 0.)
        thermal_after = float(np.sum(mass * const.cp * (land.Tatm + ocean.Tatm)))
        self.assertAlmostEqual(thermal_after, thermal_before, places=7)

    def test_wtg_flux_form_moisture_is_positive_and_conservative(self):
        p = np.array([12500., 37500., 62500., 87500.])
        edges = pressure_interfaces(p)
        omega = np.array([0., -0.12, 0.08, 0.04, 0.])
        uniform = np.full(4, 0.01)
        q_land, q_ocean, _ = transport_moisture_pair(
            uniform, uniform, omega, p, 3600., cfl=0.45,
        )
        np.testing.assert_allclose(q_land, uniform, atol=1e-15)
        np.testing.assert_allclose(q_ocean, uniform, atol=1e-15)

        initial_land = np.array([1e-6, 3e-5, 2e-3, 1.5e-2])
        initial_ocean = np.array([2e-6, 5e-5, 3e-3, 1.8e-2])
        initial_water = np.sum(np.diff(edges) * (initial_land + initial_ocean))
        q_land, q_ocean, nsubsteps = transport_moisture_pair(
            initial_land, initial_ocean, omega, p, 86400., cfl=0.45,
        )
        final_water = np.sum(np.diff(edges) * (q_land + q_ocean))
        self.assertGreater(nsubsteps, 1)
        self.assertGreaterEqual(float(np.min(q_land)), 0.)
        self.assertGreaterEqual(float(np.min(q_ocean)), 0.)
        self.assertAlmostEqual(float(final_water), float(initial_water), places=10)

    def test_original_rce_and_experiment_entry_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rce_mean.nc"
            with self.assertWarns(UserWarning):
                temperature, humidity, surface, lev, mean = rcm(
                    30, 2, path, mean_days=2,
                )
            self.assertTrue(path.exists())
            self.assertTrue(path.with_name("rce_mean_daily.nc").exists())
            self.assertFalse(mean.attrs["equilibrated"])
            control, state = run_dcm_spinup(
                30, temperature, humidity, surface, lev, spinup_days=2,
                restart_mean_days=2, ocean_MLD=20., lh_r=0.6,
            )
            forced = run_abrupt4xco2(
                30, state, lev, forced_days=1, ocean_MLD=20., lh_r=0.6,
                restart_mean_days=2,
            )
            self.assertAlmostEqual(control.attrs["co2_ppm"], 300.)
            self.assertAlmostEqual(forced.attrs["co2_ppm"], 1200.)
            np.testing.assert_array_equal(forced.control_q, state["q"])

    def test_parallel_worker_limit_and_empty_sweep(self):
        model = create_column(10, 1.)
        arguments = dict(
            num_lev=10, Tatm_rce=model.Tatm.copy(), qatm_rce=model.q.copy(),
            Ts_rce=float(np.asarray(model.Ts).squeeze()), lev=model.lev.copy(),
            ocean_MLD_list=(), spinup_days=2, restart_mean_days=2, forced_days=1,
        )
        with self.assertRaises(ValueError):
            run_parameter_sweep(**arguments, workers=MAX_WORKERS + 1)
        with self.assertRaises(TypeError):
            run_parameter_sweep(**arguments, workers=2.0)
        self.assertEqual(run_parameter_sweep(**arguments, workers=MAX_WORKERS), [])


if __name__ == "__main__":
    unittest.main()
