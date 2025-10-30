#!/usr/bin/env python3
import sys
import importlib.util

# Import kd_v1-0.py
spec = importlib.util.spec_from_file_location(
    "kd_v1_0",
    "/Users/merplateau/Projects/18Oct-Share/PythonScripts/kd_v1-0.py"
)
kd_module = importlib.util.module_from_spec(spec)
sys.modules["kd_v1_0"] = kd_module
spec.loader.exec_module(kd_module)
solve_dispersion_relation = kd_module.solve_dispersion_relation

# Test initial conditions
B0 = 1.7
n0 = 1e18
T0_eV = 3.0
v0 = 10000.0
nu_coll = 5e3
omega = 3.3e6
Z = 40

print("Testing initial point wave number solver...")
print(f"B0 = {B0} T")
print(f"n0 = {n0} m^-3")
print(f"T0 = {T0_eV} eV")
print(f"v0 = {v0} m/s")
print(f"nu = {nu_coll} rad/s")
print(f"omega = {omega} rad/s")

try:
    results = solve_dispersion_relation(
        B0=B0,
        n_target=n0,
        T_par_eV=T0_eV,
        T_perp_eV=T0_eV,
        v_target=v0,
        nu_target=nu_coll,
        w_target=omega,
        Z=Z,
        target_steps_s1=20,
        target_steps_s2=12
    )
    
    print(f"\nSuccess! k_final = {results['k_final']}")
    print(f"gamma = {results['k_final'].imag}")
    
except Exception as e:
    print(f"\nFailed with error: {e}")
    import traceback
    traceback.print_exc()
