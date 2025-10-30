#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D VASIMR ICRH Simulation - Initial Value Problem
Uses hot plasma dispersion relation solver

Author: Claude Code
Date: 2025
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import os

# Add path to wave number solver
sys.path.append('/Users/merplateau/Projects/18Oct-Share/PythonScripts')

# Import module with hyphen in name using importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "kd_v1_0",
    "/Users/merplateau/Projects/18Oct-Share/PythonScripts/kd_v1-0.py"
)
kd_module = importlib.util.module_from_spec(spec)
sys.modules["kd_v1_0"] = kd_module
spec.loader.exec_module(kd_module)
solve_dispersion_relation = kd_module.solve_dispersion_relation

# ============================================================================
# Physical Constants
# ============================================================================
class PhysicalConstants:
    """Physical constants in SI units"""
    c = 3e8              # Speed of light [m/s]
    e = 1.602e-19        # Elementary charge [C]
    kB = 1.38e-23        # Boltzmann constant [J/K]
    eps0 = 8.854e-12     # Vacuum permittivity [F/m]
    m_p = 1.67e-27       # Proton mass [kg]
    m_e = 9.109e-31      # Electron mass [kg]

constants = PhysicalConstants()

# ============================================================================
# Simulation Parameters
# ============================================================================

# Spatial domain
z0 = 0.0              # Start position [m]
z1 = 0.1              # Middle position [m]
z2 = 0.3              # End position [m]

# Magnetic field boundary conditions
B_z1 = 1.4            # Magnetic field at z1 [T]
B_z2 = 0.8            # Magnetic field at z2 [T]

# Calculate linear magnetic field parameters: B(z) = B0 + alpha * z
# Using two boundary conditions:
# B(z1) = B0 + alpha * z1 = 1.4
# B(z2) = B0 + alpha * z2 = 0.8
alpha = (B_z2 - B_z1) / (z2 - z1)  # = (0.8 - 1.4) / (0.3 - 0.1) = -3
B0 = B_z1 - alpha * z1             # = 1.4 - (-3) * 0.1 = 1.7 T

print(f"Magnetic field parameters: B(z) = {B0:.2f} + ({alpha:.2f})*z")
print(f"Verification: B({z1}) = {B0 + alpha*z1:.2f} T, B({z2}) = {B0 + alpha*z2:.2f} T")

# Initial conditions
v0 = 10000.0          # Inlet flow velocity [m/s]
T0_eV = 3.0           # Inlet temperature [eV]
n0 = 1e18             # Inlet density [m^-3]
P_in = 1000.0        # Input power [W]

# Physical parameters
omega = 3.3e6         # Drive frequency [rad/s]
nu_coll = 5e3         # Collision frequency [rad/s]
Z = 40                # Ion charge number (Argon)
m_i = Z * constants.m_p  # Ion mass [kg]

# Calculate flux invariant (particle flux conservation)
B_initial = B0 + alpha * z0
Gamma0 = (n0 * v0) / B_initial

print(f"\nInitial conditions:")
print(f"  v0 = {v0:.2e} m/s")
print(f"  T0 = {T0_eV:.2f} eV")
print(f"  n0 = {n0:.2e} m^-3")
print(f"  P_in = {P_in:.2e} W")
print(f"  B0 = {B_initial:.2f} T")
print(f"  Gamma0 = {Gamma0:.2e} (m^-3 T m/s)")

# ============================================================================
# Magnetic Field Functions
# ============================================================================

def B_field(z):
    """
    Magnetic field strength as a function of z

    Args:
        z: Position [m]

    Returns:
        B: Magnetic field strength [T]
    """
    return B0 + alpha * z

def dB_dz(z):
    """
    Magnetic field gradient

    Args:
        z: Position [m]

    Returns:
        dB/dz: Magnetic field gradient [T/m]
    """
    return alpha

# ============================================================================
# Auxiliary Functions
# ============================================================================

def calculate_density(v_parallel, B):
    """
    Calculate density from flux conservation

    n(z) = Gamma0 * B(z) / v_parallel(z)
    """
    return (Gamma0 * B) / v_parallel

def calculate_collision_term(n, T_perp, T_parallel):
    """
    Calculate collision term Q_coll

    Q_coll = nu_coll * n * (T_perp - T_parallel)
    """
    return nu_coll * n * (T_perp - T_parallel)

# Cache for wave solver results
gamma_cache = {}
last_successful_gamma = 0.004  # Default fallback value

def calculate_wave_damping(B, n, T_perp, T_parallel, v):
    """
    Call wave number solver to calculate damping rate gamma = Im(k_parallel)
    With caching and timeout protection

    Args:
        B: Magnetic field strength [T]
        n: Density [m^-3]
        T_perp: Perpendicular temperature [eV]
        T_parallel: Parallel temperature [eV]
        v: Flow velocity [m/s]

    Returns:
        gamma: Damping rate (i.e. Im(k_parallel)) [1/m]
    """
    global last_successful_gamma

    # Create cache key (rounded to avoid floating point precision issues)
    cache_key = (round(B, 3), round(n/1e17, 2), round(T_perp, 2),
                 round(T_parallel, 2), round(v/1000, 1))

    # Check cache first
    if cache_key in gamma_cache:
        return gamma_cache[cache_key]

    try:
        # Redirect stdout to suppress verbose output
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Call wave number solver with reduced verbosity
            results = solve_dispersion_relation(
                B0=B,
                n_target=n,
                T_par_eV=T_parallel,
                T_perp_eV=T_perp,
                v_target=v,
                nu_target=nu_coll,
                w_target=omega,
                Z=Z,
                target_steps_s1=15,  # Reduced from 20
                target_steps_s2=10   # Reduced from 12
            )

        # Extract damping rate
        k_final = results['k_final']
        gamma = np.imag(k_final)

        # Cache the result
        gamma_cache[cache_key] = gamma
        last_successful_gamma = gamma

        return gamma

    except Exception as e:
        print(f"  Warning: Wave solver failed, using last successful gamma={last_successful_gamma:.6f}")
        # Return last successful gamma
        return last_successful_gamma

def calculate_volume_heating_rate(gamma, P, B):
    """
    Calculate volume heating rate Q_perp

    Q_perp = (2 * gamma * P * B) / B_initial
    """
    return (2 * gamma * P * B) / B_initial

# ============================================================================
# ODE System Derivative Function
# ============================================================================

# Global variables for caching and debugging
call_count = 0
last_z = -1

def derivative_function(z, Y):
    """
    Derivative function for ODE system dY/dz = F(z, Y)

    State vector Y = [v_parallel, T_perp, T_parallel, P]

    Args:
        z: Current position [m]
        Y: State vector [v_parallel (m/s), T_perp (eV), T_parallel (eV), P (W)]

    Returns:
        dY_dz: Derivative vector [dv/dz, dT_perp/dz, dT_parallel/dz, dP/dz]
    """
    global call_count, last_z
    call_count += 1

    # Unpack state variables
    v_parallel = Y[0]
    T_perp = Y[1]
    T_parallel = Y[2]
    P = Y[3]

    # Print progress (every certain distance)
    if abs(z - last_z) > 0.01 or call_count == 1:
        print(f"\nSolving... z = {z:.4f} m, v = {v_parallel:.2e} m/s, "
              f"T_perp = {T_perp:.2f} eV, T_par = {T_parallel:.2f} eV, P = {P:.2e} W")
        last_z = z

    # Step 1: Calculate auxiliary quantities
    B = B_field(z)
    dBdz = dB_dz(z)
    n = calculate_density(v_parallel, B)

    # Prevent negative temperatures and other non-physical values
    if T_perp <= 0 or T_parallel <= 0 or v_parallel <= 0 or P < 0:
        print(f"Warning: Non-physical values detected at z={z:.4f}")
        return np.zeros(4)

    Q_coll = calculate_collision_term(n, T_perp, T_parallel)
    gamma = calculate_wave_damping(B, n, T_perp, T_parallel, v_parallel)
    Q_perp = calculate_volume_heating_rate(gamma, P, B)

    RHS_coll_perp = (Q_perp - Q_coll) / n
    RHS_coll_para = Q_coll / n

    # Step 2: Build 3x3 linear system
    # Right-hand side vector R
    R = np.array([
        -(T_perp / B) * dBdz,
        RHS_coll_perp - (2 * v_parallel * T_perp / B) * dBdz,
        RHS_coll_para + (v_parallel * T_parallel / B) * dBdz
    ])

    # Coefficient matrix M
    # Note: T_perp and T_parallel are in eV, need to convert to energy units (J)
    T_perp_J = T_perp * constants.e
    T_parallel_J = T_parallel * constants.e

    M = np.array([
        [(m_i * v_parallel**2 - T_parallel_J) / v_parallel, 0, 1 * constants.e],
        [-T_perp_J, v_parallel * constants.e, 0],
        [-T_parallel_J, 0, v_parallel * constants.e]
    ])

    # Step 3: Solve for fluid derivatives
    try:
        Y_fluid_prime = np.linalg.solve(M, R)
        dv_dz = Y_fluid_prime[0]
        dT_perp_dz = Y_fluid_prime[1]
        dT_parallel_dz = Y_fluid_prime[2]
    except np.linalg.LinAlgError:
        print(f"Warning: Linear system solve failed at z={z:.4f}")
        dv_dz = 0
        dT_perp_dz = 0
        dT_parallel_dz = 0

    # Step 4: Calculate wave derivative
    dP_dz = -2 * gamma * P

    return np.array([dv_dz, dT_perp_dz, dT_parallel_dz, dP_dz])

# ============================================================================
# Initial Value Problem Solution
# ============================================================================

def solve_1D_VASIMR():
    """
    Solve 1D VASIMR ICRH model as an initial value problem

    Returns:
        solution: solve_ivp solution object
    """
    # Set initial conditions
    Y0 = np.array([v0, T0_eV, T0_eV, P_in])

    # Set solution range
    z_span = [z0, z2]

    # Set solution parameters
    # Use RK45 method with appropriate tolerance
    print("\n" + "="*70)
    print("Starting 1D VASIMR ICRH Model Solution")
    print("="*70)

    solution = solve_ivp(
        derivative_function,
        z_span,
        Y0,
        method='RK45',
        dense_output=True,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.01  # Maximum step size 1 cm
    )

    if solution.success:
        print("\n" + "="*70)
        print("Solution Successful!")
        print("="*70)
        print(f"Total derivative function calls: {call_count}")
        print(f"Generated {len(solution.t)} solution points")
    else:
        print("\n" + "="*70)
        print("Solution Failed!")
        print("="*70)
        print(f"Error message: {solution.message}")

    return solution

# ============================================================================
# Post-processing and Visualization
# ============================================================================

def post_process_and_plot(solution):
    """
    Post-process and visualize results

    Args:
        solution: solve_ivp solution object
    """
    # Extract solution
    z_points = solution.t
    Y_solution = solution.y

    v_parallel = Y_solution[0, :]
    T_perp = Y_solution[1, :]
    T_parallel = Y_solution[2, :]
    P = Y_solution[3, :]

    # Calculate derived quantities
    B_values = B_field(z_points)
    n_values = calculate_density(v_parallel, B_values)

    # Calculate thrust (approximate)
    thrust = m_i * n_values[-1] * v_parallel[-1]**2  # Simplified formula

    print(f"\nFinal state (z = {z_points[-1]:.3f} m):")
    print(f"  Flow velocity: {v_parallel[-1]:.2e} m/s")
    print(f"  T_perp: {T_perp[-1]:.2f} eV")
    print(f"  T_parallel: {T_parallel[-1]:.2f} eV")
    print(f"  Power: {P[-1]:.2e} W")
    print(f"  Density: {n_values[-1]:.2e} m^-3")
    print(f"  Magnetic field: {B_values[-1]:.2f} T")
    print(f"  Thrust density: {thrust:.2e} N/m^2")

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Flow velocity distribution
    axes[0, 0].plot(z_points * 100, v_parallel / 1000, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Position z [cm]')
    axes[0, 0].set_ylabel('Flow velocity [km/s]')
    axes[0, 0].set_title('Ion Flow Velocity vs Position')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Temperature distribution
    axes[0, 1].plot(z_points * 100, T_perp, 'r-', linewidth=2, label='T_perp')
    axes[0, 1].plot(z_points * 100, T_parallel, 'b-', linewidth=2, label='T_parallel')
    axes[0, 1].set_xlabel('Position z [cm]')
    axes[0, 1].set_ylabel('Temperature [eV]')
    axes[0, 1].set_title('Ion Temperature vs Position')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Power distribution
    axes[1, 0].plot(z_points * 100, P / 1000, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Position z [cm]')
    axes[1, 0].set_ylabel('Wave Power [kW]')
    axes[1, 0].set_title('Wave Power vs Position')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Density distribution
    axes[1, 1].plot(z_points * 100, n_values / 1e18, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Position z [cm]')
    axes[1, 1].set_ylabel('Density [10^18 m^-3]')
    axes[1, 1].set_title('Plasma Density vs Position')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Magnetic field distribution
    axes[2, 0].plot(z_points * 100, B_values, 'c-', linewidth=2)
    axes[2, 0].set_xlabel('Position z [cm]')
    axes[2, 0].set_ylabel('Magnetic Field [T]')
    axes[2, 0].set_title('Magnetic Field vs Position')
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Temperature anisotropy
    anisotropy = T_perp / T_parallel
    axes[2, 1].plot(z_points * 100, anisotropy, 'k-', linewidth=2)
    axes[2, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].set_xlabel('Position z [cm]')
    axes[2, 1].set_ylabel('T_perp / T_parallel')
    axes[2, 1].set_title('Temperature Anisotropy vs Position')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = '/Users/merplateau/Projects/18Oct-Share/PythonScripts1DTry'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '1D_VASIMR_results.png'), dpi=300)
    print(f"\nFigure saved to: {output_dir}/1D_VASIMR_results.png")

    plt.show()

    return {
        'z': z_points,
        'v_parallel': v_parallel,
        'T_perp': T_perp,
        'T_parallel': T_parallel,
        'P': P,
        'n': n_values,
        'B': B_values
    }

# ============================================================================
# Main Program
# ============================================================================

if __name__ == '__main__':
    # Solve
    solution = solve_1D_VASIMR()

    # Post-process and visualize
    if solution.success:
        results = post_process_and_plot(solution)
        print("\nSimulation complete!")
    else:
        print("\nSimulation failed, please check parameters and boundary conditions.")
