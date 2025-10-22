# -*- coding: utf-8 -*-
"""
Cold Plasma Dielectric Tensor Calculator v1.0

Calculate dielectric tensor components under cold plasma approximation:
- K_perp: Perpendicular component
- K_phi (K_g): Gyro component
- K_par: Parallel component

Reference formulas:
    K_perp = 1 - Sum[omega_pl_tilde^2 / (omega^2 - omega_cl_tilde^2)]
    K_phi = Sum[omega_pl_tilde^2 * omega_cl_tilde / (omega*(omega^2 - omega_cl_tilde^2))]
    K_par = 1 - Sum[omega_pl_tilde^2 / omega^2]

    where:
    omega_pl_tilde = sqrt(q_l^2 * n_l / (eps0 * m_l_nu))
    omega_cl_tilde = q_l * B0 / m_l_nu
    m_l_nu = m_l * (1 + i*nu_l/omega)

Based on kd_v1-0.py architecture
Date: 2025
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Physical Constants
# ============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants (SI units)"""
    c: float = 3e8              # Speed of light [m/s]
    e: float = 1.602e-19        # Elementary charge [C]
    eps0: float = 8.854e-12     # Vacuum permittivity [F/m]
    m_p: float = 1.67e-27       # Proton mass [kg]
    m_e: float = 9.109e-31      # Electron mass [kg]


# ============================================================================
# Cold Plasma Parameters
# ============================================================================

@dataclass(frozen=True)
class ColdPlasmaParameters:
    """
    Cold plasma parameters (immutable)

    Cold plasma approximation: ignore thermal motion and drift velocity
    """
    n_i: float          # Ion density [m^-3]
    n_e: float          # Electron density [m^-3]
    nu_i: float         # Ion collision frequency [rad/s]
    nu_e: float         # Electron collision frequency [rad/s]
    B0: float           # Magnetic field strength [T]
    Z: int              # Ion charge number

    def __post_init__(self):
        """Parameter validation"""
        assert self.n_i > 0, f"Ion density must be positive: {self.n_i}"
        assert self.n_e > 0, f"Electron density must be positive: {self.n_e}"
        assert self.nu_i >= 0, f"Ion collision frequency cannot be negative: {self.nu_i}"
        assert self.nu_e >= 0, f"Electron collision frequency cannot be negative: {self.nu_e}"
        assert self.B0 > 0, f"Magnetic field must be positive: {self.B0}"
        assert self.Z > 0, f"Charge number must be positive integer: {self.Z}"


# ============================================================================
# Cold Plasma Dielectric Tensor Calculation
# ============================================================================

class ColdPlasmaDielectric:
    """
    Cold plasma dielectric tensor calculation (pure functions, stateless)
    """

    def __init__(self, constants: Optional[PhysicalConstants] = None):
        self.const = constants or PhysicalConstants()

    def compute_effective_mass(self, m: float, nu: float, omega: complex) -> complex:
        """
        Compute effective mass m_nu = m*(1 + i*nu/omega)

        Args:
            m: Particle mass [kg]
            nu: Collision frequency [rad/s]
            omega: Angular frequency [rad/s]

        Returns:
            Effective mass [kg]
        """
        return m * (1.0 + 1j * nu / omega)

    def compute_plasma_frequency(self, n: float, q: float, m_eff: complex) -> complex:
        """
        Compute modified plasma frequency omega_p_tilde = sqrt(q^2 * n / (eps0 * m_nu))

        Args:
            n: Particle density [m^-3]
            q: Particle charge (with sign) [C]
            m_eff: Effective mass [kg]

        Returns:
            Modified plasma frequency [rad/s]
        """
        return np.sqrt(q**2 * n / (self.const.eps0 * m_eff))

    def compute_cyclotron_frequency(self, q: float, B0: float, m_eff: complex) -> complex:
        """
        Compute modified cyclotron frequency omega_c_tilde = q * B0 / m_nu

        Args:
            q: Particle charge (with sign) [C]
            B0: Magnetic field strength [T]
            m_eff: Effective mass [kg]

        Returns:
            Modified cyclotron frequency [rad/s]
        """
        return q * B0 / m_eff

    def compute_K_perp(self, omega: complex, params: ColdPlasmaParameters) -> complex:
        """
        Compute perpendicular dielectric tensor component K_perp

        K_perp = 1 - Sum[omega_pl_tilde^2 / (omega^2 - omega_cl_tilde^2)]

        Args:
            omega: Angular frequency [rad/s]
            params: Cold plasma parameters

        Returns:
            K_perp
        """
        K_perp = 1.0 + 0j

        # Ion mass
        m_i = params.Z * self.const.m_p

        # Ion contribution
        m_i_eff = self.compute_effective_mass(m_i, params.nu_i, omega)
        omega_pi = self.compute_plasma_frequency(params.n_i, params.Z * self.const.e, m_i_eff)
        omega_ci = self.compute_cyclotron_frequency(params.Z * self.const.e, params.B0, m_i_eff)
        K_perp -= omega_pi**2 / (omega**2 - omega_ci**2)

        # Electron contribution
        m_e_eff = self.compute_effective_mass(self.const.m_e, params.nu_e, omega)
        omega_pe = self.compute_plasma_frequency(params.n_e, -self.const.e, m_e_eff)
        omega_ce = self.compute_cyclotron_frequency(-self.const.e, params.B0, m_e_eff)
        K_perp -= omega_pe**2 / (omega**2 - omega_ce**2)

        return K_perp

    def compute_K_phi(self, omega: complex, params: ColdPlasmaParameters) -> complex:
        """
        Compute gyro dielectric tensor component K_phi (i.e. K_g)

        K_phi = Sum[omega_pl_tilde^2 * omega_cl_tilde / (omega*(omega^2 - omega_cl_tilde^2))]

        Args:
            omega: Angular frequency [rad/s]
            params: Cold plasma parameters

        Returns:
            K_phi (K_g)
        """
        K_phi = 0.0 + 0j

        # Ion mass
        m_i = params.Z * self.const.m_p

        # Ion contribution
        m_i_eff = self.compute_effective_mass(m_i, params.nu_i, omega)
        omega_pi = self.compute_plasma_frequency(params.n_i, params.Z * self.const.e, m_i_eff)
        omega_ci = self.compute_cyclotron_frequency(params.Z * self.const.e, params.B0, m_i_eff)
        K_phi += omega_pi**2 * omega_ci / (omega * (omega**2 - omega_ci**2))

        # Electron contribution
        m_e_eff = self.compute_effective_mass(self.const.m_e, params.nu_e, omega)
        omega_pe = self.compute_plasma_frequency(params.n_e, -self.const.e, m_e_eff)
        omega_ce = self.compute_cyclotron_frequency(-self.const.e, params.B0, m_e_eff)
        K_phi += omega_pe**2 * omega_ce / (omega * (omega**2 - omega_ce**2))

        return K_phi

    def compute_K_par(self, omega: complex, params: ColdPlasmaParameters) -> complex:
        """
        Compute parallel dielectric tensor component K_par

        K_par = 1 - Sum[omega_pl_tilde^2 / omega^2]

        Args:
            omega: Angular frequency [rad/s]
            params: Cold plasma parameters

        Returns:
            K_par
        """
        K_par = 1.0 + 0j

        # Ion mass
        m_i = params.Z * self.const.m_p

        # Ion contribution
        m_i_eff = self.compute_effective_mass(m_i, params.nu_i, omega)
        omega_pi = self.compute_plasma_frequency(params.n_i, params.Z * self.const.e, m_i_eff)
        K_par -= omega_pi**2 / omega**2

        # Electron contribution
        m_e_eff = self.compute_effective_mass(self.const.m_e, params.nu_e, omega)
        omega_pe = self.compute_plasma_frequency(params.n_e, -self.const.e, m_e_eff)
        K_par -= omega_pe**2 / omega**2

        return K_par

    def compute_all_components(self, omega: complex, params: ColdPlasmaParameters) -> dict:
        """
        Compute all dielectric tensor components

        Args:
            omega: Angular frequency [rad/s]
            params: Cold plasma parameters

        Returns:
            Dictionary containing K_perp, K_phi, K_par
        """
        return {
            'K_perp': self.compute_K_perp(omega, params),
            'K_phi': self.compute_K_phi(omega, params),
            'K_par': self.compute_K_par(omega, params)
        }


# ============================================================================
# Convenience Interface Functions
# ============================================================================

def compute_cold_plasma_dielectric_ion_only(
    omega: float,
    B0: float,
    n_i: float,
    nu_i: float = 0.0,
    Z: int = 40
) -> dict:
    """
    Compute cold plasma dielectric tensor components (ION ONLY, no electron contribution)

    This is appropriate for low-frequency waves where ion dynamics dominate.

    Args:
        omega: Angular frequency [rad/s]
        B0: Magnetic field strength [T]
        n_i: Ion density [m^-3]
        nu_i: Ion collision frequency [rad/s] (default 0)
        Z: Ion charge number (default 40)

    Returns:
        Dictionary containing K_perp, K_phi (K_g), K_par (ion contribution only)
    """
    const = PhysicalConstants()
    m_i = Z * const.m_p

    # Effective mass with collision
    m_i_eff = m_i * (1.0 + 1j * nu_i / omega)

    # Modified plasma and cyclotron frequencies
    omega_pi = np.sqrt((Z * const.e)**2 * n_i / (const.eps0 * m_i_eff))
    omega_ci = (Z * const.e) * B0 / m_i_eff

    # Calculate dielectric tensor components (ion only)
    K_perp = 1.0 - omega_pi**2 / (omega**2 - omega_ci**2)
    K_phi = omega_pi**2 * omega_ci / (omega * (omega**2 - omega_ci**2))
    K_par = 1.0 - omega_pi**2 / omega**2

    return {
        'K_perp': K_perp,
        'K_phi': K_phi,
        'K_par': K_par
    }


def compute_cold_plasma_dielectric(
    omega: float,
    B0: float,
    n_i: float,
    nu_i: float = 0.0,
    n_e: Optional[float] = None,
    nu_e: float = 0.0,
    Z: int = 40
) -> dict:
    """
    Compute cold plasma dielectric tensor components (convenience interface)

    Args:
        omega: Angular frequency [rad/s]
        B0: Magnetic field strength [T]
        n_i: Ion density [m^-3]
        nu_i: Ion collision frequency [rad/s] (default 0)
        n_e: Electron density [m^-3] (default Z * n_i, quasi-neutrality)
        nu_e: Electron collision frequency [rad/s] (default 0)
        Z: Ion charge number (default 40)

    Returns:
        Dictionary containing K_perp, K_phi (K_g), K_par
    """
    # Quasi-neutrality condition
    if n_e is None:
        n_e = Z * n_i

    # Create parameter object
    params = ColdPlasmaParameters(
        n_i=n_i,
        n_e=n_e,
        nu_i=nu_i,
        nu_e=nu_e,
        B0=B0,
        Z=Z
    )

    # Calculate dielectric tensor
    calculator = ColdPlasmaDielectric()
    return calculator.compute_all_components(omega, params)


# ============================================================================
# Main Program (Test Example)
# ============================================================================

if __name__ == '__main__':
    # Test parameters
    omega = 3.3e6      # Frequency [rad/s]
    B0 = 1.42           # Magnetic field [T]
    n_i = 1e18         # Ion density [m^-3]
    nu_i = 1e5         # Ion collision frequency [rad/s]
    Z = 40             # Charge number

    print("="*70)
    print("Cold Plasma Dielectric Tensor Calculation Test")
    print("="*70)
    print(f"Parameters:")
    print(f"  omega = {omega:.2e} rad/s")
    print(f"  B0 = {B0:.2f} T")
    print(f"  n_i = {n_i:.2e} m^-3")
    print(f"  nu_i = {nu_i:.2e} rad/s")
    print(f"  Z = {Z}")
    print(f"  n_e = {Z * n_i:.2e} m^-3 (quasi-neutrality)")
    print("="*70)

    # Calculate
    result = compute_cold_plasma_dielectric(
        omega=omega,
        B0=B0,
        n_i=n_i,
        nu_i=nu_i,
        Z=Z
    )

    print(f"\nDielectric tensor components:")
    print(f"  K_perp = {result['K_perp'].real:.6f} + {result['K_perp'].imag:.6f}j")
    print(f"  K_phi = {result['K_phi'].real:.6f} + {result['K_phi'].imag:.6f}j")
    print(f"  K_par = {result['K_par'].real:.6f} + {result['K_par'].imag:.6f}j")
    print("="*70)
