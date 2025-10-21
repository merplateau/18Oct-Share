# -*- coding: utf-8 -*-
"""
热等离子体动理学色散关系求解器 - 重构版本 v1.0

核心改进：
1. 不可变参数设计（PlasmaParameters）
2. 动态派生量计算（PlasmaState 使用 @property）
3. 纯函数式物理计算（DispersionFunction）
4. 解耦的同伦控制器（HomotopyContinuation）
5. 完整的介电张量计算

作者：重构自 1012-claude-1+v-4.py
日期：2025
"""

from __future__ import annotations  # 支持 Python 3.7+ 的类型注解

import numpy as np
from scipy import special, optimize
from dataclasses import dataclass, replace
from typing import Optional, Protocol, Callable
from abc import ABC, abstractmethod
import time


# ============================================================================
# 第一部分：物理常数和参数容器
# ============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """物理常数（SI单位制）"""
    c: float = 3e8              # 光速 [m/s]
    e: float = 1.602e-19        # 元电荷 [C]
    kB: float = 1.38e-23        # 玻尔兹曼常数 [J/K]
    eps0: float = 8.854e-12     # 真空介电常数 [F/m]
    m_p: float = 1.67e-27       # 质子质量 [kg]


@dataclass(frozen=True)
class PlasmaParameters:
    """
    等离子体参数（不可变）

    所有参数采用SI单位：
    - 密度: m^-3
    - 温度: K（不是eV！）
    - 速度: m/s
    - 频率: rad/s
    """
    n: float          # 密度 [m^-3]
    T_par: float      # 平行温度 [K]
    T_perp: float     # 垂直温度 [K]
    v: float          # 漂移速度 [m/s]
    nu: float         # 碰撞频率 [rad/s]

    def __post_init__(self):
        """参数合法性验证"""
        assert self.n > 0, f"密度必须为正: {self.n}"
        assert self.T_par > 0, f"平行温度必须为正: {self.T_par}"
        assert self.T_perp > 0, f"垂直温度必须为正: {self.T_perp}"
        assert self.nu >= 0, f"碰撞频率不能为负: {self.nu}"

    def evolve(self, **changes) -> 'PlasmaParameters':
        """创建新参数对象（不可变更新模式）"""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        """转换为字典（用于绘图等场景）"""
        return {
            'n': self.n,
            'T_par': self.T_par,
            'T_perp': self.T_perp,
            'v': self.v,
            'nu': self.nu
        }


# ============================================================================
# 第二部分：等离子体状态（参数 + 派生量）
# ============================================================================

class PlasmaState:
    """
    等离子体状态：封装参数和派生量

    派生量通过 @property 动态计算，无需手动更新
    """

    def __init__(self, params: PlasmaParameters, B0: float, Z: int,
                 constants: PhysicalConstants = None):
        """
        Args:
            params: 等离子体参数
            B0: 磁场强度 [T]
            Z: 离子电荷数
            constants: 物理常数（默认使用标准值）
        """
        self._params = params
        self._B0 = B0
        self._Z = Z
        self._const = constants or PhysicalConstants()
        self._m_i = self._Z * self._const.m_p  # 离子质量
        self._q_i = self._const.e              # 离子电荷

    @property
    def params(self) -> PlasmaParameters:
        """当前等离子体参数"""
        return self._params

    @property
    def B0(self) -> float:
        """磁场强度"""
        return self._B0

    @property
    def constants(self) -> PhysicalConstants:
        """物理常数"""
        return self._const

    @property
    def w_p(self) -> float:
        """等离子体频率 [rad/s]"""
        return np.sqrt(self._params.n * self._q_i**2 /
                      (self._const.eps0 * self._m_i))

    @property
    def w_c(self) -> float:
        """回旋频率 [rad/s]"""
        return self._q_i * self._B0 / self._m_i

    @property
    def v_th_par(self) -> float:
        """平行热速度 [m/s]"""
        return np.sqrt(2 * self._const.kB * self._params.T_par / self._m_i)

    @property
    def v_th_perp(self) -> float:
        """垂直热速度 [m/s]"""
        return np.sqrt(2 * self._const.kB * self._params.T_perp / self._m_i)

    def with_params(self, new_params: PlasmaParameters) -> 'PlasmaState':
        """创建新状态（不可变更新）"""
        return PlasmaState(new_params, self._B0, self._Z, self._const)

    def __repr__(self):
        return (f"PlasmaState(n={self._params.n:.2e}, "
                f"T_par={self._params.T_par*self._const.kB/self._const.e:.1f}eV, "
                f"v={self._params.v:.1e}, nu={self._params.nu:.1e})")


# ============================================================================
# 第三部分：色散函数和介电张量（纯函数）
# ============================================================================

class DispersionFunction:
    """
    色散关系和介电张量计算（纯函数，无状态）

    所有方法都是静态方法，输入确定输出，无副作用
    """

    @staticmethod
    def plasma_dispersion_Z(zeta: complex) -> complex:
        """
        等离子体色散函数 Z(ζ) = i√π W(ζ)

        其中 W(ζ) 是 Faddeeva 函数
        """
        return 1j * np.sqrt(np.pi) * special.wofz(zeta)
    
    @staticmethod
    def plasma_dispersion_Z_adapt(zeta: complex) -> complex:
        """
        等离子体色散函数 Z_(ζ)
        """
        if np.imag(zeta) >= 0:
            Z_val = 1j * np.sqrt(np.pi) * special.wofz(zeta)
        else:
            zeta = np.conj(zeta)
            Z_val = 1j * np.sqrt(np.pi) * special.wofz(zeta)
            Z_val = np.conj(Z_val)
        return Z_val

    @staticmethod
    def compute_zeta_plus1(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算 ζ_+1 = (ω + iν - k∥v - ωc) / (k∥v_th,∥)

        对应于 m = +1 回旋谐波
        """
        p = state.params
        w_eff = w + 1j * p.nu
        return (w_eff - k * p.v - state.w_c) / (k * state.v_th_par)

    @staticmethod
    def compute_zeta_minus1(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算 ζ_-1 = (ω + iν - k∥v + ωc) / (k∥v_th,∥)

        对应于 m = -1 回旋谐波
        """
        p = state.params
        w_eff = w + 1j * p.nu
        return (w_eff - k * p.v + state.w_c) / (k * state.v_th_par)

    @staticmethod
    def compute_zeta_zero(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算 ζ_0 = (ω + iν - k∥v) / (k∥v_th,∥)

        对应于 m = 0（无回旋效应）
        """
        p = state.params
        w_eff = w + 1j * p.nu
        return (w_eff - k * p.v) / (k * state.v_th_par)

    @staticmethod
    def compute_A_plus1(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算 A_+1 系数（用于色散关系）

        物理意义：m=+1 回旋谐波对介电张量的贡献
        """
        p = state.params
        w_eff = w + 1j * p.nu

        zeta = DispersionFunction.compute_zeta_plus1(w, k, state)
        Z_val = DispersionFunction.plasma_dispersion_Z(zeta)

        term1 = (p.T_perp - p.T_par) / (w_eff * p.T_par)
        term2_factor = (zeta * p.T_perp / (w_eff * p.T_par) +
                       state.w_c / (w_eff * k * state.v_th_par))

        return term1 + term2_factor * Z_val
    
    @staticmethod
    def compute_A_plus1_4diel(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算 A_+1 系数（用于色散关系）

        物理意义：m=+1 回旋谐波对介电张量的贡献
        """
        p = state.params
        w_eff = w + 1j * p.nu

        zeta = DispersionFunction.compute_zeta_plus1(w, k, state)
        Z_val = DispersionFunction.plasma_dispersion_Z_adapt(zeta)

        term1 = (p.T_perp - p.T_par) / (w_eff * p.T_par)
        term2_factor = (zeta * p.T_perp / (w_eff * p.T_par) +
                       state.w_c / (w_eff * k * state.v_th_par))

        return term1 + term2_factor * Z_val

    @staticmethod
    def compute_A_minus1(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算 A_-1 系数（用于介电张量）

        物理意义：m=-1 回旋谐波对介电张量的贡献
        """
        p = state.params
        w_eff = w + 1j * p.nu

        zeta = DispersionFunction.compute_zeta_minus1(w, k, state)
        Z_val = DispersionFunction.plasma_dispersion_Z_adapt(zeta)

        term1 = (p.T_perp - p.T_par) / (w_eff * p.T_par)
        term2_factor = (zeta * p.T_perp / (w_eff * p.T_par) -
                       state.w_c / (w_eff * k * state.v_th_par))

        return term1 + term2_factor * Z_val

    @staticmethod
    def compute_B_zero(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算 B_0 系数（用于 K_par）

        物理意义：平行传播时的纵向响应
        """
        p = state.params
        w_eff = w + 1j * p.nu

        zeta = DispersionFunction.compute_zeta_zero(w, k, state)
        Z_val = DispersionFunction.plasma_dispersion_Z_adapt(zeta)

        # 注意：这里使用 p.v（当前值），因为是物理计算s
        term1 = (w_eff * p.T_perp - k * p.v * p.T_par) / (w_eff * k * p.T_par)
        term2_factor = zeta * p.T_perp / (k * p.T_par)

        return term1 + term2_factor * Z_val

    @staticmethod
    def compute_K_perp(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算垂直介电张量分量 K_⊥

        K_⊥ = 1 + (ωp²/2ω) * (A_-1 + A_+1)
        """
        A_plus1 = DispersionFunction.compute_A_plus1_4diel(w, k, state)
        A_minus1 = DispersionFunction.compute_A_minus1(w, k, state)

        return 1.0 + (state.w_p**2 / (2 * w)) * (A_minus1 + A_plus1)

    @staticmethod
    def compute_K_g(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算陀螺介电张量分量 K_g（非对角分量）

        K_g = (ωp²/2ω) * (A_-1 - A_+1)
        """
        A_plus1 = DispersionFunction.compute_A_plus1_4diel(w, k, state)
        A_minus1 = DispersionFunction.compute_A_minus1(w, k, state)

        return (state.w_p**2 / (2 * w)) * (A_minus1 - A_plus1)

    @staticmethod
    def compute_K_par(w: complex, k: complex, state: PlasmaState) -> complex:
        """
        计算平行介电张量分量 K_∥

        K_∥ = 1 + (2ωp²ω_eff)/(k²v_th²ω) * [v/ω_eff + B_0(ζ_0)]
        """
        p = state.params
        w_eff = w + 1j * p.nu

        B_zero = DispersionFunction.compute_B_zero(w, k, state)

        coeff = (2 * state.w_p**2 * w_eff) / (k**2 * state.v_th_par**2 * w)
        term_in_bracket = p.v / w_eff + B_zero

        return 1.0 + coeff * term_in_bracket

    @classmethod
    def evaluate_dispersion(cls, w: complex, k: complex,
                           state: PlasmaState) -> complex:
        """
        计算左旋波色散关系 D(ω,k) = 0

        D(ω,k) = k²c² - ω² - ωp² ω A_+1(ω,k)

        Returns:
            色散函数值（求根时应趋于0）
        """
        if np.abs(k) < 1e-15:
            return np.inf + 0j

        A_plus1 = cls.compute_A_plus1(w, k, state)
        c = state.constants.c

        return k**2 * c**2 - w**2 - state.w_p**2 * w * A_plus1


# ============================================================================
# 第四部分：数值求解器
# ============================================================================

class DispersionSolver:
    """
    色散关系数值求解器（与物理细节解耦）

    使用 Levenberg-Marquardt 方法求解复数方程
    """

    def __init__(self, dispersion_func: type = DispersionFunction,
                 tolerance: float = 1e-6,
                 max_residual: float = 1e12):
        """
        Args:
            dispersion_func: 色散函数类
            tolerance: 求解容差
            max_residual: 最大残差阈值
        """
        self.disp_func = dispersion_func
        self.tol = tolerance
        self.max_res = max_residual

    def solve_for_k(self, w: complex, state: PlasmaState,
                   k_guess: complex) -> tuple[Optional[complex], int]:
        """
        给定频率 ω 和等离子体状态，求解波数 k

        Args:
            w: 频率（可以是复数）
            state: 等离子体状态
            k_guess: 初始猜测值

        Returns:
            (k_solution, num_function_evaluations)
            如果求解失败，k_solution 为 None
        """
        def residual(k_vec):
            """残差函数：将复数方程转换为实数方程组"""
            k = k_vec[0] + 1j * k_vec[1]
            D = self.disp_func.evaluate_dispersion(w, k, state)
            return [np.real(D), np.imag(D)]

        k0_vec = [np.real(k_guess), np.imag(k_guess)]

        try:
            result = optimize.root(
                residual, k0_vec,
                method='lm',
                options={'xtol': self.tol}
            )

            if result.success:
                k_sol = result.x[0] + 1j * result.x[1]
                # 验证解的质量
                residual_norm = np.abs(
                    self.disp_func.evaluate_dispersion(w, k_sol, state)
                )
                if residual_norm < self.max_res:
                    return k_sol, result.nfev

            return None, result.nfev

        except Exception:
            return None, 0

    def get_cold_plasma_guess(self, w: float, state: PlasmaState) -> complex:
        """
        获取冷等离子体近似作为初始猜测

        对于左旋波：k ≈ √(ω² + ωp²) / c
        """
        return np.sqrt(w**2 + state.w_p**2) / state.constants.c


# ============================================================================
# 第五部分：预测-校正算法
# ============================================================================

class Predictor:
    """预测器：根据历史数据外推下一个k值"""

    def predict_next_step(self, k_history: list, param_history: list,
                         step_size: float) -> complex:
        """
        一阶外推预测

        Args:
            k_history: 波数历史
            param_history: 参数历史（可以是标量或向量）
            step_size: 步长

        Returns:
            预测的下一个k值
        """
        if len(k_history) < 2:
            return k_history[-1]

        dk = k_history[-1] - k_history[-2]
        dp = param_history[-1] - param_history[-2]

        # 处理向量参数
        is_vector = isinstance(dp, np.ndarray)

        if is_vector:
            dp_norm = np.linalg.norm(dp)
        else:
            dp_norm = np.abs(dp)

        if dp_norm < 1e-12:
            return k_history[-1]

        dk_dp = dk / dp if not is_vector else dk / dp_norm
        k_predicted = k_history[-1] + dk_dp * step_size

        return k_predicted


class AdaptiveStepController:
    """
    自适应步长控制器

    根据预测误差、收敛速度、曲率等因素动态调整步长
    """

    def __init__(self, target_total_steps: int = 20,
                 phase_name: str = "",
                 max_absolute_step: Optional[float] = None):
        self.target_steps = target_total_steps
        self.phase_name = phase_name
        self.max_absolute_step = max_absolute_step

        # 自适应参数
        self.min_step_factor = 1e-6
        self.max_step_factor = 10.0
        self.optimal_pred_error = 0.05
        self.fast_convergence_threshold = 8
        self.slow_convergence_threshold = 20
        self.high_curvature_threshold = 0.1

    def estimate_curvature(self, k_history: list) -> float:
        """估计轨迹曲率（二阶差分）"""
        if len(k_history) < 3:
            return 0.0
        d2k = k_history[-1] - 2*k_history[-2] + k_history[-3]
        curvature = np.abs(d2k) / (np.abs(k_history[-1]) + 1e-12)
        return curvature

    def calculate_prediction_error(self, k_predicted: complex,
                                   k_solution: Optional[complex]) -> float:
        """计算相对预测误差"""
        if k_solution is None or np.abs(k_solution) < 1e-12:
            return 0.0
        error = np.abs(k_predicted - k_solution) / np.abs(k_solution)
        return error

    def evaluate_and_adjust(self, k_history: list, k_predicted: complex,
                           k_solution: Optional[complex], nfev: int,
                           current_step: float, success: bool,
                           current_progress: float) -> tuple[str, float]:
        """
        评估当前步并调整步长

        Returns:
            ('accept', new_step) 或 ('retry', reduced_step)
        """
        if not success:
            if self.phase_name:
                print(f"  [{self.phase_name}] 求解失败，步长减半重试")
            return 'retry', current_step * 0.5

        # 1. 预测误差因子
        pred_error = self.calculate_prediction_error(k_predicted, k_solution)
        if pred_error < 0.01:
            error_factor = 1.5
        elif pred_error < self.optimal_pred_error:
            error_factor = 1.2
        elif pred_error < 0.2:
            error_factor = 1.0
        elif pred_error < 0.5:
            error_factor = 0.8
        else:
            error_factor = 0.6

        # 2. 收敛速度因子
        if nfev < self.fast_convergence_threshold:
            convergence_factor = 1.3
        elif nfev > self.slow_convergence_threshold:
            convergence_factor = 0.7
        else:
            convergence_factor = 1.0

        # 3. 曲率因子
        curvature = self.estimate_curvature(k_history)
        if curvature > self.high_curvature_threshold:
            curvature_factor = 0.7
        else:
            curvature_factor = 1.0

        # 4. 进度因子
        if current_progress > 0.8:
            progress_factor = 1.3
        else:
            progress_factor = 1.0

        # 综合调整
        total_factor = (error_factor * convergence_factor *
                       curvature_factor * progress_factor)
        new_step = current_step * total_factor

        # 限幅
        max_step = current_step * self.max_step_factor
        min_step = current_step * self.min_step_factor
        new_step = np.clip(new_step, min_step, max_step)

        if self.max_absolute_step is not None:
            new_step = min(new_step, self.max_absolute_step)

        # 调试信息（每5步输出一次）
        if len(k_history) % 5 == 0 and self.phase_name:
            print(f"  [{self.phase_name}] 步{len(k_history)}: "
                  f"pred_err={pred_error:.3f}, nfev={nfev}, "
                  f"curv={curvature:.3e}, factor={total_factor:.2f}")

        return 'accept', new_step


# ============================================================================
# 第六部分：同伦路径协议
# ============================================================================

class HomotopyPath(Protocol):
    """同伦路径协议（接口定义）"""
    def __call__(self, lambda_val: float) -> tuple[float, PlasmaParameters]:
        """
        给定同伦参数 λ ∈ [0,1]，返回对应的 (频率, 参数)
        """
        ...


class FrequencyHomotopyPath:
    """频率同伦路径：ω从 w_start 到 w_target，其他参数固定"""

    def __init__(self, w_start: float, w_target: float,
                 fixed_params: PlasmaParameters):
        self.w_start = w_start
        self.w_target = w_target
        self.fixed_params = fixed_params

    def __call__(self, lambda_val: float) -> tuple[float, PlasmaParameters]:
        w = self.w_start + lambda_val * (self.w_target - self.w_start)
        return w, self.fixed_params


class VectorHomotopyPath:
    """参数向量同伦：多个参数同时从起始值到目标值，频率固定"""

    def __init__(self, w_fixed: float,
                 params_start: PlasmaParameters,
                 params_target: PlasmaParameters,
                 varying_fields: list[str]):
        self.w = w_fixed
        self.p_start = params_start
        self.p_target = params_target
        self.varying = set(varying_fields)

    def __call__(self, lambda_val: float) -> tuple[float, PlasmaParameters]:
        # 线性插值
        new_values = {}
        for field in self.varying:
            val_start = getattr(self.p_start, field)
            val_target = getattr(self.p_target, field)
            new_values[field] = val_start + lambda_val * (val_target - val_start)

        params = self.p_start.evolve(**new_values)
        return self.w, params


class VectorWithFrequencyPath:
    """
    向量同伦（包含频率w）：频率和参数同时从起始值到目标值

    用于单阶段同伦策略（当 w_target < w_ref 时）
    """

    def __init__(self, w_start: float, w_target: float,
                 params_start: PlasmaParameters,
                 params_target: PlasmaParameters,
                 varying_fields: list[str]):
        """
        Args:
            w_start: 起始频率
            w_target: 目标频率
            params_start: 起始参数
            params_target: 目标参数
            varying_fields: 需要同伦的参数名列表（不包括'w'）
        """
        self.w_start = w_start
        self.w_target = w_target
        self.p_start = params_start
        self.p_target = params_target
        self.varying = set(varying_fields)

    def __call__(self, lambda_val: float) -> tuple[float, PlasmaParameters]:
        # 频率线性插值
        w = self.w_start + lambda_val * (self.w_target - self.w_start)

        # 参数线性插值
        new_values = {}
        for field in self.varying:
            val_start = getattr(self.p_start, field)
            val_target = getattr(self.p_target, field)
            new_values[field] = val_start + lambda_val * (val_target - val_start)

        params = self.p_start.evolve(**new_values)
        return w, params


# ============================================================================
# 第七部分：同伦延拓控制器
# ============================================================================

@dataclass
class HomotopySolution:
    """同伦求解结果"""
    k_history: list[complex]
    w_history: list[float]
    state_history: list[PlasmaState]
    iterations: list[int]
    success: bool
    metadata: dict


class HomotopyContinuation:
    """同伦延拓主控制器"""

    def __init__(self, solver: DispersionSolver,
                 predictor: Predictor,
                 step_controller: AdaptiveStepController):
        self.solver = solver
        self.predictor = predictor
        self.step_controller = step_controller

    def execute(self, path: HomotopyPath,
               initial_state: PlasmaState,
               k_initial: complex,
               target_steps: int = 20) -> HomotopySolution:
        """
        执行同伦延拓

        Args:
            path: 同伦路径
            initial_state: 初始等离子体状态
            k_initial: 初始波数
            target_steps: 目标步数

        Returns:
            HomotopySolution 对象
        """
        # 初始化历史记录
        w_init, params_init = path(0.0)
        state_init = initial_state.with_params(params_init)

        # 求解初始点（用冷等离子体解作为猜测，求出热等离子体解）
        k_solved_init, nfev_init = self.solver.solve_for_k(w_init, state_init, k_initial)

        if k_solved_init is None:
            raise RuntimeError(f"初始点求解失败 (w={w_init:.2e}, k_guess={k_initial})")

        # 保存初始解（热等离子体解，不是冷等离子体猜测）
        k_hist = [k_solved_init]
        w_hist = [w_init]
        state_hist = [state_init]
        lambda_hist = [0.0]
        iter_hist = [nfev_init]

        # 初始化步长
        current_lambda = 0.0
        initial_step_size = 1.0 / (target_steps * 2)
        current_step_size = initial_step_size

        while current_lambda < 1.0:
            # 1. 预测
            k_predicted = self.predictor.predict_next_step(
                k_hist, lambda_hist, current_step_size
            )

            # 2. 尝试校正
            retry_count = 0
            max_retries = 10
            is_accepted = False

            while not is_accepted and retry_count < max_retries:
                # 确保不超过终点
                if current_lambda + current_step_size >= 1.0:
                    current_step_size = 1.0 - current_lambda
                    # 重新预测
                    k_predicted = self.predictor.predict_next_step(
                        k_hist, lambda_hist, current_step_size
                    )

                next_lambda = current_lambda + current_step_size
                w_next, params_next = path(next_lambda)
                state_next = initial_state.with_params(params_next)

                # 求解
                k_sol, nfev = self.solver.solve_for_k(w_next, state_next, k_predicted)

                # 评估步长
                success = k_sol is not None
                current_progress = len(k_hist) / target_steps
                action, new_step = self.step_controller.evaluate_and_adjust(
                    k_hist, k_predicted, k_sol, nfev,
                    current_step_size, success, current_progress
                )

                if action == 'accept':
                    # 接受并前进
                    k_hist.append(k_sol)
                    w_hist.append(w_next)
                    state_hist.append(state_next)
                    lambda_hist.append(next_lambda)
                    iter_hist.append(nfev)

                    current_lambda = next_lambda
                    current_step_size = new_step
                    is_accepted = True

                elif action == 'retry':
                    # 减小步长重试
                    current_step_size = new_step
                    retry_count += 1

            if retry_count >= max_retries:
                print(f"警告：在 λ={current_lambda:.3f} 处重试次数过多")
                return HomotopySolution(
                    k_history=k_hist,
                    w_history=w_hist,
                    state_history=state_hist,
                    iterations=iter_hist,
                    success=False,
                    metadata={'failure_lambda': current_lambda}
                )

        return HomotopySolution(
            k_history=k_hist,
            w_history=w_hist,
            state_history=state_hist,
            iterations=iter_hist,
            success=True,
            metadata={}
        )


# ============================================================================
# 第八部分：自适应同伦策略
# ============================================================================

@dataclass
class ReferenceValues:
    """参考值配置"""
    w_ref_factor: float = 0.9      # w_ref = 0.9 * w_c
    v_ref: float = 1e4              # 参考速度 [m/s]
    T_ref_eV: float = 10.0          # 参考温度 [eV]
    n_ref: float = 1e17             # 参考密度 [m^-3]
    nu_ref_factor: float = 0.01     # nu_ref = w_target / 100


class AdaptiveHomotopyStrategy:
    """
    自适应同伦策略选择器

    根据目标参数智能决定：
    - 单阶段 vs 两阶段
    - 哪些参数需要同伦，哪些直接用目标值
    """

    def __init__(self, reference: ReferenceValues = None):
        self.ref = reference or ReferenceValues()

    def select_paths(self, target_state: PlasmaState,
                    w_target: float) -> list[HomotopyPath]:
        """
        选择同伦路径

        Returns:
            路径列表（1个或2个元素）
        """
        w_c = target_state.w_c
        w_ref = self.ref.w_ref_factor * w_c

        # 构建参考参数
        const = target_state.constants
        T_ref_K = self.ref.T_ref_eV * const.e / const.kB

        ref_params = PlasmaParameters(
            n=self.ref.n_ref,
            T_par=T_ref_K,
            T_perp=T_ref_K,
            v=self.ref.v_ref,
            nu=w_target * self.ref.nu_ref_factor
        )

        target_params = target_state.params

        print("\n" + "="*70)
        print("自适应同伦策略选择")
        print("="*70)
        print(f"参考频率: w_ref = {w_ref:.2e} rad/s ({self.ref.w_ref_factor}*w_c)")
        print(f"目标频率: w_target = {w_target:.2e} rad/s")
        print(f"频率比: w_target/w_c = {w_target/w_c:.3f}")

        if w_target < w_ref:
            # 单阶段策略：频率和参数同时同伦
            print("\n[策略] w_target < w_ref → 单阶段向量同伦（包含频率）")
            return self._single_stage_path(w_ref, w_target, ref_params, target_params)
        else:
            # 两阶段策略
            print("\n[策略] w_target >= w_ref → 两阶段同伦")
            return self._two_stage_paths(w_ref, w_target, ref_params, target_params)

    def _single_stage_path(self, w_ref: float, w_target: float,
                          ref_params: PlasmaParameters,
                          target_params: PlasmaParameters) -> list[HomotopyPath]:
        """
        单阶段向量同伦路径（包含频率）

        适用场景：w_target < w_ref
        所有参数（包括频率）同时从参考值到目标值
        """
        # 决定哪些参数需要同伦（应用智能决策）
        stage_params = self._decide_stage1_params(ref_params, target_params)

        print(f"\n单阶段起始参数:")
        print(f"  w = {w_ref:.2e} rad/s")
        print(f"  v = {stage_params.v:.2e} m/s")
        print(f"  T_par = {stage_params.T_par:.2e} K")
        print(f"  n = {stage_params.n:.2e} m^-3")
        print(f"  nu = {stage_params.nu:.2e} rad/s")

        # 决定哪些参数需要变化
        varying_fields = []

        # 频率总是需要同伦（从 w_ref 到 w_target）
        print(f"\n同伦参数:")
        print(f"  w: {w_ref:.2e} → {w_target:.2e} rad/s")

        # v 的同伦判断
        if abs(stage_params.v - target_params.v) > 1e-6:
            varying_fields.append('v')
            print(f"  v: {stage_params.v:.2e} → {target_params.v:.2e} m/s")

        # T 的同伦判断
        if abs(stage_params.T_par - target_params.T_par) > 1e-6:
            varying_fields.append('T_par')
            varying_fields.append('T_perp')
            const = PhysicalConstants()
            T_start_eV = stage_params.T_par * const.kB / const.e
            T_target_eV = target_params.T_par * const.kB / const.e
            print(f"  T_par: {T_start_eV:.2f} → {T_target_eV:.2f} eV")

        # n 的同伦判断
        if abs(stage_params.n - target_params.n) > 1e-6:
            varying_fields.append('n')
            print(f"  n: {stage_params.n:.2e} → {target_params.n:.2e} m^-3")

        # nu 总是需要同伦（从 w_ref/100 到 nu_target）
        if abs(stage_params.nu - target_params.nu) > 1e-6:
            varying_fields.append('nu')
            print(f"  nu: {stage_params.nu:.2e} → {target_params.nu:.2e} rad/s")

        if not varying_fields:
            print("\n注意：除频率外，无其他参数需要同伦")

        # 创建单阶段向量同伦路径（包含频率）
        path = VectorWithFrequencyPath(
            w_ref, w_target,
            stage_params, target_params,
            varying_fields
        )

        return [path]

    def _two_stage_paths(self, w_ref: float, w_target: float,
                        ref_params: PlasmaParameters,
                        target_params: PlasmaParameters) -> list[HomotopyPath]:
        """
        两阶段同伦路径

        阶段1: 频率同伦，其他参数智能固定
        阶段2: 参数向量同伦
        """
        # 决定阶段1的参数
        stage1_params = self._decide_stage1_params(ref_params, target_params)

        print(f"\n阶段1固定参数:")
        print(f"  v = {stage1_params.v:.2e} m/s")
        print(f"  T_par = {stage1_params.T_par:.2e} K")
        print(f"  n = {stage1_params.n:.2e} m^-3")
        print(f"  nu = {stage1_params.nu:.2e} rad/s")

        # 阶段1路径
        path1 = FrequencyHomotopyPath(w_ref, w_target, stage1_params)

        # 决定阶段2需要变化的参数
        varying_fields = []
        if abs(stage1_params.v - target_params.v) > 1e-6:
            varying_fields.append('v')
        if abs(stage1_params.T_par - target_params.T_par) > 1e-6:
            varying_fields.append('T_par')
            varying_fields.append('T_perp')
        if abs(stage1_params.n - target_params.n) > 1e-6:
            varying_fields.append('n')
        if abs(stage1_params.nu - target_params.nu) > 1e-6:
            varying_fields.append('nu')

        print(f"\n阶段2变化参数: {varying_fields if varying_fields else '无（跳过阶段2）'}")

        if not varying_fields:
            return [path1]

        # 阶段2路径
        path2 = VectorHomotopyPath(w_target, stage1_params, target_params, varying_fields)

        return [path1, path2]

    def _decide_stage1_params(self, ref_params: PlasmaParameters,
                             target_params: PlasmaParameters) -> PlasmaParameters:
        """
        决策逻辑：阶段1使用哪些值

        规则：若 target > ref，则阶段1直接用 target（避免阶段2反向同伦）
        """
        updates = {}

        # v 的决策
        if target_params.v > self.ref.v_ref:
            updates['v'] = target_params.v
            print(f"\n[决策] v_target > v_ref → 阶段1使用目标值")

        # T 的决策
        const = PhysicalConstants()
        T_ref_K = self.ref.T_ref_eV * const.e / const.kB
        if target_params.T_par > T_ref_K:
            updates['T_par'] = target_params.T_par
            updates['T_perp'] = target_params.T_perp
            print(f"[决策] T_target > T_ref → 阶段1使用目标值")

        # n 的决策
        if target_params.n > self.ref.n_ref:
            updates['n'] = target_params.n
            print(f"[决策] n_target > n_ref → 阶段1使用目标值")

        return ref_params.evolve(**updates)


# ============================================================================
# 第九部分：主求解接口
# ============================================================================

def solve_dispersion_relation(
    B0: float,
    n_target: float,
    T_par_eV: float,
    T_perp_eV: float,
    v_target: float,
    nu_target: float,
    w_target: float,
    Z: int = 40,
    target_steps_s1: int = 20,
    target_steps_s2: int = 12
) -> dict:
    """
    求解热等离子体色散关系（主接口）

    Args:
        B0: 磁场强度 [T]
        n_target: 等离子体密度 [m^-3]
        T_par_eV: 平行温度 [eV]
        T_perp_eV: 垂直温度 [eV]
        v_target: 漂移速度 [m/s]
        nu_target: 碰撞频率 [rad/s]
        w_target: 目标频率 [rad/s]
        Z: 离子电荷数
        target_steps_s1: 阶段1目标步数
        target_steps_s2: 阶段2目标步数

    Returns:
        包含所有结果的字典
    """
    print("\n" + "="*70)
    print("热等离子体色散关系求解器 v1.0")
    print("="*70)

    start_time = time.time()

    # 1. 创建物理常数和参数
    constants = PhysicalConstants()

    # 温度转换：eV → K
    T_par_K = T_par_eV * constants.e / constants.kB
    T_perp_K = T_perp_eV * constants.e / constants.kB

    target_params = PlasmaParameters(
        n=n_target,
        T_par=T_par_K,
        T_perp=T_perp_K,
        v=v_target,
        nu=nu_target
    )

    # 2. 创建初始状态
    initial_state = PlasmaState(target_params, B0, Z, constants)

    print(f"\n目标参数:")
    print(f"  密度 n = {n_target:.2e} m^-3")
    print(f"  温度 T_par = {T_par_eV:.2f} eV, T_perp = {T_perp_eV:.2f} eV")
    print(f"  速度 v = {v_target:.2e} m/s")
    print(f"  碰撞频率 nu = {nu_target:.2e} rad/s")
    print(f"  目标频率 w = {w_target:.2e} rad/s")
    print(f"\n派生量:")
    print(f"  回旋频率 w_c = {initial_state.w_c:.3e} rad/s")
    print(f"  等离子体频率 w_p = {initial_state.w_p:.3e} rad/s")
    print(f"  频率比 w/w_c = {w_target/initial_state.w_c:.3f}")

    # 3. 创建求解器和控制器
    disp_func = DispersionFunction
    solver = DispersionSolver(disp_func)
    predictor = Predictor()

    # 4. 选择同伦策略
    strategy = AdaptiveHomotopyStrategy()
    paths = strategy.select_paths(initial_state, w_target)

    # 5. 执行同伦
    solutions = []
    w_init, params_init = paths[0](0.0)
    state_init = initial_state.with_params(params_init)
    k_current = solver.get_cold_plasma_guess(w_init, state_init)

    for i, path in enumerate(paths, 1):
        print(f"\n{'='*70}")
        print(f"执行阶段 {i}/{len(paths)}")
        print(f"{'='*70}")

        w_start, params_start = path(0.0)
        state_start = initial_state.with_params(params_start)

        target_steps = target_steps_s1 if i == 1 else target_steps_s2
        phase_name = f"Stage{i}"

        step_controller = AdaptiveStepController(
            target_total_steps=target_steps,
            phase_name=phase_name,
            max_absolute_step=initial_state.w_c / 3.0 if i == 1 else None
        )

        homotopy = HomotopyContinuation(solver, predictor, step_controller)

        solution = homotopy.execute(path, state_start, k_current, target_steps)
        solutions.append(solution)

        if not solution.success:
            raise RuntimeError(f"阶段 {i} 同伦失败")

        print(f"\n阶段 {i} 完成，共 {len(solution.k_history)} 步")
        k_current = solution.k_history[-1]

    end_time = time.time()

    # 6. 合并结果
    all_k = []
    all_w = []
    all_states = []
    all_iters = []
    trans_indices = [0]

    for sol in solutions:
        if all_k:  # 跳过第一个点（重复）
            all_k.extend(sol.k_history[1:])
            all_w.extend(sol.w_history[1:])
            all_states.extend(sol.state_history[1:])
            all_iters.extend(sol.iterations[1:])
        else:
            all_k.extend(sol.k_history)
            all_w.extend(sol.w_history)
            all_states.extend(sol.state_history)
            all_iters.extend(sol.iterations)
        trans_indices.append(len(all_k))

    # 7. 计算最终解的介电张量
    k_final = all_k[-1]
    w_final = all_w[-1]
    state_final = all_states[-1]

    K_perp = disp_func.compute_K_perp(w_final, k_final, state_final)
    K_g = disp_func.compute_K_g(w_final, k_final, state_final)
    K_par = disp_func.compute_K_par(w_final, k_final, state_final)

    zeta_plus1_final = disp_func.compute_zeta_plus1(w_final, k_final, state_final)
    zeta_minus1_final = disp_func.compute_zeta_minus1(w_final, k_final, state_final)
    zeta_zero_final = disp_func.compute_zeta_zero(w_final, k_final, state_final)
    Z_plus1_final = disp_func.plasma_dispersion_Z_adapt(zeta_plus1_final)
    Z_minus1_final = disp_func.plasma_dispersion_Z_adapt(zeta_minus1_final)
    Z_zero_final = disp_func.plasma_dispersion_Z_adapt(zeta_zero_final)

    print("\n" + "="*70)
    print("求解完成！")
    print("="*70)
    print(f"计算时间: {end_time - start_time:.3f} 秒")
    print(f"总步数: {len(all_k)}")
    print(f"阶段划分: {trans_indices}")
    print(f"\n最终解:")
    print(f"  波数 k = {np.real(k_final):.6f} + {np.imag(k_final):.6f}j")
    print(f"  宗量 zeta_plus1 = {np.real(zeta_plus1_final):.6f} + {np.imag(zeta_plus1_final):.6f}j")
    print(f"  等离子体函数 Z_plus1 = {np.real(Z_plus1_final):.6f} + {np.imag(Z_plus1_final):.6f}j")
    print(f"  宗量 zeta_minus1 = {np.real(zeta_minus1_final):.6f} + {np.imag(zeta_minus1_final):.6f}j")
    print(f"  等离子体函数 Z_minus1 = {np.real(Z_minus1_final):.6f} + {np.imag(Z_minus1_final):.6f}j")
    print(f"  宗量 zeta_zero = {np.real(zeta_zero_final):.6f} + {np.imag(zeta_zero_final):.6f}j")
    print(f"  等离子体函数 Z_zero = {np.real(Z_zero_final):.6f} + {np.imag(Z_zero_final):.6f}j")
    print(f"\n介电张量分量:")
    print(f"  K_⊥ = {np.real(K_perp):.6f} + {np.imag(K_perp):.6f}j")
    print(f"  K_g = {np.real(K_g):.6f} + {np.imag(K_g):.6f}j")
    print(f"  K_∥ = {np.real(K_par):.6f} + {np.imag(K_par):.6f}j")

    # 8. 计算衍生物理量
    print("\n计算衍生物理量...")

    # 8.1 zeta 参数历史
    zeta_history = np.array([
        disp_func.compute_zeta_plus1(all_w[i], all_k[i], all_states[i])
        for i in range(len(all_k))
    ])

    # 8.2 相速度历史
    phase_velocity_history = np.real(np.array(all_w) / np.array(all_k))

    # 8.3 阻尼率历史（γ = -Im(k) * v_phase）
    damping_rate_history = -np.imag(np.array(all_k)) * phase_velocity_history

    # 8.4 参数演化历史
    n_history = np.array([s.params.n for s in all_states])
    T_par_history_eV = np.array([s.params.T_par * constants.kB / constants.e for s in all_states])
    T_perp_history_eV = np.array([s.params.T_perp * constants.kB / constants.e for s in all_states])
    v_history = np.array([s.params.v for s in all_states])
    nu_history = np.array([s.params.nu for s in all_states])

    # 8.5 色散关系残差（验证解的精度）
    dispersion_residual_history = np.array([
        np.abs(disp_func.evaluate_dispersion(all_w[i], all_k[i], all_states[i]))
        for i in range(len(all_k))
    ])

    print(f"  平均残差: {np.mean(dispersion_residual_history):.2e}")
    print(f"  最大残差: {np.max(dispersion_residual_history):.2e}")

    # 9. 构建返回结果
    return {
        # 基本历史数据
        'k_history': np.array(all_k),
        'w_history': np.array(all_w),
        'state_history': all_states,
        'iterations': np.array(all_iters),

        # 阶段信息
        'trans_indices': trans_indices,
        'num_stages': len(solutions),

        # 最终解
        'k_final': k_final,
        'w_final': w_final,
        'state_final': state_final,

        # 介电张量（最终值）
        'K_perp': K_perp,
        'K_g': K_g,
        'K_par': K_par,

        # 衍生物理量历史
        'zeta_history': zeta_history,
        'phase_velocity_history': phase_velocity_history,
        'damping_rate_history': damping_rate_history,
        'dispersion_residual_history': dispersion_residual_history,

        # 参数演化历史
        'n_history': n_history,
        'T_par_history_eV': T_par_history_eV,
        'T_perp_history_eV': T_perp_history_eV,
        'v_history': v_history,
        'nu_history': nu_history,

        # 元数据
        'computation_time': end_time - start_time,
        'constants': constants
    }


# ============================================================================
# 主程序（示例）
# ============================================================================

if __name__ == '__main__':
    # 物理参数
    B0 = 0.1          # 磁场 [T]
    n_p = 1e18        # 密度 [m^-3]
    T_i_eV = 30.0      # 温度 [eV]
    v_i = 10.0       # 速度 [m/s]
    nu_i = 1e5      # 碰撞频率 [rad/s]
    w_target = 3.3e6    # 目标频率 [rad/s]
    Z_number = 40     # 离子电荷数

    # 求解
    results = solve_dispersion_relation(
        B0=B0,
        n_target=n_p,
        T_par_eV=T_i_eV,
        T_perp_eV=T_i_eV,
        v_target=v_i,
        nu_target=nu_i,
        w_target=w_target,
        Z=Z_number,
        target_steps_s1=20,
        target_steps_s2=12
    )

    # 绘图（如果绘图模块可用）
    try:
        from kdlogdraw_v1_999 import plot_all_results
        plot_all_results(results)
    except ImportError:
        print("\n注意：绘图模块 kdlogdraw_v1_0.py 未找到，跳过绘图")
