# -*- coding: utf-8 -*-
"""
该脚本使用根轨迹追踪法高效地求解热等离子体动理学色散关系。

改进版本 (1012-claude-1+v-3):
- 统一的五参数同伦架构
- 智能参考值策略：自动决定哪些参数在阶段1直接使用目标值

五个同伦参数：
1. 频率 w: 总是在阶段1同伦
2. 离子流向速度 v: ref=10000, 若target>ref则阶段1用target，否则阶段2同伦
3. 平行温度 T_par: ref=10eV, 若target>ref则阶段1用target，否则阶段2同伦
4. 碰撞频率 nu: ref=w/100, 总是在阶段2同伦
5. 离子密度 n: ref=1e17, 若target>ref则阶段1用target，否则阶段2同伦
"""

import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import time

class Predictor:
    """预测器：根据历史数据和给定的步长进行一阶外推"""
    def predict_next_step(self, k_history, param_history, step_size):
        """根据给定的步长预测下一个k值"""
        if len(k_history) < 2:
            k_predicted = k_history[-1]
        else:
            dk = k_history[-1] - k_history[-2]
            dp = param_history[-1] - param_history[-2]

            is_vector = isinstance(dp, np.ndarray)

            if (is_vector and np.linalg.norm(dp) < 1e-12) or (not is_vector and np.abs(dp) < 1e-12):
                k_predicted = k_history[-1]
            else:
                dk_dp = dk / dp
                k_predicted = k_history[-1] + dk_dp * step_size

        return k_predicted

class Corrector:
    def __init__(self, solver_function):
        self.solve_func = solver_function

    def correct_to_solution(self, k_predicted, **kwargs):
        k_sol, nfev = self.solve_func(k_predicted, **kwargs)
        success = k_sol is not None
        return k_sol, nfev, success

class AdaptiveStepController:
    """自适应步长控制器：纯数学策略，无物理先验"""
    def __init__(self, target_total_steps=20, phase_name="", max_absolute_step=None):
        self.target_steps = target_total_steps
        self.phase_name = phase_name
        self.min_step_factor = 1e-6
        self.max_step_factor = 10.0
        self.max_absolute_step = max_absolute_step

        # 自适应参数
        self.optimal_pred_error = 0.05
        self.fast_convergence_threshold = 8
        self.slow_convergence_threshold = 20
        self.high_curvature_threshold = 0.1

    def estimate_curvature(self, k_history):
        """估计曲率（二阶差分）"""
        if len(k_history) < 3:
            return 0.0
        d2k = k_history[-1] - 2*k_history[-2] + k_history[-3]
        curvature = np.abs(d2k) / (np.abs(k_history[-1]) + 1e-12)
        return curvature

    def calculate_prediction_error(self, k_predicted, k_solution):
        """计算相对预测误差"""
        if k_solution is None or np.abs(k_solution) < 1e-12:
            return 0.0
        error = np.abs(k_predicted - k_solution) / np.abs(k_solution)
        return error

    def evaluate_and_adjust(self, k_history, k_predicted, k_solution, nfev,
                           current_step, success, current_progress):
        """综合评估并调整步长"""
        if not success:
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

        # 4. 全局进度因子
        if current_progress > 0.8:
            progress_factor = 1.3
        else:
            progress_factor = 1.0

        # 综合调整
        total_factor = error_factor * convergence_factor * curvature_factor * progress_factor
        new_step = current_step * total_factor

        # 限幅
        max_step = current_step * self.max_step_factor
        min_step = current_step * self.min_step_factor
        new_step = np.clip(new_step, min_step, max_step)

        if self.max_absolute_step is not None:
            new_step = min(new_step, self.max_absolute_step)

        # 调试信息
        if len(k_history) % 5 == 0:
            print(f"  [{self.phase_name}] 步{len(k_history)}: pred_err={pred_error:.3f}, "
                  f"nfev={nfev}, curv={curvature:.3e}, factor={total_factor:.2f}")

        return 'accept', new_step

class DispersionSolver:
    def __init__(self, B0, n_p, T_par_i, T_perp_i, v_i=10000, nu_i=64400, Z_number=40):
        self.c = 3e8
        self.kB = 1.38e-23
        self.e = 1.602e-19
        self.m_i = Z_number * 1.67e-27
        self.eps0 = 8.854e-12
        self.q_i = self.e
        self.Z_number = Z_number

        # 目标值
        self.n_target = n_p
        self.T_par_target = T_par_i
        self.T_perp_target = T_perp_i
        self.v_target = v_i
        self.nu_target = nu_i
        self.B0 = B0

        # 当前值（会在同伦过程中变化）
        self.current_n = n_p
        self.current_T_par = T_par_i
        self.current_T_perp = T_perp_i
        self.current_v = v_i
        self.current_nu = nu_i

        # 更新派生量
        self._update_derived_quantities()

    def _update_derived_quantities(self):
        """更新依赖于当前参数的派生量"""
        self.n_i = self.current_n
        self.T_par_i = self.current_T_par
        self.T_perp_i = self.current_T_perp
        self.w_p_i = np.sqrt(self.n_i * self.q_i**2 / (self.eps0 * self.m_i))
        self.w_c_i = self.q_i * self.B0 / self.m_i
        self.v_th_par_i = np.sqrt(2 * self.kB * self.T_par_i / self.m_i)

    def _Z_plus(self, z):
        return 1j * np.sqrt(np.pi) * special.wofz(z)

    def _fun_A(self, w, k_par):
        w_eff = w + 1j * self.current_nu
        zeta = (w_eff - k_par * self.current_v - self.w_c_i) / (k_par * self.v_th_par_i)
        term1 = (self.T_perp_i - self.T_par_i) / (w_eff * self.T_par_i)
        term2_factor = ((w_eff - k_par * self.current_v - self.w_c_i) * self.T_perp_i +
                       self.w_c_i * self.T_par_i) / (k_par * self.v_th_par_i * self.T_par_i * w_eff)
        return term1 + term2_factor * self._Z_plus(zeta)

    def _hotPDE(self, k_par, w):
        if np.isclose(k_par, 0):
            return np.inf
        return k_par**2 * self.c**2 - w**2 - self.w_p_i**2 * w * self._fun_A(w, k_par)

    def _solve_single_k(self, k0, w):
        try:
            sol = optimize.root(
                lambda k: [np.real(self._hotPDE(k[0] + 1j*k[1], w)),
                           np.imag(self._hotPDE(k[0] + 1j*k[1], w))],
                [np.real(k0), np.imag(k0)], method='lm', options={'xtol': 1e-6})
            if sol.success and np.abs(self._hotPDE(sol.x[0] + 1j*sol.x[1], w)) < 1e12:
                return sol.x[0] + 1j*sol.x[1], sol.nfev
        except Exception:
            return None, 0
        return None, sol.nfev if 'sol' in locals() else 0

    def _get_cold_plasma_solution(self, w):
        return np.sqrt(w**2 + self.w_p_i**2) / self.c

    def _calculate_zeta(self, k, w):
        return (w + 1j*self.current_nu - k * self.current_v - self.w_c_i) / (k * self.v_th_par_i)

    def _solve_homotopy_scalar(self, k_start, w_start, w_target, params_fixed, target_steps=12):
        """
        阶段1：频率同伦

        params_fixed: dict, 包含 {'n', 'T_par', 'T_perp', 'v', 'nu'}
        """
        print(f"\n{'='*60}")
        print(f"阶段1：频率同伦 (w: {w_start:.2e} -> {w_target:.2e})")
        print(f"{'='*60}")
        print(f"固定参数:")
        print(f"  n   = {params_fixed['n']:.2e} m^-3")
        print(f"  T_par = {params_fixed['T_par']/self.e*self.kB:.2f} eV")
        print(f"  v   = {params_fixed['v']:.2e} m/s")
        print(f"  nu  = {params_fixed['nu']:.2e} rad/s")
        print(f"目标步数: {target_steps}")

        # 设置当前参数
        self.current_n = params_fixed['n']
        self.current_T_par = params_fixed['T_par']
        self.current_T_perp = params_fixed['T_perp']
        self.current_v = params_fixed['v']
        self.current_nu = params_fixed['nu']
        self._update_derived_quantities()

        k_hist, param_hist, pred_hist, iter_hist, zeta_hist = [k_start], [w_start], [np.nan], [], []
        _, nfev = self._solve_single_k(k_start, w_start)
        iter_hist.append(nfev)
        zeta_hist.append(self._calculate_zeta(k_start, w_start))

        # 初始步长
        total_range = w_target - w_start
        initial_step_size = total_range / (target_steps * 5)
        print(f"初始步长: {initial_step_size:.2e} (总范围: {total_range:.2e})")

        predictor = Predictor()
        corrector = Corrector(lambda k_pred, w: self._solve_single_k(k_pred, w=w))
        max_freq_step = self.w_c_i / 3.0
        step_controller = AdaptiveStepController(target_total_steps=target_steps, phase_name="Stage1-Freq",
                                                max_absolute_step=max_freq_step)

        current_w = w_start
        current_step_size = initial_step_size

        while current_w < w_target:
            k_pred = predictor.predict_next_step(k_hist, param_hist, current_step_size)

            is_accepted = False
            retry_count = 0
            max_retries = 10

            while not is_accepted and retry_count < max_retries:
                if current_w + current_step_size >= w_target:
                    current_step_size = w_target - current_w
                    k_pred = predictor.predict_next_step(k_hist, param_hist, current_step_size)

                next_w = current_w + current_step_size
                k_sol, nfev, success = corrector.correct_to_solution(k_pred, w=next_w)

                current_progress = len(k_hist) / target_steps
                action, payload = step_controller.evaluate_and_adjust(
                    k_hist, k_pred, k_sol, nfev, current_step_size, success, current_progress)

                if action == 'accept':
                    k_hist.append(k_sol)
                    param_hist.append(next_w)
                    pred_hist.append(k_pred)
                    iter_hist.append(nfev)
                    zeta_hist.append(self._calculate_zeta(k_sol, next_w))
                    current_w = next_w
                    current_step_size = payload
                    is_accepted = True
                elif action == 'retry':
                    current_step_size = payload
                    k_pred = predictor.predict_next_step(k_hist, param_hist, current_step_size)
                    retry_count += 1

            if retry_count >= max_retries:
                raise RuntimeError(f"阶段1在 w={current_w:.2e} 处重试次数过多")

        print(f"阶段1完成，共 {len(k_hist)} 步\n")
        return k_hist, param_hist, pred_hist, iter_hist, zeta_hist

    def _solve_homotopy_vector(self, w_fixed, k_start, params_start, params_target, param_names, target_steps=8):
        """
        阶段2：多参数向量同伦

        params_start: dict of parameter values at start
        params_target: dict of parameter values at end
        param_names: list of parameter names to vary (subset of ['n', 'T_par', 'v', 'nu'])
        """
        print(f"\n{'='*60}")
        print(f"阶段2：向量同伦 @ w = {w_fixed:.2e}")
        print(f"{'='*60}")
        print(f"同伦参数: {param_names}")
        print(f"起始值 -> 目标值:")
        for name in param_names:
            if name == 'T_par':
                print(f"  {name}: {params_start[name]/self.e*self.kB:.2f} eV -> {params_target[name]/self.e*self.kB:.2f} eV")
            else:
                print(f"  {name}: {params_start[name]:.2e} -> {params_target[name]:.2e}")
        print(f"目标步数: {target_steps}")

        # 构建向量
        p_start = np.array([params_start[name] for name in param_names])
        p_target = np.array([params_target[name] for name in param_names])

        k_hist, lambda_hist, pred_hist, iter_hist, zeta_hist = [k_start], [0.0], [np.nan], [], []
        params_hist = {name: [params_start[name]] for name in param_names}

        # 设置初始参数
        for name, val in params_start.items():
            if name == 'n':
                self.current_n = val
            elif name == 'T_par':
                self.current_T_par = val
            elif name == 'T_perp':
                self.current_T_perp = val
            elif name == 'v':
                self.current_v = val
            elif name == 'nu':
                self.current_nu = val
        self._update_derived_quantities()

        _, nfev = self._solve_single_k(k_start, w_fixed)
        iter_hist.append(nfev)
        zeta_hist.append(self._calculate_zeta(k_start, w_fixed))

        predictor = Predictor()
        corrector = Corrector(lambda k_pred, **kwargs: self._update_params_and_solve(w_fixed, k_pred, **kwargs))
        step_controller = AdaptiveStepController(target_total_steps=target_steps, phase_name="Stage2-Vector")

        current_lambda = 0.0
        initial_step_lambda = min(0.15, 1.0 / target_steps * 1.5)
        current_step_lambda = initial_step_lambda

        while current_lambda < 1.0:
            remaining_distance = 1.0 - current_lambda
            remaining_steps = max(target_steps - len(k_hist) + 1, 2)
            suggested_step = remaining_distance / remaining_steps

            k_pred = predictor.predict_next_step(k_hist, lambda_hist, current_step_lambda)

            is_accepted = False
            retry_count = 0
            max_retries = 10

            while not is_accepted and retry_count < max_retries:
                if current_lambda + current_step_lambda >= 1.0:
                    current_step_lambda = 1.0 - current_lambda

                next_lambda = current_lambda + current_step_lambda
                next_p = p_start + next_lambda * (p_target - p_start)

                # 构建参数字典
                next_params = {name: next_p[i] for i, name in enumerate(param_names)}
                k_sol, nfev, success = corrector.correct_to_solution(k_pred, **next_params)

                current_progress = len(k_hist) / target_steps
                action, payload = step_controller.evaluate_and_adjust(
                    k_hist, k_pred, k_sol, nfev, current_step_lambda, success, current_progress)

                if action == 'accept':
                    k_hist.append(k_sol)
                    lambda_hist.append(next_lambda)
                    pred_hist.append(k_pred)
                    for i, name in enumerate(param_names):
                        params_hist[name].append(next_p[i])
                    iter_hist.append(nfev)
                    zeta_hist.append(self._calculate_zeta(k_sol, w_fixed))
                    current_lambda = next_lambda

                    adaptive_step = payload
                    suggested_step = max((1.0 - current_lambda) / remaining_steps, 0.05)
                    current_step_lambda = min(adaptive_step, suggested_step * 1.2)

                    is_accepted = True
                elif action == 'retry':
                    current_step_lambda = payload
                    k_pred = predictor.predict_next_step(k_hist, lambda_hist, current_step_lambda)
                    retry_count += 1

            if retry_count >= max_retries:
                raise RuntimeError(f"阶段2在 λ={current_lambda:.3f} 处重试次数过多")

        print(f"阶段2完成，共 {len(k_hist)} 步\n")
        return k_hist, params_hist, pred_hist, iter_hist, zeta_hist

    def _update_params_and_solve(self, w, k_pred, **params):
        """更新参数并求解"""
        for name, val in params.items():
            if name == 'n':
                self.current_n = val
            elif name == 'T_par':
                self.current_T_par = val
            elif name == 'T_perp':
                self.current_T_perp = val
            elif name == 'v':
                self.current_v = val
            elif name == 'nu':
                self.current_nu = val
        self._update_derived_quantities()
        return self._solve_single_k(k_pred, w)

    def _solve_homotopy_vector_with_w(self, k_start, params_start, params_target, param_names, target_steps=20):
        """
        向量同伦（包含频率w）

        params_start: dict of parameter values at start (including 'w')
        params_target: dict of parameter values at end (including 'w')
        param_names: list of parameter names to vary (must include 'w')
        """
        print(f"\n{'='*60}")
        print(f"单阶段向量同伦（包含频率w）")
        print(f"{'='*60}")
        print(f"同伦参数: {param_names}")
        print(f"起始值 -> 目标值:")
        for name in param_names:
            if name == 'w':
                print(f"  {name}: {params_start[name]:.2e} rad/s -> {params_target[name]:.2e} rad/s")
            elif name == 'T_par':
                print(f"  {name}: {params_start[name]/self.e*self.kB:.2f} eV -> {params_target[name]/self.e*self.kB:.2f} eV")
            else:
                print(f"  {name}: {params_start[name]:.2e} -> {params_target[name]:.2e}")
        print(f"目标步数: {target_steps}")

        # 构建向量
        p_start = np.array([params_start[name] for name in param_names])
        p_target = np.array([params_target[name] for name in param_names])

        k_hist, lambda_hist, pred_hist, iter_hist, zeta_hist = [k_start], [0.0], [np.nan], [], []
        params_hist = {name: [params_start[name]] for name in param_names}

        # 设置初始参数
        w_current = params_start['w']
        for name, val in params_start.items():
            if name == 'w':
                continue  # w单独处理
            elif name == 'n':
                self.current_n = val
            elif name == 'T_par':
                self.current_T_par = val
            elif name == 'T_perp':
                self.current_T_perp = val
            elif name == 'v':
                self.current_v = val
            elif name == 'nu':
                self.current_nu = val
        self._update_derived_quantities()

        _, nfev = self._solve_single_k(k_start, w_current)
        iter_hist.append(nfev)
        zeta_hist.append(self._calculate_zeta(k_start, w_current))

        predictor = Predictor()
        corrector = Corrector(lambda k_pred, **kwargs: self._update_params_and_solve_with_w(k_pred, **kwargs))
        step_controller = AdaptiveStepController(target_total_steps=target_steps, phase_name="Vector+w")

        current_lambda = 0.0
        initial_step_lambda = min(0.15, 1.0 / target_steps * 1.5)
        current_step_lambda = initial_step_lambda

        while current_lambda < 1.0:
            remaining_distance = 1.0 - current_lambda
            remaining_steps = max(target_steps - len(k_hist) + 1, 2)
            suggested_step = remaining_distance / remaining_steps

            k_pred = predictor.predict_next_step(k_hist, lambda_hist, current_step_lambda)

            is_accepted = False
            retry_count = 0
            max_retries = 10

            while not is_accepted and retry_count < max_retries:
                if current_lambda + current_step_lambda >= 1.0:
                    current_step_lambda = 1.0 - current_lambda

                next_lambda = current_lambda + current_step_lambda
                next_p = p_start + next_lambda * (p_target - p_start)

                # 构建参数字典（包括w）
                next_params = {name: next_p[i] for i, name in enumerate(param_names)}
                k_sol, nfev, success = corrector.correct_to_solution(k_pred, **next_params)

                current_progress = len(k_hist) / target_steps
                action, payload = step_controller.evaluate_and_adjust(
                    k_hist, k_pred, k_sol, nfev, current_step_lambda, success, current_progress)

                if action == 'accept':
                    k_hist.append(k_sol)
                    lambda_hist.append(next_lambda)
                    pred_hist.append(k_pred)
                    for i, name in enumerate(param_names):
                        params_hist[name].append(next_p[i])
                    iter_hist.append(nfev)
                    zeta_hist.append(self._calculate_zeta(k_sol, next_params['w']))
                    current_lambda = next_lambda

                    adaptive_step = payload
                    suggested_step = max((1.0 - current_lambda) / remaining_steps, 0.05)
                    current_step_lambda = min(adaptive_step, suggested_step * 1.2)

                    is_accepted = True
                elif action == 'retry':
                    current_step_lambda = payload
                    k_pred = predictor.predict_next_step(k_hist, lambda_hist, current_step_lambda)
                    retry_count += 1

            if retry_count >= max_retries:
                raise RuntimeError(f"向量同伦在 λ={current_lambda:.3f} 处重试次数过多")

        print(f"向量同伦完成，共 {len(k_hist)} 步\n")
        return k_hist, params_hist, pred_hist, iter_hist, zeta_hist

    def _update_params_and_solve_with_w(self, k_pred, **params):
        """更新参数并求解（包括w）"""
        w = None
        for name, val in params.items():
            if name == 'w':
                w = val
            elif name == 'n':
                self.current_n = val
            elif name == 'T_par':
                self.current_T_par = val
            elif name == 'T_perp':
                self.current_T_perp = val
            elif name == 'v':
                self.current_v = val
            elif name == 'nu':
                self.current_nu = val

        if w is None:
            raise ValueError("w must be provided in params")

        self._update_derived_quantities()
        return self._solve_single_k(k_pred, w)

    def solve_dispersion_adaptive(self, w_target, target_steps_s1=12, target_steps_s2=8):
        """
        统一的五参数同伦求解

        智能策略：
        - 如果 w_target < 0.9*w_ci: 直接采用单阶段向量同伦，w也作为同伦参数
        - 如果 w_target >= 0.9*w_ci: 两阶段同伦
          - 阶段1: 频率同伦 (w: 0.9*w_ci -> w_target)
          - 阶段2: 向量同伦 (其他参数: 参考值 -> 目标值)
        - v, T_par, n: 若target > ref，直接用target，否则参与同伦
        - nu: 总是参与同伦（无论大小）
        """
        # 参考值定义
        v_ref = 10000.0
        T_par_ref = 10 * self.e / self.kB  # 10 eV
        T_perp_ref = T_par_ref  # 假设各向同性
        n_ref = 1e17
        nu_ref = w_target / 100.0
        w_ref = 0.9 * self.w_c_i

        print("\n" + "="*70)
        print("五参数统一同伦架构")
        print("="*70)
        print(f"参考值:")
        print(f"  w_ref     = {w_ref:.2e} rad/s (0.9*w_ci)")
        print(f"  v_ref     = {v_ref:.2e} m/s")
        print(f"  T_par_ref = {T_par_ref/self.e*self.kB:.2f} eV")
        print(f"  n_ref     = {n_ref:.2e} m^-3")
        print(f"  nu_ref    = {nu_ref:.2e} rad/s")
        print(f"\n目标值:")
        print(f"  w_target     = {w_target:.2e} rad/s")
        print(f"  v_target     = {self.v_target:.2e} m/s")
        print(f"  T_par_target = {self.T_par_target/self.e*self.kB:.2f} eV")
        print(f"  n_target     = {self.n_target:.2e} m^-3")
        print(f"  nu_target    = {self.nu_target:.2e} rad/s")

        # 判断是否需要两阶段同伦
        if w_target < w_ref:
            print(f"\n[模式选择] w_target < w_ref → 采用单阶段向量同伦，w也参与同伦")
            print(f"           从 (w={w_ref:.2e}, ...) 同伦到 (w={w_target:.2e}, ...)")

            # 决策各参数的起始值和是否参与同伦
            params_start = {}
            params_target_dict = {}
            params_vary = ['w']  # w必然参与同伦

            # w的起始和目标
            params_start['w'] = w_ref
            params_target_dict['w'] = w_target

            # v的决策
            if self.v_target > v_ref:
                params_start['v'] = self.v_target
                params_target_dict['v'] = self.v_target
                print(f"[决策] v_target > v_ref → 直接使用目标值 {self.v_target:.2e}")
            else:
                params_start['v'] = v_ref
                params_target_dict['v'] = self.v_target
                params_vary.append('v')
                print(f"[决策] v_target <= v_ref → 参与同伦")

            # T_par的决策
            if self.T_par_target > T_par_ref:
                params_start['T_par'] = self.T_par_target
                params_start['T_perp'] = self.T_perp_target
                params_target_dict['T_par'] = self.T_par_target
                params_target_dict['T_perp'] = self.T_perp_target
                print(f"[决策] T_par_target > T_par_ref → 直接使用目标值 {self.T_par_target/self.e*self.kB:.2f} eV")
            else:
                params_start['T_par'] = T_par_ref
                params_start['T_perp'] = T_perp_ref
                params_target_dict['T_par'] = self.T_par_target
                params_target_dict['T_perp'] = self.T_perp_target
                params_vary.append('T_par')
                print(f"[决策] T_par_target <= T_par_ref → 参与同伦")

            # n的决策
            if self.n_target > n_ref:
                params_start['n'] = self.n_target
                params_target_dict['n'] = self.n_target
                print(f"[决策] n_target > n_ref → 直接使用目标值 {self.n_target:.2e}")
            else:
                params_start['n'] = n_ref
                params_target_dict['n'] = self.n_target
                params_vary.append('n')
                print(f"[决策] n_target <= n_ref → 参与同伦")

            # nu总是参与同伦
            params_start['nu'] = nu_ref
            params_target_dict['nu'] = self.nu_target
            params_vary.append('nu')
            print(f"[决策] nu 总是参与同伦")

            print(f"\n同伦参数: {params_vary}")
            print(f"总目标步数: {target_steps_s1 + target_steps_s2}")

            # 初始化
            self.current_n = params_start['n']
            self.current_T_par = params_start['T_par']
            self.current_T_perp = params_start['T_perp']
            self.current_v = params_start['v']
            self.current_nu = params_start['nu']
            self._update_derived_quantities()

            k_start, _ = self._solve_single_k(self._get_cold_plasma_solution(w_ref), w_ref)
            if k_start is None:
                raise RuntimeError("无法在参考频率和参考参数下找到解")

            # 执行单阶段向量同伦（包含w）
            k_hist, params_hist, pred_hist, iter_hist, zeta_hist = self._solve_homotopy_vector_with_w(
                k_start, params_start, params_target_dict, params_vary,
                target_steps=target_steps_s1 + target_steps_s2)

            # 构建历史数据
            w_hist = params_hist['w'] if 'w' in params_hist else [w_target] * len(k_hist)
            n_hist = params_hist['n'] if 'n' in params_hist else [params_start['n']] * len(k_hist)
            T_par_hist = params_hist['T_par'] if 'T_par' in params_hist else [params_start['T_par']] * len(k_hist)
            v_hist = params_hist['v'] if 'v' in params_hist else [params_start['v']] * len(k_hist)
            nu_hist = params_hist['nu'] if 'nu' in params_hist else [params_start['nu']] * len(k_hist)

            all_histories = {
                'k': k_hist, 'w': w_hist, 'n': n_hist, 'T_par': T_par_hist,
                'v': v_hist, 'nu': nu_hist, 'pred': pred_hist, 'iter': iter_hist, 'zeta': zeta_hist
            }

            # 单阶段的元信息用于绘图
            meta_info = {
                'num_stages': 1,
                'trans_idx': 0,
                'stage1_params': params_vary,
                'stage1_start': params_start,
                'stage1_target': params_target_dict,
                'w_ci': self.w_c_i,
                'e': self.e,
                'kB': self.kB
            }

            return all_histories, meta_info

        else:
            print(f"\n[模式选择] w_target >= w_ref → 采用两阶段同伦")
            print(f"           阶段1: w同伦 ({w_ref:.2e} -> {w_target:.2e})")
            print(f"           阶段2: 其他参数向量同伦")

            # 决策逻辑：阶段1使用哪些值
            params_stage1 = {}
            params_stage2_vary = []  # 需要在阶段2同伦的参数

            # v的决策
            if self.v_target > v_ref:
                params_stage1['v'] = self.v_target
                print(f"\n[决策] v_target > v_ref → 阶段1使用目标值 {self.v_target:.2e}")
            else:
                params_stage1['v'] = v_ref
                params_stage2_vary.append('v')
                print(f"\n[决策] v_target <= v_ref → 阶段1使用参考值，阶段2同伦")

            # T_par的决策
            if self.T_par_target > T_par_ref:
                params_stage1['T_par'] = self.T_par_target
                params_stage1['T_perp'] = self.T_perp_target
                print(f"[决策] T_par_target > T_par_ref → 阶段1使用目标值 {self.T_par_target/self.e*self.kB:.2f} eV")
            else:
                params_stage1['T_par'] = T_par_ref
                params_stage1['T_perp'] = T_perp_ref
                params_stage2_vary.append('T_par')
                print(f"[决策] T_par_target <= T_par_ref → 阶段1使用参考值，阶段2同伦")

            # n的决策
            if self.n_target > n_ref:
                params_stage1['n'] = self.n_target
                print(f"[决策] n_target > n_ref → 阶段1使用目标值 {self.n_target:.2e}")
            else:
                params_stage1['n'] = n_ref
                params_stage2_vary.append('n')
                print(f"[决策] n_target <= n_ref → 阶段1使用参考值，阶段2同伦")

            # nu总是阶段2同伦
            params_stage1['nu'] = nu_ref
            params_stage2_vary.append('nu')
            print(f"[决策] nu 总是在阶段2同伦")

            print(f"\n阶段2将同伦的参数: {params_stage2_vary}")
            print(f"总目标步数: {target_steps_s1} (阶段1) + {target_steps_s2} (阶段2) = {target_steps_s1 + target_steps_s2}")

            # 初始化
            self.current_n = params_stage1['n']
            self.current_T_par = params_stage1['T_par']
            self.current_T_perp = params_stage1['T_perp']
            self.current_v = params_stage1['v']
            self.current_nu = params_stage1['nu']
            self._update_derived_quantities()

            k_start, _ = self._solve_single_k(self._get_cold_plasma_solution(w_ref), w_ref)
            if k_start is None:
                raise RuntimeError("无法在起始频率和参考参数下找到解")

            # 阶段1：频率同伦
            k_h1, w_h1, pred_h1, i_h1, z_h1 = self._solve_homotopy_scalar(
                k_start, w_ref, w_target, params_stage1, target_steps=target_steps_s1)

            # 阶段2：向量同伦（仅对需要变化的参数）
            k_start_s2 = k_h1[-1]

            if len(params_stage2_vary) == 0:
                # 无需阶段2
                print("所有参数已在阶段1达到目标，跳过阶段2")
                k_h2 = [k_start_s2]
                params_h2 = {name: [params_stage1[name]] for name in ['n', 'T_par', 'v', 'nu']}
                pred_h2 = [np.nan]
                i_h2 = []
                z_h2 = []
            else:
                # 构建阶段2的起始和目标参数
                params_s2_start = dict(params_stage1)  # 从阶段1的结束状态开始
                params_s2_target = dict(params_stage1)  # 复制一份，然后更新需要变化的

                # 更新目标值
                for name in params_stage2_vary:
                    if name == 'v':
                        params_s2_target['v'] = self.v_target
                    elif name == 'T_par':
                        params_s2_target['T_par'] = self.T_par_target
                        params_s2_target['T_perp'] = self.T_perp_target
                    elif name == 'n':
                        params_s2_target['n'] = self.n_target
                    elif name == 'nu':
                        params_s2_target['nu'] = self.nu_target

                k_h2, params_h2, pred_h2, i_h2, z_h2 = self._solve_homotopy_vector(
                    w_target, k_start_s2, params_s2_start, params_s2_target, params_stage2_vary,
                    target_steps=target_steps_s2)

                # 补充未变化的参数历史
                for name in ['n', 'T_par', 'T_perp', 'v', 'nu']:
                    if name not in params_h2:
                        params_h2[name] = [params_stage1[name]] * len(k_h2)

            # 合并历史
            trans_idx = len(k_h1)
            k_hist = k_h1 + k_h2[1:]
            w_hist = w_h1 + [w_target] * (len(k_h2) - 1)

            # 合并参数历史
            n_hist = [params_stage1['n']] * len(k_h1) + params_h2['n'][1:]
            T_par_hist = [params_stage1['T_par']] * len(k_h1) + params_h2['T_par'][1:]
            v_hist = [params_stage1['v']] * len(k_h1) + params_h2['v'][1:]
            nu_hist = [params_stage1['nu']] * len(k_h1) + params_h2['nu'][1:]

            pred_hist = pred_h1 + pred_h2[1:]
            iter_hist = i_h1 + i_h2[1:] if len(i_h2) > 0 else i_h1
            zeta_hist = z_h1 + z_h2[1:] if len(z_h2) > 0 else z_h1

            all_histories = {
                'k': k_hist, 'w': w_hist, 'n': n_hist, 'T_par': T_par_hist,
                'v': v_hist, 'nu': nu_hist, 'pred': pred_hist, 'iter': iter_hist, 'zeta': zeta_hist
            }

            # 两阶段的元信息用于绘图
            meta_info = {
                'num_stages': 2,
                'trans_idx': trans_idx,
                'stage2_params': params_stage2_vary,
                'stage2_start': params_s2_start if len(params_stage2_vary) > 0 else {},
                'stage2_target': params_s2_target if len(params_stage2_vary) > 0 else {},
                'w_start': w_ref,
                'w_target': w_target,
                'w_ci': self.w_c_i,
                'e': self.e,
                'kB': self.kB
            }

            return all_histories, meta_info

if __name__ == '__main__':
    e, kB = 1.602e-19, 1.38e-23
    n_p, T_i_ev, B0, Z_number = 1e18, 3, 1.4, 40
    T_par_i = T_i_ev * e / kB
    T_perp_i = T_i_ev * e / kB
    w_target = 1e6

    v_i_target = 400
    nu_i_target = 1.8e5

    start_time = time.time()
    solver = DispersionSolver(B0, n_p, T_par_i, T_perp_i, Z_number=Z_number,
                              v_i=v_i_target, nu_i=nu_i_target)
    w_ci = solver.w_c_i
    print(f"\n离子回旋频率 w_ci = {w_ci:.3e} rad/s")
    print(f"目标频率 w = {w_target:.3e} rad/s (w/w_ci = {w_target/w_ci:.3f})")

    # 可调整目标步数
    histories, meta_info = solver.solve_dispersion_adaptive(w_target, target_steps_s1=20, target_steps_s2=12)

    end_time = time.time()
    print("\n" + "="*70)
    print(f"Computation completed!")
    print(f"  Time elapsed: {end_time - start_time:.5f} s")
    print(f"  Total steps: {len(histories['k'])}")
    print(f"  Stage transition index: {meta_info['trans_idx']}")
    print("="*70)

    final_k = histories['k'][-1]
    print(f"\nFinal solution: k = {np.real(final_k):.4f} + {np.imag(final_k):.4f}j")

    # Extract data
    k_array = np.array(histories['k'])
    w_array = np.array(histories['w'])
    iter_array = np.array(histories['iter'])
    zeta_array = np.array(histories['zeta'])
    n_total = len(k_array)
    trans_idx = meta_info['trans_idx']
    num_stages = meta_info['num_stages']

    # Color scheme
    color_real = '#4A7BB7'  # Blue for real part
    color_imag = '#CC334C'  # Red for imaginary part

    # ========== Figure 1: Wavenumber vs. Homotopy Parameter ==========
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    if num_stages == 1:
        # Single-stage vector homotopy
        lambda_vals = np.linspace(0, 1, n_total)

        ax1.plot(lambda_vals, np.real(k_array), 'o-', color=color_real,
                markersize=4, linewidth=1.5, label='Re(k)')
        ax1.plot(lambda_vals, np.imag(k_array), 'o-', color=color_imag,
                markersize=4, linewidth=1.5, label='Im(k)')

        ax1.set_xlabel(r'Homotopy parameter $\lambda$', fontsize=12)
        ax1.set_ylabel('Wavenumber k', fontsize=12)
        ax1.set_title('Kinetic Dispersion: Single-Stage Vector Homotopy', fontsize=13)

        # Add parameter info as text below x-axis
        param_names = meta_info['stage1_params']
        param_start = meta_info['stage1_start']
        param_target = meta_info['stage1_target']

        param_text_lines = []
        for pname in param_names:
            if pname == 'w':
                param_text_lines.append(f"$\\omega$: {param_start[pname]:.2e} → {param_target[pname]:.2e} rad/s")
            elif pname == 'T_par':
                T_start_eV = param_start[pname] / e * kB
                T_target_eV = param_target[pname] / e * kB
                param_text_lines.append(f"$T_{{||}}$: {T_start_eV:.1f} → {T_target_eV:.1f} eV")
            elif pname == 'v':
                param_text_lines.append(f"$v$: {param_start[pname]:.1e} → {param_target[pname]:.1e} m/s")
            elif pname == 'nu':
                param_text_lines.append(f"$\\nu$: {param_start[pname]:.1e} → {param_target[pname]:.1e} rad/s")
            elif pname == 'n':
                param_text_lines.append(f"$n$: {param_start[pname]:.1e} → {param_target[pname]:.1e} m⁻³")

        param_text = '; '.join(param_text_lines)
        ax1.text(0.5, -0.18, param_text, transform=ax1.transAxes,
                ha='center', va='top', fontsize=10, style='italic')

        ax1.legend(loc='best', fontsize=11)
        ax1.grid(False)

    else:
        # Two-stage homotopy: single plot with combined x-axis
        # Construct combined x-axis values
        w_normalized_s1 = w_array[:trans_idx] / w_ci
        lambda_vals_s2 = np.linspace(0, 1, n_total - trans_idx + 1)

        # Create combined x array: normalize stage 1 to [0, trans_idx-1], stage 2 to [trans_idx-1, n_total-1]
        x_combined = np.arange(n_total)

        # Plot continuous lines
        ax1.plot(x_combined, np.real(k_array), 'o-', color=color_real,
                markersize=4, linewidth=1.5, label='Re(k)')
        ax1.plot(x_combined, np.imag(k_array), 'o-', color=color_imag,
                markersize=4, linewidth=1.5, label='Im(k)')

        # Add vertical dividing line
        ax1.axvline(trans_idx - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)

        # Set up dual x-axis labels
        # Stage 1 ticks - reduce number to avoid overlap
        n_ticks_s1 = min(4, max(2, trans_idx))
        if trans_idx > 0:
            tick_indices_s1 = np.linspace(0, trans_idx-1, n_ticks_s1, dtype=int)
            tick_labels_s1 = [f"{w_array[i]/w_ci:.2f}" for i in tick_indices_s1]
        else:
            tick_indices_s1 = []
            tick_labels_s1 = []

        # Stage 2 ticks - reduce number to avoid overlap
        n_ticks_s2 = min(4, max(2, n_total - trans_idx + 1))
        if n_total > trans_idx:
            tick_indices_s2 = np.linspace(trans_idx, n_total-1, n_ticks_s2, dtype=int)
            # Calculate corresponding lambda values
            tick_labels_s2 = [f"{(i - trans_idx + 1) / (n_total - trans_idx):.2f}"
                             for i in tick_indices_s2]
        else:
            tick_indices_s2 = []
            tick_labels_s2 = []

        # Combine ticks
        all_tick_indices = list(tick_indices_s1) + list(tick_indices_s2)
        all_tick_labels = list(tick_labels_s1) + list(tick_labels_s2)

        ax1.set_xticks(all_tick_indices)
        ax1.set_xticklabels(all_tick_labels, fontsize=9, rotation=0)

        # Add x-axis label with two parts
        ax1.set_ylabel('Wavenumber k', fontsize=12)
        ax1.set_title('Kinetic Dispersion: Two-Stage Homotopy', fontsize=13)

        # Add text labels for the two regions using text annotations
        # Position these between the plot and the tick labels
        if trans_idx > 0:
            ax1.text((trans_idx-1)/2, -0.15, r'$\omega / \omega_{ci}$',
                    ha='center', va='top', fontsize=11, weight='bold',
                    transform=ax1.transAxes)
        if n_total > trans_idx:
            # Calculate the relative position for stage 2 label
            stage2_center_rel = (trans_idx - 0.5) / n_total + ((n_total - trans_idx) / 2) / n_total
            ax1.text(stage2_center_rel, -0.15, r'$\lambda$',
                    ha='center', va='top', fontsize=11, weight='bold',
                    transform=ax1.transAxes)

        # Add parameter info for stage 2
        if len(meta_info['stage2_params']) > 0:
            param_names_s2 = meta_info['stage2_params']
            param_start_s2 = meta_info['stage2_start']
            param_target_s2 = meta_info['stage2_target']

            param_text_lines = []
            for pname in param_names_s2:
                if pname == 'T_par':
                    T_start_eV = param_start_s2[pname] / e * kB
                    T_target_eV = param_target_s2[pname] / e * kB
                    param_text_lines.append(f"$T_{{||}}$: {T_start_eV:.1f} → {T_target_eV:.1f} eV")
                elif pname == 'v':
                    param_text_lines.append(f"$v$: {param_start_s2[pname]:.1e} → {param_target_s2[pname]:.1e} m/s")
                elif pname == 'nu':
                    param_text_lines.append(f"$\\nu$: {param_start_s2[pname]:.1e} → {param_target_s2[pname]:.1e} rad/s")
                elif pname == 'n':
                    param_text_lines.append(f"$n$: {param_start_s2[pname]:.1e} → {param_target_s2[pname]:.1e} m⁻³")

            param_text = 'Stage 2: ' + '; '.join(param_text_lines)
            ax1.text(0.98, 0.02, param_text, transform=ax1.transAxes,
                    ha='right', va='bottom', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax1.legend(loc='best', fontsize=11)
        ax1.grid(False)

    # ========== Figure 2: Root Locus in Complex k-plane ==========
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    x_re, y_im = np.real(k_array), np.imag(k_array)

    if num_stages == 1:
        # Single stage
        param_names = meta_info['stage1_params']
        param_label = ', '.join([p if p != 'T_par' else '$T_{||}$' for p in param_names])

        ax2.plot(x_re, y_im, 'o-', color='black', markersize=5,
                linewidth=1.5, label=f'Stage 1: {param_label} homotopy')

    else:
        # Two stages
        # Stage 1
        ax2.plot(x_re[:trans_idx], y_im[:trans_idx], 'o-', color='black',
                markersize=5, linewidth=1.5, label='Stage 1: $\\omega$ homotopy')
        # Stage 2
        if trans_idx < n_total:
            # Get stage 2 parameter labels
            if len(meta_info['stage2_params']) > 0:
                param_names_s2 = meta_info['stage2_params']
                param_label_s2 = ', '.join([p if p != 'T_par' else '$T_{||}$' for p in param_names_s2])
                stage2_label = f'Stage 2: {param_label_s2} homotopy'
            else:
                stage2_label = 'Stage 2: vector homotopy'

            ax2.plot(x_re[trans_idx-1:], y_im[trans_idx-1:], 'o--', color='gray',
                    markersize=5, linewidth=1.5, label=stage2_label)

    ax2.set_xlabel('Re(k)', fontsize=12)
    ax2.set_ylabel('Im(k)', fontsize=12)
    ax2.set_title('Root Locus in Complex k-Plane', fontsize=13)
    ax2.grid(False)
    ax2.legend(loc='best', fontsize=10)

    # Set adaptive axis limits with small margins
    re_min, re_max = np.min(x_re), np.max(x_re)
    im_min, im_max = np.min(y_im), np.max(y_im)

    re_range = re_max - re_min
    im_range = im_max - im_min

    # Add 5% margin on each side
    margin_re = 0.05 * re_range if re_range > 0 else 0.1
    margin_im = 0.05 * im_range if im_range > 0 else 0.1

    # Calculate ranges with margins
    re_with_margin = re_range + 2 * margin_re
    im_with_margin = im_range + 2 * margin_im

    # Use the larger range for both axes to create a square plot
    max_range = max(re_with_margin, im_with_margin)

    # Calculate centers
    re_center = (re_min + re_max) / 2
    im_center = (im_min + im_max) / 2

    # Set equal ranges centered on data
    ax2.set_xlim(re_center - max_range/2, re_center + max_range/2)
    ax2.set_ylim(im_center - max_range/2, im_center + max_range/2)

    ax2.set_aspect('equal', adjustable='box')

    # ========== Figure 3: Solver Performance & Zeta ==========
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    p_indices = np.arange(n_total)

    # Top panel: Iterations
    ax3.plot(p_indices, iter_array, 'o-', color='black', markersize=4, linewidth=1.2)
    if trans_idx > 0 and trans_idx < n_total:
        ax3.axvline(trans_idx - 0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.6)
    ax3.set_ylabel('Solver iterations', fontsize=12)
    ax3.set_title('Solver Performance vs. Point Index', fontsize=13)
    ax3.grid(False)

    # Bottom panel: Zeta
    ax4.plot(p_indices, np.real(zeta_array), 'o-', color=color_real,
            markersize=4, linewidth=1.2, label=r'Re($\zeta$)')
    ax4.plot(p_indices, np.imag(zeta_array), 'o-', color=color_imag,
            markersize=4, linewidth=1.2, label=r'Im($\zeta$)')
    if trans_idx > 0 and trans_idx < n_total:
        ax4.axvline(trans_idx - 0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.6)
    ax4.set_xlabel('Point index', fontsize=12)
    ax4.set_ylabel(r'$\zeta$ parameter', fontsize=12)
    ax4.grid(False)
    ax4.legend(loc='best', fontsize=11)

    plt.tight_layout()
    plt.show()
