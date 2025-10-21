# -*- coding: utf-8 -*-
"""
热等离子体色散关系求解结果绘图模块 v1.0

独立的绘图功能，从主求解器解耦

作者：重构自 1012-claude-1+v-4.py
日期：2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ============================================================================
# 配色方案
# ============================================================================

class ColorScheme:
    """统一的颜色方案"""
    REAL = '#4A7BB7'      # 蓝色：实部
    IMAG = '#CC334C'      # 红色：虚部
    STAGE1 = 'black'      # 黑色：阶段1
    STAGE2 = 'gray'       # 灰色：阶段2
    DIVIDER = 'black'     # 黑色虚线：阶段分界


# ============================================================================
# 主绘图函数
# ============================================================================

def plot_all_results(results: dict, save_path: Optional[str] = None):
    """
    绘制所有结果图

    Args:
        results: solve_dispersion_relation() 返回的结果字典
        save_path: 保存路径（可选，不提供则显示）
    """
    # 提取数据
    k_array = results['k_history']
    w_array = results['w_history']
    state_history = results['state_history']
    iter_array = results['iterations']
    trans_indices = results['trans_indices']
    num_stages = results['num_stages']
    constants = results['constants']

    # 计算 zeta 历史（用于诊断）
    zeta_array = np.array([
        _compute_zeta_for_plot(w_array[i], k_array[i], state_history[i])
        for i in range(len(k_array))
    ])

    n_total = len(k_array)
    trans_idx = trans_indices[1] if len(trans_indices) > 1 else n_total

    # 计算 w_c（用于归一化）
    w_c = state_history[0].w_c

    # 创建三个图形
    fig1 = plot_wavenumber_vs_homotopy(
        k_array, w_array, w_c, trans_idx, num_stages, constants
    )

    fig2 = plot_root_locus(
        k_array, trans_idx, num_stages
    )

    fig3 = plot_solver_diagnostics(
        k_array, iter_array, zeta_array, trans_idx, n_total
    )

    # 显示或保存
    if save_path:
        fig1.savefig(f"{save_path}_wavenumber.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"{save_path}_locus.png", dpi=300, bbox_inches='tight')
        fig3.savefig(f"{save_path}_diagnostics.png", dpi=300, bbox_inches='tight')
        print(f"\n图形已保存到 {save_path}_*.png")
    else:
        plt.show()


def _compute_zeta_for_plot(w: complex, k: complex, state) -> complex:
    """计算用于绘图的 zeta 值（ζ_+1）"""
    p = state.params
    w_eff = w + 1j * p.nu
    return (w_eff - k * p.v - state.w_c) / (k * state.v_th_par)


# ============================================================================
# 图1：波数 vs 同伦参数
# ============================================================================

def plot_wavenumber_vs_homotopy(k_array, w_array, w_c, trans_idx, num_stages, constants):
    """
    绘制波数随同伦参数的变化

    单阶段：横轴为 λ
    两阶段：横轴为阶段1的 ω/ωc，阶段2的 λ
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    n_total = len(k_array)

    if num_stages == 1:
        # 单阶段绘图：频率 + 参数向量同伦
        lambda_vals = np.linspace(0, 1, n_total)
        ax.plot(lambda_vals, np.real(k_array), 'o-', color=ColorScheme.REAL,
               markersize=4, linewidth=1.5, label='Re(k)')
        ax.plot(lambda_vals, np.imag(k_array), 'o-', color=ColorScheme.IMAG,
               markersize=4, linewidth=1.5, label='Im(k)')
        ax.set_xlabel(r'Homotopy parameter $\lambda$', fontsize=12)
        ax.set_title('Kinetic Dispersion: Single-Stage Vector Homotopy (with $\omega$)', fontsize=13)

        # 添加参数信息（显示频率范围）
        w_start = w_array[0]
        w_end = w_array[-1]
        param_text = f"$\omega$: {w_start/w_c:.2f}$\omega_c$ → {w_end/w_c:.2f}$\omega_c$"
        ax.text(0.5, -0.15, param_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=10, style='italic')

    else:
        # 两阶段绘图
        x_combined = np.arange(n_total)

        # 绘制连续线
        ax.plot(x_combined, np.real(k_array), 'o-', color=ColorScheme.REAL,
               markersize=4, linewidth=1.5, label='Re(k)')
        ax.plot(x_combined, np.imag(k_array), 'o-', color=ColorScheme.IMAG,
               markersize=4, linewidth=1.5, label='Im(k)')

        # 添加阶段分界线
        if trans_idx > 0 and trans_idx < n_total:
            ax.axvline(trans_idx - 0.5, color=ColorScheme.DIVIDER,
                      linestyle='--', linewidth=2, alpha=0.7)

        # 设置双轴标签
        # 阶段1：ω/ωc
        if trans_idx > 0:
            n_ticks_s1 = min(4, max(2, trans_idx))
            tick_indices_s1 = np.linspace(0, trans_idx-1, n_ticks_s1, dtype=int)
            tick_labels_s1 = [f"{w_array[i]/w_c:.2f}" for i in tick_indices_s1]
        else:
            tick_indices_s1 = []
            tick_labels_s1 = []

        # 阶段2：λ
        if n_total > trans_idx:
            n_ticks_s2 = min(4, max(2, n_total - trans_idx + 1))
            tick_indices_s2 = np.linspace(trans_idx, n_total-1, n_ticks_s2, dtype=int)
            tick_labels_s2 = [f"{(i - trans_idx + 1) / (n_total - trans_idx):.2f}"
                            for i in tick_indices_s2]
        else:
            tick_indices_s2 = []
            tick_labels_s2 = []

        # 合并刻度
        all_tick_indices = list(tick_indices_s1) + list(tick_indices_s2)
        all_tick_labels = list(tick_labels_s1) + list(tick_labels_s2)

        ax.set_xticks(all_tick_indices)
        ax.set_xticklabels(all_tick_labels, fontsize=9)

        # 添加轴标签说明
        if trans_idx > 0:
            ax.text(0.25, -0.15, r'$\omega / \omega_{ci}$',
                   ha='center', va='top', fontsize=11, weight='bold',
                   transform=ax.transAxes)
        if n_total > trans_idx:
            stage2_center_rel = (trans_idx - 0.5) / n_total + ((n_total - trans_idx) / 2) / n_total
            ax.text(stage2_center_rel, -0.15, r'$\lambda$',
                   ha='center', va='top', fontsize=11, weight='bold',
                   transform=ax.transAxes)

        ax.set_title('Kinetic Dispersion: Two-Stage Homotopy', fontsize=13)

    ax.set_ylabel('Wavenumber k', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig


# ============================================================================
# 图2：复平面根轨迹
# ============================================================================

def plot_root_locus(k_array, trans_idx, num_stages):
    """
    绘制复k平面的根轨迹
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    x_re = np.real(k_array)
    y_im = np.imag(k_array)
    n_total = len(k_array)

    if num_stages == 1:
        ax.plot(x_re, y_im, 'o-', color=ColorScheme.STAGE1,
               markersize=5, linewidth=1.5, label='Single stage')

    else:
        # 阶段1
        if trans_idx > 0:
            ax.plot(x_re[:trans_idx], y_im[:trans_idx], 'o-',
                   color=ColorScheme.STAGE1, markersize=5, linewidth=1.5,
                   label='Stage 1: $\\omega$ homotopy')

        # 阶段2
        if trans_idx < n_total:
            ax.plot(x_re[trans_idx-1:], y_im[trans_idx-1:], 'o--',
                   color=ColorScheme.STAGE2, markersize=5, linewidth=1.5,
                   label='Stage 2: parameter homotopy')

    ax.set_xlabel('Re(k)', fontsize=12)
    ax.set_ylabel('Im(k)', fontsize=12)
    ax.set_title('Root Locus in Complex k-Plane', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # 设置等比例坐标轴（正方形图）
    re_min, re_max = np.min(x_re), np.max(x_re)
    im_min, im_max = np.min(y_im), np.max(y_im)

    re_range = re_max - re_min
    im_range = im_max - im_min

    margin_re = 0.05 * re_range if re_range > 0 else 0.1
    margin_im = 0.05 * im_range if im_range > 0 else 0.1

    re_with_margin = re_range + 2 * margin_re
    im_with_margin = im_range + 2 * margin_im

    max_range = max(re_with_margin, im_with_margin)

    re_center = (re_min + re_max) / 2
    im_center = (im_min + im_max) / 2

    ax.set_xlim(re_center - max_range/2, re_center + max_range/2)
    ax.set_ylim(im_center - max_range/2, im_center + max_range/2)
    ax.set_aspect('equal', adjustable='box')

    return fig


# ============================================================================
# 图3：求解器性能诊断
# ============================================================================

def plot_solver_diagnostics(k_array, iter_array, zeta_array, trans_idx, n_total):
    """
    绘制求解器性能和 zeta 参数

    上图：迭代次数
    下图：zeta 的实部和虚部
    """
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    p_indices = np.arange(n_total)

    # 上图：迭代次数
    ax_top.plot(p_indices, iter_array, 'o-', color='black',
               markersize=4, linewidth=1.2)

    if trans_idx > 0 and trans_idx < n_total:
        ax_top.axvline(trans_idx - 0.5, color=ColorScheme.DIVIDER,
                      linestyle='--', linewidth=1.5, alpha=0.6)

    ax_top.set_ylabel('Solver iterations', fontsize=12)
    ax_top.set_title('Solver Performance vs. Point Index', fontsize=13)
    ax_top.grid(True, alpha=0.3)

    # 下图：zeta 参数
    ax_bottom.plot(p_indices, np.real(zeta_array), 'o-',
                  color=ColorScheme.REAL, markersize=4, linewidth=1.2,
                  label=r'Re($\zeta_{+1}$)')
    ax_bottom.plot(p_indices, np.imag(zeta_array), 'o-',
                  color=ColorScheme.IMAG, markersize=4, linewidth=1.2,
                  label=r'Im($\zeta_{+1}$)')

    if trans_idx > 0 and trans_idx < n_total:
        ax_bottom.axvline(trans_idx - 0.5, color=ColorScheme.DIVIDER,
                         linestyle='--', linewidth=1.5, alpha=0.6)

    ax_bottom.set_xlabel('Point index', fontsize=12)
    ax_bottom.set_ylabel(r'$\zeta_{+1}$ parameter', fontsize=12)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc='best', fontsize=11)

    plt.tight_layout()

    return fig


# ============================================================================
# 独立测试（如果直接运行此文件）
# ============================================================================

if __name__ == '__main__':
    print("这是一个绘图模块，请从 kd_v1-0.py 中导入使用")
    print("\n示例用法：")
    print("    from kdlogdraw_v1_0 import plot_all_results")
    print("    results = solve_dispersion_relation(...)")
    print("    plot_all_results(results)")
