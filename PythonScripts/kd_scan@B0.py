# -*- coding: utf-8 -*-
"""
热等离子体色散关系求解器 - B0扫描脚本

扫描磁场 B0，绘制：
1. 波数 k 随 B0 的变化
2. 介电张量分量 K_perp, K_g, K_par 随 B0 的变化

作者：基于 kd_v1-0.py
日期：2025
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

# 导入带连字符的模块 kd_v1-0.py
spec = importlib.util.spec_from_file_location("kd_v1_0", "/Users/merplateau/Projects/18Oct-Share/PythonScripts/kd_v1-0.py")
kd_module = importlib.util.module_from_spec(spec)
sys.modules["kd_v1_0"] = kd_module
spec.loader.exec_module(kd_module)
solve_dispersion_relation = kd_module.solve_dispersion_relation

# 设置中文字体支持和LaTeX渲染
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC']  # 宋体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False  # 关闭完整LaTeX（避免依赖问题）
plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体渲染数学公式
plt.rcParams['font.family'] = 'sans-serif'

# ============================================================================
# 扫描参数设置
# ============================================================================

# B0 扫描范围
B0_values = np.linspace(1.2, 1.5, 300)  # 从 0.5 T 到 2.0 T，20个点

# 固定物理参数（与 kd_v1-0.py 保持一致）
n_p = 1e18        # 密度 [m^-3]
T_i_eV = 3     # 温度 [eV]
v_i = 20000.0        # 速度 [m/s]
nu_i = 5e4        # 碰撞频率 [rad/s]
w_target = 3.3e6  # 目标频率 [rad/s]
Z_number = 40     # 离子电荷数

print("="*70)
print("B0 扫描开始")
print("="*70)
print(f"B0 范围: {B0_values[0]:.2f} T 到 {B0_values[-1]:.2f} T")
print(f"扫描点数: {len(B0_values)}")
print(f"\n固定参数:")
print(f"  n = {n_p:.2e} m^-3")
print(f"  T = {T_i_eV:.2f} eV")
print(f"  v = {v_i:.2e} m/s")
print(f"  nu = {nu_i:.2e} rad/s")
print(f"  w = {w_target:.2e} rad/s")
print(f"  Z = {Z_number}")
print("="*70)

# ============================================================================
# 执行扫描
# ============================================================================

# 存储结果
k_final_array = []
K_perp_array = []
K_g_array = []
K_par_array = []
zeta_plus1_array = []
zeta_minus1_array = []
zeta_zero_array = []
eta_array = []
successful_B0 = []

for i, B0 in enumerate(B0_values):
    print(f"\n[{i+1}/{len(B0_values)}] 正在计算 B0 = {B0:.3f} T ...")

    try:
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

        # 提取最终值
        k_final_array.append(results['k_final'])
        K_perp_array.append(results['K_perp'])
        K_g_array.append(results['K_g'])
        K_par_array.append(results['K_par'])
        zeta_plus1_array.append(results['zeta_plus1_final'])
        zeta_minus1_array.append(results['zeta_minus1_final'])
        zeta_zero_array.append(results['zeta_zero_final'])

        # 计算 eta = (w - k_par * v_i - w_ci) / w_ci
        # 物理常数
        m_p = 1.67e-27  # 质子质量 [kg]
        e = 1.602e-19   # 元电荷 [C]
        m_i = Z_number * m_p  # 离子质量
        q_i = e  # 离子电荷
        w_ci = q_i * B0 / m_i  # 回旋频率 [rad/s]
        k_par = np.real(results['k_final'])  # 取波数的实部作为平行波数
        eta = (w_target - k_par * v_i - w_ci) / w_ci
        eta_array.append(eta)

        successful_B0.append(B0)

        print(f"  ✓ 成功: k = {results['k_final']:.6f}, η = {eta:.6f}")

    except Exception as e:
        print(f"  ✗ 失败: {str(e)}")
        continue

print("\n" + "="*70)
print(f"扫描完成！成功点数: {len(successful_B0)}/{len(B0_values)}")
print("="*70)

# 转换为 numpy 数组
successful_B0 = np.array(successful_B0)
k_final_array = np.array(k_final_array)
K_perp_array = np.array(K_perp_array)
K_g_array = np.array(K_g_array)
K_par_array = np.array(K_par_array)
zeta_plus1_array = np.array(zeta_plus1_array)
zeta_minus1_array = np.array(zeta_minus1_array)
zeta_zero_array = np.array(zeta_zero_array)
eta_array = np.array(eta_array)

# ============================================================================
# 绘图模板函数
# ============================================================================

def complex_B_1(x_data, y_data, ylabel, title, filename, var_name, y_use_log=False):
    """
    统一的复数数据绘图模板

    参数:
        x_data: 横坐标数据 (通常是 B0)
        y_data: 纵坐标复数数据
        ylabel: 纵坐标标签（LaTeX格式）
        title: 图表标题
        filename: 保存文件名
        var_name: 变量名称（用于图例，LaTeX格式）
        y_use_log: 是否使用对数坐标（默认False）
    """
    plt.figure(figsize=(5, 5))

    # 绘制实部和虚部
    plt.plot(x_data, np.real(y_data), 'o-', color='#1055C9',
             linewidth=2, markersize=0, label=rf'Re({var_name})')
    plt.plot(x_data, np.imag(y_data), 'o-', color='#BF092F',
             linewidth=2, markersize=0, label=rf'Im({var_name})')

    # 设置标签和标题
    plt.xlabel(r'$B_0$ [T]', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20, fontweight='bold')

    # 设置对数坐标
    if y_use_log:
        plt.yscale('log')

    # 刻度字体和坐标轴反转
    plt.tick_params(labelsize=20)
    plt.gca().invert_xaxis()

    # 图例样式
    legend = plt.legend(fontsize=12, frameon=True, fancybox=False,
                       edgecolor='black', framealpha=1.0)
    legend.get_frame().set_linewidth(1.0)

    # 其他设置
    plt.grid(False)
    plt.tight_layout()

    # 保存图片
    plt.savefig(filename, dpi=300)
    print(f"  ✓ 已保存: {filename}")


def complex_B_2(x_data, y_perp, y_phi, y_par, y_k, y_zeta_plus1, filename, y_use_log=False,
                axes_width=4.0, axes_height=4.0,
                n_i=None, T_i_par=None, T_i_perp=None, v_i=None, nu_i=None, w=None, Z=40):
    """
    上下双子图绘图模板

    上图：三个介电张量分量（K_perp, K_phi, K_par）- 双纵轴
    下图：波数 k（左轴）+ zeta_plus1（右轴）- 双纵轴

    参数:
        x_data: 横坐标数据 (通常是 B0)
        y_perp: K_perp 复数数据
        y_phi: K_phi 复数数据
        y_par: K_par 复数数据
        y_k: 波数 k 复数数据
        y_zeta_plus1: zeta_plus1 复数数据
        filename: 保存文件名
        y_use_log: 是否使用对数坐标（默认False）
        axes_width: 每个 axes 绘图区域宽度（英寸），默认 4.0
        axes_height: 每个 axes 绘图区域高度（英寸），默认 4.0
        n_i: 离子密度 [m^-3]
        T_i_par: 平行温度 [eV]
        T_i_perp: 垂直温度 [eV]
        v_i: 漂移速度 [m/s]
        nu_i: 碰撞频率 [rad/s]
        w: 目标频率 [rad/s]
        Z: 离子电荷数，默认 40
    """
    # 计算回旋共振磁场强度 B_ci
    B_ci = None
    if w is not None and Z is not None:
        # 物理常数
        m_p = 1.67e-27  # 质子质量 [kg]
        e = 1.602e-19   # 元电荷 [C]

        m_i = Z * m_p   # 离子质量
        q_i = e         # 离子电荷（单电荷）

        # 回旋共振：w = w_c = q_i * B_ci / m_i
        # 所以 B_ci = w * m_i / q_i
        B_ci = w * m_i / q_i

        # 检查是否在扫描范围内
        B_min = np.min(x_data)
        B_max = np.max(x_data)
        if B_ci < B_min or B_ci > B_max:
            B_ci = None  # 不在范围内，不画线

    # 估算边距（英寸）
    left_margin = 1.2      # 左侧ylabel + 刻度
    right_margin = 1.2     # 右侧ylabel + 刻度
    bottom_margin = 2    # 底部xlabel + 刻度 + 参数文本
    top_margin = 0.3       # 顶部（无标题）
    middle_gap = 0.3       # 两个子图之间的间距

    # 计算 figure 总尺寸（上下两个 5×5 的绘图区域）
    fig_width = axes_width + left_margin + right_margin
    fig_height = 2 * axes_height + bottom_margin + top_margin + middle_gap

    # 创建 figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 上图（介电张量）的位置 [left, bottom, width, height]
    upper_rect = [
        left_margin / fig_width,
        (bottom_margin + axes_height + middle_gap) / fig_height,
        axes_width / fig_width,
        axes_height / fig_height
    ]

    # 下图（波数）的位置
    lower_rect = [
        left_margin / fig_width,
        bottom_margin / fig_height,
        axes_width / fig_width,
        axes_height / fig_height
    ]

    # ========== 上图：介电张量 ==========
    ax1 = fig.add_axes(upper_rect)

    # 左轴：K_perp 和 K_phi
    # K_perp - 实线
    line1 = ax1.plot(x_data, np.real(y_perp), ':', color='#1055C9',
                     linewidth=2, label=r'Re($K_\perp$)')
    line2 = ax1.plot(x_data, np.imag(y_perp), ':', color='#BF092F',
                     linewidth=2, label=r'Im($K_\perp$)')

    # K_phi - 虚线
    line3 = ax1.plot(x_data, np.real(y_phi), '--', color='#1055C9',
                     linewidth=2, label=r'Re($K_\phi$)')
    line4 = ax1.plot(x_data, np.imag(y_phi), '--', color='#BF092F',
                     linewidth=2, label=r'Im($K_\phi$)')

    # 左轴设置（上图不显示 xlabel）
    ax1.set_ylabel(r'$K_\perp$, $K_\phi$', fontsize=20)
    ax1.tick_params(axis='both', labelsize=20, labelbottom=False)  # 隐藏横坐标刻度标签
    ax1.invert_xaxis()

    # 左轴科学计数法设置
    from matplotlib.ticker import ScalarFormatter
    formatter_left = ScalarFormatter(useMathText=False, useOffset=True)
    formatter_left.set_scientific(True)
    formatter_left.set_powerlimits((-2, 2))
    ax1.yaxis.set_major_formatter(formatter_left)
    ax1.yaxis.get_offset_text().set_fontsize(20)  # 科学计数法量级字体大小
    # 强制刷新以应用格式
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2), useMathText=False)

    if y_use_log:
        ax1.set_yscale('log')
    else:
        # 左轴：让0在中间
        left_data = np.concatenate([np.real(y_perp), np.imag(y_perp),
                                    np.real(y_phi), np.imag(y_phi)])
        left_max_abs = np.max(np.abs(left_data))
        ax1.set_ylim(-left_max_abs * 1.05, left_max_abs * 1.05)

    # 右轴：K_par
    ax2 = ax1.twinx()

    # K_par - 点线
    line5 = ax2.plot(x_data, np.real(y_par), '-', color='#1055C9',
                     linewidth=2.5, label=r'Re($K_\parallel$)')
    line6 = ax2.plot(x_data, np.imag(y_par), '-', color='#BF092F',
                     linewidth=2.5, label=r'Im($K_\parallel$)')

    # 右轴设置
    ax2.set_ylabel(r'$K_\parallel$', fontsize=20)
    ax2.tick_params(axis='y', labelsize=20)

    # 右轴科学计数法设置
    formatter_right = ScalarFormatter(useMathText=False, useOffset=True)
    formatter_right.set_scientific(True)
    formatter_right.set_powerlimits((-2, 2))
    ax2.yaxis.set_major_formatter(formatter_right)
    ax2.yaxis.get_offset_text().set_fontsize(20)  # 科学计数法量级字体大小
    # 强制刷新以应用格式
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2), useMathText=False)

    if y_use_log:
        ax2.set_yscale('log')
    else:
        # 右轴：让0在中间
        right_data = np.concatenate([np.real(y_par), np.imag(y_par)])
        right_max_abs = np.max(np.abs(right_data))
        ax2.set_ylim(-right_max_abs * 1.05, right_max_abs * 1.05)

    # 合并图例
    lines = line1 + line2 + line3 + line4 + line5 + line6
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, fontsize=12, frameon=True,
                       fancybox=False, edgecolor='black', framealpha=1.0,
                       loc='best')
    legend.get_frame().set_linewidth(1.0)

    # 画回旋共振磁场强度的垂直线（上图）
    if B_ci is not None:
        ax1.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # ========== 下图：波数 k（左轴）+ zeta_plus1（右轴）==========
    ax3 = fig.add_axes(lower_rect)

    # 左轴：绘制波数 k（实线）
    line_k1 = ax3.plot(x_data, np.real(y_k), '-', color='#1055C9',
                       linewidth=2, label=r'Re($k$)')
    line_k2 = ax3.plot(x_data, np.imag(y_k), '-', color='#BF092F',
                       linewidth=2, label=r'Im($k$)')

    # 左轴设置
    ax3.set_xlabel(r'$B_0$ [T]', fontsize=20)
    ax3.set_ylabel(r'$k$', fontsize=20)
    ax3.tick_params(axis='both', labelsize=20)
    ax3.invert_xaxis()

    # 波数图不使用科学计数法
    formatter_k = ScalarFormatter(useOffset=False)
    formatter_k.set_scientific(False)
    ax3.yaxis.set_major_formatter(formatter_k)

    # 左轴：让0在中间
    k_data = np.concatenate([np.real(y_k), np.imag(y_k)])
    k_max_abs = np.max(np.abs(k_data))
    ax3.set_ylim(-k_max_abs * 1.05, k_max_abs * 1.05)

    # 右轴：zeta_plus1
    ax4 = ax3.twinx()

    # 绘制 zeta_plus1（虚线）
    line_zeta1 = ax4.plot(x_data, np.real(y_zeta_plus1), '--', color='#1055C9',
                          linewidth=2, label=r'Re($\zeta_{+1}$)')
    line_zeta2 = ax4.plot(x_data, np.imag(y_zeta_plus1), '--', color='#BF092F',
                          linewidth=2, label=r'Im($\zeta_{+1}$)')

    # 右轴设置
    ax4.set_ylabel(r'$\zeta_{+1}$', fontsize=20)
    ax4.tick_params(axis='y', labelsize=20)

    # 右轴：让0在中间
    zeta_data = np.concatenate([np.real(y_zeta_plus1), np.imag(y_zeta_plus1)])
    zeta_max_abs = np.max(np.abs(zeta_data))
    ax4.set_ylim(-zeta_max_abs * 1.05, zeta_max_abs * 1.05)

    # 合并下图图例
    lines_lower = line_k1 + line_k2 + line_zeta1 + line_zeta2
    labels_lower = [l.get_label() for l in lines_lower]
    legend3 = ax3.legend(lines_lower, labels_lower, fontsize=12, frameon=True,
                        fancybox=False, edgecolor='black', framealpha=1.0, loc='best')
    legend3.get_frame().set_linewidth(1.0)

    # 画回旋共振磁场强度的垂直线（下图）
    if B_ci is not None:
        ax3.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 添加参数文本（在下图底部下方，距离 xlabel 0.2 英寸）
    if n_i is not None and w is not None:
        # 第一行：ion=Ar, ω=3.3MHz, n_i=...
        w_MHz = w /(2*np.pi*1e3)
        n_i_sci = f"{n_i:.1e}"
        line1 = f"ion=Ar, $\omega$={w_MHz:.1f}kHz, $n_i$={n_i_sci} m$^{{-3}}$"

        # 第二行：T_i∥=...eV, T_i⊥=...eV, ν_i=...Hz
        if T_i_par is not None and T_i_perp is not None and nu_i is not None:
            nu_i_Hz = nu_i / (2 * np.pi * 1e3)  # rad/s -> Hz
            line2 = f"$T_{{i\parallel}}$={T_i_par:.1f}eV, $T_{{i\perp}}$={T_i_perp:.1f}eV, $\\nu_i$={nu_i_Hz:.1f}kHz"
        else:
            line2 = ""

        # 第三行：V_i∥=...m/s
        if v_i is not None:
            line3 = f"$V_{{i\parallel}}$={v_i:.1f}m/s"
        else:
            line3 = ""

        # 组合文本
        param_text = line1
        if line2:
            param_text += "\n" + line2
        if line3:
            param_text += "\n" + line3

        # 计算文本位置（下图底部边缘 - 0.2 英寸）
        # lower_rect[1] 是下图底部在 figure 中的相对位置
        # 需要将 0.2 英寸转换为 figure 的相对坐标
        text_gap_inch = 0.9
        text_y_position = (bottom_margin - text_gap_inch) / fig_height

        # 在图底部添加文本
        fig.text(0.5, text_y_position, param_text, ha='center', fontsize=20,
                verticalalignment='top')  # 使用 top 对齐，从上往下排列

    # 其他设置
    ax1.grid(False)
    ax3.grid(False)
    ax4.grid(False)
    # 注意：不使用 tight_layout()，因为已手动指定 axes 位置

    # 保存图片
    plt.savefig(filename, dpi=300)
    print(f"  ✓ 已保存: {filename}")
    print(f"      每个 axes 绘图区域: {axes_width} × {axes_height} 英寸")
    print(f"      图片总尺寸: {fig_width:.2f} × {fig_height:.2f} 英寸 = {int(fig_width*300)} × {int(fig_height*300)} 像素 @ 300dpi")


def real_B_1(x_data, y_data, ylabel, filename, axes_width=4.0, axes_height=4.0):
    """
    单子图实数数据绘图模板

    用于绘制实数数据（非复数），强制纵轴0在中间

    参数:
        x_data: 横坐标数据 (通常是 B0)
        y_data: 纵坐标实数数据
        ylabel: 纵坐标标签（LaTeX格式）
        filename: 保存文件名
        axes_width: axes 绘图区域宽度（英寸），默认 4.0
        axes_height: axes 绘图区域高度（英寸），默认 4.0
    """
    # 估算边距（英寸），参考 complex_B_2
    left_margin = 1.2      # 左侧ylabel + 刻度
    right_margin = 0.3     # 右侧留白（单轴图不需要右侧ylabel）
    bottom_margin = 0.8    # 底部xlabel + 刻度
    top_margin = 0.3       # 顶部留白

    # 计算 figure 总尺寸
    fig_width = axes_width + left_margin + right_margin
    fig_height = axes_height + bottom_margin + top_margin

    # 创建 figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # axes 位置 [left, bottom, width, height]
    axes_rect = [
        left_margin / fig_width,
        bottom_margin / fig_height,
        axes_width / fig_width,
        axes_height / fig_height
    ]

    # 创建 axes
    ax = fig.add_axes(axes_rect)

    # 绘制实数数据（蓝色实线）
    ax.plot(x_data, y_data, '-', color='#1055C9', linewidth=2)

    # 设置标签
    ax.set_xlabel(r'$B_0$ [T]', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.invert_xaxis()

    # 强制纵轴0在中间
    y_max_abs = np.max(np.abs(y_data))
    ax.set_ylim(-y_max_abs * 1.05, y_max_abs * 1.05)

    # 添加水平参考线 y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 其他设置
    ax.grid(False)

    # 保存图片
    plt.savefig(filename, dpi=300)
    print(f"  ✓ 已保存: {filename}")
    print(f"      axes 绘图区域: {axes_width} × {axes_height} 英寸")
    print(f"      图片总尺寸: {fig_width:.2f} × {fig_height:.2f} 英寸 = {int(fig_width*300)} × {int(fig_height*300)} 像素 @ 300dpi")


# ============================================================================
# 绘图
# ============================================================================

print("\n开始绘图...")

# 图1：波数 k vs B0
complex_B_1(
    x_data=successful_B0,
    y_data=k_final_array,
    ylabel=r'$k$ [m$^{-1}$]',
    title=r'波数 $k$ 随磁场 $B_0$ 的变化',
    filename='k_vs_B0.png',
    var_name=r'$k$',
    y_use_log=False
)

# 图2：介电张量分量 K_perp vs B0
complex_B_1(
    x_data=successful_B0,
    y_data=K_perp_array,
    ylabel=r'$K_\perp$',
    title=r'介电张量垂直分量 $K_\perp$ 随磁场 $B_0$ 的变化',
    filename='K_perp_vs_B0.png',
    var_name=r'$K_\perp$',
    y_use_log=False
)

# 图3：介电张量分量 K_g vs B0
complex_B_1(
    x_data=successful_B0,
    y_data=K_g_array,
    ylabel=r'$K_\phi$',
    title=r'介电张量角向分量 $K_\phi$ 随磁场 $B_0$ 的变化',
    filename='K_phi_vs_B0.png',
    var_name=r'$K_\phi$',
    y_use_log=False
)

# 图4：介电张量分量 K_par vs B0
complex_B_1(
    x_data=successful_B0,
    y_data=K_par_array,
    ylabel=r'$K_\parallel$',
    title=r'介电张量平行分量 $K_\parallel$ 随磁场 $B_0$ 的变化',
    filename='K_par_vs_B0.png',
    var_name=r'$K_\parallel$',
    y_use_log=False
)

# 图5：宗量 zeta_+1 vs B0
complex_B_1(
    x_data=successful_B0,
    y_data=zeta_plus1_array,
    ylabel=r'$\zeta_{+1}$',
    title=r'宗量 $\zeta_{+1}$ 随磁场 $B_0$ 的变化',
    filename='zeta_plus1_vs_B0.png',
    var_name=r'$\zeta_{+1}$',
    y_use_log=False
)

# 图6：宗量 zeta_-1 vs B0
complex_B_1(
    x_data=successful_B0,
    y_data=zeta_minus1_array,
    ylabel=r'$\zeta_{-1}$',
    title=r'宗量 $\zeta_{-1}$ 随磁场 $B_0$ 的变化',
    filename='zeta_minus1_vs_B0.png',
    var_name=r'$\zeta_{-1}$',
    y_use_log=False
)

# 图7：宗量 zeta_0 vs B0
complex_B_1(
    x_data=successful_B0,
    y_data=zeta_zero_array,
    ylabel=r'$\zeta_{0}$',
    title=r'宗量 $\zeta_{0}$ 随磁场 $B_0$ 的变化',
    filename='zeta_zero_vs_B0.png',
    var_name=r'$\zeta_{0}$',
    y_use_log=False
)

# 图8：介电张量 + 波数 + zeta 综合图（上下双子图）
complex_B_2(
    x_data=successful_B0,
    y_perp=K_perp_array,
    y_phi=K_g_array,
    y_par=K_par_array,
    y_k=k_final_array,
    y_zeta_plus1=zeta_plus1_array,
    filename='K_all_vs_B0.png',
    y_use_log=False,
    n_i=n_p,
    T_i_par=T_i_eV,
    T_i_perp=T_i_eV,
    v_i=v_i,
    nu_i=nu_i,
    w=w_target,
    Z=Z_number
)

# 图9：η vs B0
real_B_1(
    x_data=successful_B0,
    y_data=eta_array,
    ylabel=r'$\eta = \frac{\omega - k_{\parallel} V_{i\parallel} - \omega_{ci}}{\omega_{ci}}$',
    filename='eta_vs_B0.png'
)

print("\n" + "="*70)
print("所有图像已生成完成！")
print("="*70)
print("输出文件:")
print("  1. k_vs_B0.png - 波数随磁场变化")
print("  2. K_perp_vs_B0.png - 垂直介电张量分量")
print("  3. K_phi_vs_B0.png - 角向介电张量分量")
print("  4. K_par_vs_B0.png - 平行介电张量分量")
print("  5. zeta_plus1_vs_B0.png - 宗量 zeta_+1 随磁场变化")
print("  6. zeta_minus1_vs_B0.png - 宗量 zeta_-1 随磁场变化")
print("  7. zeta_zero_vs_B0.png - 宗量 zeta_0 随磁场变化")
print("  8. K_all_vs_B0.png - 介电张量分量、波数、zeta综合图（上下双子图）")
print("  9. eta_vs_B0.png - η 参数随磁场变化")
print("="*70)

plt.show()
