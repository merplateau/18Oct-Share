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
# 【配置区】- 只需修改这里即可切换比较参数
# ============================================================================

# 选择比较参数类型（只需修改这一行）
# 可选: 'density', 'temperature', 'velocity', 'collision'
COMPARISON_PARAM = 'temperature'

# 为比较参数设置三个值（从小到大）
PARAM_VALUES = [3, 10, 30]  # 示例: 密度的三个值

# ============================================================================
# 扫描参数设置
# ============================================================================

# B0 扫描范围
B0_values = np.linspace(1.2, 1.5, 300)  # 从 0.5 T 到 2.0 T，300个点

# 基准物理参数（比较参数会被PARAM_VALUES覆盖）
BASE_PARAMS = {
    'density': 1e18,        # 密度 [m^-3]
    'temperature': 3,       # 温度 [eV]
    'velocity': 10000.0,    # 速度 [m/s]
    'collision': 1e4        # 碰撞频率 [rad/s]
}

w_target = 3.3e6  # 目标频率 [rad/s]
Z_number = 40     # 离子电荷数

# ============================================================================
# 参数映射系统（自动处理，无需修改）
# ============================================================================

# 参数名称映射
PARAM_NAME_MAP = {
    'density': {'var': 'n_target', 'symbol': 'n_i', 'unit': 'm^{-3}'},
    'temperature': {'var': 'T_par_eV', 'symbol': 'T_i', 'unit': 'eV'},
    'velocity': {'var': 'v_target', 'symbol': 'V_{i\\parallel}', 'unit': 'm/s'},
    'collision': {'var': 'nu_target', 'symbol': '\\nu_i', 'unit': 'Hz'}
}

# 验证配置
if COMPARISON_PARAM not in PARAM_NAME_MAP:
    raise ValueError(f"无效的比较参数: {COMPARISON_PARAM}. 可选: {list(PARAM_NAME_MAP.keys())}")

if len(PARAM_VALUES) != 3:
    raise ValueError(f"必须提供3个比较参数值，当前提供了 {len(PARAM_VALUES)} 个")

print("="*70)
print("三参数比较 B0 扫描")
print("="*70)
print(f"比较参数: {COMPARISON_PARAM}")
print(f"参数值: {PARAM_VALUES}")
print(f"\nB0 范围: {B0_values[0]:.2f} T 到 {B0_values[-1]:.2f} T")
print(f"扫描点数: {len(B0_values)}")
print(f"\n基准参数:")
print(f"  密度 = {BASE_PARAMS['density']:.2e} m^-3")
print(f"  温度 = {BASE_PARAMS['temperature']:.2f} eV")
print(f"  速度 = {BASE_PARAMS['velocity']:.2e} m/s")
print(f"  碰撞频率 = {BASE_PARAMS['collision']:.2e} rad/s")
print(f"  角频率 = {w_target:.2e} rad/s")
print(f"  离子电荷数 = {Z_number}")
print("="*70)

# ============================================================================
# 执行扫描
# ============================================================================

# 存储三组结果（对应三个比较参数值）
results_list = []

# 物理常数
m_p = 1.67e-27  # 质子质量 [kg]
e = 1.602e-19   # 元电荷 [C]
m_i = Z_number * m_p  # 离子质量
q_i = e  # 离子电荷

# 外层循环：比较参数的三个值
for param_idx, param_value in enumerate(PARAM_VALUES):
    print(f"\n{'='*70}")
    print(f"【第 {param_idx+1}/3 组】比较参数 {COMPARISON_PARAM} = {param_value}")
    print('='*70)

    # 构建当前参数组合
    current_params = BASE_PARAMS.copy()
    current_params[COMPARISON_PARAM] = param_value

    # 提取实际参数值
    n_p = current_params['density']
    T_i_eV = current_params['temperature']
    v_i = current_params['velocity']
    nu_i = current_params['collision']

    # 存储当前组的结果
    k_final_array = []
    K_perp_array = []
    K_g_array = []
    K_par_array = []
    zeta_plus1_array = []
    zeta_minus1_array = []
    zeta_zero_array = []
    delta_array = []
    successful_B0 = []

    # 内层循环：扫描 B0
    for i, B0 in enumerate(B0_values):
        print(f"[{i+1}/{len(B0_values)}] 正在计算 B0 = {B0:.3f} T ...", end=' ')

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

            # 计算 delta = (w - k_par * v_i - w_ci) / w_ci
            w_ci = q_i * B0 / m_i  # 回旋频率 [rad/s]
            k_par = np.real(results['k_final'])  # 取波数的实部作为平行波数
            delta = (w_target - k_par * v_i - w_ci) / w_ci
            delta_array.append(delta)

            successful_B0.append(B0)

            print(f"✓ k = {results['k_final']:.6f}")

        except Exception as e:
            print(f"✗ 失败: {str(e)}")
            continue

    # 转换为 numpy 数组并保存到结果列表
    results_list.append({
        'param_value': param_value,
        'B0': np.array(successful_B0),
        'k_final': np.array(k_final_array),
        'K_perp': np.array(K_perp_array),
        'K_g': np.array(K_g_array),
        'K_par': np.array(K_par_array),
        'zeta_plus1': np.array(zeta_plus1_array),
        'zeta_minus1': np.array(zeta_minus1_array),
        'zeta_zero': np.array(zeta_zero_array),
        'delta': np.array(delta_array)
    })

    print(f"✓ 第 {param_idx+1} 组完成！成功点数: {len(successful_B0)}/{len(B0_values)}")

print("\n" + "="*70)
print("所有扫描完成！")
print("="*70)

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


def triple_comparison_plot(results_list, data_key, ylabel, filename,
                            param_name, param_symbol, param_unit, var_name, w=None, Z=None):
    """
    三参数比较绘图函数

    绘制三组数据的实部和虚部，用不同线型区分

    参数:
        results_list: 包含三组结果的列表
        data_key: 要绘制的数据键名 ('k_final', 'K_perp', 等)
        ylabel: 纵坐标标签
        filename: 保存文件名
        param_name: 比较参数名称
        param_symbol: 参数符号（LaTeX格式）
        param_unit: 参数单位
        var_name: 变量名称（用于图例，LaTeX格式）
        w: 目标频率 [rad/s]
        Z: 离子电荷数
    """
    plt.figure(figsize=(5, 5))

    # 线型: 实线, 虚线, 点线
    linestyles = ['-', '--', ':']

    # 颜色: 蓝色(实部), 红色(虚部)
    color_real = '#1055C9'
    color_imag = '#BF092F'

    # 计算回旋共振磁场强度 B_ci
    B_ci = None
    if w is not None and Z is not None:
        # 物理常数
        m_p = 1.67e-27  # 质子质量 [kg]
        e = 1.602e-19   # 元电荷 [C]
        m_i = Z * m_p   # 离子质量
        q_i = e         # 离子电荷（单电荷）

        # 回旋共振：w = w_c = q_i * B_ci / m_i
        B_ci = w * m_i / q_i

        # 检查是否在扫描范围内
        all_B0 = np.concatenate([results['B0'] for results in results_list])
        B_min = np.min(all_B0)
        B_max = np.max(all_B0)
        if B_ci < B_min or B_ci > B_max:
            B_ci = None  # 不在范围内，不画线

    # 绘制三组数据
    for idx, (results, linestyle) in enumerate(zip(results_list, linestyles)):
        param_val = results['param_value']
        x_data = results['B0']
        y_data = results[data_key]

        # 格式化参数值标签
        if param_unit == 'm^{-3}':
            label_val = f"{param_val:.1e}"
        elif param_unit == 'eV':
            label_val = f"{param_val:.1f}"
        elif param_unit == 'm/s':
            label_val = f"{param_val:.1e}"
        elif param_unit == 'Hz':
            label_val = f"{param_val:.1e}"
        else:
            label_val = f"{param_val}"

        # 绘制实部
        plt.plot(x_data, np.real(y_data), linestyle=linestyle, color=color_real,
                linewidth=2, markersize=0,
                label=rf'Re({var_name}), ${param_symbol}$={label_val}')

        # 绘制虚部
        plt.plot(x_data, np.imag(y_data), linestyle=linestyle, color=color_imag,
                linewidth=2, markersize=0,
                label=rf'Im({var_name}), ${param_symbol}$={label_val}')

    # 设置标签
    plt.xlabel(r'$B_0$ [T]', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    # 刻度字体和坐标轴反转
    plt.tick_params(labelsize=20)
    plt.gca().invert_xaxis()

    # 画回旋共振磁场强度的垂直线
    if B_ci is not None:
        plt.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 图例样式
    legend = plt.legend(fontsize=10, frameon=True, fancybox=False,
                       edgecolor='black', framealpha=1.0, loc='best')
    legend.get_frame().set_linewidth(1.0)

    # 其他设置
    plt.grid(False)
    plt.tight_layout()

    # 保存图片
    plt.savefig(filename, dpi=300)
    print(f"  ✓ 已保存: {filename}")


def triple_comparison_real_plot(results_list, data_key, ylabel, filename,
                                param_name, param_symbol, param_unit, w=None, Z=None):
    """
    三参数比较绘图函数 - 实数数据版本

    绘制三组实数数据，用不同线型区分

    参数:
        results_list: 包含三组结果的列表
        data_key: 要绘制的数据键名 ('delta', 等)
        ylabel: 纵坐标标签
        filename: 保存文件名
        param_name: 比较参数名称
        param_symbol: 参数符号（LaTeX格式）
        param_unit: 参数单位
        w: 目标频率 [rad/s]
        Z: 离子电荷数
    """
    # 估算边距（英寸）
    left_margin = 1.2
    right_margin = 0.3
    bottom_margin = 0.8
    top_margin = 0.3
    axes_width = 4.0
    axes_height = 4.0

    # 计算 figure 总尺寸
    fig_width = axes_width + left_margin + right_margin
    fig_height = axes_height + bottom_margin + top_margin

    # 创建 figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # axes 位置
    axes_rect = [
        left_margin / fig_width,
        bottom_margin / fig_height,
        axes_width / fig_width,
        axes_height / fig_height
    ]

    ax = fig.add_axes(axes_rect)

    # 计算回旋共振磁场强度 B_ci
    B_ci = None
    if w is not None and Z is not None:
        # 物理常数
        m_p = 1.67e-27  # 质子质量 [kg]
        e = 1.602e-19   # 元电荷 [C]
        m_i = Z * m_p   # 离子质量
        q_i = e         # 离子电荷（单电荷）

        # 回旋共振：w = w_c = q_i * B_ci / m_i
        B_ci = w * m_i / q_i

        # 检查是否在扫描范围内
        all_B0 = np.concatenate([results['B0'] for results in results_list])
        B_min = np.min(all_B0)
        B_max = np.max(all_B0)
        if B_ci < B_min or B_ci > B_max:
            B_ci = None  # 不在范围内，不画线

    # 线型: 实线, 虚线, 点线
    linestyles = ['-', '--', ':']
    color = '#1055C9'

    # 绘制三组数据
    for idx, (results, linestyle) in enumerate(zip(results_list, linestyles)):
        param_val = results['param_value']
        x_data = results['B0']
        y_data = results[data_key]

        # 格式化参数值标签
        if param_unit == 'm^{-3}':
            label_val = f"{param_val:.1e}"
        elif param_unit == 'eV':
            label_val = f"{param_val:.1f}"
        elif param_unit == 'm/s':
            label_val = f"{param_val:.1e}"
        elif param_unit == 'Hz':
            label_val = f"{param_val:.1e}"
        else:
            label_val = f"{param_val}"

        ax.plot(x_data, y_data, linestyle=linestyle, color=color, linewidth=2,
                label=rf'${param_symbol}$={label_val}')

    # 设置标签
    ax.set_xlabel(r'$B_0$ [T]', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.invert_xaxis()

    # 自动匹配y轴范围（不强制0在中间）
    # matplotlib会自动设置合适的范围

    # 添加水平参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 画回旋共振磁场强度的垂直线
    if B_ci is not None:
        ax.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 图例
    legend = ax.legend(fontsize=12, frameon=True, fancybox=False,
                      edgecolor='black', framealpha=1.0, loc='best')
    legend.get_frame().set_linewidth(1.0)

    # 其他设置
    ax.grid(False)

    # 保存图片
    plt.savefig(filename, dpi=300)
    print(f"  ✓ 已保存: {filename}")


def triple_comparison_delta_plot(results_list, data_key, ylabel, filename,
                                 param_name, param_symbol, param_unit, w=None, Z=None):
    """
    三参数比较绘图函数 - delta专用版本（继承自triple_comparison_real_plot）

    在父函数基础上，额外绘制冷等离子体的delta作为背景参考线
    冷等离子体delta: delta_cold = (w - w_ci) / w_ci (相当于k=0的情况)

    参数:
        results_list: 包含三组结果的列表
        data_key: 要绘制的数据键名 ('delta')
        ylabel: 纵坐标标签
        filename: 保存文件名
        param_name: 比较参数名称
        param_symbol: 参数符号（LaTeX格式）
        param_unit: 参数单位
        w: 目标频率 [rad/s]
        Z: 离子电荷数
    """
    # 估算边距（英寸）- 继承自父函数
    left_margin = 1.2
    right_margin = 0.3
    bottom_margin = 0.8
    top_margin = 0.3
    axes_width = 4.0
    axes_height = 4.0

    # 计算 figure 总尺寸
    fig_width = axes_width + left_margin + right_margin
    fig_height = axes_height + bottom_margin + top_margin

    # 创建 figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # axes 位置
    axes_rect = [
        left_margin / fig_width,
        bottom_margin / fig_height,
        axes_width / fig_width,
        axes_height / fig_height
    ]

    ax = fig.add_axes(axes_rect)

    # 计算回旋共振磁场强度 B_ci（用于垂直线）
    B_ci = None
    if w is not None and Z is not None:
        # 物理常数
        m_p = 1.67e-27  # 质子质量 [kg]
        e = 1.602e-19   # 元电荷 [C]
        m_i = Z * m_p   # 离子质量
        q_i = e         # 离子电荷（单电荷）

        # 回旋共振：w = w_c = q_i * B_ci / m_i
        B_ci = w * m_i / q_i

        # 检查是否在扫描范围内
        all_B0 = np.concatenate([results['B0'] for results in results_list])
        B_min = np.min(all_B0)
        B_max = np.max(all_B0)
        if B_ci < B_min or B_ci > B_max:
            B_ci = None  # 不在范围内，不画线

    # 线型: 实线, 虚线, 点线
    linestyles = ['-', '--', ':']
    color = '#1055C9'

    # 绘制三组热等离子体数据
    for idx, (results, linestyle) in enumerate(zip(results_list, linestyles)):
        param_val = results['param_value']
        x_data = results['B0']
        y_data = results[data_key]

        # 格式化参数值标签
        if param_unit == 'm^{-3}':
            label_val = f"{param_val:.1e}"
        elif param_unit == 'eV':
            label_val = f"{param_val:.1f}"
        elif param_unit == 'm/s':
            label_val = f"{param_val:.1e}"
        elif param_unit == 'Hz':
            label_val = f"{param_val:.1e}"
        else:
            label_val = f"{param_val}"

        ax.plot(x_data, y_data, linestyle=linestyle, color=color, linewidth=2,
                label=rf'${param_symbol}$={label_val}')

    # 设置标签
    ax.set_xlabel(r'$B_0$ [T]', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.invert_xaxis()

    # 自动匹配y轴范围（不强制0在中间）- 继承自父函数
    # matplotlib会自动设置合适的范围

    # 添加水平参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 画回旋共振磁场强度的垂直线
    if B_ci is not None:
        ax.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 【新增功能】绘制冷等离子体delta作为背景参考
    if w is not None and Z is not None:
        # 获取所有B0数据点
        all_B0 = np.concatenate([results['B0'] for results in results_list])
        B0_range = np.linspace(np.min(all_B0), np.max(all_B0), 500)

        # 计算冷等离子体delta: delta_cold = (w - w_ci) / w_ci
        # 其中 w_ci = q_i * B0 / m_i
        m_p = 1.67e-27  # 质子质量 [kg]
        e = 1.602e-19   # 元电荷 [C]
        m_i = Z * m_p   # 离子质量
        q_i = e         # 离子电荷

        w_ci_range = q_i * B0_range / m_i
        delta_cold = (w - w_ci_range) / w_ci_range

        # 用灰色实线绘制冷等离子体delta（作为背景参考）
        # 超出当前y轴范围的部分会被自动裁剪
        ax.plot(B0_range, delta_cold, '-', color='gray', linewidth=1.5, alpha=0.4,
                label=r'Cold Plasma')

    # 图例
    legend = ax.legend(fontsize=12, frameon=True, fancybox=False,
                      edgecolor='black', framealpha=1.0, loc='best')
    legend.get_frame().set_linewidth(1.0)

    # 其他设置
    ax.grid(False)

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

# 获取参数信息
param_info = PARAM_NAME_MAP[COMPARISON_PARAM]
param_symbol = param_info['symbol']
param_unit = param_info['unit']

# 图1：波数 k vs B0（三参数比较）
triple_comparison_plot(
    results_list=results_list,
    data_key='k_final',
    ylabel=r'$k$ [m$^{-1}$]',
    filename='k_vs_B0.png',
    param_name=COMPARISON_PARAM,
    param_symbol=param_symbol,
    param_unit=param_unit,
    var_name=r'$k$',
    w=w_target,
    Z=Z_number
)

# 图2：介电张量分量 K_perp vs B0（三参数比较）
triple_comparison_plot(
    results_list=results_list,
    data_key='K_perp',
    ylabel=r'$K_\perp$',
    filename='K_perp_vs_B0.png',
    param_name=COMPARISON_PARAM,
    param_symbol=param_symbol,
    param_unit=param_unit,
    var_name=r'$K_\perp$',
    w=w_target,
    Z=Z_number
)

# 图3：介电张量分量 K_g vs B0（三参数比较）
triple_comparison_plot(
    results_list=results_list,
    data_key='K_g',
    ylabel=r'$K_\phi$',
    filename='K_phi_vs_B0.png',
    param_name=COMPARISON_PARAM,
    param_symbol=param_symbol,
    param_unit=param_unit,
    var_name=r'$K_\phi$',
    w=w_target,
    Z=Z_number
)

# 图4：介电张量分量 K_par vs B0（三参数比较）
triple_comparison_plot(
    results_list=results_list,
    data_key='K_par',
    ylabel=r'$K_\parallel$',
    filename='K_par_vs_B0.png',
    param_name=COMPARISON_PARAM,
    param_symbol=param_symbol,
    param_unit=param_unit,
    var_name=r'$K_\parallel$',
    w=w_target,
    Z=Z_number
)

# 图5：宗量 zeta_+1 vs B0（三参数比较）
triple_comparison_plot(
    results_list=results_list,
    data_key='zeta_plus1',
    ylabel=r'$\zeta_{+1}$',
    filename='zeta_plus1_vs_B0.png',
    param_name=COMPARISON_PARAM,
    param_symbol=param_symbol,
    param_unit=param_unit,
    var_name=r'$\zeta_{+1}$',
    w=w_target,
    Z=Z_number
)

# 图6：宗量 zeta_-1 vs B0（三参数比较）
triple_comparison_plot(
    results_list=results_list,
    data_key='zeta_minus1',
    ylabel=r'$\zeta_{-1}$',
    filename='zeta_minus1_vs_B0.png',
    param_name=COMPARISON_PARAM,
    param_symbol=param_symbol,
    param_unit=param_unit,
    var_name=r'$\zeta_{-1}$',
    w=w_target,
    Z=Z_number
)

# 图7：宗量 zeta_0 vs B0（三参数比较）
triple_comparison_plot(
    results_list=results_list,
    data_key='zeta_zero',
    ylabel=r'$\zeta_{0}$',
    filename='zeta_zero_vs_B0.png',
    param_name=COMPARISON_PARAM,
    param_symbol=param_symbol,
    param_unit=param_unit,
    var_name=r'$\zeta_{0}$',
    w=w_target,
    Z=Z_number
)

# 图8：δ vs B0（三参数比较 + 冷等离子体背景）
triple_comparison_delta_plot(
    results_list=results_list,
    data_key='delta',
    ylabel=r'$\delta_{IC}$',
    filename='delta_vs_B0.png',
    param_name=COMPARISON_PARAM,
    param_symbol=param_symbol,
    param_unit=param_unit,
    w=w_target,
    Z=Z_number
)

# 【已屏蔽】图9：介电张量 + 波数 + zeta 综合图（上下双子图）
# 如需启用，请取消下面的注释
# complex_B_2(
#     x_data=results_list[0]['B0'],
#     y_perp=results_list[0]['K_perp'],
#     y_phi=results_list[0]['K_g'],
#     y_par=results_list[0]['K_par'],
#     y_k=results_list[0]['k_final'],
#     y_zeta_plus1=results_list[0]['zeta_plus1'],
#     filename='K_all_vs_B0.png',
#     y_use_log=False,
#     n_i=BASE_PARAMS['density'],
#     T_i_par=BASE_PARAMS['temperature'],
#     T_i_perp=BASE_PARAMS['temperature'],
#     v_i=BASE_PARAMS['velocity'],
#     nu_i=BASE_PARAMS['collision'],
#     w=w_target,
#     Z=Z_number
# )

print("\n" + "="*70)
print("所有图像已生成完成！")
print("="*70)
print("输出文件:")
print("  1. k_vs_B0.png - 波数随磁场变化（三参数比较）")
print("  2. K_perp_vs_B0.png - 垂直介电张量分量（三参数比较）")
print("  3. K_phi_vs_B0.png - 角向介电张量分量（三参数比较）")
print("  4. K_par_vs_B0.png - 平行介电张量分量（三参数比较）")
print("  5. zeta_plus1_vs_B0.png - 宗量 zeta_+1（三参数比较）")
print("  6. zeta_minus1_vs_B0.png - 宗量 zeta_-1（三参数比较）")
print("  7. zeta_zero_vs_B0.png - 宗量 zeta_0（三参数比较）")
print("  8. delta_vs_B0.png - δ 参数（三参数比较）")
print(f"\n比较参数: {COMPARISON_PARAM}")
print(f"参数值: {PARAM_VALUES}")
print("="*70)

plt.show()
