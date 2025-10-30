# -*- coding: utf-8 -*-
"""
热等离子体色散关系求解器 - 参数连续变化扫描脚本

扫描磁场 B0，四个物理参数随磁场线性变化，绘制：
1. delta vs B0
2. 波数 k vs B0

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
# 【配置区】- 参数随磁场连续变化
# ============================================================================

# B0 扫描范围
B0_values = np.linspace(0.4, 0.75, 300)  # 从 0.8 T 到 1.5 T，300个点

# 参数范围设置：起始值和结束值（随B0线性变化）
PARAMS_RANGE = {
    'density': (2e16, 1e17),        # 密度范围 [m^-3]
    'temperature': (3, 3),         # 温度范围 [eV]
    'velocity': (60000, 10000),      # 速度范围 [m/s]
    'collision': (1e4, 1e4)         # 碰撞频率范围 [rad/s]
}

# 固定参数
w_target = 1.65e6  # 目标频率 [rad/s]
Z_number = 40     # 离子电荷数

print("="*70)
print("参数连续变化 B0 扫描")
print("="*70)
print(f"B0 范围: {B0_values[0]:.2f} T 到 {B0_values[-1]:.2f} T")
print(f"扫描点数: {len(B0_values)}")
print(f"\n参数变化范围:")
print(f"  密度: {PARAMS_RANGE['density'][0]:.2e} → {PARAMS_RANGE['density'][1]:.2e} m^-3")
print(f"  温度: {PARAMS_RANGE['temperature'][0]:.2f} → {PARAMS_RANGE['temperature'][1]:.2f} eV")
print(f"  速度: {PARAMS_RANGE['velocity'][0]:.2e} → {PARAMS_RANGE['velocity'][1]:.2e} m/s")
print(f"  碰撞频率: {PARAMS_RANGE['collision'][0]:.2e} → {PARAMS_RANGE['collision'][1]:.2e} rad/s")
print(f"\n固定参数:")
print(f"  角频率 = {w_target:.2e} rad/s")
print(f"  离子电荷数 = {Z_number}")
print("="*70)

# ============================================================================
# 执行扫描
# ============================================================================

# 物理常数
m_p = 1.67e-27  # 质子质量 [kg]
e = 1.602e-19   # 元电荷 [C]
m_i = Z_number * m_p  # 离子质量
q_i = e  # 离子电荷

# 存储结果
k_final_array = []
delta_array = []
zeta_plus1_array = []
successful_B0 = []
successful_n_p = []
successful_T_i = []
successful_v_i = []
successful_nu_i = []

print(f"\n开始扫描...")

# 循环扫描 B0
for i, B0 in enumerate(B0_values):
    # 计算当前B0对应的参数值（线性插值）
    # progress = (B0 - B0_values[0]) / (B0_values[-1] - B0_values[0])
    progress = (i / (len(B0_values) - 1)) if len(B0_values) > 1 else 0

    n_p = PARAMS_RANGE['density'][0] + progress * (PARAMS_RANGE['density'][1] - PARAMS_RANGE['density'][0])
    T_i_eV = PARAMS_RANGE['temperature'][0] + progress * (PARAMS_RANGE['temperature'][1] - PARAMS_RANGE['temperature'][0])
    v_i = PARAMS_RANGE['velocity'][0] + progress * (PARAMS_RANGE['velocity'][1] - PARAMS_RANGE['velocity'][0])
    nu_i = PARAMS_RANGE['collision'][0] + progress * (PARAMS_RANGE['collision'][1] - PARAMS_RANGE['collision'][0])

    print(f"[{i+1}/{len(B0_values)}] B0={B0:.3f}T, n={n_p:.2e}, T={T_i_eV:.2f}eV, v={v_i:.0f}m/s, nu={nu_i:.2e} ...", end=' ')

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

        # 提取波数
        k_final_array.append(results['k_final'])

        # 计算 delta = (w - k_par * v_i - w_ci) / w_ci
        w_ci = q_i * B0 / m_i  # 回旋频率 [rad/s]
        k_par = np.real(results['k_final'])  # 取波数的实部作为平行波数
        delta = (w_target - k_par * v_i - w_ci) / w_ci
        delta_array.append(delta)

        # 提取 zeta_plus1
        zeta_plus1 = results['zeta_plus1_final']
        zeta_plus1_array.append(zeta_plus1)

        # 记录成功的数据点
        successful_B0.append(B0)
        successful_n_p.append(n_p)
        successful_T_i.append(T_i_eV)
        successful_v_i.append(v_i)
        successful_nu_i.append(nu_i)

        print(f"✓")

    except Exception as e:
        print(f"✗ {str(e)}")
        continue

print(f"\n扫描完成！成功点数: {len(successful_B0)}/{len(B0_values)}")

# 转换为 numpy 数组
successful_B0 = np.array(successful_B0)
k_final_array = np.array(k_final_array)
delta_array = np.array(delta_array)
zeta_plus1_array = np.array(zeta_plus1_array)
successful_n_p = np.array(successful_n_p)
successful_T_i = np.array(successful_T_i)
successful_v_i = np.array(successful_v_i)
successful_nu_i = np.array(successful_nu_i)

# ============================================================================
# 计算 dk/dz 相关量
# ============================================================================

# 物理空间参数：变化发生在 z=0 到 z=0.3m
z_start = 0.0  # m
z_end = 0.3    # m
delta_z = z_end - z_start

# 磁场从 z=0 线性下降到 z=0.3
# z=0: B0 = B0_max (0.75 T)
# z=0.3: B0 = B0_min (0.4 T)
B0_at_z0 = B0_values[-1]  # 0.75 T
B0_at_z03 = B0_values[0]  # 0.4 T
dB0_dz = (B0_at_z03 - B0_at_z0) / delta_z  # T/m (负值，因为下降)

print(f"\n物理空间信息:")
print(f"  磁场变化范围: z = {z_start} m 到 z = {z_end} m")
print(f"  B0(z=0) = {B0_at_z0:.3f} T")
print(f"  B0(z=0.3) = {B0_at_z03:.3f} T")
print(f"  dB0/dz = {dB0_dz:.3f} T/m")

# 提取 Re(k)
k_real = np.real(k_final_array)

# 计算 d Re(k) / dB0 使用后向差分
dk_dB0 = np.zeros_like(k_real)
dk_dB0[0] = 0  # 第一个点无法计算后向差分
for i in range(1, len(k_real)):
    dk_dB0[i] = (k_real[i] - k_real[i-1]) / (successful_B0[i] - successful_B0[i-1])

# 计算 d Re(k) / dz = (d Re(k) / dB0) * (dB0 / dz)
dk_dz = dk_dB0 * dB0_dz

# 计算 (1/Re(k)^2) * (d Re(k) / dz)
# 避免除以零
k_real_safe = np.where(np.abs(k_real) < 1e-15, 1e-15, k_real)
normalized_dk_dz = (1.0 / k_real_safe**2) * dk_dz

# ============================================================================
# 绘图函数
# ============================================================================

def plot_k_vs_B0(B0_data, k_data, w, Z):
    """
    波数 k vs B0 绘图函数

    参数:
        B0_data: 磁场强度数据
        k_data: 波数数据（复数）
        w: 目标频率 [rad/s]
        Z: 离子电荷数
    """
    # 创建figure
    plt.figure(figsize=(5, 5))

    # 颜色: 蓝色(实部), 红色(虚部)
    color_real = '#1055C9'
    color_imag = '#BF092F'

    # 绘制实部和虚部
    plt.plot(B0_data, np.real(k_data), '-', color=color_real,
            linewidth=2, markersize=0, label=r'Re($k$)')
    plt.plot(B0_data, np.imag(k_data), '-', color=color_imag,
            linewidth=2, markersize=0, label=r'Im($k$)')

    # 计算回旋共振磁场强度
    B_ci = None
    if w is not None and Z is not None:
        m_p = 1.67e-27
        e = 1.602e-19
        m_i = Z * m_p
        q_i = e
        B_ci = w * m_i / q_i

        B_min = np.min(B0_data)
        B_max = np.max(B0_data)
        if B_min <= B_ci <= B_max:
            plt.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 设置标签
    plt.xlabel(r'$B_0$ [T]', fontsize=20)
    plt.ylabel(r'$k$ [m$^{-1}$]', fontsize=20)

    # 刻度字体和坐标轴反转
    plt.tick_params(labelsize=20)
    plt.gca().invert_xaxis()

    # 图例样式（强制左上角）
    legend = plt.legend(fontsize=12, frameon=True, fancybox=False,
                       edgecolor='black', framealpha=1.0, loc='upper left')
    legend.get_frame().set_linewidth(1.0)

    # 其他设置
    plt.grid(False)
    plt.tight_layout()

    # 保存图片
    plt.savefig('k_vs_B0.png', dpi=300)
    print(f"  ✓ 已保存: k_vs_B0.png")
    plt.close()


def plot_delta_vs_B0(B0_data, delta_data, w, Z, params_range):
    """
    delta vs B0 绘图函数（带冷等离子体背景线）

    参数:
        B0_data: 磁场强度数据
        delta_data: delta数据（实数）
        w: 目标频率 [rad/s]
        Z: 离子电荷数
        params_range: 参数范围字典
    """
    # 估算边距（英寸）
    left_margin = 0.8
    right_margin = 0.3
    bottom_margin = 2.4  # 增加底部边距以容纳参数文本
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

    # 颜色
    color = '#1055C9'

    # 绘制热等离子体delta
    ax.plot(B0_data, delta_data, '-', color=color, linewidth=2,
            label=r'Hot Plasma')

    # 计算回旋共振磁场强度
    B_ci = None
    if w is not None and Z is not None:
        m_p = 1.67e-27
        e = 1.602e-19
        m_i = Z * m_p
        q_i = e
        B_ci = w * m_i / q_i

        B_min = np.min(B0_data)
        B_max = np.max(B0_data)
        if B_min <= B_ci <= B_max:
            ax.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # 绘制冷等离子体delta作为背景参考
        B0_range = np.linspace(B_min, B_max, 500)
        w_ci_range = q_i * B0_range / m_i
        delta_cold = (w - w_ci_range) / w_ci_range

        ax.plot(B0_range, delta_cold, '-', color='gray', linewidth=1.5, alpha=0.4,
                label=r'Cold Plasma')

    # 设置标签
    ax.set_xlabel(r'$B_0$ [T]', fontsize=20)
    ax.set_ylabel(r'$\delta_{IC}$', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.invert_xaxis()

    # 添加水平参考线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 图例（强制左上角）
    legend = ax.legend(fontsize=12, frameon=True, fancybox=False,
                      edgecolor='black', framealpha=1.0, loc='upper left')
    legend.get_frame().set_linewidth(1.0)

    # 其他设置
    ax.grid(False)

    # 添加参数说明文本（在图下方）
    # 由于B0轴反向，参数范围也反向显示
    # 格式化参数文本（4行）
    line1 = rf'$n_i$@{params_range["density"][1]:.0e}$\rightarrow${params_range["density"][0]:.0e} m$^{{-3}}$'
    line2 = rf'$T_i$@{params_range["temperature"][1]:.0e}$\rightarrow${params_range["temperature"][0]:.0e} eV'
    line3 = rf'$V_{{i\parallel}}$@{params_range["velocity"][1]:.0e}$\rightarrow${params_range["velocity"][0]:.0e} m/s'
    line4 = rf'$\nu_i$@{params_range["collision"][1]:.0e}$\rightarrow${params_range["collision"][0]:.0e} rad/s'

    # 组合多行文本
    param_text = line1 + "\n" + line2 + "\n" + line3 + "\n" + line4

    # 计算文本位置（图底部下方0.9英寸处）
    text_gap_inch = 0.9
    text_y_position = (bottom_margin - text_gap_inch) / fig_height

    # 在图底部添加文本（居中对齐，从上往下排列）
    fig.text(0.5, text_y_position, param_text, ha='center', fontsize=20,
            verticalalignment='top')

    # 保存图片
    plt.savefig('delta_vs_B0.png', dpi=300)
    print(f"  ✓ 已保存: delta_vs_B0.png")
    plt.close()


def plot_zeta_plus1_vs_B0(B0_data, zeta_plus1_data, w, Z):
    """
    zeta_plus1 vs B0 绘图函数

    参数:
        B0_data: 磁场强度数据
        zeta_plus1_data: zeta_plus1数据（复数）
        w: 目标频率 [rad/s]
        Z: 离子电荷数
    """
    # 创建figure
    plt.figure(figsize=(5, 5))

    # 颜色: 蓝色(实部), 红色(虚部)
    color_real = '#1055C9'
    color_imag = '#BF092F'

    # 绘制实部和虚部
    plt.plot(B0_data, np.real(zeta_plus1_data), '-', color=color_real,
            linewidth=2, markersize=0, label=r'Re($\zeta_{+1}$)')
    plt.plot(B0_data, np.imag(zeta_plus1_data), '-', color=color_imag,
            linewidth=2, markersize=0, label=r'Im($\zeta_{+1}$)')

    # 计算回旋共振磁场强度
    B_ci = None
    if w is not None and Z is not None:
        m_p = 1.67e-27
        e = 1.602e-19
        m_i = Z * m_p
        q_i = e
        B_ci = w * m_i / q_i

        B_min = np.min(B0_data)
        B_max = np.max(B0_data)
        if B_min <= B_ci <= B_max:
            plt.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 设置标签
    plt.xlabel(r'$B_0$ [T]', fontsize=20)
    plt.ylabel(r'$\zeta_{+1}$', fontsize=20)

    # 刻度字体和坐标轴反转
    plt.tick_params(labelsize=20)
    plt.gca().invert_xaxis()

    # 图例样式（强制左上角）
    legend = plt.legend(fontsize=12, frameon=True, fancybox=False,
                       edgecolor='black', framealpha=1.0, loc='upper left')
    legend.get_frame().set_linewidth(1.0)

    # 其他设置
    plt.grid(False)
    plt.tight_layout()

    # 保存图片
    plt.savefig('zeta_plus1_vs_B0.png', dpi=300)
    print(f"  ✓ 已保存: zeta_plus1_vs_B0.png")
    plt.close()


def plot_normalized_dk_dz_vs_B0(B0_data, normalized_dk_dz_data, w, Z):
    """
    归一化波数空间导数 vs B0 绘图函数

    绘制 (1/Re(k)^2) * (d Re(k)/dz)

    参数:
        B0_data: 磁场强度数据
        normalized_dk_dz_data: (1/Re(k)^2) * (d Re(k)/dz) 数据
        w: 目标频率 [rad/s]
        Z: 离子电荷数
    """
    # 创建figure
    plt.figure(figsize=(5, 5))

    # 颜色: 蓝色
    color = '#1055C9'

    # 绘制归一化导数
    plt.plot(B0_data, normalized_dk_dz_data, '-', color=color,
            linewidth=2, markersize=0)

    # 计算回旋共振磁场强度
    B_ci = None
    if w is not None and Z is not None:
        m_p = 1.67e-27
        e = 1.602e-19
        m_i = Z * m_p
        q_i = e
        B_ci = w * m_i / q_i

        B_min = np.min(B0_data)
        B_max = np.max(B0_data)
        if B_min <= B_ci <= B_max:
            plt.axvline(x=B_ci, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 添加水平参考线 y=0
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 设置标签
    plt.xlabel(r'$B_0$ [T]', fontsize=20)
    plt.ylabel(r'$\frac{1}{\mathrm{Re}(k)^2}\frac{d\mathrm{Re}(k)}{dz}$ [m$^{-1}$]', fontsize=20)

    # 刻度字体和坐标轴反转
    plt.tick_params(labelsize=20)
    plt.gca().invert_xaxis()

    # 其他设置
    plt.grid(False)
    plt.tight_layout()

    # 保存图片
    plt.savefig('normalized_dk_dz_vs_B0.png', dpi=300)
    print(f"  ✓ 已保存: normalized_dk_dz_vs_B0.png")
    plt.close()


# ============================================================================
# 绘图
# ============================================================================

print("\n开始绘图...")

# 图1：波数 k vs B0
plot_k_vs_B0(
    B0_data=successful_B0,
    k_data=k_final_array,
    w=w_target,
    Z=Z_number
)

# 图2：delta vs B0（带冷等离子体背景）
plot_delta_vs_B0(
    B0_data=successful_B0,
    delta_data=delta_array,
    w=w_target,
    Z=Z_number,
    params_range=PARAMS_RANGE
)

# 图3：zeta_plus1 vs B0
plot_zeta_plus1_vs_B0(
    B0_data=successful_B0,
    zeta_plus1_data=zeta_plus1_array,
    w=w_target,
    Z=Z_number
)

# 图4：归一化 dk/dz vs B0
plot_normalized_dk_dz_vs_B0(
    B0_data=successful_B0,
    normalized_dk_dz_data=normalized_dk_dz,
    w=w_target,
    Z=Z_number
)

print("\n" + "="*70)
print("所有图像已生成完成！")
print("="*70)
print("输出文件:")
print("  1. k_vs_B0.png - 波数随磁场变化（参数连续变化）")
print("  2. delta_vs_B0.png - δ参数随磁场变化（参数连续变化，含冷等离子体背景）")
print("  3. zeta_plus1_vs_B0.png - ζ_+1参数随磁场变化（参数连续变化）")
print("  4. normalized_dk_dz_vs_B0.png - (1/Re(k)²)·(dRe(k)/dz) 随磁场变化")
print("="*70)

plt.show()
