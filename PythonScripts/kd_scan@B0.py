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

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 扫描参数设置
# ============================================================================

# B0 扫描范围
B0_values = np.linspace(0.1, 2, 200)  # 从 0.5 T 到 2.0 T，20个点

# 固定物理参数（与 kd_v1-0.py 保持一致）
n_p = 1e18        # 密度 [m^-3]
T_i_eV = 30     # 温度 [eV]
v_i = 10000.0        # 速度 [m/s]
nu_i = 1e5        # 碰撞频率 [rad/s]
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
        successful_B0.append(B0)

        print(f"  ✓ 成功: k = {results['k_final']:.6f}")

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

# ============================================================================
# 绘图
# ============================================================================

print("\n开始绘图...")

# 图1：波数 k vs B0
plt.figure(figsize=(10, 6))
plt.plot(successful_B0, np.real(k_final_array), 'o-', color='blue',
         linewidth=2, markersize=6, label='Re(k)')
plt.plot(successful_B0, np.imag(k_final_array), 'o-', color='red',
         linewidth=2, markersize=6, label='Im(k)')
plt.xlabel('B₀ [T]', fontsize=14)
plt.ylabel('k [m⁻¹]', fontsize=14)
plt.title('波数 k 随磁场 B₀ 的变化', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('k_vs_B0.png', dpi=300)
print("  ✓ 已保存: k_vs_B0.png")

# 图2：介电张量分量 K_perp vs B0
plt.figure(figsize=(10, 6))
plt.plot(successful_B0, np.real(K_perp_array), 'o-', color='blue',
         linewidth=2, markersize=6, label='Re(K_⊥)')
plt.plot(successful_B0, np.imag(K_perp_array), 'o-', color='red',
         linewidth=2, markersize=6, label='Im(K_⊥)')
plt.xlabel('B₀ [T]', fontsize=14)
plt.ylabel('K_⊥', fontsize=14)
plt.title('介电张量垂直分量 K_⊥ 随磁场 B₀ 的变化', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('K_perp_vs_B0.png', dpi=300)
print("  ✓ 已保存: K_perp_vs_B0.png")

# 图3：介电张量分量 K_g vs B0
plt.figure(figsize=(10, 6))
plt.plot(successful_B0, np.real(K_g_array), 'o-', color='blue',
         linewidth=2, markersize=6, label='Re(K_g)')
plt.plot(successful_B0, np.imag(K_g_array), 'o-', color='red',
         linewidth=2, markersize=6, label='Im(K_g)')
plt.xlabel('B₀ [T]', fontsize=14)
plt.ylabel('K_g', fontsize=14)
plt.title('介电张量陀螺分量 K_g 随磁场 B₀ 的变化', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('K_g_vs_B0.png', dpi=300)
print("  ✓ 已保存: K_g_vs_B0.png")

# 图4：介电张量分量 K_par vs B0
plt.figure(figsize=(10, 6))
plt.plot(successful_B0, np.real(K_par_array), 'o-', color='blue',
         linewidth=2, markersize=6, label='Re(K_∥)')
plt.plot(successful_B0, np.imag(K_par_array), 'o-', color='red',
         linewidth=2, markersize=6, label='Im(K_∥)')
plt.xlabel('B₀ [T]', fontsize=14)
plt.ylabel('K_∥', fontsize=14)
plt.title('介电张量平行分量 K_∥ 随磁场 B₀ 的变化', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('K_par_vs_B0.png', dpi=300)
print("  ✓ 已保存: K_par_vs_B0.png")

print("\n" + "="*70)
print("所有图像已生成完成！")
print("="*70)
print("输出文件:")
print("  1. k_vs_B0.png - 波数随磁场变化")
print("  2. K_perp_vs_B0.png - 垂直介电张量分量")
print("  3. K_g_vs_B0.png - 陀螺介电张量分量")
print("  4. K_par_vs_B0.png - 平行介电张量分量")
print("="*70)

plt.show()
