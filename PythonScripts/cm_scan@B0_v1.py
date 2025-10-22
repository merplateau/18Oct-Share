# -*- coding: utf-8 -*-
"""
冷等离子体介电张量 - B0扫描脚本

扫描磁场 B0，绘制介电张量分量随 B0 的变化：
1. K_perp（垂直分量）
2. K_phi/K_g（陀螺分量）
3. K_par（平行分量）

作者：基于 cm_v1.py
日期：2025
"""

import numpy as np
import matplotlib.pyplot as plt
from cm_v1 import compute_cold_plasma_dielectric_ion_only

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 扫描参数设置
# ============================================================================

# B0 扫描范围
B0_values = np.linspace(0.5, 2, 100)  # 从 1.2 T 到 1.5 T，30个点

# 固定物理参数
omega = 3.3e6      # 频率 [rad/s]
n_i = 1e18         # 离子密度 [m^-3]
nu_i = 1e5         # 离子碰撞频率 [rad/s]
Z = 40             # 离子电荷数

print("="*70)
print("冷等离子体 B0 扫描开始")
print("="*70)
print(f"B0 范围: {B0_values[0]:.2f} T 到 {B0_values[-1]:.2f} T")
print(f"扫描点数: {len(B0_values)}")
print(f"\n固定参数:")
print(f"  omega = {omega:.2e} rad/s")
print(f"  n_i = {n_i:.2e} m^-3")
print(f"  nu_i = {nu_i:.2e} rad/s")
print(f"  Z = {Z}")
print("="*70)

# ============================================================================
# 执行扫描
# ============================================================================

print("\n正在计算冷等离子体介电张量...")

K_perp_array = []
K_phi_array = []
K_par_array = []

for i, B0 in enumerate(B0_values):
    if (i+1) % 10 == 0:
        print(f"  进度: {i+1}/{len(B0_values)} (B0 = {B0:.3f} T)")

    result = compute_cold_plasma_dielectric_ion_only(
        omega=omega,
        B0=B0,
        n_i=n_i,
        nu_i=nu_i,
        Z=Z
    )

    K_perp_array.append(result['K_perp'])
    K_phi_array.append(result['K_phi'])
    K_par_array.append(result['K_par'])

# 转换为 numpy 数组
K_perp_array = np.array(K_perp_array)
K_phi_array = np.array(K_phi_array)
K_par_array = np.array(K_par_array)

print("  ✓ 计算完成")

# ============================================================================
# 绘图
# ============================================================================

print("\n开始绘图...")

# 图1：K_perp vs B0
plt.figure(figsize=(10, 6))
plt.plot(B0_values, np.real(K_perp_array), 'o-', color='blue',
         linewidth=2, markersize=6, label='Re(K_⊥)')
plt.plot(B0_values, np.imag(K_perp_array), 'o-', color='red',
         linewidth=2, markersize=6, label='Im(K_⊥)')
plt.xlabel('B₀ [T]', fontsize=14)
plt.ylabel('K_⊥', fontsize=14)
plt.title('冷等离子体：垂直分量 K_⊥ 随磁场 B₀ 的变化', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cm_K_perp_vs_B0.png', dpi=300)
print("  ✓ 已保存: cm_K_perp_vs_B0.png")

# 图2：K_phi (K_g) vs B0
plt.figure(figsize=(10, 6))
plt.plot(B0_values, np.real(K_phi_array), 'o-', color='blue',
         linewidth=2, markersize=6, label='Re(K_φ)')
plt.plot(B0_values, np.imag(K_phi_array), 'o-', color='red',
         linewidth=2, markersize=6, label='Im(K_φ)')
plt.xlabel('B₀ [T]', fontsize=14)
plt.ylabel('K_φ (K_g)', fontsize=14)
plt.title('冷等离子体：陀螺分量 K_φ 随磁场 B₀ 的变化', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cm_K_phi_vs_B0.png', dpi=300)
print("  ✓ 已保存: cm_K_phi_vs_B0.png")

# 图3：K_par vs B0
plt.figure(figsize=(10, 6))
plt.plot(B0_values, np.real(K_par_array), 'o-', color='blue',
         linewidth=2, markersize=6, label='Re(K_∥)')
plt.plot(B0_values, np.imag(K_par_array), 'o-', color='red',
         linewidth=2, markersize=6, label='Im(K_∥)')
plt.xlabel('B₀ [T]', fontsize=14)
plt.ylabel('K_∥', fontsize=14)
plt.title('冷等离子体：平行分量 K_∥ 随磁场 B₀ 的变化', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cm_K_par_vs_B0.png', dpi=300)
print("  ✓ 已保存: cm_K_par_vs_B0.png")

print("\n" + "="*70)
print("所有图像已生成完成！")
print("="*70)
print("输出文件:")
print("  1. cm_K_perp_vs_B0.png - 垂直介电张量分量")
print("  2. cm_K_phi_vs_B0.png - 陀螺介电张量分量")
print("  3. cm_K_par_vs_B0.png - 平行介电张量分量")
print("="*70)

# 输出统计信息
print("\n数值范围统计:")
print(f"K_perp: Re [{np.real(K_perp_array).min():.2f}, {np.real(K_perp_array).max():.2f}], "
      f"Im [{np.imag(K_perp_array).min():.2f}, {np.imag(K_perp_array).max():.2f}]")
print(f"K_phi:  Re [{np.real(K_phi_array).min():.2f}, {np.real(K_phi_array).max():.2f}], "
      f"Im [{np.imag(K_phi_array).min():.2f}, {np.imag(K_phi_array).max():.2f}]")
print(f"K_par:  Re [{np.real(K_par_array).min():.2f}, {np.real(K_par_array).max():.2f}], "
      f"Im [{np.imag(K_par_array).min():.2f}, {np.imag(K_par_array).max():.2f}]")

plt.show()
