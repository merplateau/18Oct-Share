"""
冷等离子体介电张量色散关系
绘制介电张量三个分量在离子回旋共振频率附近的变化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 物理常数
e = 1.602e-19  # 电子电荷 (C)
m_e = 9.109e-31  # 电子质量 (kg)
m_i = 1.673e-27  # 质子质量 (kg) - 氢离子
epsilon_0 = 8.854e-12  # 真空介电常数 (F/m)
c = 3e8  # 光速 (m/s)

# 等离子体参数
B0 = 1.0  # 磁场强度 (T)
n_e = 1e19  # 电子密度 (m^-3)
n_i = n_e  # 准中性条件

# 特征频率
omega_ce = e * B0 / m_e  # 电子回旋频率
omega_ci = e * B0 / m_i  # 离子回旋频率
omega_pe = np.sqrt(n_e * e**2 / (m_e * epsilon_0))  # 电子等离子体频率
omega_pi = np.sqrt(n_i * e**2 / (m_i * epsilon_0))  # 离子等离子体频率

print(f"电子回旋频率 ω_ce = {omega_ce/2/np.pi/1e9:.3f} GHz")
print(f"离子回旋频率 ω_ci = {omega_ci/2/np.pi/1e6:.3f} MHz")
print(f"电子等离子体频率 ω_pe = {omega_pe/2/np.pi/1e9:.3f} GHz")
print(f"离子等离子体频率 ω_pi = {omega_pi/2/np.pi/1e6:.3f} MHz")

# 频率范围：在离子回旋共振频率附近
omega_min = 0.1 * omega_ci
omega_max = 2.0 * omega_ci
omega = np.linspace(omega_min, omega_max, 2000)

# 冷等离子体介电张量分量
def compute_dielectric_tensor(omega, omega_ce, omega_ci, omega_pe, omega_pi):
    """
    计算冷等离子体介电张量分量

    在冷等离子体近似下,介电张量为:
    ε = [[ε₁, -iε₂, 0],
         [iε₂, ε₁, 0],
         [0, 0, ε₃]]

    其中 (Stix参数):
    S = ε₁ = 1 - ω_pe²/(ω²-ω_ce²) - ω_pi²/(ω²-ω_ci²)
    D = ε₂ = (ω_pe²·ω_ce)/(ω(ω²-ω_ce²)) + (ω_pi²·ω_ci)/(ω(ω²-ω_ci²))
    P = ε₃ = 1 - ω_pe²/ω² - ω_pi²/ω²
    """

    # 避免除零,添加小的阻尼项
    nu = 1e-3 * omega_ci  # 小的碰撞频率

    # S分量 (ε₁)
    S = 1 - omega_pe**2 / (omega**2 - omega_ce**2 + 1j*nu*omega) \
          - omega_pi**2 / (omega**2 - omega_ci**2 + 1j*nu*omega)

    # D分量 (ε₂)
    D = omega_pe**2 * omega_ce / (omega * (omega**2 - omega_ce**2 + 1j*nu*omega)) \
      + omega_pi**2 * omega_ci / (omega * (omega**2 - omega_ci**2 + 1j*nu*omega))

    # P分量 (ε₃)
    P = 1 - omega_pe**2 / (omega**2 + 1j*nu*omega) \
          - omega_pi**2 / (omega**2 + 1j*nu*omega)

    return S, D, P

# 计算介电张量分量
S, D, P = compute_dielectric_tensor(omega, omega_ce, omega_ci, omega_pe, omega_pi)

# 提取实部和虚部
S_real = np.real(S)
S_imag = np.imag(S)
D_real = np.real(D)
D_imag = np.imag(D)
P_real = np.real(P)
P_imag = np.imag(P)

# 归一化频率
omega_norm = omega / omega_ci

# 绘图
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('冷等离子体介电张量分量在离子回旋共振频率附近的变化', fontsize=16, fontweight='bold')

# 第一行: S分量 (ε₁)
axes[0, 0].plot(omega_norm, S_real, 'b-', linewidth=2, label='Re(S)')
axes[0, 0].axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='ω_ci')
axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[0, 0].set_xlabel('ω/ω_ci', fontsize=12)
axes[0, 0].set_ylabel('Re(ε₁)', fontsize=12)
axes[0, 0].set_title('S分量 - 实部', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)
axes[0, 0].set_ylim([-50, 50])

axes[0, 1].plot(omega_norm, S_imag, 'r-', linewidth=2, label='Im(S)')
axes[0, 1].axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='ω_ci')
axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[0, 1].set_xlabel('ω/ω_ci', fontsize=12)
axes[0, 1].set_ylabel('Im(ε₁)', fontsize=12)
axes[0, 1].set_title('S分量 - 虚部', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)

# 第二行: D分量 (ε₂)
axes[1, 0].plot(omega_norm, D_real, 'b-', linewidth=2, label='Re(D)')
axes[1, 0].axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='ω_ci')
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[1, 0].set_xlabel('ω/ω_ci', fontsize=12)
axes[1, 0].set_ylabel('Re(ε₂)', fontsize=12)
axes[1, 0].set_title('D分量 - 实部', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)
axes[1, 0].set_ylim([-50, 50])

axes[1, 1].plot(omega_norm, D_imag, 'r-', linewidth=2, label='Im(D)')
axes[1, 1].axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='ω_ci')
axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[1, 1].set_xlabel('ω/ω_ci', fontsize=12)
axes[1, 1].set_ylabel('Im(ε₂)', fontsize=12)
axes[1, 1].set_title('D分量 - 虚部', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)

# 第三行: P分量 (ε₃)
axes[2, 0].plot(omega_norm, P_real, 'b-', linewidth=2, label='Re(P)')
axes[2, 0].axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='ω_ci')
axes[2, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[2, 0].set_xlabel('ω/ω_ci', fontsize=12)
axes[2, 0].set_ylabel('Re(ε₃)', fontsize=12)
axes[2, 0].set_title('P分量 - 实部', fontsize=13, fontweight='bold')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend(fontsize=10)

axes[2, 1].plot(omega_norm, P_imag, 'r-', linewidth=2, label='Im(P)')
axes[2, 1].axvline(x=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='ω_ci')
axes[2, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[2, 1].set_xlabel('ω/ω_ci', fontsize=12)
axes[2, 1].set_ylabel('Im(ε₃)', fontsize=12)
axes[2, 1].set_title('P分量 - 虚部', fontsize=13, fontweight='bold')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('dielectric_tensor_components.png', dpi=300, bbox_inches='tight')
print("\n图像已保存为: dielectric_tensor_components.png")
plt.show()

# 额外绘制一个组合图,展示三个分量的实部
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(omega_norm, S_real, 'b-', linewidth=2, label='S = ε₁ (垂直分量)')
ax.plot(omega_norm, D_real, 'g-', linewidth=2, label='D = ε₂ (交叉分量)')
ax.plot(omega_norm, P_real, 'r-', linewidth=2, label='P = ε₃ (平行分量)')
ax.axvline(x=1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='离子回旋频率 ω_ci')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('归一化频率 ω/ω_ci', fontsize=14)
ax.set_ylabel('介电张量分量实部', fontsize=14)
ax.set_title('冷等离子体介电张量分量对比', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='best')
ax.set_ylim([-50, 50])
plt.tight_layout()
plt.savefig('dielectric_tensor_comparison.png', dpi=300, bbox_inches='tight')
print("图像已保存为: dielectric_tensor_comparison.png")
plt.show()

print("\n脚本执行完毕!")
