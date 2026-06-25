import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.linalg import cho_factor, cho_solve
from tqdm import tqdm

PANTOGRAPH = 1  # 受电弓模型 1–3; 1 = DSA380; 2 = DSA250;
RIGID_OVERHEAD_CONTACT_SYSTEM = 1  # 刚性接触网参数 1–3
SPEED_KMH = 130  # 列车速度 [km/h]
NM = 100  # 模态数
DT_BASE = 0.00005  # 时间步 [s]
G = 9.8  # 重力 [m/s²]

# KS 为受电弓–接触网之间的接触弹簧刚度 (接触界面参数), 不属于刚性接触网基础设施,
# 故作为独立常量, 不并入接触网预设表.
KS = 82300.0  # contact spring stiffness [N/m]

# TOL_SUP 仅为接头布点的几何安全距离
TOL_SUP = 1.5  # [m]

# 瑞利阻尼(pc.m)
ALPHA_C = 0.0124
BETA_C = 0.0001

# 通常 1–2 次收敛;
CONTACT_ITERS_MAX = 5

STABLE_START = 0.3
STABLE_END = 0.7

# 接触线波磨不平顺默认参数
WEAR_AMPLITUDE = 1.0e-3  # 接触线磨耗幅值 A_w [m]，参考范围 [0.2, 3] mm, 间隔为0.2mm
WEAR_WAVELENGTH = 0.6  # 接触线波磨波长 λ_w [m]，参考范围 [0.4, 1.2] m, 间隔为0.2m


def contact_wire_wear(x, A_w: float = WEAR_AMPLITUDE, lambda_w: float = WEAR_WAVELENGTH):
    r"""
    $$
    W_{cw} = \frac{1}{2} A_w [1 - \cos(\frac{2 \pi l}{\lambda_w})]
    $$

    $W_{cw}$: 接触线磨耗深度
    $A_w$: 接触线磨耗幅值
    $\lambda_w$: 波磨的波长
    $l$: 接触线上沿线长度坐标
    """
    return 0.5 * A_w * (1.0 - np.cos(2.0 * np.pi * x / lambda_w))


def rigid_overhead_contact_system_params(rigid_overhead_contact_system: int, N_spans: int = 30):
    """Return (L, N, rhoA, EI, KEQ, MEQ, MZ, L_MZ) for the selected overhead contact system preset.

    悬挂 / 接头参数 (KEQ 支撑等效刚度, MEQ 支撑等效质量, MZ 84,
    L_MZ 汇流排单段长度) 作为刚性接触网基础设施的一部分随预设一并返回.
    接触弹簧刚度 KS 属于受电弓–接触网接触界面, 不在此处 (见模块常量 KS).
    """
    table = {
        # L,    N,  rhoA, EI,       KEQ,  MEQ, MZ,   L_MZ
        1: (
            8.0,
            30,
            8.1,
            1.7e5,
            6.7e7,
            7.0,
            2.84,
            12.0,
        ),  # A deep learning-based surrogate model for dynamic interaction assessment of high-speed overhead conductor rail system case 1
        2: (
            8.5,
            30,
            7.1,
            2.7e5,
            6e5,
            7.0,
            2.84,
            12.0,
        ),  # A deep learning-based surrogate model for dynamic interaction assessment of high-speed overhead conductor rail system case 2
        3: (8, 30, 7.25, 2.69e5, 6e4, 7.0, 2.84, 12.0),  # 陈龙 西南交大博士大论文
    }
    if rigid_overhead_contact_system not in table:
        raise ValueError(
            f'RIGID_OVERHEAD_CONTACT_SYSTEM must be 1–3, the current value is {rigid_overhead_contact_system}.'
        )
    L, N_default, rhoA, EI, KEQ, MEQ, MZ, L_MZ = table[rigid_overhead_contact_system]
    N = N_spans if N_spans is not None else N_default
    return L, N, rhoA, EI, KEQ, MEQ, MZ, L_MZ


def pantograph_params(ptype: int, v_kmh: float):
    """
    Return (m1,m2,m3, k1,k2,k3, c1,c2,c3, F0) for the selected pantograph.
    Mass order: m1 = bow head (top), m2 = intermediate, m3 = lower frame.
    Stiffness order: k1 = bow head (top), k2 = intermediate, k3 = lower frame.
    Damping order: c1 = bow head (top), c2 = intermediate, c3 = lower frame.
    F0: uplift force.
    """
    f_aero = 0.00047 * v_kmh**2
    table = {
        1: (7.12, 6.00, 5.80, 9430.0, 14100.0, 0.1, 0, 0, 70.0, 120.0),  # DSA380
        2: (7.51, 5.855, 4.645, 8380.0, 6200.0, 80.0, 0, 0, 70, 120.0),  # DSA250
    }
    if ptype not in table:
        raise ValueError(f'PANTOGRAPH must be 1–2, the current value is {ptype}')
    return table[ptype]


def compute_busbar_positions(LS: float, x_j: np.ndarray, l_mz: float, tol_sup: float = TOL_SUP) -> np.ndarray:
    all_sup = np.concatenate([[0.0], x_j, [LS]])
    x_mz = np.arange(l_mz, LS - l_mz / 2.0, l_mz, dtype=float)

    for k in range(len(x_mz)):
        dists = np.abs(x_mz[k] - all_sup)
        idx = int(np.argmin(dists))
        if dists[idx] < tol_sup:
            violating = all_sup[idx]
            x_mz[k] = violating + tol_sup if x_mz[k] >= violating else violating - tol_sup
            # Extreme case: still in conflict → fall back to the midpoint of the enclosing span
            if np.min(np.abs(x_mz[k] - all_sup)) < tol_sup:
                below = all_sup[all_sup < x_mz[k]].max()
                above = all_sup[all_sup > x_mz[k]].min()
                x_mz[k] = 0.5 * (below + above)
    return x_mz


def run_simulation(
    rigid_overhead_contact_system: int = RIGID_OVERHEAD_CONTACT_SYSTEM,
    pantograph: int = PANTOGRAPH,
    speed_kmh: float = SPEED_KMH,
    NM: int = NM,
    N_spans: int = 30,
    dt_base: float = DT_BASE,
    irregularity: bool = False,
    wear_amplitude: float = WEAR_AMPLITUDE,
    wear_wavelength: float = WEAR_WAVELENGTH,
    verbose: bool = True,
):

    L, N, rhoA, EI, KEQ, MEQ, MZ, L_MZ = rigid_overhead_contact_system_params(rigid_overhead_contact_system, N_spans)
    m1, m2, m3, k1, k2, k3, c1, c2, c3, F0 = pantograph_params(pantograph, speed_kmh)

    v = speed_kmh / 3.6  # [m/s]
    dt = dt_base
    LS = L * N

    t_total = LS / v
    t_vec = np.arange(0, t_total + dt, dt)
    n_steps = len(t_vec)
    x_vec = v * t_vec

    if verbose:
        print('=' * 60)
        print('POCS Simulation')
        print('=' * 60)
        print(f'  Catenary preset  : {rigid_overhead_contact_system}   (L={L} m, N={N} spans, LS={LS} m)')
        print(f'  Pantograph type  : {pantograph}')
        print(f'  Speed            : {speed_kmh} km/h  ({v:.2f} m/s)')
        print(f'  Time step dt     : {dt:.6f} s')
        print(f'  Total steps      : {n_steps:,}')
        print(f'  Retained modes   : {NM}')
        if irregularity:
            print(f'  Irregularity     : ON  (A_w={wear_amplitude * 1e3:.3f} mm, λ_w={wear_wavelength:.3f} m)')
        else:
            print('  Irregularity     : OFF')

    M_p = np.diag([m1, m2, m3])
    C_p = np.array(
        [
            [c1, -c1, 0],
            [-c1, c1 + c2, -c2],
            [0, -c2, c2 + c3],
        ],
        dtype=float,
    )
    K_p = np.array(
        [
            [k1, -k1, 0],
            [-k1, k1 + k2, -k2],
            [0, -k2, k2 + k3],
        ],
        dtype=float,
    )
    F_p = np.array([0.0, 0.0, F0])

    modes = np.arange(1, NM + 1, dtype=float)
    norm_factor = np.sqrt(2.0 / (rhoA * LS))
    omega_n = (modes * np.pi / LS) ** 2 * np.sqrt(EI / rhoA)

    x_j = L * np.arange(1, N, dtype=float)
    Phi_sup = norm_factor * np.sin(np.outer(modes * np.pi / LS, x_j))  # (NM, N-1)
    M_add_sup = MEQ * Phi_sup @ Phi_sup.T
    K_add_sup = KEQ * Phi_sup @ Phi_sup.T

    x_mz = compute_busbar_positions(LS, x_j, l_mz=L_MZ)
    Phi_mz = norm_factor * np.sin(np.outer(modes * np.pi / LS, x_mz))  # (NM, Nmz)
    M_add_mz = MZ * Phi_mz @ Phi_mz.T

    M_cat = np.eye(NM) + M_add_sup + M_add_mz
    K_cat = np.diag(omega_n**2) + K_add_sup
    # 瑞利阻尼按论文附录 D 式 (D-12): C = α·M + β·K, 含支撑/接头的满矩阵.
    C_cat = ALPHA_C * M_cat + BETA_C * K_cat

    int_sin = (LS / (modes * np.pi)) * (1.0 - np.cos(modes * np.pi))
    F_grav_beam = -rhoA * G * norm_factor * int_sin
    F_grav_sup = -MEQ * G * Phi_sup.sum(axis=1)
    F_grav_mz = -MZ * G * Phi_mz.sum(axis=1)
    F_gravity = F_grav_beam + F_grav_sup + F_grav_mz

    q_static = np.linalg.solve(K_cat, F_gravity)

    n_dof = 3 + NM
    M_sys = np.zeros((n_dof, n_dof))
    C_sys = np.zeros((n_dof, n_dof))
    K_sys = np.zeros((n_dof, n_dof))
    M_sys[:3, :3] = M_p
    M_sys[3:, 3:] = M_cat
    C_sys[:3, :3] = C_p
    C_sys[3:, 3:] = C_cat
    K_sys[:3, :3] = K_p
    K_sys[3:, 3:] = K_cat

    F_base = np.zeros(n_dof)
    F_base[:3] = F_p
    F_base[3:] = F_gravity

    # Woodbury 预计算: Newmark 常量系数 + 无接触系统矩阵 P 的 Cholesky 分解.
    # 每步有效矩阵 Kt = P + KS·w wᵀ (秩-1 接触), 用 Sherman-Morrison 把 O(n³) 解降为 O(n²).
    beta_nm, gamma_nm = 0.25, 0.5
    a0 = 1.0 / (beta_nm * dt * dt)
    a1 = gamma_nm / (beta_nm * dt)
    a2 = 1.0 / (beta_nm * dt)
    a3 = 1.0 / (2.0 * beta_nm) - 1.0
    a4 = gamma_nm / beta_nm - 1.0
    a5 = dt * (gamma_nm / (2.0 * beta_nm) - 1.0)
    a6 = dt * (1.0 - gamma_nm)
    a7 = gamma_nm * dt
    cP = cho_factor(K_sys + a0 * M_sys + a1 * C_sys, check_finite=False)  # 一次性分解, 各步复用
    w_buf = np.zeros(n_dof)
    w_buf[0] = 1.0  # w = [1, 0, 0, -φ]; 接触步内原地写入 w[3:]

    def phi_at(x):
        return norm_factor * np.sin(modes * np.pi * x / LS)

    Y = np.zeros(n_dof)
    V = np.zeros(n_dof)

    # Catenary starts at static gravity equilibrium
    Y[3:] = q_static

    # Pantograph initial state: solve static balance with contact spring engaged at x = 0
    phi_0 = phi_at(0.0)
    u_c_0 = phi_0 @ q_static
    K_p_static = K_p.copy()
    K_p_static[0, 0] += KS
    F_p_static = F_p + np.array([KS * u_c_0, 0.0, 0.0])
    Y[:3] = np.linalg.solve(K_p_static, F_p_static)

    # Consistent initial acceleration including the contact-spring assembly at x = 0
    Kc0 = np.zeros((n_dof, n_dof))
    Kc0[0, 0] = KS
    Kc0[0, 3:] = -KS * phi_0
    Kc0[3:, 0] = -KS * phi_0
    Kc0[3:, 3:] = KS * np.outer(phi_0, phi_0)
    A = np.linalg.solve(M_sys, F_base - C_sys @ V - (K_sys + Kc0) @ Y)

    contact_force = np.zeros(n_steps)
    y_pantograph = np.zeros(n_steps)
    y_rigid_overhead_contact_system = np.zeros(n_steps)

    rel_0 = Y[0] - u_c_0
    contact_force[0] = KS * rel_0 if rel_0 > 0 else 0.0
    y_pantograph[0] = Y[0]
    y_rigid_overhead_contact_system[0] = u_c_0

    t_start = time.time()
    for k in tqdm(range(1, n_steps), desc='Simulating', unit='step', disable=not verbose):
        xc = x_vec[k]
        phi = phi_at(xc)

        # 接触线波磨磨耗深度 w_cw(x)，关闭不平顺时恒为 0 (eq. 5-1, 5-3)
        w_cw = contact_wire_wear(xc, wear_amplitude, wear_wavelength) if irregularity else 0.0

        # Prediction step decides contact state for the current assembly
        Y_pred = Y + dt * V + 0.5 * dt * dt * A
        in_contact = (Y_pred[0] - phi @ Y_pred[3:] - w_cw) > 0.0

        F_eff = F_base
        if in_contact and w_cw != 0.0:  # 磨耗等效为接触弹簧的常量预压量，加到激励向量上
            F_eff = F_base.copy()
            F_eff[0] += KS * w_cw
            F_eff[3:] -= KS * phi * w_cw

        # Newmark 有效载荷; 有效矩阵 Kt = P + KS·w wᵀ, P 已预分解 → Sherman-Morrison
        # 有效载荷仅依赖上一步状态 (Y, V, A), 与本步接触状态无关 → 在迭代外计算一次.
        Ft = F_eff + M_sys @ (a0 * Y + a2 * V + a3 * A) + C_sys @ (a1 * Y + a4 * V + a5 * A)
        u = cho_solve(cP, Ft, check_finite=False)

        # Active-set 迭代: 用当前假设的接触状态求解, 再用解出的间隙重新判定;
        # 状态不变即收敛.
        w_buf[3:] = -phi
        z = cho_solve(cP, w_buf, check_finite=False)  # 接触列向量, 不随状态变 → 复用
        converged = False
        for _ in range(CONTACT_ITERS_MAX):
            if in_contact:
                Y_new = u - KS * (w_buf @ u) / (1.0 + KS * (w_buf @ z)) * z
            else:
                Y_new = u
            # 用解出的状态重新判定 (考虑磨耗预压 w_cw)
            u_c_new = phi @ Y_new[3:]
            in_contact_new = (Y_new[0] - u_c_new - w_cw) > 0.0
            if in_contact_new == in_contact:
                converged = True
                break  # 状态自洽
            # 状态翻转: u 的磨耗激励项需匹配新状态 (z = P⁻¹w 已在循环外算好)
            # u_with_wear − u_no_wear = KS·w_cw·z, 翻转时 ± 该差值即可切换 RHS
            if w_cw != 0.0:
                u = u + KS * w_cw * z if in_contact_new else u - KS * w_cw * z
            in_contact = in_contact_new  # 翻转状态, 重新求解
        if not converged:
            # 极端工况 (过大 dt/KS/波磨) 下 active-set 振荡未收敛.
            # 取接触解: 单边约束由接触力截断 (rel>0 ? KS·rel : 0) 兜底, 避免漏算接触.
            Y_new = u - KS * (w_buf @ u) / (1.0 + KS * (w_buf @ z)) * z
            u_c_new = phi @ Y_new[3:]
        A_new = a0 * (Y_new - Y) - a2 * V - a3 * A
        V = V + a6 * A + a7 * A_new
        A = A_new
        Y = Y_new

        rel = Y[0] - u_c_new - w_cw
        contact_force[k] = KS * rel if rel > 0 else 0.0
        y_pantograph[k] = Y[0]
        y_rigid_overhead_contact_system[k] = u_c_new

    elapsed = time.time() - t_start
    if verbose:
        print(f'\nSimulation complete – {elapsed:.1f} s wall time')

    def compute_stats(fc_arr):
        if len(fc_arr) == 0:
            return {}
        return {
            'mean_N': float(fc_arr.mean()),
            'std_N': float(fc_arr.std()),
            'max_N': float(fc_arr.max()),
            'min_N': float(fc_arr.min()),
            'loss_of_contact_pct': float(100 * (fc_arr == 0).mean()),
        }

    stats_full = compute_stats(contact_force)

    i_start = int(n_steps * STABLE_START)
    i_end = int(n_steps * STABLE_END)
    contact_force_stable = contact_force[i_start:i_end]
    x_stable = x_vec[i_start:i_end]
    y_panto_stable = y_pantograph[i_start:i_end]
    y_cat_stable = y_rigid_overhead_contact_system[i_start:i_end]
    stats_stable = compute_stats(contact_force_stable)

    if verbose:
        print(f'\n--- Contact force statistics (stable window {int(STABLE_START * 100)}%–{int(STABLE_END * 100)}%) ---')
        for k, v in stats_stable.items():
            print(f'  {k:<28}: {v:.3f}')
        print()

    return {
        'x_vec': x_vec,
        't_vec': t_vec,
        'contact_force': contact_force,
        'y_pantograph': y_pantograph,
        'y_rigid_overhead_contact_system': y_rigid_overhead_contact_system,
        'x_stable': x_stable,
        't_stable': t_vec[i_start:i_end],
        'contact_force_stable': contact_force_stable,
        'y_pantograph_stable': y_panto_stable,
        'y_rigid_overhead_contact_system_stable': y_cat_stable,
        'stats_full': stats_full,
        'stats_stable': stats_stable,
    }


def _use_cjk_font() -> bool:
    """尝试启用一个可用的中文字体; 找不到则返回 False (改用英文标签)."""
    candidates = [
        'Noto Sans CJK SC',
        'Microsoft YaHei',
        'SimHei',
        'WenQuanYi Zen Hei',
        'Source Han Sans SC',
        'Arial Unicode MS',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            return True
    return False


def plot_results(
    results: dict,
    pantograph: int = PANTOGRAPH,
    rigid_overhead_contact_system: int = RIGID_OVERHEAD_CONTACT_SYSTEM,
    speed_kmh: float = SPEED_KMH,
    out_dir: str | Path = './result/pc_plots',
    show: bool = True,
):
    """绘制全部段与稳定段的弓网接触力、弓头位移、刚性接触网位移.

    3 行 (三个物理量) × 2 列 (左: 全部段, 右: 稳定段), 横轴为沿线位置 x [m].
    图保存到 `out_dir`, 文件名按 (受电弓, 接触网, 速度) 区分.
    """
    cjk = _use_cjk_font()
    if cjk:
        col_titles = ('全部段', f'稳定段 ({int(STABLE_START * 100)}%–{int(STABLE_END * 100)}%)')
        xlabel = '沿线位置 x [m]'
        rows = [
            ('弓网接触力 [N]', 'contact_force', 'contact_force_stable'),
            ('弓头位移 [m]', 'y_pantograph', 'y_pantograph_stable'),
            ('刚性接触网位移 [m]', 'y_rigid_overhead_contact_system', 'y_rigid_overhead_contact_system_stable'),
        ]
        sup = f'受电弓 {pantograph} · 接触网 {rigid_overhead_contact_system} · {int(round(speed_kmh))} km/h'
    else:
        col_titles = ('Full run', f'Stable window ({int(STABLE_START * 100)}%–{int(STABLE_END * 100)}%)')
        xlabel = 'Position x [m]'
        rows = [
            ('Contact force [N]', 'contact_force', 'contact_force_stable'),
            ('Pantograph disp. [m]', 'y_pantograph', 'y_pantograph_stable'),
            ('OCS disp. [m]', 'y_rigid_overhead_contact_system', 'y_rigid_overhead_contact_system_stable'),
        ]
        sup = f'Pantograph {pantograph} · OCS {rigid_overhead_contact_system} · {int(round(speed_kmh))} km/h'

    x_full = results['x_vec']
    x_stab = results['x_stable']

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex='col')
    for r, (ylabel, key_full, key_stab) in enumerate(rows):
        axes[r, 0].plot(x_full, results[key_full], lw=0.7, color='C0')
        axes[r, 1].plot(x_stab, results[key_stab], lw=0.7, color='C1')
        axes[r, 0].set_ylabel(ylabel)
        for c in (0, 1):
            axes[r, c].grid(alpha=0.3)
    for c in (0, 1):
        axes[0, c].set_title(col_titles[c])
        axes[-1, c].set_xlabel(xlabel)
    fig.suptitle(sup)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'p{pantograph}_c{rigid_overhead_contact_system}_{int(round(speed_kmh))}kmh'
    fig_path = out_dir / f'pc_response_{suffix}.png'
    fig.savefig(fig_path, dpi=150)
    print(f'Figure saved → {fig_path}')
    if show:
        plt.show()
    plt.close(fig)
    return fig_path


if __name__ == '__main__':
    results = run_simulation(
        rigid_overhead_contact_system=RIGID_OVERHEAD_CONTACT_SYSTEM,
        pantograph=PANTOGRAPH,
        speed_kmh=SPEED_KMH,
        NM=NM,
        verbose=True,
    )
    plot_results(
        results,
        pantograph=PANTOGRAPH,
        rigid_overhead_contact_system=RIGID_OVERHEAD_CONTACT_SYSTEM,
        speed_kmh=SPEED_KMH,
    )
