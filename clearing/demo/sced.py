"""
SCED 求解器
"""

import numpy as np
import pyomo.environ as pyo
from pypower.api import case30

# ----- 供外部调用的主函数 -----
def solve_sced(mpc=None, u_vector=None, solver_name="gurobi", ref_bus=0, tee=False):
    """
    求解 SCED.
    参数:
      - mpc: pypower case dict (如果为 None, 将自动读 case30())
      - u_vector: numpy array 或 list，长度 = ng，包含 scuc 得到的 0/1 开机状态。
                  如果为 None，函数会尝试从本目录的 scuc.py 导入 solve_scuc 并执行以获得 u。
      - solver_name: 优先求解器（默认 'gurobi'）
      - ref_bus: 参考母线索引（0-based），用于固定 theta
      - tee: 是否打印求解器输出
    返回:
      - Pg_vals: numpy array (ng,) 发电机出力（p.u.）
      - LMPs: numpy array (nb,) 每个节点的边际电价（dual of nodal balance）
    """

    # 读取 mpc
    if mpc is None:
        mpc = case30()

    baseMVA = mpc['baseMVA']
    bus = mpc['bus']
    gen = mpc['gen']
    gencost = mpc['gencost']
    branch = mpc['branch']

    nb = bus.shape[0]
    ng = gen.shape[0]
    nl = branch.shape[0]

    # 基本数据（p.u.）
    Pd = bus[:, 2] / baseMVA           # 注：pypower bus[:,2] 是 MW 有功负荷
    gen_bus = gen[:, 0].astype(int) - 1
    Pgmin = gen[:, 9] / baseMVA
    Pgmax = gen[:, 8] / baseMVA

    # 取成本系数：gencost 格式 [model,start,shutdown,n,c2,c1,c0]
    a = gencost[:, 4]   # quadratic
    b = gencost[:, 5]   # linear
    c = gencost[:, 6]   # constant
    # ignore constant term / startup here

    # 计算网络连接与参数（用于 DC）
    from_bus = branch[:, 0].astype(int) - 1
    to_bus = branch[:, 1].astype(int) - 1
    x = branch[:, 3]    # reactance (per unit)
    rateA = branch[:, 5] / baseMVA  # 支路热极限，p.u.

    # 如果没有提供 u_vector，则尝试导入 scuc.solve_scuc
    if u_vector is None:
        try:
            import scuc
            print("尝试调用 scuc.solve_scuc() 获取启停计划 u...")
            scu_u = scuc.solve_scuc(mpc)
            # scuc.solve_scuc 可能直接返回 numpy arrays or lists; 兼容处理
            u_vector = np.array(scu_u).astype(float)
            print("成功获取 u（来自 scuc）")
        except Exception as e:
            raise RuntimeError(
                "没有提供 u_vector，且无法自动调用 scuc.solve_scuc() 获取启停计划。"
                "请先运行 SCUC 并把 u（0/1）传入本函数，或确保 scuc.py 可导入并返回 (Pg,u)。\n"
                f"底层错误: {e}"
            )

    u_vector = np.asarray(u_vector, dtype=float)
    if u_vector.shape[0] != ng:
        raise ValueError(f"u_vector 长度 ({u_vector.shape[0]}) 与机组数 ({ng}) 不匹配。")

    # --- 构建 Pyomo 模型 ---
    model = pyo.ConcreteModel()
    model.G = pyo.RangeSet(0, ng-1)
    model.B = pyo.RangeSet(0, nb-1)
    model.L = pyo.RangeSet(0, nl-1)

    # 变量
    model.Pg = pyo.Var(model.G, within=pyo.NonNegativeReals)   # 发电
    model.theta = pyo.Var(model.B, within=pyo  .Reals)           # 节点相角
    model.f = pyo.Var(model.L, within=pyo.Reals)               # 支路潮流（正向 from->to）

    # 目标：二次函数 sum(a Pg^2 + b Pg)
    def obj_rule(m):
        return sum(a[g] * m.Pg[g]**2 + b[g] * m.Pg[g] + c[g] for g in range(ng))
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # 引入 u 的常数参数到模型（以便表达 Pgmin*u 的右端是常数）
    # Pyomo Param 需要指定 mutable=False（默认）以便在建模时当常数使用
    model.u_par = pyo.Param(model.G, initialize={i: float(u_vector[i]) for i in range(ng)}, mutable=False)

    # Pg 出力约束：Pgmin * u <= Pg <= Pgmax * u
    def pg_lower(m, g):
        return m.Pg[g] >= Pgmin[g] * m.u_par[g]
    def pg_upper(m, g):
        return m.Pg[g] <= Pgmax[g] * m.u_par[g]
    model.pg_lower = pyo.Constraint(model.G, rule=pg_lower)
    model.pg_upper = pyo.Constraint(model.G, rule=pg_upper)

    # 参考母线角度固定
    def ref_theta(m):
        return m.theta[ref_bus] == 0.0
    model.ref_con = pyo.Constraint(rule=ref_theta)

    # 支路潮流定义： f_l = (theta_i - theta_j) / x_l
    def flow_def(m, l):
        i = from_bus[l]
        j = to_bus[l]
        return m.f[l] == (m.theta[i] - m.theta[j]) / x[l]
    model.flow_def = pyo.Constraint(model.L, rule=flow_def)

    # 支路容量限制
    def flow_limit(m, l):
        return (-rateA[l], m.f[l], rateA[l])
    model.flow_limit = pyo.Constraint(model.L, rule=flow_limit)

    # 节点功率平衡： sum_{g at i} Pg_g - Pd_i == sum_{l: i->*} f_l - sum_{l: *->i} f_l
    # 构造节点到支路的 incidence（符号）关系
    # 对每个节点 i，找出以 i 为 from 的支路以及以 i 为 to 的支路
    from_inc = {i: [] for i in range(nb)}
    to_inc   = {i: [] for i in range(nb)}
    for l in range(nl):
        from_inc[from_bus[l]].append(l)
        to_inc[to_bus[l]].append(l)

    def nodal_balance(m, i):
        gen_sum = sum(m.Pg[g] for g in range(ng) if gen_bus[g] == i)
        inflow = sum(m.f[l] for l in to_inc[i])    # flows coming into node (from other->i)
        outflow = sum(m.f[l] for l in from_inc[i]) # flows leaving node (i->other)
        return gen_sum - Pd[i] + inflow - outflow == 0
    model.nodal_balance = pyo.Constraint(model.B, rule=nodal_balance)

    # --- dual suffix 用于读取 LMP ---
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # --- 求解 ---
    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError(f"求解器 {solver_name} 不可用，请安装 Gurobi/CPLEX/IPOPT 等支持对偶导出的求解器。")

    # 对于 Gurobi，可以设置参数以确保导出对偶（通常可直接）
    res = solver.solve(model, tee=tee)

    # 检查求解状态
    status = res.solver.status
    term = res.solver.termination_condition
    print(f"求解状态: {status}, 终止条件: {term}")

    if (str(status).lower() not in ("ok","warning")) and (str(term).lower() not in ("optimal","optimal_tolerance")):
        print("注意：求解器没有返回最优解，继续但请谨慎解读结果。")

    # 提取 Pg 值
    Pg_vals = np.array([pyo.value(model.Pg[g]) for g in range(ng)], dtype=float)

    # 提取 LMPs：节点功率平衡约束的 duals（对偶）
    # 注意：有些求解器返回的是基于目标单位的对偶；若你需要 $/MWh，可乘以 baseMVA / 1 等进行单位换算
    LMPs = np.zeros(nb, dtype=float)
    for i in range(nb):
        con = model.nodal_balance[i]
        try:
            dual_val = model.dual.get(con, None)
            if dual_val is None:
                # 有时 duals 以 model.dual[con] 可索引
                dual_val = model.dual[con]
        except Exception:
            dual_val = None

        if dual_val is None:
            # 如果没有对偶信息，设为 NaN 并提示
            LMPs[i] = float('nan')
        else:
            LMPs[i] = float(dual_val)

    # 打印结果（p.u. 单位）
    print("\n--- SCED 结果（p.u. 单位） ---")
    for g in range(ng):
        print(f"Gen {g+1} @ bus {gen_bus[g]+1:2d}: Pg = {Pg_vals[g]:8.4f} p.u. (u={int(u_vector[g])})")
    print("\n--- 节点边际电价 LMPs（来自 nodal balance 的对偶） ---")
    for i in range(nb):
        print(f"Bus {i+1:2d}: LMP = {LMPs[i]:.4f}")

    return Pg_vals, LMPs


# 如果直接运行 sced.py，就尝试调用 scuc 并求解
if __name__ == "__main__":
    print("读取 case30 数据...")
    mpc = case30()
    Pg_sced, LMPs = solve_sced(mpc=mpc)
