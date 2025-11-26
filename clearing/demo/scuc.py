"""
SCUC 求解器
"""

import pyomo.environ as pyo
from pypower.api import case30
import numpy as np
import utils

def load_case_data():
    return case30()


def build_scuc_model(mpc):
    """构建 Pyomo 模型"""
    m = pyo.ConcreteModel()

    baseMVA = mpc['baseMVA']
    bus = mpc['bus']
    gen = mpc['gen']
    gencost = mpc['gencost']
    branch = mpc['branch']

    nb = bus.shape[0]
    ng = gen.shape[0]
    nl = branch.shape[0]

    buses = range(nb)
    gens = range(ng)
    lines = range(nl)

    Pd = bus[:, 2] / baseMVA
    gen_bus = gen[:, 0].astype(int) - 1
    Pgmin = gen[:, 9] / baseMVA
    Pgmax = gen[:, 8] / baseMVA
    Fmax = branch[:, 5] / baseMVA

    startup_cost = gencost[:, 1]
    c2 = gencost[:, 4]
    c1 = gencost[:, 5]
    c0 = gencost[:, 6]

    PTDF = utils.build_PTDF(mpc)

    # === 变量 ===
    m.Pg = pyo.Var(gens, within=pyo.NonNegativeReals)
    m.u = pyo.Var(gens, within=pyo.Binary)

    # === 目标函数 ===
    def obj_rule(m):
        return sum(c2[g]*m.Pg[g]**2 + c1[g]*m.Pg[g] + c0[g]*m.u[g] for g in gens) + sum(startup_cost[g]*m.u[g] for g in gens)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # === 出力约束 ===
    def upper_rule(m, g):
        return m.Pg[g] <= Pgmax[g] * m.u[g]
    def lower_rule(m, g):
        return m.Pg[g] >= Pgmin[g] * m.u[g]
    m.gen_upper = pyo.Constraint(gens, rule=upper_rule)
    m.gen_lower = pyo.Constraint(gens, rule=lower_rule)

    # === 系统功率平衡 ===
    def power_balance_rule(m):
        return sum(m.Pg[g] for g in gens) == sum(Pd)
    m.balance = pyo.Constraint(rule=power_balance_rule)

    # === 潮流约束 ===
    def flow_rule(m, l):
        flow = sum(PTDF[l, i] * (sum(m.Pg[g] for g in gens if gen_bus[g] == i) - Pd[i])
                   for i in buses)
        return (-Fmax[l], flow, Fmax[l])
    m.flow = pyo.Constraint(lines, rule=flow_rule)

    return m


def solve_scuc(mpc, solver_name='gurobi'):
    print("构建模型...")
    model = build_scuc_model(mpc)

    print("求解模型...")
    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError(f"Solver '{solver_name}' 不可用。")

    result = solver.solve(model, tee=False)

    print("\n=== SCUC 求解结果 ===")
    for g in model.Pg:
        print(f"Gen {g+1}: Pg = {pyo.value(model.Pg[g]):.3f} p.u. | u = {int(pyo.value(model.u[g]))}")
    print(f"\n总成本: {pyo.value(model.obj):.3f}")

    # 以列表形式返回变量的求解结果u
    return [int(pyo.value(model.u[g])) for g in model.u]


if __name__ == "__main__":
    print("加载 IEEE 30-bus 数据...")
    mpc = load_case_data()

    # 将mpc[gencost]中的第二列全部改为0.001
    # mpc['gencost'][:, 1] = 0.001

    solve_scuc(mpc)
