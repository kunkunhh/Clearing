"""
SCUC 求解器 - 使用新的数据存储方案
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import utils2

import dataloader

def build_scuc_model(data, Pd, Pgmin, Pgmax, T):
    m = pyo.ConcreteModel()
    
    buses = data['buses']
    lines = data['lines']
    gens = data['gens']
    print('build_scuc_model:len(buses)=', len(buses))
    print('build_scuc_model:len(lines)=', len(lines))
    print('build_scuc_model:len(gens)=', len(gens))
    
    # 节点、发电机、线路索引
    bus_indices = range(len(buses))
    gen_indices = range(len(gens))
    line_indices = range(len(lines))
    time_indices = range(T)
    
    # 获取初始出力（用于爬坡约束）
    Pglast = np.array([gen.last_output for gen in gens])
    ramp_limit = np.array([gen.ramp * gen.capacity * 60 for gen in gens])
    
    # 发电机连接节点
    gen_bus = np.array([gen.bus_id - 1 for gen in gens])
    
    c1 = np.array([gen.outcost for gen in gens])          # 可变成本系数 (万元/MW)
    start_cost = np.array([gen.startcost for gen in gens]) # 原始启动成本 (万元/次) 
    
    PTDF = utils2.build_PTDF(buses, lines)
    
    # === 变量 ===
    m.Pg = pyo.Var(gen_indices, time_indices, within=pyo.NonNegativeReals)
    m.u = pyo.Var(gen_indices, time_indices, within=pyo.Binary)
    # 新增：启动事件变量 
    m.v = pyo.Var(gen_indices, range(1, T), within=pyo.Binary)  # v[g,t]表示t时刻启动
    
    # === 目标函数 ===
    m.obj = pyo.Objective(
        expr=sum(
            c1[g] * m.Pg[g, t]  # 可变成本
            + (start_cost[g] * m.v[g, t] if t >= 1 else 0)  # t>=1的启动成本
            + (0 * start_cost[g] * m.u[g, 0] if t == 0 else 0)  # t=0的启动惩罚
            for g in gen_indices for t in time_indices
        ),
        sense=pyo.minimize
    )   
    
    # === 出力上下界 === 
    def upper_rule(m, g, t):
        return m.Pg[g, t] <= Pgmax[g, t] * m.u[g, t]
    m.gen_upper = pyo.Constraint(gen_indices, time_indices, rule=upper_rule)

    def lower_rule(m, g, t):
        return m.Pg[g, t] >= Pgmin[g, t] * m.u[g, t]
    m.gen_lower = pyo.Constraint(gen_indices, time_indices, rule=lower_rule)

    # === 功率平衡 === 
    def balance_rule(m, t):
        return sum(m.Pg[g, t] for g in gen_indices) == sum(Pd[i][t] for i in bus_indices)
    m.balance = pyo.Constraint(time_indices, rule=balance_rule)

    # === 线路潮流 === 
    def flow_rule(m, l, t):
        flow = 0
        for i in bus_indices:
            gen_at_bus = sum(m.Pg[g, t] for g in gen_indices if gen_bus[g] == i)
            flow += PTDF[l, i] * (gen_at_bus - Pd[i][t])
        return (-lines[l].capacity, flow, lines[l].capacity)
    m.flow = pyo.Constraint(line_indices, time_indices, rule=flow_rule)

    # === 爬坡约束 === 
    def ramp_rule(m, g, t):
        if t == 0:
            BIG_M = 1e9
            return (-BIG_M, m.Pg[g, 0] - Pglast[g], BIG_M)
        else:
            return (-ramp_limit[g], m.Pg[g, t] - m.Pg[g, t-1], ramp_limit[g])
    m.ramp = pyo.Constraint(gen_indices, time_indices, rule=ramp_rule)

    # === 预留容量/旋转备用约束 ===
    gen_capacity = np.array([gen.capacity for gen in gens])  
    def reserve_rule(m, t):
        """开机总容量 >= 1.05 * 该时段总负荷"""
        total_load = sum(Pd[i][t] for i in bus_indices)
        total_online_capacity = sum(gen_capacity[g] * m.u[g, t] for g in gen_indices)
        return total_online_capacity >= 1.05 * total_load
    m.reserve_con = pyo.Constraint(time_indices, rule=reserve_rule)

    # === 启动事件约束 (仅t>=1) ===
    def startup_rule(m, g, t):
        """当且仅当 t 时刻开机且 t-1 时刻停机时，v[g,t]=1"""
        return m.v[g, t] >= m.u[g, t] - m.u[g, t-1]
    m.startup_con = pyo.Constraint(gen_indices, range(1, T), rule=startup_rule)
    
    # 确保 v 不会过度激活
    def v_upper_rule(m, g, t):
        """v[g,t] 不能超过 1 - u[g,t-1]（上一时刻停机才可能启动）"""
        return m.v[g, t] <= 1 - m.u[g, t-1]
    m.v_upper = pyo.Constraint(gen_indices, range(1, T), rule=v_upper_rule)

    return m


def solve_scuc(data, T, solver_name='gurobi'):

    # 这里days设置为[0]，只考虑第一天的数据
    (Pd, Pgmin, Pgmax) = utils2.calculate_PdPg(data, T, range(114,115), range(T))

    print("构建SCUC模型...")
    model = build_scuc_model(data, Pd, Pgmin, Pgmax, T)
    
    print("求解模型...")
    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError("求解器不可用。")
    
    try:
        result = solver.solve(model, tee=False)
        
        # 检查求解状态
        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("\n=== SCUC 求解结果 ===")
    
            total_cost = 0
            total_gen = 0
            
            for g in range(len(data['gens'])):  
                for t in range(T):  
                    pg_value = pyo.value(model.Pg[g, t])
                    u_value = pyo.value(model.u[g, t])
                    gen = data['gens'][g]
                    
                    # 可变成本
                    time_cost = pg_value * gen.outcost
                    
                    # 仅当t>=1且发生启动时添加启动成本
                    if t >= 1:
                        v_value = pyo.value(model.v[g, t])
                        if v_value is not None and v_value > 0.5:  # 确认是启动事件
                            time_cost += gen.startcost
                    
                    total_cost += time_cost
                    total_gen += pg_value
            
            # 添加目标函数值验证（重要！）
            model_obj = pyo.value(model.obj)
            print(f"[验证] 模型目标函数值: {model_obj:.2f} 万元")
            print(f"[验证] 手动计算总成本: {total_cost:.2f} 万元")
            if abs(total_cost - model_obj) > 1e-3:
                print(f"警告: 成本计算差异 {abs(total_cost - model_obj):.2f} 万元")
            
            print(f"\n总发电量: {total_gen:.1f} MW")
            print(f"总成本: {total_cost:.2f} 万元")
            
            # 获取启停状态
            u_status = [
                [int(pyo.value(model.u[g, t])) for t in range(T)]
                for g in range(len(data['gens']))
            ]
            # 注意：u_status[g][t] 表示机组g在时刻t的状态
            # 若需要按时间组织（t, g），则转置；否则保持原结构
            print('机组启停状态 (机组×时间):')
            for g, status in enumerate(u_status):
                print(f"  机组 {g+1}: {status}")
            
            # u_status 转置然后转为list
            u_status = list(map(list, zip(*u_status)))
            
            return u_status
            
        else:
            print(f"求解失败: {result.solver.termination_condition}")
            return None
            
    except Exception as e:
        print(f"求解过程中出错: {e}")
        import traceback
        traceback.print_exc()  # 添加详细错误堆栈
        return None

