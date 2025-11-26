"""
SCUC 求解器 - 使用新的数据存储方案
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import utils2

import dataloader

def build_scuc_model(data):
    """构建 Pyomo 模型 - 使用新的数据对象"""
    m = pyo.ConcreteModel()
    
    buses = data['buses']
    lines = data['lines']
    gens = data['gens']
    print('len(buses):',len(buses))
    print('len(lines):',len(lines))
    print('len(gens):',len(gens))
    
    # 节点、发电机、线路索引
    bus_indices = range(len(buses))
    gen_indices = range(len(gens))
    line_indices = range(len(lines))
    
    Pd = np.array([bus.load for bus in buses])   
    Pgmin = np.array([gen.min_output_rate * gen.capacity for gen in gens]) 
    Pgmax = np.array([gen.upper for gen in gens]) 
    Pglast = np.array([gen.last_output for gen in gens])
    ramp_limit = np.array([gen.ramp*gen.capacity*60 for gen in gens])
    Fmax = np.array([line.capacity for line in lines]) 
    for i in range(len(gens)):
        if Pgmin[i] > Pgmax[i]:
            raise ValueError(f"发电机 {gens[i].name} 的最小出力大于最大出力")
    
    # 发电机连接节点
    gen_bus = []
    for gen in gens:
        bus_id = gen.bus_id
        gen_bus.append(bus_id - 1)  # 转换为0-based索引
    
    
    # 成本系数（线性成本）
    c1 = np.array([gen.outcost for gen in gens]) 
    c0 = np.array([gen.startcost*0.04 for gen in gens]) 
    
    # 构建PTDF矩阵
    PTDF = utils2.build_PTDF(buses, lines)
    
    # === 变量 ===
    m.Pg = pyo.Var(gen_indices, within=pyo.NonNegativeReals, bounds=(0, max(Pgmax)))
    m.u = pyo.Var(gen_indices, within=pyo.Binary)
    
    # === 目标函数 ===
    def obj_rule(m):
        return sum(c1[g] * m.Pg[g]  + c0[g] * m.u[g] for g in gen_indices)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # === 发电机出力约束 ===
    def upper_rule(m, g):
        return m.Pg[g] <= Pgmax[g] * m.u[g]
    
    def lower_rule(m, g):
        return m.Pg[g] >= Pgmin[g] * m.u[g]
    
    m.gen_upper = pyo.Constraint(gen_indices, rule=upper_rule)
    m.gen_lower = pyo.Constraint(gen_indices, rule=lower_rule)
    
    # === 系统功率平衡 ===
    def power_balance_rule(m):
        total_gen = sum(m.Pg[g] for g in gen_indices)
        total_load = sum(Pd)
        return total_gen == total_load
    
    m.balance = pyo.Constraint(rule=power_balance_rule)
    
    # === 线路潮流约束 ===
    def flow_rule(m, l):
        if l < PTDF.shape[0]:
            # 计算线路潮流
            flow = 0
            for i in range(len(buses) - 1):  # 去掉参考节点
                gen_power_at_bus = sum(m.Pg[g] for g in gen_indices if gen_bus[g] == i)
                flow += PTDF[l, i] * (gen_power_at_bus - Pd[i])
            
            return (-Fmax[l], flow, Fmax[l])
        else:
            return pyo.Constraint.Skip
    
    m.flow = pyo.Constraint(line_indices, rule=flow_rule)

    # 添加爬坡约束
    def ramp_rule(m, g):
        return (-ramp_limit[g], m.Pg[g] - Pglast[g], ramp_limit[g])
    if Pglast[0] >= 0:
        m.ramp_constraints = pyo.Constraint(range(len(gens)), rule=ramp_rule)
    
    return m

def solve_scuc(data, solver_name='gurobi'):
    """求解SCUC问题"""
    # 其实就是跑了一下求解，然后进行输出
    print("构建SCUC模型...")
    model = build_scuc_model(data)
    
    print("求解模型...")
    solver = pyo.SolverFactory(solver_name)
    
    if not solver.available():
        raise RuntimeError("求解器不可用。")
    
    try:
        result = solver.solve(model, tee=False)
        
        # 检查求解状态
        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("\n=== SCUC 求解结果 ===")
            
            # 显示发电机状态
            total_cost = 0
            total_gen = 0
            
            # print("\n发电机出力计划:")
            for g in model.Pg:
                pg_value = pyo.value(model.Pg[g])
                u_value = pyo.value(model.u[g])
                gen_cost = pyo.value(model.Pg[g]) * data['gens'][g].outcost + pyo.value(model.u[g]) * data['gens'][g].startcost*0.04
                
                # print(f"Gen {g+1}: Pg = {pg_value:.3f} p.u. ({pg_value:.1f} MW) | "
                #       f"u = {int(u_value)} | 成本 = {gen_cost:.2f} 万元")
                
                total_cost += gen_cost
                total_gen += pg_value 
            
            print(f"总发电量: {total_gen:.1f} MW")
            print(f"总成本: {total_cost:.2f} 万元")
            print(f"目标函数值: {pyo.value(model.obj):.2f}")
            
            # 返回发电机启停状态
            u_status = [int(pyo.value(model.u[g])) for g in model.u]
            print(f"发电机启停状态: {u_status}")
            return u_status
            
        else:
            print(f"求解失败: {result.solver.termination_condition}")
            return None
            
    except Exception as e:
        print(f"求解过程中出错: {e}")
        return None



if __name__ == "__main__":
    print("加载新的数据存储方案...")
    data = dataloader.load_case_data()
    
    # 求解SCUC
    u_status = solve_scuc(data)
    
    if u_status:        
        # 保存结果
        results = {
            'generator_status': u_status,
            'total_generation': sum(pyo.value(build_scuc_model(data).Pg[g]) * 100 for g in range(len(data['gens']))),
            'total_cost': pyo.value(build_scuc_model(data).obj)
        }
        
        print("\nSCUC求解完成!")
    else:
        print("SCUC求解失败")