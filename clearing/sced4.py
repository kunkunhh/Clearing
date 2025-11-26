"""
SCED 求解器 - 基于SCUC的机组启停状态进行经济调度
包含节点边际电价(LMP)计算
"""

import pyomo.environ as pyo
import numpy as np
import utils2
import dataloader
from pyomo.core.base.suffix import Suffix

def build_sced_model(data, generator_status, storage_net_injection=None):
    """构建SCED模型 - 基于SCUC的机组启停状态"""
    m = pyo.ConcreteModel()
    m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    if storage_net_injection is None:
        storage_net_injection = np.zeros(len(data['buses']))
    
    buses = data['buses']
    lines = data['lines']
    gens = data['gens']
    
    # 基本数据
    Pd = np.array([bus.load for bus in buses])   
    Pgmin = np.array([gen.min_output_rate * gen.capacity for gen in gens]) 
    Pgmax = np.array([gen.upper for gen in gens]) 
    Pglast = np.array([gen.last_output for gen in gens])
    ramp_limit = np.array([gen.ramp*gen.capacity*60 for gen in gens])
    Fmax = np.array([line.capacity for line in lines]) 
    c1 = np.array([gen.outcost for gen in gens]) 
    
    # 发电机连接节点
    gen_bus = [gen.bus_id - 1 for gen in gens]  # 转换为0-based索引
    
    # 构建PTDF矩阵
    PTDF = utils2.build_PTDF(buses, lines)
    
    # 变量
    m.Pg = pyo.Var(range(len(gens)), within=pyo.NonNegativeReals, bounds=(0, max(Pgmax)))
    # 注：这里bounds=(0, max(Pgmax))这样写是安全的，因为下面我们给出了出力约束
    
    # 目标函数
    m.obj = pyo.Objective(expr=sum(c1[g] * m.Pg[g] for g in range(len(gens))), sense=pyo.minimize)
    
    # 发电机出力约束
    def gen_rule(m, g):
        if generator_status[g] == 1:
            return (Pgmin[g], m.Pg[g], Pgmax[g])
        else:
            return m.Pg[g] == 0
    
    m.gen_constraints = pyo.Constraint(range(len(gens)), rule=gen_rule)
    
    # 系统功率平衡
    def balance_rule(m):
        total_storage_injection = sum(storage_net_injection[i] for i in range(len(buses)))
        return sum(m.Pg[g] for g in range(len(gens))) == sum(Pd) - total_storage_injection
    m.balance = pyo.Constraint(rule=balance_rule)
    
    # 线路潮流约束
    def flow_rule(m, l):
        if l < PTDF.shape[0]:
            flow = 0
            for i in range(len(buses)):
                gen_power_at_bus = sum(m.Pg[g] for g in range(len(gens)) if gen_bus[g] == i)
                # 新增：加入储能净注入
                net_injection = gen_power_at_bus - Pd[i] + storage_net_injection[i]  # 关键修改
                flow += PTDF[l, i] * net_injection
            return (-Fmax[l], flow, Fmax[l])
        return pyo.Constraint.Skip
    
    # 添加爬坡约束
    def ramp_rule(m, g):
        return (-ramp_limit[g], m.Pg[g] - Pglast[g], ramp_limit[g])
    if Pglast[0] >= 0:
        m.ramp_constraints = pyo.Constraint(range(len(gens)), rule=ramp_rule)
    

    m.flow = pyo.Constraint(range(len(lines)), rule=flow_rule)
    
    return m

def calculate_lmp(model, data):
    """计算节点边际电价(LMP)"""
    buses = data['buses']
    lines = data['lines']
    PTDF = utils2.build_PTDF(buses, lines)
    
    # 获取对偶变量
    system_marginal_price = model.dual[model.balance]
    congestion_prices = []
    for l in range(len(lines)):
        congestion_price = model.dual[model.flow[l]]
        congestion_prices.append(congestion_price)
    
    # 计算LMP
    lmp_values = []
    for i in range(len(buses)):
        congestion_component = 0.0
        for l in range(len(lines)):
            if l < PTDF.shape[0] and i < PTDF.shape[1]:
                congestion_component += congestion_prices[l] * PTDF[l, i]
        lmp_values.append(system_marginal_price - congestion_component)
    
    return lmp_values, system_marginal_price, congestion_prices

def solve_sced(data, generator_status, storage_ch=None, storage_dis=None, solver_name='gurobi'):
    # 同理，仅solve+显示结果
    """求解SCED问题并计算LMP"""

    # 1. 计算节点级储能净注入
    storage_net_injection = np.zeros(len(data['buses']))
    if storage_ch is not None and storage_dis is not None:
        for bus_idx in range(len(data['buses'])):
            total_ch = sum(storage_ch[bus_idx])  # 该节点总充电功率
            total_dis = sum(storage_dis[bus_idx])  # 该节点总放电功率
            storage_net_injection[bus_idx] = total_dis - total_ch  # 净注入
    
    # 2. 构建模型时传入净注入
    model = build_sced_model(data, generator_status, storage_net_injection)
    model = build_sced_model(data, generator_status)


    solver = pyo.SolverFactory(solver_name)
    
    result = solver.solve(model, tee=False)
    
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print("SCED求解失败")
        return None
    
    # 显示结果
    print("\n=== SCED 求解结果 ===")
    
    total_cost = 0
    total_gen = 0
    pg_results = []
    
    # print("发电机经济调度结果:")
    for g in model.Pg:
        pg_value = pyo.value(model.Pg[g])
        gen_cost = pg_value * data['gens'][g].outcost
        status = "运行" if generator_status[g] == 1 else "停机"
        
        # print(f"Gen {g+1}: Pg = {pg_value:.3f} p.u. | 状态 = {status} | 成本 = {gen_cost:.2f} 万元")
        
        total_cost += gen_cost
        total_gen += pg_value
        pg_results.append(pg_value)
    
    print(f"总发电量: {total_gen:.1f} MW")
    print(f"总运行成本: {total_cost:.2f} 万元")
    
    # 计算LMP
    lmp_values, smp, congestion_prices = calculate_lmp(model, data)
    
    print("=== 节点边际电价(LMP)分析 ===")
    print(f"系统边际价格(SMP): {smp*10000:.2f} 元/MWh")
    print('-'*50+'\n')
    
    # print("\n各节点边际电价(LMP):")
    # for i, bus in enumerate(data['buses']):
    #     print(f"节点 {bus.id}: LMP = {lmp_values[i]*10000:.2f} 元/MWh")
    
    # print("\n线路阻塞价格:")
    # for l, line in enumerate(data['lines']):
    #     print(f"线路 {l+1}: 阻塞价格 = {congestion_prices[l]*10000:.2f} 元/MWh")
    
    return {
        'pg_results': pg_results,
        'lmp_values': lmp_values,
        'system_marginal_price': smp,
        'congestion_prices': congestion_prices,
        'total_cost': total_cost
    }

