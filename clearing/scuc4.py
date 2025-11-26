"""
SCUC 求解器 - 使用新的数据存储方案
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import utils2
import scipy.io

import dataloader

# t=0启动惩罚系数
begin_punish_factor = 0

def build_scuc_model(data, Pd, Pgmin, Pgmax, T):    
    m = pyo.ConcreteModel()
    
    buses = data['buses']
    lines = data['lines']
    gens = data['gens']
    storages = data['storages']
    print('build_scuc_model:len(buses)=', len(buses))
    print('build_scuc_model:len(lines)=', len(lines))
    print('build_scuc_model:len(gens)=', len(gens))
    print('build_scuc_model:len(storages)=', len(storages))
    
    # 节点、发电机、线路索引
    bus_indices = range(len(buses))
    gen_indices = range(len(gens))
    line_indices = range(len(lines))
    storage_indices = range(len(storages))
    time_indices = range(T)
    
    # 获取初始出力（用于爬坡约束）
    Pglast = np.array([gen.last_output for gen in gens])
    ramp_limit = np.array([gen.ramp * gen.capacity * 60 for gen in gens])
    
    # 发电机连接节点
    gen_bus = np.array([gen.bus_id - 1 for gen in gens])
    
    c1 = np.array([gen.outcost for gen in gens])          # 可变成本系数 (万元/MW)
    start_cost = np.array([gen.startcost for gen in gens]) # 原始启动成本 (万元/次) 
    
    PTDF = utils2.build_PTDF(buses, lines)

    if storages:
        storage_bus = np.array([s.bus_id - 1 for s in storages])  # 节点索引(0-based)
        storage_capacity = np.array([s.capacity for s in storages])  # 能量容量(MWh)
        storage_eff = np.sqrt(np.array([s.efficiency for s in storages]))  # 单程效率(假设往返效率=eff^2)
        init_soc = np.array([s.soc for s in storages])  # 初始SOC比例
        # 最大充放电功率 = 容量 (1C速率，1小时充满/放完)
        max_charge_rate = np.array([s.capacity/s.storage_time for s in storages])
        max_discharge_rate = max_charge_rate.copy()

    
    # === 变量 ===
    m.Pg = pyo.Var(gen_indices, time_indices, within=pyo.NonNegativeReals)
    m.u = pyo.Var(gen_indices, time_indices, within=pyo.Binary)
    # 新增：启动事件变量 
    m.v = pyo.Var(gen_indices, range(1, T), within=pyo.Binary)  # v[g,t]表示t时刻启动
    if storages:
        m.ch = pyo.Var(storage_indices, time_indices, within=pyo.NonNegativeReals)  # 充电功率(MW)
        m.dis = pyo.Var(storage_indices, time_indices, within=pyo.NonNegativeReals) # 放电功率(MW)
        m.soc = pyo.Var(storage_indices, time_indices, within=pyo.NonNegativeReals) # SOC(能量,MWh)
    
    # === 目标函数 ===
    m.obj = pyo.Objective(
        expr=sum(
            c1[g] * m.Pg[g, t]  # 可变成本
            + (start_cost[g] * m.v[g, t] if t >= 1 else 0)  # t>=1的启动成本
            + (begin_punish_factor * start_cost[g] * m.u[g, 0] if t == 0 else 0)  # t=0的启动惩罚，值为0，允许全启动
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
        total_gen = sum(m.Pg[g, t] for g in gen_indices)
        total_load = sum(Pd[i][t] for i in bus_indices)
        
        storage_net = 0
        if storages:
            total_ch = sum(m.ch[s, t] for s in storage_indices)
            total_dis = sum(m.dis[s, t] for s in storage_indices)
            storage_net = total_dis - total_ch
        
        return total_gen + storage_net == total_load
    m.balance = pyo.Constraint(time_indices, rule=balance_rule) 

    # === 线路潮流 === 
    def flow_rule(m, l, t):
        flow = 0
        for i in bus_indices:
            # 常规机组注入
            gen_at_bus = sum(m.Pg[g, t] for g in gen_indices if gen_bus[g] == i)
            
            # 储能净注入 (放电-充电)
            storage_net_injection = 0
            if storages:
                ch_at_bus = sum(m.ch[s, t] for s in storage_indices if storage_bus[s] == i)
                dis_at_bus = sum(m.dis[s, t] for s in storage_indices if storage_bus[s] == i)
                storage_net_injection = dis_at_bus - ch_at_bus  # 放电为正注入，充电为负
            
            # 节点净注入 = 发电 - 负荷 + 储能净注入
            net_injection = gen_at_bus - Pd[i][t] + storage_net_injection
            flow += PTDF[l, i] * net_injection
        
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
    # 旋转备用也能尽量减少sced步无解
    gen_capacity = np.array([gen.capacity for gen in gens])  
    def reserve_rule(m, t):
        """开机总容量 >= 1.05 * 该时段总负荷"""
        total_load = sum(Pd[i][t] for i in bus_indices)
        total_online_capacity = sum(gen_capacity[g] * m.u[g, t] for g in gen_indices)
        return total_online_capacity >= 1.1 * total_load
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

    if storages:
        
        # (1) 初始SOC约束 - 保持不变
        def init_soc_rule(m, s):
            return m.soc[s, 0] == init_soc[s] * storage_capacity[s]
        m.init_soc_con = pyo.Constraint(storage_indices, rule=init_soc_rule)
        
        # (2) 能量平衡约束 - 修正版本
        def storage_balance_rule(m, s, t):
            if t == 0:
                # t=0 时刻：从初始SOC开始，考虑第0时刻的充放电
                return (m.soc[s, 0] == init_soc[s] * storage_capacity[s] 
                        + m.ch[s, 0] * storage_eff[s]   # 第0时刻充电
                        - m.dis[s, 0] / storage_eff[s]) # 第0时刻放电
            else:
                # t≥1 时刻：正常递推
                return (m.soc[s, t] == m.soc[s, t-1] 
                        + m.ch[s, t] * storage_eff[s]   
                        - m.dis[s, t] / storage_eff[s])
            
        m.storage_balance = pyo.Constraint(storage_indices, time_indices, rule=storage_balance_rule)

        
        # (3) 充放电功率限制
        def charge_limit_rule(m, s, t):
            return m.ch[s, t] <= max_charge_rate[s]
        m.charge_limit = pyo.Constraint(storage_indices, time_indices, rule=charge_limit_rule)
        
        def discharge_limit_rule(m, s, t):
            return m.dis[s, t] <= max_discharge_rate[s]
        m.discharge_limit = pyo.Constraint(storage_indices, time_indices, rule=discharge_limit_rule)
        
        # (4) SOC安全边界
        def soc_limits_rule(m, s, t):
            return (0, m.soc[s, t], storage_capacity[s])
        m.soc_limits = pyo.Constraint(storage_indices, time_indices, rule=soc_limits_rule)
        
        # (5) 周期结束SOC要求 (t=T-1)
        def end_soc_rule(m, s):
            return m.soc[s, T-1] == 0.5 * storage_capacity[s]  # 恢复至50% SOC
        m.end_soc = pyo.Constraint(storage_indices, rule=end_soc_rule)

    return m


def solve_scuc(data, T, day, solver_name='gurobi'):

    load_data_mat = scipy.io.loadmat('data_PROSPECT43/load_PROSPECT43.mat')
    vre_data_mat = scipy.io.loadmat('data_PROSPECT43/vre_PROSPECT43.mat')

    # 这里只考虑第一天的数据
    (Pd, Pgmin, Pgmax) = utils2.calculate_PdPg(data, T, range(day,day+1), range(T), load_data_mat, vre_data_mat)

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
                    else:
                        u_value = pyo.value(model.u[g, t])
                        if u_value is not None and u_value > 0.5:  # 确认是启动事件
                            time_cost += gen.startcost * begin_punish_factor
                    
                    total_cost += time_cost
                    total_gen += pg_value
            
            # 添加目标函数值验证
            model_obj = pyo.value(model.obj)
            print(f"[验证] 模型目标函数值: {model_obj:.2f} 万元")
            print(f"[验证] 手动计算总成本: {total_cost:.2f} 万元")
            if abs(total_cost - model_obj) > 1e-3:
                print(f"警告: 成本计算差异 {abs(total_cost - model_obj):.2f} 万元")
            
            print(f"\n总发电量: {total_gen:.1f} MW")
            print(f"总成本: {total_cost:.2f} 万元")
            
            storage_results = None
            storages = data['storages']
            storage_indices = range(len(storages))
            if storages and result.solver.status == pyo.SolverStatus.ok:
                storage_ch = [[pyo.value(model.ch[s, t]) for t in range(T)] for s in storage_indices]
                storage_dis = [[pyo.value(model.dis[s, t]) for t in range(T)] for s in storage_indices]
                storage_soc = [[pyo.value(model.soc[s, t]) for t in range(T)] for s in storage_indices]
                
                # 转置为时间×储能格式
                storage_ch = list(map(list, zip(*storage_ch)))
                storage_dis = list(map(list, zip(*storage_dis)))
                storage_soc = list(map(list, zip(*storage_soc)))
                
                storage_results = {
                    'charge': storage_ch,    # 每小时充电功率 [t][storage_id]
                    'discharge': storage_dis, # 每小时放电功率
                    'soc': storage_soc        # 每小时SOC (MWh)
                }


            # 获取启停状态
            u_status = [
                [int(pyo.value(model.u[g, t])) for t in range(T)]
                for g in range(len(data['gens']))
            ]
            # 注意：u_status[g][t] 表示机组g在时刻t的状态
            # 若需要按时间组织（t, g），则转置；否则保持原结构

            print('scuc4.solve_scuc: 机组启停状态 (机组×时间):(略。Search Code:136436)')
            # for g, status in enumerate(u_status):
            #     print(f"  机组 {g+1}: {status}")

            # 输出储能结果
            if storages:
                print(storage_results['charge'])
                print(storage_results['discharge'])
                print(storage_results['soc'])
            
            # u_status 转置然后转为list
            u_status = list(map(list, zip(*u_status)))
            
            return (u_status, storage_results) if storages else u_status
            
        else:
            print(f"求解失败: {result.solver.termination_condition}")
            return None
            
    except Exception as e:
        print(f"求解过程中出错: {e}")
        import traceback
        traceback.print_exc()  # 添加详细错误堆栈
        return None

