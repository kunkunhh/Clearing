import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataloader import buses, lines, gens
from matplotlib.colors import Normalize
import dataloader


def build_PTDF(buses, lines, ref_bus_id=1):
    """生成 DC 功率传输分布因子矩阵 PTDF - 使用新的数据格式"""
    nb = len(buses)  # 节点数量
    nl = len(lines)  # 线路数量
    
    # 创建节点ID到索引的映射
    bus_id_to_idx = {bus.id: i for i, bus in enumerate(buses)}
    
    # 提取线路参数
    from_bus = [bus_id_to_idx[line.from_bus_id] for line in lines]
    to_bus = [bus_id_to_idx[line.to_bus_id] for line in lines]
    x = [line.X for line in lines]
    
    # 构造B矩阵
    B = np.zeros((nb, nb))
    for i in range(nl):
        B[from_bus[i], to_bus[i]] -= 1/x[i]
        B[to_bus[i], from_bus[i]] -= 1/x[i]
        B[from_bus[i], from_bus[i]] += 1/x[i]
        B[to_bus[i], to_bus[i]] += 1/x[i]
    
    # 处理参考节点
    ref_idx = bus_id_to_idx[ref_bus_id]
    B_red = np.delete(np.delete(B, ref_idx, axis=0), ref_idx, axis=1)
    B_inv = np.linalg.inv(B_red)
    
    # 计算PTDF矩阵
    PTDF = np.zeros((nl, nb))
    for k in range(nl):
        e = np.zeros(nb)
        e[from_bus[k]] = 1
        e[to_bus[k]] = -1
        e_red = np.delete(e, ref_idx)
        f = (1/x[k]) * e_red @ B_inv
        PTDF[k, :] = np.insert(f, ref_idx, 0)
    
    # print(f"PTDF矩阵形状: {PTDF.shape}")

    # # 可视化B矩阵与PTDF矩阵
    # plt.figure(figsize=(10, 8))
    # plt.subplot(1, 2, 1)
    # plt.spy(B, markersize=1)
    # plt.title('B Matrix')
    # plt.subplot(1, 2, 2)
    # plt.spy(PTDF, markersize=1)
    # plt.title('PTDF Matrix')
    # plt.show()

    return PTDF



def calculate_PdPg(data, T, days, hours, load_data_mat, vre_data_mat):
    if T != len(days)*len(hours):
        raise ValueError("utils2.calculate_PdPg: T should be equal to the total number of hours.")
    data = dataloader.load_case_data()
    Pd = [[0 for _ in range(T)] for _ in range(len(data['buses']))]
    Pgmax = [[0 for _ in range(T)] for _ in range(len(data['gens']))]
    Pgmin = [[0 for _ in range(T)] for _ in range(len(data['gens']))]
    t = 0
    for day in days:
        for hour in hours:
            for i,bus in enumerate(data['buses']):
                bus.calculate_load(day,hour,load_data_mat)
                Pd[i][t] = bus.load
            for j,gen in enumerate(data['gens']):
                gen.calculate_upper(day,hour,vre_data_mat)
                Pgmin[j][t] = gen.min_output_rate * gen.capacity
                Pgmax[j][t] = gen.upper
            t += 1
    # 将Pd, Pgmin, Pgmax都转化为numpy array
    Pd = np.array(Pd)
    Pgmin = np.array(Pgmin)
    Pgmax = np.array(Pgmax)
            
    return Pd, Pgmin, Pgmax