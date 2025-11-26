"""
读取mpc, 计算PTDF矩阵
"""

import pyomo.environ as pyo
from pypower.api import case30
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sced

def build_PTDF(mpc):
    """生成 DC 功率传输分布因子矩阵 PTDF"""
    branch = mpc['branch']
    bus = mpc['bus']
    nb = bus.shape[0]
    nl = branch.shape[0]

    from_bus = branch[:, 0].astype(int) - 1
    to_bus = branch[:, 1].astype(int) - 1
    x = branch[:, 3]

    # 构造B矩阵
    B = np.zeros((nb, nb))
    for i in range(nl):
        B[from_bus[i], to_bus[i]] -= 1/x[i]
        B[to_bus[i], from_bus[i]] -= 1/x[i]
        B[from_bus[i], from_bus[i]] += 1/x[i]
        B[to_bus[i], to_bus[i]] += 1/x[i]
    
    print(B)

    ref = 0
    B_red = np.delete(np.delete(B, ref, axis=0), ref, axis=1)
    B_inv = np.linalg.inv(B_red)

    PTDF = np.zeros((nl, nb))
    for k in range(nl):
        e = np.zeros(nb)
        e[from_bus[k]] = 1
        e[to_bus[k]] = -1
        e = np.delete(e, ref)
        f = (1/x[k]) * e @ B_inv
        PTDF[k, :] = np.insert(f, ref, 0)

    print('type(PTDF):',type(PTDF))
    print('PTDF.shape:',PTDF.shape)
    # print(PTDF)

    # 可视化B矩阵与PTDF矩阵
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.spy(B, markersize=1)
    plt.title('B Matrix')
    plt.subplot(1, 2, 2)
    plt.spy(PTDF, markersize=1)
    plt.title('PTDF Matrix')
    plt.show()

    return PTDF
    




def visualize_sced(mpc, Pg_vals, LMPs, title="IEEE 30-bus SCED LMP Map"):
    """
    绘制节点LMP、电价分布及阻塞线路
    """
    baseMVA = mpc['baseMVA']
    bus = mpc['bus']
    branch = mpc['branch']
    gen = mpc['gen']

    nb = bus.shape[0]
    ng = gen.shape[0]
    nl = branch.shape[0]

    from_bus = branch[:, 0].astype(int) - 1
    to_bus = branch[:, 1].astype(int) - 1
    gen_bus = gen[:, 0].astype(int) - 1

    Pd = bus[:, 2] / baseMVA
    Fmax = branch[:, 5] / baseMVA

    # 构造 PTDF
    PTDF = build_PTDF(mpc)

    # 节点注入功率 = 发电 - 负荷
    Pinj = np.zeros(nb)
    for g in range(ng):
        Pinj[gen_bus[g]] += Pg_vals[g]
    Pinj -= Pd

    # 计算潮流 (p.u.)
    flows = PTDF @ Pinj

    # 阻塞线路判断（接近限额）
    congested = np.abs(flows) >= 0.99 * Fmax

    # 构建网络拓扑
    G = nx.Graph()
    G.add_nodes_from(range(nb))
    edges = [(int(f), int(t)) for f, t in zip(from_bus, to_bus)]
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42, k=0.3)

    # ==================== 绘图 ====================
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(title, fontsize=14)

    node_colors = np.array(LMPs, dtype=float)
    cmap = plt.cm.coolwarm
    vmin, vmax = np.nanmin(node_colors), np.nanmax(node_colors)

    # --- 绘制线路 ---
    for l, (f, t) in enumerate(edges):
        if congested[l]:
            nx.draw_networkx_edges(G, pos, edgelist=[(f, t)],
                                   width=3.5, edge_color='red', alpha=0.9, ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(f, t)],
                                   width=1.2, edge_color='gray', alpha=0.5, ax=ax)

    # --- 绘制节点 ---
    nx.draw_networkx_nodes(
        G, pos, node_size=500, node_color=node_colors,
        cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.95,
        edgecolors='black', ax=ax
    )

    # --- 绘制节点标签 ---
    labels = {i: f"{i+1}\n{node_colors[i]:.2f}" for i in range(nb)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color='black', ax=ax)

    # --- 发电机节点标注（优化） ---
    for g in range(ng):
        i = gen_bus[g]
        Pg_MW = Pg_vals[g] * baseMVA
        PMAX = gen[g, 8]  # 机组最大出力 (MW)
        PMIN = gen[g, 9]  # 机组最小出力 (MW)

        x, y = pos[i]

        # 判断机组状态
        if Pg_MW >= 0.98 * PMAX:
            color = 'red'     # 满载
        elif Pg_MW <= 0.02 * PMAX:
            color = 'gray'    # 停机
        else:
            color = 'orange'  # 正常运行

        ax.scatter(x, y, s=800, facecolors='none', edgecolors=color, linewidths=2.5)

        # 在节点上方标注机组出力
        ax.text(x, y + 0.07, f"G{g+1}\n{Pg_MW:.1f} MW",
                ha='center', va='bottom', fontsize=8, color='darkblue', fontweight='bold')


    # --- 颜色条 ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("LMP (system units)", fontsize=10)

    # --- 图例 ---
    import matplotlib.lines as mlines
    line_normal = mlines.Line2D([], [], color='gray', linewidth=1.5, label='Normal Line')
    line_cong = mlines.Line2D([], [], color='red', linewidth=3, label='Congested Line')
    ax.legend(handles=[line_normal, line_cong], loc='upper left')

    ax.axis("off")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    mpc = case30()
    PTDF = build_PTDF(mpc)
    print(PTDF)
    print(PTDF.shape)