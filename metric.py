import numpy as np

def get_ps(popfun):
    num = popfun.shape[0]
    ps = []
    for i in range(num):
        flag = True
        for j in range(num):
            if all(popfun[j] <= popfun[i]) and any(popfun[j] < popfun[i]):
                flag = False
                break
        if flag:
            ps.append(i)
    return popfun[ps]

def IGD(pareto_front, solutions):
    return np.mean([np.min(np.linalg.norm(solutions - pf, axis=1)) for pf in pareto_front])

def GD(solutions, pareto_front):
    return np.mean([np.min(np.linalg.norm(pareto_front - s, axis=1)) for s in solutions])

def spread(solutions, pareto_front):
    solutions = solutions[np.lexsort((solutions[:, 1], solutions[:, 0]))]  # 按照第一列排序
    d = np.mean([np.linalg.norm(solutions[i] - solutions[i - 1]) for i in range(1, len(solutions))])
    d_f = np.linalg.norm(solutions[0] - pareto_front[np.argmin(pareto_front[:, 0])])
    d_l = np.linalg.norm(solutions[-1] - pareto_front[np.argmax(pareto_front[:, 0])])
    return (d_f + d_l + sum(abs(np.linalg.norm(solutions[i] - solutions[i - 1]) - d) for i in range(1, len(solutions)))) / (d_f + d_l + (len(solutions) - 1) * d)




def evaluate_pareto(data):
            objs = np.array(data)
            v_max, v_min = objs.max(axis=0), objs.min(axis=0)
            norm_objs = (objs - v_min) / (v_max - v_min + 1e-6)
            pareto = get_ps(norm_objs)
            gd = GD(norm_objs, pareto) # 计算算法的GD值
            gd='%.2e'% gd
            igd = IGD(pareto, norm_objs) # 计算算法的IGD值
            igd='%.2e'% igd
            deta=spread(norm_objs, pareto) # 计算算法的spread值
            deta='%.2e'% deta    
            return gd, igd, deta

# import json

# # 读取reward数据
# with open("HFSD//record//record.json", "r") as f:
#     record = json.load(f)

# reward = record['reward']
# print(evaluate_pareto(reward))