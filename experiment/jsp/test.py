import pandas as pd
import numpy as np
import json

with open("resurt_all.json", "r") as f:
    my_resurt = json.load(f)
data_ur = my_resurt["ur"]
data_tards = my_resurt["tards"]


def process_tard(ur):
    data_from = ["EDD", "MS", "SRO", "CR", "noname_2", "RL"]
    df_dict = pd.read_excel(
        f"result_{ur}.xlsx", sheet_name=["sum", "maximum", "tardy rate"]
    )
    df_sum = df_dict["sum"]
    df_maximum = df_dict["maximum"]
    df_tard_rate = df_dict["tardy rate"]
    ur_data = data_tards[str(ur)]
    print(ur_data["RL"])
    print(ur_data["MS"])
    print(ur_data["SRO"])
    mydata = np.array([ur_data[f"{x}"] for x in data_from])
    print(mydata[-1, :, :])
    my_maximum = np.max(mydata, axis=-1)
    my_sum = np.sum(mydata, axis=-1)
    # 统计最后一维的非零元素的数量
    non_zero_counts = np.count_nonzero(mydata, axis=-1)
    # 计算非零概率：非零数量 / 总元素数量
    my_tard_rate = non_zero_counts / mydata.shape[-1]
    for i, data_f in enumerate(data_from):
        df_sum[f"{data_f}"] = my_sum[i, :]
        df_maximum[f"{data_f}"] = my_maximum[i, :]
        df_tard_rate[f"{data_f}"] = my_tard_rate[i, :]

    with pd.ExcelWriter(f"my_tardy_result{ur}.xlsx") as writer:
        df_sum.to_excel(writer, sheet_name="sum", index=False)
        df_maximum.to_excel(writer, sheet_name="maximum", index=False)
        df_tard_rate.to_excel(writer, sheet_name="tardy rate", index=False)


process_tard(70)
