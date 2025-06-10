import random
import numpy as np
import pickle


random.seed(0)
np.random.seed(0)


def Instance_Generator(M_num, E_ave, New_insert):
    """
    生成初始作业和新插入作业的加工信息
    """
    Initial_Job_num=5
    job_info_list = []
    Op_num = [random.randint(1, 20) for _ in range(New_insert + Initial_Job_num)]

    A1 = [0 for _ in range(Initial_Job_num)]
    A = np.random.exponential(E_ave, size=New_insert)
    A = [int(A[i]) for i in range(len(A))]
    A1.extend(A)

    for i in range(Initial_Job_num + New_insert):
        Job_i = []
        for j in range(Op_num[i]):
            k = random.randint(1, 4)
            T = list(range(M_num))
            random.shuffle(T)
            T = T[0 : k + 1]

            proc = {}
            process_time = random.randint(1, 20)
            for M_i in T:
                proc[M_i] = process_time
            Job_i.append(proc)

        job_info_list.append(
            {
                "id": i,
                "process_num": Op_num[i],
                "process_list": Job_i,
                "Insert_time": A1[i],
            }
        )

    return job_info_list, A1


def generate_and_save_job_data(machine_num, E_ave, job_num, num_instances=10):
    all_data = []
    for _ in range(num_instances):
        job_info_list, arrive_time = Instance_Generator(machine_num, E_ave, job_num)
        instance_data = {
            "machine_num": machine_num,
            "E_ave": E_ave,
            "job_num": job_num,
            "arrival_times": arrive_time,
            "job_info_list": job_info_list,
        }
        all_data.append(instance_data)

    # 保存数据到文件
    with open(f"HFSD/8_12_16/{machine_num}_{E_ave}_{job_num}.pkl", "wb") as f:
        pickle.dump(all_data, f)

def add_new_machines_same_time(job_info_list, new_mch_start_id, new_mch_num, prob=0.5):
    for job in job_info_list:
        for op in job["process_list"]:
            if len(op) == 0:
                continue
            existing_times = list(op.values())
            for i in range(new_mch_num):
                mch_id = new_mch_start_id + i
                if mch_id not in op and random.random() < prob:
                    sampled_time = random.choice(existing_times)
                    op[mch_id] = sampled_time
    return job_info_list

def add_machines_to_saved_data(old_path, new_path, new_mch_start_id, new_mch_num, prob=0.5):
    with open(old_path, "rb") as f:
        data = pickle.load(f)
    for instance in data:
        instance["job_info_list"] = add_new_machines_same_time(
            instance["job_info_list"],
            new_mch_start_id,
            new_mch_num,
            prob
        )
        # 更新机器总数
        instance["machine_num"] += new_mch_num

    # 保存到新文件
    with open(new_path, "wb") as f:
        pickle.dump(data, f)

    print(f"已添加新机器并保存至 {new_path}")



M_num = [4,8]
job_nums = [20, 30, 40]
E_aves = [50, 100, 200]
origin_m = 8
# #生成新机器
for job_num in job_nums:
    for E_ave in E_aves:
        generate_and_save_job_data(origin_m, E_ave, job_num)
        print(f"数据已保存：HFDS/8_12_16/{origin_m}_{E_ave}_{job_num}.pkl")

# # # 增加机器
for m in M_num:
    for job_num in job_nums:
        for E_ave in E_aves:
            # old_path = f"HFSD/job/{origin_m}_{E_ave}_{job_num}.pkl"
            # new_path = f"HFSD/job/{origin_m+m}_{E_ave}_{job_num}.pkl"
            old_path = f"HFSD/8_12_16/{origin_m}_{E_ave}_{job_num}.pkl"
            new_path = f"HFSD/8_12_16/{origin_m+m}_{E_ave}_{job_num}.pkl"
            add_machines_to_saved_data(
                old_path,
                new_path,
                origin_m,
                m,
                prob=0.5  # 所有工序都尽可能添加新机器
            )