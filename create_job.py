import json
import numpy as np

np.random.seed(42)


def create_jsp_job():
    """
    生成FJS数据集
    """
    job_list = []
    for i in range(10):
        data_dict = {
            "type": i + 1,
            "process_num": np.random.randint(5, 13),
            "process_list": [],
        }
        for j in range(data_dict["process_num"]):
            data_dict["process_list"].append(
                {np.random.randint(1, 11): np.random.randint(5, 10)}
            )
        job_list.append(data_dict)
    with open("jsp.json", "w") as f:
        json.dump(job_list, f)


def create_fjsp_diff_job():
    """
    生成FJSP_DIFF数据集
    """
    job_list = []
    for i in range(20):
        data_dict = {
            "type": i + 1,
            "process_num": np.random.randint(5, 13),
            "process_list": [],
        }
        for j in range(data_dict["process_num"]):
            M = np.random.choice(
                np.arange(1, 11), size=np.random.randint(2, 11), replace=False
            )
            P = {}
            for m in M:
                P[int(m)] = np.random.randint(5, 10)
            data_dict["process_list"].append(P)
        job_list.append(data_dict)
    with open("fjsp_diff.json", "w") as f:
        json.dump(job_list, f)


def create_fjsp_same_job():
    """
    生成FJSP_DIFF数据集
    """
    job_list = []
    for i in range(20):
        data_dict = {
            "type": i + 1,
            "process_num":10,
            "process_list": [],
        }
        for j in range(data_dict["process_num"]):
            M = np.random.choice(
                np.arange(1, 11), size=np.random.randint(1, 4), replace=False
            )
            T = np.random.randint(1, 20)
            P = {}
            for m in M:
                P[int(m)] = T
            data_dict["process_list"].append(P)
        job_list.append(data_dict)
    with open("fjsp_same.json", "w") as f:
        json.dump(job_list, f)

def create_cfjsp_diff_job():
    """
    生成FJSP_DIFF数据集
    """
    job_list = []
    for i in range(10):
        data_dict = {
            "type": i + 1,
            "process_num": np.random.randint(5, 13),
            "process_list": [],
        }
        for j in range(data_dict["process_num"]):
            P = {}
            for m in range(1, 11):
                P[m] = np.random.randint(5, 10)
            data_dict["process_list"].append(P)
        job_list.append(data_dict)
    with open("cfjsp_diff.json", "w") as f:
        json.dump(job_list, f)


def create_cfjsp_same_job():
    """
    生成FJSP_DIFF数据集
    """
    job_list = []
    for i in range(10):
        data_dict = {
            "type": i + 1,
            "process_num": np.random.randint(5, 13),
            "process_list": [],
        }
        for j in range(data_dict["process_num"]):
            P = {}
            T = np.random.randint(5, 10)
            for m in range(1, 11):
                P[m] = T
            data_dict["process_list"].append(P)
        job_list.append(data_dict)
    with open("cfjsp_same.json", "w") as f:
        json.dump(job_list, f)


def create_test_fjsp_same_job():
    """
    生成FJSP_DIFF数据集
    """
    job_list = []
    for i in range(20):
        data_dict = {
            "type": i + 1,
            "process_num": np.random.randint(10, 30),
            "process_list": [],
        }
        for j in range(data_dict["process_num"]):
            M = np.random.choice(
                np.arange(1, 20), size=np.random.randint(1, 20), replace=False
            )
            T = np.random.randint(5, 35)
            P = {}
            for m in M:
                P[int(m)] = T
            data_dict["process_list"].append(P)
        job_list.append(data_dict)
    with open("fjsp_same_test.json", "w") as f:
        json.dump(job_list, f)


if __name__ == "__main__":
    # create_jsp_job()
    # create_fjsp_diff_job()
    create_fjsp_same_job()
    # create_cfjsp_diff_job()
    # create_cfjsp_same_job()
    # create_test_fjsp_same_job()
