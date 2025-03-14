import random
import numpy as np

INF = 1e9


def SPT(jobs, machine_id: int) -> int:
    """当前工序最短处理时间优先"""
    min_t = INF
    aim_job = None
    for job in jobs:
        c_t = job.get_t_process(machine_id)
        aim_job, min_t = (job, c_t) if c_t < min_t else (aim_job, min_t)
    return aim_job


def LPT(jobs, machine_id: int) -> int:
    """当前工序最长处理时间优先"""
    max_t = -INF
    aim_job = None
    for job in jobs:
        c_t = job.get_t_process(machine_id)
        aim_job, max_t = (job, c_t) if c_t > max_t else (aim_job, max_t)
    return aim_job


def SRPT(jobs, machine_id: int) -> int:
    """平均剩余最短处理时间优先"""
    min_t = INF
    aim_job = None
    for job in jobs:
        c_t = job.get_remaining_avg_time()
        aim_job, min_t = (job, c_t) if c_t < min_t else (aim_job, min_t)
    return aim_job


def LRPT(jobs, machine_id: int) -> int:
    """平均剩余最长处理时间优先"""
    max_t = -INF
    aim_job = None
    for job in jobs:
        c_t = job.get_remaining_avg_time()
        aim_job, max_t = (job, c_t) if c_t > max_t else (aim_job, max_t)
    return aim_job


def FIFO(jobs, machine_id: int) -> int:
    insert_time = INF
    aim_job = None
    for job in jobs:
        aim_job, insert_time = (
            (job, job.insert_time)
            if job.insert_time < insert_time
            else (aim_job, insert_time)
        )
    return aim_job


def noname(jobs, machine, machines_UR: list[float]):
    sorted_jobs = sorted(jobs, key=lambda job: job.get_t_process(machine.id))
    avg_Ur = np.mean(machines_UR)
    U = machines_UR[machine.id - 1]
    if U >= avg_Ur:
        return sorted_jobs[0]
    return sorted_jobs[-1]


def noname_2(jobs, machine, machines_UR):
    sorted_Jobs = sorted(jobs, key=lambda job: job.get_t_process(machine.id))
    sorted_UR = machines_UR.copy()
    sorted_UR.sort(reverse=True)
    current_machine_ur = machines_UR[machine.id - 1]
    rank = sorted_UR.index(current_machine_ur)
    percentile = rank / len(sorted_UR)
    job_index = int(percentile * len(sorted_Jobs))
    return sorted_Jobs[job_index]
