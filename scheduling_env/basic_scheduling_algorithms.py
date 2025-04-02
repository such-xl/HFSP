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


def FIFO(jobs) -> int:
    insert_time = INF
    aim_job = None
    for job in jobs:
        aim_job, insert_time = (
            (job, job.insert_time)
            if job.insert_time < insert_time
            else (aim_job, insert_time)
        )
    return aim_job
def EDD(jobs):
    """
        (Earliest Due Date) 优先处理截止日期最早的作业
    """
    min_EDD = INF
    aim_job = None
    for job in jobs:
        EDD = job.due_time
        aim_job, min_EDD = (job, EDD) if EDD < min_EDD else (aim_job, min_EDD)
    return aim_job
def MS(jobs,time_step):
    """
    选择剩余时间最少的作业
    剩余时间 = 截止日期 - 当前时间 - 剩余处理时间
    """
    
    min_ms = INF
    aim_job = None
    for job in jobs:
        ms = (job.due_time-time_step)-job.get_remaining_avg_time()
        aim_job, min_ms = (job, ms) if ms < min_ms else (aim_job, min_ms)
    return aim_job

def SRO(jobs,time_step):
    """
        每剩余工序最小松弛时间规则 (Slack per Remaining Operation, S/RO)
        Slack_j = (作业j的交付日期 d_j - 当前时间 t) - 作业j的剩余总处理时间 p_rem_j
    """
    min_sro = INF
    aim_job = None
    for job in jobs:
        sro = (job.due_time-time_step-job.get_remaining_avg_time()) / (job.process_num-job.progress+1)
        aim_job,min_sro =(job,sro) if sro < min_sro else (aim_job, min_sro)
    return aim_job
def CR(jobs,time_step):
    """
        优先处理关键比率最小的作业
    """
    min_cr = INF
    aim_job = None
    for job in jobs:
        cr = (job.due_time-time_step)/job.get_remaining_avg_time()
        aim_job, min_cr = (job, cr) if cr < min_cr else (aim_job, min_cr)
    return aim_job
EDD,MS,SRO,CR



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
