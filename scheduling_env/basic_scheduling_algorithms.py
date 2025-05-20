import random
import numpy as np

INF = 1e9


def SPT(jobs,time_step, machine_id: int) -> int:
    """当前工序最短处理时间优先"""
    sorted_jobs = sorted(jobs, key=lambda job: job.get_t_process(machine_id))
    return sorted_jobs

def LPT(jobs,time_stpe, machine_id: int) -> int:
    """当前工序最长处理时间优先"""
    max_t = -INF
    aim_job = None
    sorted_jobs = sorted(jobs, key=lambda job: job.get_t_process(machine_id),reverse=True)
    return sorted_jobs


def SRPT(jobs,time_step,macine_id) -> int:
    """平均剩余最短处理时间优先"""
    sorted_jobs = sorted(jobs, key=lambda job: job.get_remaining_avg_time())
    return sorted_jobs


def LRPT(jobs, time_step,machine_id: int) -> int:
    """平均剩余最长处理时间优先"""
    sorted_jobs = sorted(jobs,key=lambda job: job.get_remaining_avg_time(),reverse=True)
    return sorted_jobs


def FIFO(jobs,time_step,machine_id) -> int:
    sorted_jobs = sorted(jobs, key=lambda job: job.insert_time)
    return sorted_jobs


def EDD(jobs,time_step,machine_id):
    """
    (Earliest Due Date) 优先处理截止日期最早的作业
    """
    sorted_jobs = sorted(jobs, key=lambda job: job.due_time)
    return sorted_jobs


def MS(jobs, time_step,machine_id):
    """
    选择剩余时间最少的作业
    剩余时间 = 截止日期 - 当前时间 - 剩余处理时间
    """
    sorted_jobs = sorted(
        jobs, key=lambda job: (job.due_time - time_step) - job.get_remaining_avg_time()
    )
    return sorted_jobs


def SRO(jobs, time_step,machine_id):
    """
    每剩余工序最小松弛时间规则 (Slack per Remaining Operation, S/RO)
    Slack_j = (作业j的交付日期 d_j - 当前时间 t) - 作业j的剩余总处理时间 p_rem_j
    """
    sorted_jobs = sorted(
        jobs,
        key=lambda job: (job.due_time - time_step - job.get_remaining_avg_time())
        / (job.process_num - job.progress + 1),
    )
    return sorted_jobs


def CR(jobs, time_step,machine_id):
    """
    优先处理关键比率最小的作业
    """
    sorted_jobs = sorted(
        jobs, key=lambda job: (job.due_time - time_step) / job.get_remaining_avg_time()
    )
    return sorted_jobs


def ATC(jobs, time_step,machine_id, k=2.0):
    """
    Apparent Tardiness Cost (ATC) Rule
    k: 控制参数，推荐范围2.0~4.0
    """
    avg_p = sum(job.get_remaining_avg_time() for job in jobs) / len(jobs)

    def atc_priority(job):
        p_j = job.get_remaining_avg_time()
        slack = max(job.due_time - time_step - p_j, 0)
        exp_term = np.exp(-slack / (k * avg_p)) if avg_p > 0 else 1
        return -(1 / p_j) * exp_term  # 取负用于升序排序

    sorted_jobs = sorted(jobs, key=atc_priority)
    return sorted_jobs


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

PDR_RULES = {
    0:EDD,
    1:CR,
    2:SRO,
    3:ATC,
    4:MS,
    5:FIFO,
    6:SPT,
    7:LPT,
    8:LRPT,
    9:SRPT
}