import random
import numpy as np
from scheduling_env.job import Job
from scheduling_env.machine import Machine

INF = 1e9


def FIFO(jobs: list[Job], machine_id: int) -> int:
    min_t = INF
    action = -1
    for i, job in enumerate(jobs):
        if job.is_wating_for_machine() and job.match_machine(machine_id):
            c_t = job.insert_time
        else:
            continue
        action, min_t = (i, c_t) if c_t < min_t else (action, min_t)
    return action


def CR(jobs: list[Job], machine_id: int, current_time: int) -> int:
    """当前工序最短剩余时间优先"""
    min_t = INF
    action = -1
    for i, job in enumerate(jobs):
        if job.is_wating_for_machine() and job.match_machine(machine_id):
            c_t = (job.due_time - current_time) / job.get_remaining_avg_time()
        else:
            continue
        action, min_t = (i, c_t) if c_t < min_t else (action, min_t)
    return action


# 最早截至日期优先
def EDD(jobs: list[Job], machine_id: int) -> int:
    """当前工序最早截止时间优先"""
    min_t = INF
    action = -1
    for i, job in enumerate(jobs):
        if job.is_wating_for_machine() and job.match_machine(machine_id):
            c_t = job.due_time
        else:
            continue
        action, min_t = (i, c_t) if c_t < min_t else (action, min_t)
    return action


def SPT(jobs: list[Job], machine_id: int) -> int:
    """当前工序最短处理时间优先"""
    min_t = INF
    action = -1
    for i, job in enumerate(jobs):
        if job.is_wating_for_machine() and job.match_machine(machine_id):
            c_t = job.get_t_process(machine_id)
        else:
            continue
        action, min_t = (i, c_t) if c_t < min_t else (action, min_t)
    return action


def LPT(jobs: list[Job], machine_id: int) -> int:
    """当前工序最长处理时间优先"""
    max_t = -INF
    action = -1
    for i, job in enumerate(jobs):
        if job.is_wating_for_machine() and job.match_machine(machine_id):
            c_t = job.get_t_process(machine_id)
        else:
            continue
        action, max_t = (i, c_t) if c_t > max_t else (action, max_t)
    return action


# 作业剩余最短处理时间优先
def SRPT(jobs: list[Job], machine_id: int) -> int:
    min_t = INF
    action = -1
    for i, job in enumerate(jobs):
        if job.is_wating_for_machine() and job.match_machine(machine_id):
            c_t = job.get_remaining_avg_time()
        else:
            continue
        action, min_t = (i, c_t) if c_t < min_t else (action, min_t)
    # if action < 0:
    #     raise ValueError('Action is not valid(Action < 0)')
    return action


def LRPT(jobs: list[Job], machine_id: int) -> int:
    """平均剩余最长处理时间优先"""
    max_t = -INF
    action = -1
    for i, job in enumerate(jobs):
        if job.is_wating_for_machine() and job.match_machine(machine_id):
            c_t = job.get_remaining_avg_time()
        else:
            continue
        action, max_t = (i, c_t) if c_t > max_t else (action, max_t)
    # if action < 0:
    #     raise ValueError('Action is not valid(Action < 0)')
    return action


# 随机
def Random(jobs: list, machine_id: int) -> int:
    """随机选择一个作业"""
    job_index = []
    action = -1
    for i, job in enumerate(jobs):
        if job.is_wating_for_machine() and job.match_machine(machine_id):
            job_index.append(i)
    action = random.choice(job_index) if len(job_index) > 0 else -1
    return action
