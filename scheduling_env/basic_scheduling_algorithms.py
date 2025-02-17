import random
from scheduling_env.job import Job
INF = 1e9
def SPT(jobs: list[Job], machine_id: int) -> int:
    '''当前工序最短处理时间优先'''
    min_t = INF
    action = -1
    for i,job  in enumerate(jobs):
        if not job.is_wating_for_machine() or not job.match_machine(machine_id):
            continue
        c_t = job.get_t_process(machine_id)
        action = i if c_t < min_t else action
    # if action < 0:
    #     raise ValueError('Action is not valid(Action < 0)')
    return action    
def LPT(jobs: list[Job], machine_id: int) -> int:
    '''当前工序最长处理时间优先'''
    max_t = -INF
    action = -1
    for i,job  in enumerate(jobs):
        if not job.is_wating_for_machine() or not job.match_machine(machine_id):
            continue
        c_t = job.get_t_process(machine_id)
        action = i if c_t > max_t else action
    # if action < 0:
    #     raise ValueError('Action is not valid(Action < 0)')
    return action

# 平均剩余最短处理时间优先
def SRPT(jobs: list[Job], machine_id: int) -> int:
    min_t = INF
    action = -1
    for i,job  in enumerate(jobs):
        if not job.is_wating_for_machine() or not job.match_machine(machine_id):
            continue
        c_t = job.get_remaining_avg_time()
        action = i if c_t < min_t else action
    # if action < 0:
    #     raise ValueError('Action is not valid(Action < 0)')
    return action

def LRPT(jobs: list[Job], machine_id: int) -> int:
    """平均剩余最长处理时间优先"""
    max_t = -INF
    action = -1
    for i,job  in enumerate(jobs):
        if not job.is_wating_for_machine() or not job.match_machine(machine_id):
            continue
        c_t = job.get_remaining_avg_time()
        action = i if c_t > max_t else action
    # if action < 0:
    #     raise ValueError('Action is not valid(Action < 0)')
    return action

def random_action(jobs:list) -> int:
    return random.randint(0,len(jobs))



