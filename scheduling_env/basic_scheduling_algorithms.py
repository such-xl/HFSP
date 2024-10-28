import random
def sjf(jobs: list, machine_id: int) -> int:
    '''短作业优先'''
    action = 0
    if len(jobs) == 0:
        return 29
    min_t = jobs[action].get_t_process(machine_id)
    for i in range(1,len(jobs)):
        c_t = jobs[i].get_t_process(machine_id)
        if c_t < min_t:
            min_t = c_t
            action = i
    return action    
def random_action(jobs:list) -> int:
    return random.randint(0,len(jobs))



