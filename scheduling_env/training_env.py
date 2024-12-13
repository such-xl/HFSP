'''
    多智能体作业调度训练环境
    1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
    2: 判断所有job是否完成 over if done else repeat 1
'''

import math
import random
from .job import Job,JobList,JobStatus
from .machine import Machine,MachineList,MachineStatus

class TrainingEnv():
    # 初始化环境
    def __init__(self,action_dim,reward_type,max_machine_num,max_job_num) -> None:
        self._action_space = (0,action_dim-1)
        self._action_dim = action_dim 
        self._machine_num = 0        #总agent数
        self._job_num  = 0         #总作业数
        self._max_machine_num = max_machine_num
        self._max_job_num = max_job_num
        self._draw_data = None        #画图信息
        self._time_step = 0
        self._reward_type = reward_type
        self._job_list:list[Job] = None 
        self._machines = None
        self._jobs:JobList = JobList()
        self._current_machine = None
    def get_jobs_from_file(self, jobs_path:str):
        self._machine_num = self._jobs.fetch_jobs_from_file(jobs_path)
        self._job_num = self._jobs.length
        self._machines = MachineList(self._machine_num)
        machine:Machine = self._machines.head
        self._machine_list = []
        while machine:
            self._machine_list.append(machine)
            machine = machine.next

    def is_decision_machine(self,agent_id):
        """是否是需要做出决策的agent，当agent只能选择空闲时，则不需要做出决策"""
        job :Job = self._jobs.head
        while job:
            if job.is_wating_for_machine() and job.match_machine(agent_id):
                return True
            job = job.next
        return False
    def get_decision_machines(self):
        """获取需要做出决策的agent"""
        decision_machines = []
        machine:Machine = self._machines.head
        while machine:
            if machine.is_idle() and not machine.step_decision_made(self._time_step) and self.is_decision_machine(machine.id):
                decision_machines.append(machine)
            machine = machine.next
        return decision_machines
    def execute_action(self,machine,action):
        """
            执行动作
        """
        if action == self._action_dim - 1:
            return None
        machine.load_job(self._job_list[action],self._time_step)
        return self.get_state()

    def run(self):
        """
            所有忙碌agent和job更新一个time_step,使得必产生空闲机器
            在内添加随机时间
        """
        # 更新one timestep时序
        min_run_timestep = 1
        machine:Machine = self._machines.head
        while machine:
            if machine.is_idle():
                machine = machine.next
                continue
            job:Job = machine.job
            machine.run(min_run_timestep)
            machine = machine.next
        self._time_step += min_run_timestep
        done = True
        job:Job = self._jobs.head
        while job:
            if not job.is_completed():
                done =  False
                break
            job = job.next
        truncated = False if self._time_step < 400 else True
        while not done and not truncated and not self.is_any_machine_need_to_decision(): # 没有结束且没有空闲机器，继续
            done,truncated = self.run()
        return done,truncated
    def reset(self,jobs_path:str):
        """
            重置环境
            reutrn:
                state: 当前job环境状态
                machine_action: 决策机器的状态
                action_mask: 机器的动作mask
        """
        self._jobs = JobList()
        self._machines = None
        self.get_jobs_from_file(jobs_path) #从文件中获取job和machine信息
        self._job_list = []

        # job_list
        job:Job = self._jobs.head
        while job:
            self._job_list.append(job)
            job = job.next
        decision_machines = self.get_decision_machines()
        x = random.randint(0,len(decision_machines)-1)
        self._current_machine = decision_machines[x]
        self._time_step = 0
        state,action_mask = self.get_state()
        return state,action_mask
    def get_state(self):
        job:Job = self._jobs.head
        state = []
        action_mask = []
        while job:
            state.append(job.get_state())
            action_mask.append(True if job.is_wating_for_machine() and job.match_machine(self._current_machine.id) else False)
            job = job.next
        action_mask.append(True)
        machine_state = self._current_machine.get_state()
        machine_state.extend([0 for _ in range(len(state[0])-len(machine_state))])
        state.append(machine_state)
        return state,action_mask
    
    def step(self,action,scale_factor):

        if action == self._action_dim - 1: # 采样空闲动作，不对环境作出改变
            self._current_machine.update_decision_time(self._time_step)
        else:
            self._current_machine.load_job(self._job_list[action],self._time_step)
        done,truncated = False,False
        if not self.is_any_machine_need_to_decision(): # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done,truncated = self.run()
        # 要么结束，要么有机器需要采样动作
        decision_machines = self.get_decision_machines()
        if len(decision_machines):
            x = random.randint(0,len(decision_machines)-1)
            self._current_machine = decision_machines[x]
        state,action_mask = self.get_state()
        # reward = -0.1 if action==self._action_dim-1 else 0
        if truncated:
            reward = -1000
        else:
            reward = -0.01 if action==self._action_dim-1 else 0
            reward = reward if not done else 1000/self._time_step
        return state,action_mask,reward,done,truncated

    def is_any_machine_need_to_decision(self):
        machine:Machine = self._machines.head
        while machine:
            if machine.is_idle() and not machine.step_decision_made(self._time_step) and self.is_decision_machine(machine.id):
                return True
            machine = machine.next
        return False
    def reward_func_0(self,scale_factor,done):
        """
          每步返回-1的奖励
        """ 
        return 1 if done else -0.01
    def reward_func_1(self):
        """
            返回in_progress_jobs的相对延迟的率
        """
        count = 0
        delay_rate = 0
        for job in self._job_list:
            if job.is_completed():
                continue
            count += 1
            delay_rate = max(delay_rate,job.current_progress_remaining_time())
        return -delay_rate * 0.01 if count else (1/self._time_step)*300
    def reward_func_2(self,scale_factor,actions):
        """
            机器平均空闲率
        """
        idle_machine:Machine = self._idle_machines.head
        busy_machine:Machine = self._busy_machines.head

        idle_rate = 0
        while idle_machine:
            idle_rate += idle_machine.get_idle_time(self.time_step)
            idle_machine = idle_machine.next
        while busy_machine:
            idle_rate += busy_machine.get_idle_time(self.time_step)
            busy_machine = busy_machine.next
        machine_counts = self._idle_machines.length + self._busy_machines.length
        r1 = -idle_rate/machine_counts*scale_factor if machine_counts else 0
        uncompleted_job:Job = self._uncompleted_jobs.head
        latest_finnish = 0
        while uncompleted_job:
            latest_finnish = max(latest_finnish,uncompleted_job.current_progress_remaining_time())
            uncompleted_job = uncompleted_job.next

        r2 = -math.log(latest_finnish*scale_factor+1) if latest_finnish else 0
        r3 = -actions.count(self._action_dim-1)/len(actions)
        return (r1*0.2+r2*0.7+r3*0.1) + 0.5
    @property
    def action_space(self):
        return self._action_space
    @property
    def jobs_num(self):
        return self._jobs_num
 
    @property
    def draw_data(self):
        return self._draw_data
    @property
    def time_step(self):
        return self._time_step