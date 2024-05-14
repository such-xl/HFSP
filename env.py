import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from scheduling_env.job import JobList
from scheduling_env.machine import MachineList
import time
class ScheduleEnv(gym.Env):
    def __init__(self):
        super(ScheduleEnv, self).__init__()
        self._action_spaces = [spaces.Box(0,1)]  # action_space
        self._observation_spaces = None  #
        self._jobList = JobList()
        self._machineList = any


    def step(self,machine,act):
        # 根据智能体动作更新环境状态并返回每个智能体的观测、奖励、done、info
        obs = []
        rewards = []
        dones = []
        infos = []
        if act == 0:
            print(f'机器{machine.ID} 选择空闲')
            return False
        job = self._jobList.jobList[act-1]
        # agent 装载任务
        if machine.isIdle():
            machine.execute(job.ID,job.status,job.getTProcess(machine.ID))
            job.load(machine.ID)
            print(f'加载任务{job.ID}到机器{machine.ID}')
        # agent 执行一个单位时间
        else:
            machine.processingOneStep()
            job.processingOneStep()
            print('')
        return self._jobList.isDone()
    def __getJobs(self, path):

        self._jobList.decodeRawJobFlie(path)
        
    def reset(self):
        path = os.path.dirname(os.path.abspath(__file__))
        path += '/schedulingEnv/data/Job_Data/Barnes/Text/mt10c1.fjs'
        self.__getJobs(path)
        self._machineList = MachineList(self.jobList.machineNum)
    def render(self, mode='human', close=False):
        # 可视化环境状态（可选）
        pass

    def close(self):
        # 清理资源
        pass
    @property
    def jobList(self):
        return self._jobList
    @property
    def machineList(self):
        return self._machineList
# 使用环境
env = ScheduleEnv()
env.reset()
print(env._action_spaces)
'''
# 环境交互实例
jobList = env.jobList
machineList = env.machineList
flag = True
timestep = 0
while flag:
    print(f'时序T:{timestep}')
    done = False 
    for machine in machineList.machineList:
        # 当前agent空闲
        if machine.isIdle():
            #获取动作
            actions,processingTime = jobList.getMachineAction(machine.ID)
            #采样动作
            act = machine.sampleActiom(actions)
            done = env.step(machine,act)
        # 当前agent工作中
        elif machine.isBusy():
            done = env.step(machine,act)
            # 如果执行该操作后该agent状态为空闲，该agent立即选择装置一个新任务
            print(jobList.jobList[1].status)
            print(machine.isIdle())
            if machine.isIdle() and not done:
                #获取动作
                actions,processingTime = jobList.getMachineAction(machine.ID)
                #采样动作
                act = machine.sampleActiom(actions)
                done = env.step(machine,act)
        # 当前agent故障
        else:
            pass
        for j in jobList.jobList:
            print(f'ID{j.ID} --- {j.status} --- {j.isAccomplished()}')
        if done:
            flag = False
            break
    timestep += 1

'''
