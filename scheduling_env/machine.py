from enum import Enum
from .utils import DoublyLinkList
from .job import Job
class MachineStatus(Enum):
    FAULT = 0
    IDLE = 1
    RUNNING = 2

from .utils import Node
class Machine(Node):
    def __init__(self,id:int,joblist:Job) -> None:
        '''
            status: 0:break, 1:idle, 2:working
        '''
        super().__init__(None)
        self._id = id
        self._status = MachineStatus.IDLE
        self._job = None
        self._job_list = joblist
        self._bin_code = self.get_bin_code()
        self._begin_idle_time = 0 # 开始等待时间
        self._end_idle_time = 0 # 结束等待时间
        self._idle_time = 0 # 空闲时间
        self._t_process = 0 #当前机器加工时间
    #获取所有机器的加工状态
    def get_state(self,num_machines):
        return [1 for _ in range(num_machines)] 
    
    def get_bin_code(self):
        binary_str = bin(self._id)[2:]
        binary_list = [int(digit) for digit in binary_str]
        return binary_list
    
    def get_state_encoding(self,lenth):  
        return [0 for i in range(lenth-len(self._bin_code))] + self._bin_code
    
    def show(self):
        print(f'machine {self._id} status:{self._status} job:{self._job.id} t_process:{self._t_process} t_processed:{self._t_processed}')
    
    def is_idle(self):
        return self._status == MachineStatus.IDLE
    
    def is_fault(self):
        return self._status == MachineStatus.FAULT
    
    def is_running(self):
        return self._status == MachineStatus.RUNNING
    
    # 装载job
    def load_job(self,job,time_step):
        """把job装置至machine"""
        if self._status != MachineStatus.IDLE:
            raise ValueError('machine is not idle')
        if self._job:
            raise ValueError('machine has job')
        self._end_time = time_step # 更新结束等待时间
        self._job = job
        job.load_to_machine(self,time_step)
        self._status = MachineStatus.RUNNING
    
    def unload_job(self):
        """卸载作业"""
        if self._status != MachineStatus.RUNNING:
            raise ValueError('machine is not running')
        if not self._job:
            raise ValueError('machine has no job')
        self._job = None
        self._status = MachineStatus.IDLE
    def run(self,min_run_timestep):
        """运行 'min_run_timestep' 时序，让环境产生空闲机器"""
        self._job.run(min_run_timestep)

        if not self._job.is_on_processing(): # 如果job不在运行，则卸载job

            self.unload_job()

    def update_end_idle_time(self,time_step):
        """更新结束等待时间"""
        self._end_idle_time = time_step
        
    def update_begin_idle_time(self,time_step):
        """更新开始等待时间"""
        self._begin_idle_time = time_step

    def get_idle_time(self,time_step):
        """获取空闲时间"""
        if self._status == MachineStatus.FAULT:
            raise ValueError('machine is fault')
        idle_time = 0

        if self._status == MachineStatus.RUNNING:
            idle_time = self._end_idle_time - self._begin_idle_time
        elif self._status == MachineStatus.IDLE:
            idle_time = time_step - self._begin_idle_time
        if idle_time < 0:
            raise ValueError('idle time is negative')
        return idle_time
    @property
    def id(self):
        return self._id
    @id.setter
    def id(self, id):
        self._id = id 
    @property
    def job(self):
        return self._job
    @job.setter
    def job(self,job):
        self._job = job  


class MachineList(DoublyLinkList):
    def __init__(self,machine_num) -> None:
        super().__init__()
        for id in range(1, machine_num+1):
            self.append(Machine(id))


