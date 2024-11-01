import random
from .utils import Node
class Machine(Node):
    def __init__(self,id:int,actions:list,status:int,brain:dict) -> None:
        '''
            status: 0:break, 1:idle, 2:working
        '''
        super().__init__(None)
        self._id = id
        self._actions = actions
        self._status = status
        self._brain = brain
        self._job = None              #该机器正在加工的job
        self._job_process = 0          #正在加工的工序
        self._t_process = 0            #当前加工的工序需要的加工时间
        self._t_processed = 0          #目前已经加工当前工序的时间
        self._bin_code = self.get_bin_code()
    def get_bin_code(self):
        binary_str = bin(self._id)[2:]
        binary_list = [int(digit) for digit in binary_str]
        return binary_list

    def get_state_encoding(self,lenth):
        return [0 for i in range(lenth-len(self._bin_code))] + self._bin_code
    def show(self):
        if self._status == 2:
            print(f'已经加工作业{self._job_id}的第{self._job_process}工序{self._t_processed}s,剩余{self._t_process-self._t_processed}')
        print()
    
    # 装载job
    def load_job(self,job):
        """把job装置至machine"""
        self._job = job
        self._status = 2
        self._t_processed = 0
        self._t_process = self._job.get_t_process(self._id)

    def unload_job(self):
        """卸载作业"""
        self._job.unload_machine()
        self._t_process = 0
        self._t_processed = 0
        self._job = None
    def run(self,min_run_timestep):
        """
            运行 'min_run_timestep' 时序，让环境产生空闲机器
        """
        self._t_processed += min_run_timestep
        self._job.run(min_run_timestep)
        if self._t_processed == self._t_process: #该工序加工完成
            self._t_processed = 0
            self._t_process = 0
            self._status = 1                     #将该机器的状态设置为空闲
    @property
    def id(self):
        return self._id
    @id.setter
    def id(self, id):
        self._id = id 
    @property
    def actions(self):
        return self._actions
    @actions.setter
    def actions(self, actions):
        self._actions = actions
    @property
    def status(self):
        return self._status
    @status.setter
    def status(self, status):
        self._status = status
    @property
    def brain(self):
        return self._brain
    @brain.setter
    def brain(self, brain):
        self._brain = brain
    @property
    def t_process(self):
        return self._t_process
    @t_process.setter
    def t_process(self, t_process):
        self._t_process = t_process
    @property
    def t_processed(self):
        return self._t_processed
    @t_processed.setter
    def t_processed(self, t_processed):
        self._t_processed = t_processed
    @property
    def job(self):
        return self._job
    @job.setter
    def job(self,job):
        self._job = job

