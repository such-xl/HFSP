import random
from .utils import Node
class Machine(Node):
    def __init__(self,id:int,actions:list,status:int,brain:dict) -> None:
        '''
            status: 0:break, 1:idle, 2:working
        '''
        super().__init__(None)
        self._id = id
        self._actions=actions
        self._status = status
        self._brain = brain


        self._job_id = 0               #该机器正常加工的job的id
        self._t_process = 0            #当前加工的工序需要的加工时间
        self._t_processed = 0          #目前已经加工当前工序的时间


    def sampleActiom(self,actions):
        return random.randint(0,len(actions)-1) 
    def show(self):
        print(f'机器{self.ID},状态{self._status}',end=' ')
        if self._status == 2:
            print(f'已经加工作业{self._jobID}的第{self._jobProcess}工序{self._T_processed}s,剩余{self._T_process-self._T_processed}')
        print()

    # 装载job
    def load_job(self,job_id,t_process):
        self._job_id = job_id
        self._t_process = t_process
        self._t_processed = 0
        self._status = 2
    # 运行一个时序
    def run_a_time_step(self):
        '''
            这里可添加机器故障
        '''
        self._t_processed += 1
        if self._t_processed == self._t_process: #该工序加工完成
            self._job_id = 0
            self._t_process = 0
            self._t_processed = 0
            self._status = 1
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
    def T_process(self, t_process):
        self._t_process = t_process
    @property
    def t_processed(self):
        return self._t_processed
    @t_processed.setter
    def T_processed(self, t_processed):
        self._T_processed = t_processed
    @property
    def job_id(self):
        return self._job_id
    @job_id.setter
    def job_id(self,job_id):
        self._job_id = job_id

