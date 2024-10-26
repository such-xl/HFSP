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
        self._job_id = 0               #该机器正在加工的job的id
        self._job_process = 0          #正在加工的工序
        self._t_process = 0            #当前加工的工序需要的加工时间
        self._t_processed = 0          #目前已经加工当前工序的时间
        self._encode = [self._id,self._status,0,0,0] #id,status,加工作业id,第几道工序,加工时间
        self.bin_code = self.get_bin_code()
    def get_bin_code(self):
        binary_str = bin(self._id)[2:]
        binary_str = binary_str.zfill(5)
        binary_list = [int(digit) for digit in binary_str]
        return binary_list
    #从动作空间中采样一个动作
    def sample_action(self,obs,act_jos,act_jos_id):
        if len(act_jos) == 1:
            return 0
        return random.randint(1,len(act_jos)-1)
    def get_machine_state(self):
        return  self._encode
    def show(self):
        # print(f'机器{self._id},状态{self._status}',end=' ')
        print(self._encode)
        if self._status == 2:
            print(f'已经加工作业{self._job_id}的第{self._job_process}工序{self._t_processed}s,剩余{self._t_process-self._t_processed}')
        print()

    # 装载job
    def load_job(self,job_id,t_process,job_process):
        self._job_id = job_id
        self._job_process = job_process
        self._t_process = t_process
        self._t_processed = 0
        self._status = 2
        self._encode[1] = 2
        self._encode[2] = job_id
        self._encode[3] = job_process
        self._encode[4] = 0
    # 运行一个时序
    def run_a_time_step(self):
        '''
            这里可添加机器故障
        '''
        self._t_processed += 1
        self._encode[4] += 1
        if self._t_processed == self._t_process: #该工序加工完成
            self._job_id = 0
            self._encode[2] = 0
            self._t_process = 0
            self._encode[3] = 0
            self._encode[4] = 0
            self._t_processed = 0
            self._status = 1                     #将该机器的状态设置为空闲
            self._encode[1] = 1
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
    def t_processed(self, t_processed):
        self._t_processed = t_processed
    @property
    def job_id(self):
        return self._job_id
    @job_id.setter
    def job_id(self,job_id):
        self._job_id = job_id

