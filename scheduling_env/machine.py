import random
from utils import Node
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


        self._job_id = -1               #该机器正常加工的job的id
        self._t_process = -1            #当前加工的工序需要的加工时间
        self._t_processed = -1          #目前已经加工当前工序的时间


    def sampleActiom(self,actions):
        return random.randint(0,len(actions)-1) 
    def show(self):
        print(f'机器{self.ID},状态{self._status}',end=' ')
        if self._status == 2:
            print(f'已经加工作业{self._jobID}的第{self._jobProcess}工序{self._T_processed}s,剩余{self._T_process-self._T_processed}')
        print()
    def isIdle(self):
        return self.status == 1
    def isBusy(self):
        return self.status == 2
     
    def processingOneStep(self):
        self._T_processed += 1 
        #print(f'machine:{self._T_process,self._T_processed}')
        if self._T_processed == self._T_process:
            self._jobID = -1
            self._jobProcess = -1
            self._T_processed = -1
            self._T_process = -1
            self._status = 1
            return True 
        return False
    def execute(self,jobID,jobProcess,T_process):
        self._jobID = jobID
        self._jobProcess = jobProcess
        self._T_process = T_process
        self._T_processed = 0
        self._status = 2
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

