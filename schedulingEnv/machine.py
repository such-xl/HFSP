import random
class Machine():
    def __init__(self,ID:int,actions:list,status:int,brain:dict) -> None:
        '''
            status: 0:break, 1:idle, 2:working
        '''
        self._ID = ID
        self._actions=actions
        self._status = status
        self._brain = brain


        self._jobID = -1
        self._jobProcess = -1
        self._T_process = -1
        self._T_processed = -1


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
    def ID(self):
        return self._ID
    @ID.setter
    def ID(self, ID):
        self._ID = ID
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
    def T_process(self):
        return self._T_process
    @T_process.setter
    def T_process(self, T_process):
        self._T_process = T_process
    @property
    def T_processed(self):
        return self._T_processed
    @T_processed.setter
    def T_processed(self, T_processed):
        self._T_processed = T_processed
    @property
    def jobID(self):
        return self._jobID

class MachineList():
    def __init__(self, machineNum) -> None:
        self._machineList = [Machine(i,[],1,{}) for i in range(1,machineNum+1)]
    @property
    def machineList(self):
        return self._machineList
