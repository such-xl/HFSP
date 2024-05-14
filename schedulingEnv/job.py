from utils import Node
class Job(Node):
    def __init__(self,id:int,process_num:int,process_list:list) -> None:
        super().__init__(None)
        self._id = id #job序号,从1开始
        self._process_num = process_num         #job工序数
        self._process_list = process_list       #job工序列表[{机器1:加工时间1,机器2:加工时间2},...{}]
        self._progess = 1                       # 加工进度 代表第progess道工序待加工，0 代表加工完成
        self._machine_id = 0                     # 正在加工该job的机器id，0表示目前没有被加工
        self._T_process = -1                    # 当前工序需被加工的时间
        self._T_processed = -1                  # 当前工序已经被加工时间
    def show(self):
        print(f'作业{self._id} 工序数{self._process_num}')
        for i,p in enumerate(self._process_list,start=1):
            print(f'工序{i}')
            print(p)
 
    def is_complete(self) -> bool:
        return self._progess == 0 

    def matchMachine(self, machineID):
        if self._machineID!=-1 or self.isAccomplished(): # 该job已经完成
            return []
        # return self.process_list.matchMachine(machineID)
        k  = self.process_list.matchMachine(machineID)
        if k == False:
            print(f'job{self._id} --- status{self._status}')
            input()
        return k
    def getTProcess(self,machineID):
        return self.process_list.matchMachine(machineID)
    def load(self,machineID):
        self._machine_id = machineID
        self._T_process = self.getTProcess(machineID)
        self._T_processed = 0
    def processingOneStep(self):
        self._T_processed += 1
        #print(f'job:{self._T_process,self._T_processed}')
        if self._T_processed == self._T_process:
            self._T_processed = -1
            self._T_process = -1
            self._machine_id = -1
            self._process_list.updateStatus()
            self._status = 1
    @property
    def id(self):
        return self._id
    @property
    def process_num(self):
        return self.process_num
    @property
    def process_list(self):
        return self._process_list
    @property
    def status(self):
        return self._status
    @property
    def machine_id(self):
        return self._machine_id