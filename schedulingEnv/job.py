class Job():
    def __init__(self,id:int,process_num:int,process_list:list) -> None:
        self._id = id #job序号,从1开始
        self._process_num = process_num         #job工序数
        self._process_list = process_list       #job工序列表[{机器1:加工时间1,机器2:加工时间2},...{}]
        #self._status = 0                       # 0 未完成, 1 已完成
        self._progess = 1                       # 加工进度 代表第progess道工序待加工，0 代表加工完成
        self._machine_id = 0                     # 正在加工该job的机器id，0表示目前没有被加工
        self._T_process = -1                    # 当前工序需被加工的时间
        self._T_processed = -1                  # 当前工序已经被加工时间
        self._next = None                       # 指针域
    def show(self):
        print(f'作业{self._id} 工序数{self._process_num}')
        # for i in range(len(self._process_list)):
        #     s = f'工序{i+1} '
        #     # for j in self._process_list[i].info:
        #     # #     s += f'机器{j[0]} || 耗时{j[1]} || '
        #     # print(s)
        #     print(i)
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
    @property
    def next(self):
        return self._next
    @next.setter
    def next(self, next):
        self._next = next
   
class Process():
    def __init__(self,ordinal:int) -> None:
        self._ordinal = ordinal
        self._info = {}
    def add_info(self,key,value):
        self._info[key] = value
    def matchMachine(self, machineID):
        try:
            return self._info[machineID]
        except Exception as e:
            print(e)
            return -1
    @property
    def ordinal(self):
        return self._ordinal  
    @property
    def info(self):
        return self._info
class ProcessList():
    def __init__(self,jobID,status) -> None:
        self._processList = []
        self._jobID = jobID
        self._status = status # n:当前需要完成第n道工序
    def updateStatus(self):
        self._status += 1
    def add_process(self, process:Process):
        self._processList.append(process)
    def isCompleted(self):
        return self._status == len(self._processList)+1
    def matchMachine(self, machineID):

        return self._processList[self._status-1].matchMachine(machineID)
    @property
    def processList(self):
        return self._processList
    @property
    def jobID(self):
        return self._jobID    
class JobList():
    def __init__(self) -> None:
        self._jobList = []
        self._jobNum = 0
        self._machineNum = 0
    def decodeRawJobFlie(self,path:str):
        with open(path,'r') as f:
            job_n,machine_n = map(int,f.readline().split()[0:-1])
            self._jobNum = job_n
            self._machineNum = machine_n
            for job_i, line_str in enumerate(f, start=1):
                line = list(map(int, line_str.split()))
                #print(f'作业{job_i}')
                i,r = 1,1
                procs = ProcessList(job_i,1)
                while i < len(line):
                    #s = f'工序{r} '                                                                                                                                                                 
                    proc = Process(r)
                    for j in range(line[i]):
                        #s +=f'机器{line[i+1+j*2]} 耗时{line[i+1+j*2+1]} || '
                        proc.add_info(line[i+1+j*2],line[i+1+j*2+1])

                    procs.add_process(proc)
                    #print(s)
                    r += 1
                    i  += (1+line[i]*2)
 
                self._jobList.append(Job(job_i,r-1,procs))
    def getUncompletedJob(self):
        return [i for i in self._jobList if not i.isAccomplished()]
    def addJob(self, job:Job):
        self.jobList.append(job)
    def isDone(self):
        for i in self.jobList:
            if not i.isAccomplished():
                return False
        return True
    def getMachineAction(self,machineID):
        actions = [0]
        processingTime = [0]
        for job in self._jobList:
            print(f'jobID  {job.ID}---status  {job.status}')
            print(f'processStatus  {job.process_list._status} ------')
            if job.isAccomplished():
                continue
            processTime = job.matchMachine(machineID)
            if processTime == -1:
                continue
            actions.append(job.ID)
            processingTime.append(processTime)
        return actions, processingTime

    @property
    def jobList(self):
        return self._jobList
    @property
    def jobNum(self):
        return self._jobNum
    @property
    def machineNum(self):
        return self._machineNum
