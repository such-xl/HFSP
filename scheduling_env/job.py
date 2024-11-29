from enum import Enum
from .utils import Node,DoublyLinkList
class JobStatus(Enum):
    COMPLETED = 0
    IDLE = 1
    RUNNING = 2
class Job(Node):
    code_len = 0
    def __init__(self,id:int,process_num:int,process_list:list,insert_time:int) -> None:
        super().__init__(None)
        self._id = id #job序号,从1开始
        self._process_num = process_num         #job工序数
        self._process_list = process_list       #job工序列表[{机器1:加工时间1,机器2:加工时间2},...{}]
        self._progress = 1                       # 加工进度 代表第progess道工序待加工，0 代表加工完成
        self._status = JobStatus.IDLE
        self._machine = None                   # 正在加工该job的机器id，0表示目前没有被加工
        self._t_process = 0                    # 当前工序需被加工的时间
        self._t_processed = 0                  # 当前工序已经被加工时间
        self._insert_time = insert_time        #进入环境的时间
        self._record = []                      #记录job加工过程
    def show(self):
        print(f'job {self._id} : {self._progress}/{self._process_num} status:{self._status} machine:{self._machine.id} t_process:{self._t_process} t_processed:{self._t_processed}',self._process_list[self._progress-1])


    def get_t_process(self, machine_id):
        """
            获取当前工序在机器machine上的加工时间
        """
        return self._process_list[self._progress-1][machine_id]

    
    def match_machine(self,machine_id) -> bool:
        """
            判断当前工序是否可以被机器machine_id加工
        """
        return machine_id in self._process_list[self._progress-1]


    def get_process_remaining_time(self):
        """
            获取当前工序剩余或所需加工时间
        """
        if self._status == JobStatus.RUNNING or self._status == JobStatus.IDLE:
            return self._t_process - self._t_processed 
        raise ValueError('job is not running')
    
    def load_to_machine(self,machine,time_step):
        """将job装载至machine"""
        if self._status != JobStatus.IDLE:
            raise ValueError('job is not idle')
        if self._status == JobStatus.COMPLETED:
            raise ValueError('job is completed')
        if self._machine:
            raise ValueError('job has machine')
        self._record.append([machine.id,time_step,time_step]) 
        self._t_process = self.get_t_process(machine.id)
        self._machine = machine
        self._t_processed = 0
        self._status = JobStatus.RUNNING

    def unload_machine(self):
        """将job从machine卸载"""
        if self._status != JobStatus.RUNNING:
            raise ValueError('job is not running')
        if not self._machine:
            raise ValueError('job has no machine')
        self._record[-1][-1] += self._t_processed
        self._machine = None
        self._t_process = 0
        self._t_processed = 0
        self._progress += 1
        self._status = JobStatus.COMPLETED if self._progress == self._process_num + 1 else JobStatus.IDLE

    def is_completed(self):
        """判断是否所有工序都完成""" 
        return self._status == JobStatus.COMPLETED
    
    def is_wating_for_machine(self):
        """判断是否等待机器加工"""
        return self._status == JobStatus.IDLE
    
    def is_on_processing(self):
        """判断是否正在加工"""
        return self._status == JobStatus.RUNNING
    
    def run(self,min_run_timestep):
        """ 执行min_run_timestep 时序"""
        self._t_processed += min_run_timestep

        if min_run_timestep<=0:
            raise ValueError('min_run_timestep must be greater than 0')
        
        if self._t_processed > self._t_process:
            raise ValueError('加工时间超过了工序时间',self._t_processed,self._t_process,min_run_timestep,self._status,self._machine)
        
        if self._t_processed == self._t_process:        #当前工序加工完成
            self.unload_machine()
    def current_progress_remaining_time(self):
        """获取当前工序剩余加工时间"""
        if self._status == JobStatus.COMPLETED:
            raise ValueError('job is completed')
        reminder = 0
        if self._status == JobStatus.IDLE:
            reminder = min(self._process_list[self._progress-1].values())
        elif self._status == JobStatus.RUNNING:
            reminder = self._t_process - self._t_processed
        if reminder <= 0:
            raise ValueError('reminder is negative or zero')
        return reminder    
      

    #获取job state 编码
    def get_state_encoding(self,machine_nums):
                         
        """job state:[当前工序加工时间,当前工序绝对延时,当前工序的加工信息编码，下一道工序的加工信息编码]"""
        job_state = [self._t_processed,self.get_process_remaining_time()]
        cp_dict = self._process_list[self._progress-1] #当前工序加工信息dict

        p1 = [cp_dict.get(i+1,0) if self._status == JobStatus.IDLE else cp_dict.get(self.machine.id) if i+1 == self.machine.id else 0  for i in range(machine_nums)]
        p2 = [self._process_list[self._progress].get(i+1,0) if self._progress < self._process_num else 0 for i in range(machine_nums)]
        job_state += p1 + p2
        return job_state
    
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
    def progress(self):
        return self._progress
    @property
    def status(self):
        return self._status
    @property
    def machine(self):
        return self._machine
    @property
    def t_process(self):
        return self._t_process
    @property
    def t_processed(self):
        return self._t_processed
    @property
    def insert_time(self):
        return  self._insert_time


class JobList(DoublyLinkList):
    def __init__(self) -> None:
        super().__init__() 
    # 从文件中获取作业信息
    def fetch_jobs_from_file(self,path:str):
        machine_num: int = 0
        with open(path,'r') as f:

            _,machine_num = map(int,f.readline().split()[0:-1])

            for job_id, line_str in enumerate(f, start=1):   
                        
                line = list(map(int, line_str.split()))

                i,r = 1,1

                procs: list[dict[int,int]] = []         # 工序列表
                while i < len(line):
                    #s = f'工序{r} '                                                                                                                                                                 
                    proc: dict[int,int] = {}            # 单个工序
                    for j in range(line[i]):
                        #s +=f'机器{line[i+1+j*2]} 耗时{line[i+1+j*2+1]} || '
                        proc[line[i+1+j*2]]=line[i+1+j*2+1]

                    procs.append(proc)
                    r += 1
                    i  += (1+line[i]*2)
                self.append(Job(id=job_id,process_num=r-1,process_list=procs,insert_time=0))
        return machine_num
                

