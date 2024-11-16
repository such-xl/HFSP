from .utils import Node
class Job(Node):
    code_len = 0
    def __init__(self,id:int,process_num:int,process_list:list,insert_time:int) -> None:
        super().__init__(None)
        self._id = id #job序号,从1开始
        self._process_num = process_num         #job工序数
        self._process_list = process_list       #job工序列表[{机器1:加工时间1,机器2:加工时间2},...{}]
        self._progress = 1                       # 加工进度 代表第progess道工序待加工，0 代表加工完成
        self._status = 2                        # 0 已完成   1 加工中  2待加工
        self._machine = None                    # 正在加工该job的机器id，0表示目前没有被加工
        self._t_process = 0                    # 当前工序需被加工的时间
        self._t_processed = 0                  # 当前工序已经被加工时间
        #self._state = self.get_job_encoding()  # [当前工序加工时间,当前工序相对延时，当前工序绝对延时,当前工序的加工信息编码，下一道工序的加工信息编码] 
        self._insert_time = insert_time        #进入环境的时间
        self._pest = self._insert_time          #当前工序的实际最早开始时间
        self._prst = 0                          #当前工序实际开始时间
        self._pests = self.get_pests()            #获取每道工序全局理论最早开始时间
    def get_pests(self):
        pests = [self._insert_time]
        ct = self._insert_time
        for p in self._process_list[:-1]:
            pt = 1e9
            for t in p.values():
                pt = min(pt,t)
            ct += pt
            pests.append(ct)
        return pests

    #获取当前工序最早完成时间 
    def get_earliest_end_time(self):
        min_t = 1e9 
        for t in self._process_list[self._progress-1].values():
           min_t = min(t,min_t)
           
        return self._earliest_start_time + min_t
    

    def get_t_process(self, machine_id):
        try:
            return self._process_list[self._progress-1][machine_id]
        except:
            return False
    
    def match_machine(self,machine_id) -> bool:
        """
            判断当前工序是否可以被机器machine_id加工
        """
        try:
            return machine_id in self._process_list[self._progress-1]
        except :
            return False

    def get_process_remaining_time(self):
        """
            获取当前工序剩余加工时间
        """
        if self._status != 1:
            raise ValueError('当前工序未在加工中')

        return self._t_process - self._t_processed 
    
    def load_to_machine(self,machine,time_step):
        """将job装载至machine"""
        self._machine = machine
        self._t_process = self.get_t_process(machine.id)
        self._t_processed = 0
        self._status = 1
        self._prst = time_step

    def unload_machine(self):
        self._machine = None
    def is_completed(self):
        """
            判断是否所有工序都完成

        """ 
        return self._status == 0
    def is_wating_for_machine(self):
        return self._status == 2
    def run(self,min_run_timestep):
        """
            执行min_run_timestep 时序
        """
        self._t_processed += min_run_timestep

        if self._t_processed == self._t_process:        #当前工序加工完成
            self._t_processed = 0
            self._t_process = 0
            self._progress +=1
            if self._progress == self._process_num+1:    # 最后一道工序加工完成
                self._status = 0
                #self._pest += self._t_process  # 更新实际当前工序的最早开始时间
            else:
                self._status = 2
    def update_pest(self,time_step):
        self._pest = time_step
    
    def update_prst(self,time_step):
        self._prst = time_step
    def get_latency(self):
       return self._prst-self._pests[self._progress-1]
    def get_relative_delay(self):
        return self._prst-self._pest
    def get_delay_ratio(self):
        return (self._prst-self._pest)/(self._prst-self._pest+self._t_process)

    def current_progress_need_time(self):
        if self._status == 2:
            p = self._process_list[self._progress-1]
            return min(p.values())
        return self._t_process - self._t_processed
            

    #获取job state 编码
    def get_state_encoding(self,machine_nums):
                         
        """job state:[当前工序加工时间,当前工序绝对延时,当前工序的加工信息编码，下一道工序的加工信息编码]"""

        job_state = [self._t_processed,self._prst-self._pest]
        cp_dict = self._process_list[self._progress-1] #当前工序加工信息dict
        p1 = [cp_dict.get(i+1,0) if self._status == 2 else cp_dict.get(self.machine.id) if i+1 == self.machine.id else 0  for i in range(machine_nums)]
        p2 = [self._process_list[self._progress].get(i+1,0) if self._progress < self._process_num else 0 for i in range(machine_nums)]
        job_state += p1+p2
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
    def encode(self):
        return self._encode
    @property
    def insert_time(self):
        return  self._insert_time
    @property
    def earliest_start_time(self):
        return self._earliest_start_time
    @property
    def pest(self):
        return self._pest
    @property
    def pest(self):
        return self._pests
    