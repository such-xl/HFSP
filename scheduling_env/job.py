from .utils import Node
class Job(Node):
    def __init__(self,id:int,process_num:int,process_list:list,encode:list,insert_time:int) -> None:
        super().__init__(None)
        self._id = id #job序号,从1开始
        self._process_num = process_num         #job工序数
        self._process_list = process_list       #job工序列表[{机器1:加工时间1,机器2:加工时间2},...{}]
        self._progress = 1                       # 加工进度 代表第progess道工序待加工，0 代表加工完成
        self._status = 0                        # 0 已完成   1 加工中  2待加工
        self._machine_id = 0                     # 正在加工该job的机器id，0表示目前没有被加工
        self._t_process = 0                    # 当前工序需被加工的时间
        self._t_processed = 0                  # 当前工序已经被加工时间
        self._encode = [self._id,1,0,0]+encode              #[job_id,带加工工序或正在加工工序，加工机器，加工时间,xxx...x] 编码
        self._insert_time = insert_time        #进入环境的时间
        self._earliest_start_time = self._insert_time  #当前工序的实际最早开始时间
        self._pest = self.get_pest()            #获取每道工序全局理论最早结束时间
    def get_pest(self):
        pest = [self._insert_time]
        ct = self._insert_time
        for p in self._process_list:
            pt = 1e9
            for t in p.values():
                pt = min(pt,t)
            ct += pt
            pest.append(ct)
        return pest
    def show(self):
        # print(f'作业{self._id} 工序数{self._process_num}')
        print(len(self._encode))
        # for i,p in enumerate(self._process_list,start=1):
        #     print(f'工序{i}')
        #     print(p)

    #获取当前工序最早完成时间 
    def get_earliest_end_time(self):
        min_t = 1e9 
        for t in self._process_list[self._progress-1].values():
           min_t = min(t,min_t)
           
        return self._earliest_start_time + min_t
    

    def get_t_process(self, machine_id):
        return self._process_list[self._progress-1][machine_id]
    
    # 判断当前工序是否可被agent_i执行
    def match_machine(self,machine_id) -> bool:
        # if machine_id == 2:
        #     print("-----")
        #     print(self._process_list[self._progress-1])
        #     print(machine_id in self._process_list[self._progress-1])
        #     print("-----")
        return machine_id in self._process_list[self._progress-1]
    
    # 将job装载至machine
    def load_to_machine(self,machine_id):
        self._machine_id = machine_id
        self._t_process = self.get_t_process(machine_id)
        self._encode[-2] = machine_id
        self._t_processed = 0
        self._status = 1
    #加工一个时序
    def run_a_time_step(self):
        self._t_processed += 1
        self._encode[-1] += 1
        if self._t_processed == self._t_process:        #当前工序加工完成
            self._t_processed = 0
            self._t_process = 0
            self._machine_id = 0
            self._progress +=1
            self._encode[-3] +=1
            self._encode[-2] = 0
            self._encode[-1] = 0
            if self._progress == self._process_num+1:    # 最后一道工序加工完成
                self._progress = 0
                self._encode[-3] = 0
                self._status = 0
            else:
                self._status = 2
    #获取job state 编码
    def get_job_state(self):
        ''''''
        job_state = [self._id,self._progress,self._machine_id,self._t_processed,0]
        for p in self._process_list[self._progress-1:self._progress-1+3]:
            for key, value in p.items():
                job_state.extend([key,value])
            job_state.append(0)
        if len(job_state)<72:
            job_state.extend([0]*(72-len(job_state)))
        return job_state
    def earliest_start_time_update(self,timestep):
        self._earliest_start_time = timestep
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
    def machine_id(self):
        return self._machine_id
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
    